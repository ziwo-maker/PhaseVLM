from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.dtw_metric import dtw,accelerated_dtw
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np


warnings.filterwarnings('ignore')  # 关闭非关键告警，保持日志简洁

class Exp_Few_Shot_Forecast(Exp_Basic):
    """Few-shot 预测实验：数据加载、训练、验证与测试流程封装。"""
    def __init__(self, args):
        super(Exp_Few_Shot_Forecast, self).__init__(args)

    def _build_model(self):
        # 按 args.model 从字典中构建模型
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            device_map = getattr(self.args, "vlm_device_map", None)
            if isinstance(device_map, str) and device_map.strip().lower() in ("", "none", "null"):
                device_map = None
            if device_map is not None:
                print("VLM device_map is enabled; skipping DataParallel to avoid double sharding.")
            else:
                # 多卡训练：使用 DataParallel 包装
                model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        # flag: 'train'/'val'/'test'
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # Adam 优化器
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # 训练与评估使用 MSE
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        """验证流程：前向推理 + loss 统计。"""
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                # 输入与时间特征移动到设备
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder 输入：label_len 为已知标签，其余 pred_len 置零
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # 编码器-解码器前向
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                # 只取预测窗口与指定特征维度
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # 在 CPU 上算 loss（更稳定且不占显存）
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        """训练主循环：含早停、学习率调整与 best checkpoint 载入。"""
        # 加载训练/验证/测试数据（few-shot 任务仍使用统一的数据接口）
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # 训练检查点目录：每个 setting 单独存放
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # 计时相关变量，用于打印速度和剩余时间
        time_now = time.time()

        # 一个 epoch 内的 step 数，用于进度与时间估计
        train_steps = len(train_loader)
        # 早停策略：以验证集 loss 为准，内部保存最优模型
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # 选择优化器与损失函数
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            # 混合精度缩放器（避免梯度下溢）
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            # epoch 内部统计量
            iter_count = 0
            train_loss = []

            # 切换为训练模式（启用 Dropout/BN 统计）
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                # 清空上一轮梯度
                model_optim.zero_grad()
                # 数据搬移到 GPU/CPU 设备
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder 输入：历史标签(label_len) + 未来预测(pred_len)的零占位
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # 编码器-解码器前向 + loss 计算（支持 AMP）
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        # 仅保留预测窗口；多变量时只取最后一维或全部
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        # 以 MSE 为训练目标
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    # 同 AMP 分支一致的输出裁剪与 loss 计算
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    # 打印训练速度与剩余时间估计
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    # 混合精度反向传播与参数更新
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    # 标准反向传播与参数更新
                    loss.backward()
                    model_optim.step()

            # 每个 epoch 的训练耗时
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            # 统计本 epoch 平均训练损失
            train_loss = np.average(train_loss)
            # 验证与测试集 loss（不参与反向传播）
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            # 早停判断（内部保存最优模型）
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # 动态学习率调整（按 epoch）
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        # 载入最优 checkpoint 继续后续测试/推理
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        # 兼容 Subset：获取原始 dataset 的缩放器
        if isinstance(test_data, torch.utils.data.Subset):
            data_scaling = test_data.dataset.scale
        else:
            data_scaling = test_data.scale
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location=self.device))

        preds = []
        trues = []
        # 可视化结果目录
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                # 输入与时间特征
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder 输入：label_len + pred_len(零)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # 前向推理
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                # 转回 numpy，便于后处理与保存
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                # 需要反归一化时做 inverse_transform
                if data_scaling and self.args.inverse:
                    shape = outputs.shape
                    if isinstance(test_data, torch.utils.data.Subset):
                        outputs = test_data.dataset.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                        batch_y = test_data.dataset.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    else:
                        outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                        batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    # 采样可视化预测曲线（使用最后一个变量）
                    input = batch_x.detach().cpu().numpy()
                    if data_scaling and self.args.inverse:
                        shape = input.shape
                        if isinstance(test_data, torch.utils.data.Subset):
                            input = test_data.dataset.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                        else:
                            input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # 结果保存目录
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # DTW 计算（可选）
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1,1)
                y = trues[i].reshape(-1,1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'not calculated'

        # 常用评价指标
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse: {}, mae: {}, dtw: {}'.format(mse, mae, dtw))
        # 追加记录到文本文件
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse: {}, mae: {}, dtw: {}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        # 保存 numpy 结果文件

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
