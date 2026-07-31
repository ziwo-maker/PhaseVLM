import argparse
import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from exp_few_shot_forecasting import Exp_Few_Shot_Forecast
from src.TimeVLM.model import Model


def _str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "y")


class WindowedCSVDataset(Dataset):
    def __init__(self, data, seq_len, pred_len, label_len=0):
        if label_len > seq_len:
            raise ValueError("label_len must be <= seq_len")
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.max_start = len(self.data) - (seq_len + pred_len) + 1

    def __len__(self):
        return max(0, self.max_start)

    def __getitem__(self, idx):
        start = idx
        mid = start + self.seq_len
        end = mid + self.pred_len
        x = self.data[start:mid]
        y = self.data[mid - self.label_len:end]
        return x, y


def _parse_float(value):
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def load_csv_series(csv_path, time_col="date", id_col="intersection_id"):
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise ValueError(f"No header found in CSV: {csv_path}")
        time_col = time_col if time_col in fieldnames else None
        id_col = id_col if id_col in fieldnames else None
        feature_cols = [c for c in fieldnames if c not in {time_col, id_col}]
        rows = list(reader)

    if not feature_cols:
        raise ValueError("No numeric feature columns found in CSV.")

    if id_col:
        time_index = {}
        times = []
        data_by_time = []
        ids = []
        for row in rows:
            time_key = row[time_col] if time_col else str(len(times))
            if time_key not in time_index:
                time_index[time_key] = len(times)
                times.append(time_key)
                data_by_time.append({})
            idx = time_index[time_key]
            inter = row[id_col]
            if inter not in ids:
                ids.append(inter)
            values = [_parse_float(row[c]) for c in feature_cols]
            data_by_time[idx][inter] = values

        data = []
        for id_map in data_by_time:
            row_values = []
            for inter in ids:
                values = id_map.get(inter)
                if values is None:
                    row_values.extend([0.0] * len(feature_cols))
                else:
                    row_values.extend(values)
            data.append(row_values)
    else:
        data = [[_parse_float(row[c]) for c in feature_cols] for row in rows]

    array = np.array(data, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return array


def split_series(data, seq_len, pred_len, train_ratio=0.7, val_ratio=0.1):
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]
    min_len = seq_len + pred_len
    if len(train) < min_len:
        raise ValueError("Training split is too small for seq_len + pred_len.")
    if len(val) < min_len:
        raise ValueError("Validation split is too small for seq_len + pred_len.")
    if len(test) < min_len:
        raise ValueError("Test split is too small for seq_len + pred_len.")
    return train, val, test


def evaluate(model, loader, device, pred_len, use_amp):
    model.eval()
    criterion = torch.nn.MSELoss()
    losses = []
    use_amp = use_amp and device.type == "cuda"
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(batch_x)
                target = batch_y[:, -pred_len:, :]
                loss = criterion(outputs, target)
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("inf")


def train_from_csv(args):
    csv_path = args.csv_path or os.path.join(args.root_path, args.data_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    data = load_csv_series(csv_path, time_col=args.time_col, id_col=args.id_col)
    feature_dim = data.shape[1]
    if args.enc_in != feature_dim:
        print(f"Adjusting enc_in/dec_in/c_out to {feature_dim} based on CSV features.")
        args.enc_in = feature_dim
        args.dec_in = feature_dim
        args.c_out = feature_dim

    train_data, val_data, test_data = split_series(
        data,
        args.seq_len,
        args.pred_len,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    if args.percent < 1.0:
        keep = max(int(len(train_data) * args.percent), args.seq_len + args.pred_len)
        train_data = train_data[:keep]

    train_set = WindowedCSVDataset(train_data, args.seq_len, args.pred_len, args.label_len)
    val_set = WindowedCSVDataset(val_data, args.seq_len, args.pred_len, args.label_len)
    test_set = WindowedCSVDataset(test_data, args.seq_len, args.pred_len, args.label_len)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.use_gpu,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=args.use_gpu,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=args.use_gpu,
    )

    if args.patch_len > args.seq_len:
        args.patch_len = args.seq_len
    if args.padding is None:
        args.padding = args.stride

    device = torch.device(f"cuda:{args.gpu}" if args.use_gpu and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu)

    model = Model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss()
    # scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp and device.type == "cuda")

    best_val = float("inf")
    patience_count = 0
    checkpoint_dir = os.path.join(args.checkpoints, f"{args.model_id}_{args.des}_{args.itr}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")

    for epoch in range(args.train_epochs):
        model.train()
        epoch_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.use_amp and device.type == "cuda"):
                outputs = model(batch_x)
                target = batch_y[:, -args.pred_len:, :]
                loss = criterion(outputs, target)
            # if scaler.is_enabled():
            #     scaler.scale(loss).backward()
            #     scaler.step(optimizer)
            #     scaler.update()
            # else:
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        val_loss = evaluate(model, val_loader, device, args.pred_len, args.use_amp)
        print(f"Epoch {epoch + 1}/{args.train_epochs} - train: {train_loss:.6f} - val: {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            patience_count = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print("Early stopping triggered.")
                break

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_loss = evaluate(model, test_loader, device, args.pred_len, args.use_amp)
    print(f"Test loss: {test_loss:.6f}")


def _build_setting(args, itr):
    return "{}_{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_fs{}_{}".format(
        args.task_name,
        args.vlm_type,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.percent,
        itr,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="few_shot_forecast")
    parser.add_argument("--is_training", type=int, default=1)
    parser.add_argument("--root_path", type=str, default="./data")
    parser.add_argument("--data_path", type=str, default="traffic_counts_2s.csv")
    parser.add_argument("--csv_path", type=str, default="")
    parser.add_argument("--model_id", type=str, default="Traffic_512_96")
    parser.add_argument("--model", type=str, default="TimeVLM")
    parser.add_argument("--data", type=str, default="custom")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--seq_len", type=int, default=25)
    parser.add_argument("--label_len", type=int, default=5)
    parser.add_argument("--pred_len", type=int, default=5)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--factor", type=int, default=3)
    parser.add_argument("--enc_in", type=int, default=862)
    parser.add_argument("--dec_in", type=int, default=862)
    parser.add_argument("--c_out", type=int, default=862)
    parser.add_argument("--des", type=str, default="Exp")
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=56)
    parser.add_argument("--norm_const", type=float, default=0.4)
    parser.add_argument("--periodicity", type=int, default=24)
    parser.add_argument("--three_channel_image", type=_str2bool, default=True)
    parser.add_argument("--finetune_vlm", type=_str2bool, default=False)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.0005)

    parser.add_argument(
        "--vlm_device_map",
        type=str,
        default="",
        help="Transformers device_map for VLM (e.g., auto). Empty disables sharding.",
    )
    parser.add_argument(
        "--vlm_max_memory",
        type=str,
        default="",
        help="Max memory for VLM sharding (e.g., '0:20GiB,1:20GiB,cpu:32GiB' or '20GiB').",
    )
    parser.add_argument("--use_mem_gate", type=_str2bool, default=False)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--percent", type=float, default=0.1)
    parser.add_argument("--stride", type=int, default=30)


    parser.add_argument("--patch_memory_size", type=int, default=100)
    parser.add_argument("--learnable_image", type=_str2bool, default=False)
    parser.add_argument("--save_images", type=_str2bool, default=False)
    parser.add_argument("--use_gpu", type=_str2bool, default=True)
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--checkpoints", type=str, default="./checkpoints")
    parser.add_argument("--time_col", type=str, default="date")
    parser.add_argument("--id_col", type=str, default="intersection_id")
    parser.add_argument("--inverse", action="store_true", default=False)
    parser.add_argument("--use_dtw", type=_str2bool, default=False)
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')  # 时间特征编码方式
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')  # 时间特征频率

    # parser.add_argument('--target_data', type=str, default='ETTh2', help='target dataset type')  # 目标数据集类型
    # parser.add_argument('--target_root_path', type=str, default='./data/ETT/', help='root path of the target data file')  # 目标数据根目录
    # parser.add_argument('--target_data_path', type=str, default='ETTh2.csv', help='target data file')  # 目标数据文件

    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')  # M4子集

    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')  # 目标特征
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")  # 数据增强次数
    parser.add_argument('--seed', type=int, default=2024, help="Randomization seed")  # 随机种子
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")  # jitter增强
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")  # scaling增强
    parser.add_argument('--permutation', default=False, action="store_true", help="Equal Length Permutation preset augmentation")  # 等长置换增强
    parser.add_argument('--randompermutation', default=False, action="store_true", help="Random Length Permutation preset augmentation")  # 随机长度置换增强
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")  # 幅度扭曲增强
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")  # 时间扭曲增强
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")  # 窗口切片增强
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")  # 窗口扭曲增强
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")  # 旋转增强
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")  # SPAWNER增强
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")  # DTW扭曲增强
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")  # ShapeDTW扭曲增强
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")  # WDBA增强
    parser.add_argument('--discdtw', default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")  # 判别式DTW扭曲
    parser.add_argument('--discsdtw', default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")  # 判别式ShapeDTW扭曲
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")  # 额外标记

    parser.add_argument('--llm_model', type=str, default='GPT2', help='LLM model')  # LLM模型类型，例如LLAMA/GPT2/BERT
    parser.add_argument('--llm_dim', type=int, default='768', help='LLM model dimension')  # LLM维度，例如LLama7b:4096; GPT2-small:768; BERT-base:768

    parser.add_argument('--padding', type=int, default=30, help='padding')  # padding大小
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')  # patch长度
    parser.add_argument('--llm_layers', type=int, default=1)  # LLM层数
    parser.add_argument('--prompt_domain', type=int, default=0, help='')  # prompt领域/域ID
    parser.add_argument('--align_const', type=float, default=0.4)  # 对齐常数

    parser.add_argument('--wo_ts', type=int, default=0, help='without/with Time Series Data 1/0')  # 是否不使用时序数据(1/0)
    
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')  # DataLoader进程数
    
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')  # 损失函数
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')  # 学习率调整策略
    parser.add_argument('--vlm_type', type=str, default='CLIP', help='VLM model type, e.g. CLIP, BLIP2, etc.')
    
    return parser.parse_args()

def main():
    args = parse_args()
    if args.csv_path == "":
        args.csv_path = None
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This run is configured to require GPU.")
    if not args.use_gpu:
        raise RuntimeError("This run requires GPU. Please run with --use_gpu True.")
    if args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    else:
        args.device_ids = [args.gpu]

    args.is_training = 1
    for ii in range(args.itr):
        exp = Exp_Few_Shot_Forecast(args)
        setting = _build_setting(args, ii)
        print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
        exp.train(setting)
        print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        exp.test(setting)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
