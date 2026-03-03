from sqlite3 import Time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
import matplotlib.pyplot as plt
import numpy as np

class TimeSeriesVisualizer:
    """时序到图像转换的可视化工具"""
    
    @staticmethod
    def plot_feature_maps(features, title="Feature Maps"):
        """绘制中间层特征图"""
        if not isinstance(features, torch.Tensor):
            return
            
        # 转为 numpy 并做 0-1 归一化，便于显示
        features = features.detach().cpu().numpy()
        features = (features - features.min()) / (features.max() - features.min() + 1e-8)
        
        # 只可视化第一个 batch 的样本
        num_channels = min(4, features.shape[1])
        if num_channels == 1:
            plt.figure(figsize=(5, 5))
            plt.imshow(features[0, 0], cmap='viridis')
            plt.axis('on')
            plt.title('Channel 1')
        else:
            fig, axes = plt.subplots(1, num_channels, figsize=(15, 5))
            for i, ax in enumerate(axes):
                ax.imshow(features[0, i], cmap='viridis')
                ax.axis('off')
                ax.set_title(f'Channel {i+1}')
        plt.suptitle(title)
        plt.tight_layout()
        
        # 保存图片到本地
        import os
        os.makedirs('ts-images/ts-visualizer', exist_ok=True)
        filename = f"ts-images/ts-visualizer/{title.replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {filename}")
        plt.close()

    @staticmethod
    def plot_attention(attention_map, title="Attention Map"):
        """绘制注意力权重图"""
        if not isinstance(attention_map, torch.Tensor):
            return
            
        attention_map = attention_map.detach().cpu().numpy()
        plt.figure(figsize=(8, 8))
        plt.imshow(attention_map[0, 0], cmap='viridis')
        plt.colorbar()
        plt.title(title)
        plt.show()

class LearnableTimeSeriesToImage(nn.Module):
    """可学习的时序到图像转换模块"""
    
    def __init__(self, input_dim, hidden_dim, output_channels, image_size, periodicity):
        super(LearnableTimeSeriesToImage, self).__init__()
        # 基本超参数
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        self.image_size = image_size
        self.periodicity = periodicity

        # 1D 卷积：在时间维上提取局部模式
        self.conv1d = nn.Conv1d(in_channels=4, out_channels=hidden_dim, kernel_size=3, padding=1)

        # 2D 卷积：在 (变量维, 时间维) 上提取二维结构
        self.conv2d_1 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim // 2, kernel_size=3, padding=1)
        self.conv2d_2 = nn.Conv2d(in_channels=hidden_dim // 2, out_channels=output_channels, kernel_size=3, padding=1)


    def forward(self, x_enc):
        """将输入时序转为图像张量 [B, output_channels, H, W]"""
        # x_enc: [B, L, D]，B 批大小，L 序列长度，D 变量维度
        B, L, D = x_enc.shape
        
        # 生成周期性编码（sin/cos），用于注入时间位置信息
        time_steps = torch.arange(L, dtype=torch.float32).unsqueeze(0).repeat(B, 1).to(x_enc.device)
        periodicity_encoding = torch.cat([
            torch.sin(time_steps / self.periodicity * (2 * torch.pi)).unsqueeze(-1),
            torch.cos(time_steps / self.periodicity * (2 * torch.pi)).unsqueeze(-1)
        ], dim=-1)
        # 扩展到变量维，形状 [B, L, D, 2]
        periodicity_encoding = periodicity_encoding.unsqueeze(-2).repeat(1, 1, D, 1)
        
        # FFT 频域幅值编码，强调周期/频率成分
        x_fft = torch.fft.rfft(x_enc, dim=1)
        x_fft_mag = torch.abs(x_fft)
        # rfft 输出长度可能短于 L，补齐到 L 以便与时域对齐
        if x_fft_mag.shape[1] < L:
            pad = torch.zeros(B, L - x_fft_mag.shape[1], D, device=x_enc.device, dtype=x_fft_mag.dtype)
            x_fft_mag = torch.cat([x_fft_mag, pad], dim=1)
        x_fft_mag = x_fft_mag.unsqueeze(-1)  # [B, L, D, 1]

        # 拼接原始值 + 频域幅值 + 周期编码
        x_enc = x_enc.unsqueeze(-1)  # [B, L, D, 1]
        x_enc = torch.cat([x_enc, x_fft_mag, periodicity_encoding], dim=-1)  # [B, L, D, 4]

        # 调整维度以适配 1D 卷积：将变量维展平到 batch
        x_enc = x_enc.permute(0, 2, 3, 1)  # [B, D, 4, L]
        x_enc = x_enc.reshape(B * D, 4, L)  # [B*D, 4, L]
        x_enc = self.conv1d(x_enc)  # [B*D, hidden_dim, L]
        x_enc = x_enc.reshape(B, D, self.hidden_dim, L)  # [B, D, hidden_dim, L]

        # 2D 卷积：在 (D, L) 平面上提取结构特征
        x_enc = x_enc.permute(0, 2, 1, 3)  # [B, hidden_dim, D, L]
        x_enc = F.tanh(self.conv2d_1(x_enc))
        x_enc = F.tanh(self.conv2d_2(x_enc))
        
        # 双线性插值到目标图像尺寸
        x_enc = F.interpolate(x_enc, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        
        # 输出形状: [B, output_channels, H, W]
        return x_enc






import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableTimeSeriesToImage_hot(nn.Module):
    """
    可学习的“Time×Lane 热力图 + (raw/delta/trend) 三通道”图像生成模块

    输入:
        x_enc: [B, L, D]
            B: batch size
            L: 时间长度（例如 30s 窗口采样 1s -> 30；采样 2s -> 15）
            D: 车道数（例如 8）

    输出:
        img: [B, 3, H, W]
            3 通道图像（类似 RGB），H=W=image_size
            输出范围约在 [0,1]（方便接视觉模型/后续网络）
    """

    def __init__(
        self,
        input_dim: int,         # D：车道数
        hidden_dim: int,        # 中间特征维度
        image_size: int,        # 输出图像大小 H=W=image_size
        trend_kernel: int = 5,  # trend 平滑核大小（建议 3/5/7）
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.image_size = image_size

        # ============================================================
        # ① 生成 3 通道特征图：raw / delta / trend
        #    - raw   : 原始强度（建议 log1p 压缩长尾）
        #    - delta : 一阶差分（突出突变/边缘）
        #    - trend : 平滑趋势（突出慢变化）
        # ============================================================

        # trend 用“可学习平滑”：深度可分离 1D 卷积（每条车道独立一个平滑核）
        # 输入 [B, D, L] -> 输出 [B, D, L]
        k = int(trend_kernel)
        if k % 2 == 0:
            k += 1  # 确保是奇数，便于对称平滑
        self.trend_dw = nn.Conv1d(
            in_channels=input_dim,
            out_channels=input_dim,
            kernel_size=k,
            padding=k // 2,
            groups=input_dim,   # depthwise：每个车道单独一组卷积核
            bias=False
        )
        # 初始化为“均值滤波”(moving average)：训练开始像移动平均，之后可自己学更合适的平滑方式
        with torch.no_grad():
            self.trend_dw.weight.fill_(1.0 / k)

        # ============================================================
        # ② 让它“像你原始代码那样可训练”：
        #    先对每条车道做 1D 卷积（沿时间提局部模式），再做 2D 卷积（车道×时间提空间结构）
        # ============================================================

        # 1D 卷积：输入通道固定为 3（raw/delta/trend）
        self.conv1d = nn.Conv1d(
            in_channels=3,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1
        )

        # 2D 卷积：在 (车道 D, 时间 L) 的二维平面提取结构
        self.conv2d_1 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim // 2,
            kernel_size=3,
            padding=1
        )
        # 输出通道固定为 3（最终图像 3 通道）
        self.conv2d_2 = nn.Conv2d(
            in_channels=hidden_dim // 2,
            out_channels=3,
            kernel_size=3,
            padding=1
        )

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        """
        将输入时序转为图像张量 [B, 3, H, W]
        """
        # --------- 0) 形状检查 ----------
        if x_enc.dim() != 3:
            raise ValueError(f"x_enc 必须是 [B, L, D]，但收到 {tuple(x_enc.shape)}")
        B, L, D = x_enc.shape
        if D != self.input_dim:
            raise ValueError(f"input_dim={self.input_dim} 但输入 D={D}，请检查车道维是否一致")

        x = x_enc.float()
        eps = 1e-6

        # 强制该模块使用 float32 计算，避免 bf16 在部分设备/卷积核上不被支持。
        with torch.cuda.amp.autocast(enabled=False):
            # ============================================================
            # 1) 三通道特征构造（只保留 raw / delta / trend）
            # ============================================================

            # 1.1 raw：原始强度
            # 交通流量/队列通常非负且长尾，用 log1p 可以压缩动态范围，训练更稳
            raw = torch.log1p(torch.clamp(x, min=0.0))  # [B, L, D]

            # 1.2 delta：一阶差分（突出突变）
            # delta[t] = x[t] - x[t-1]，第 0 帧设为 0
            delta = torch.zeros_like(x)
            delta[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]  # [B, L, D]

            # 1.3 trend：平滑趋势（突出慢变化）
            # 这里用“可学习平滑”的 depthwise conv1d，初始化成移动平均
            trend = self.trend_dw(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, L, D]

            # 1.4 拼成 3 通道特征： [B, L, D, 3]
            feat = torch.stack([raw, delta, trend], dim=-1)

            # 1.5 轻量标准化（强烈建议）
            # 目的：让 raw/delta/trend 三个通道的尺度接近，避免某个通道“压死”其他通道
            # 在 (L, D) 上做均值方差归一化，保留通道维
            mean = feat.mean(dim=(1, 2), keepdim=True)
            std = feat.std(dim=(1, 2), keepdim=True) + eps
            feat = (feat - mean) / std  # [B, L, D, 3]

            # ============================================================
            # 2) 可训练网络：1D(时间) + 2D(车道×时间)
            # ============================================================

            # 2.1 适配 1D conv：把每条车道当成一条序列，卷积核共享
            # [B, L, D, 3] -> [B, D, 3, L] -> [B*D, 3, L]
            feat = feat.permute(0, 2, 3, 1).reshape(B * D, 3, L)

            # 1D 卷积提取时间局部模式
            feat = F.silu(self.conv1d(feat))  # [B*D, hidden_dim, L]

            # 还原回 batch 和车道维：
            # [B*D, hidden_dim, L] -> [B, D, hidden_dim, L]
            feat = feat.reshape(B, D, self.hidden_dim, L)

            # 2.2 适配 2D conv：把 (D, L) 当成二维平面
            # [B, D, hidden_dim, L] -> [B, hidden_dim, D, L]
            feat = feat.permute(0, 2, 1, 3)

            # 2D 卷积提取“车道之间 + 时间之间”的联合结构
            feat = F.silu(self.conv2d_1(feat))
            img = self.conv2d_2(feat)  # [B, 3, D, L]

            # ============================================================
            # 3) 插值到目标图像大小，并把值压到 [0,1] 范围
            # ============================================================

            # 3.1 双线性插值到 [B, 3, H, W]
            img = F.interpolate(
                img,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False
            )

            # 3.2 输出范围压缩到 [0,1]
            # 用 tanh 再线性映射，比 sigmoid 更不容易早期饱和
            img = torch.tanh(img)          # [-1, 1]
            img = (img + 1.0) * 0.5        # [0, 1]

            return img
