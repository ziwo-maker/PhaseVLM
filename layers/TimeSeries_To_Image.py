import einops
import torch
import torch.nn.functional as F
from pytorch_wavelets import DWTForward

def time_series_to_simple_image(x_enc, image_size, context_len, periodicity):
    """
    Convert time series data into 3-channel image tensors.
    
    Args:
        x_enc (torch.Tensor): Input time series data of shape [B, seq_len, nvars].
        image_size (int): Size of the output image (height and width).
        context_len (int): Length of the time series sequence.
        periodicity (int): Periodicity used to reshape the time series into 2D.
        
    Returns:
        torch.Tensor: Image tensors of shape [B, 3, H, W].
    """
    # 解析输入张量的三维语义：批大小、序列长度、变量数量
    B, seq_len, nvars = x_enc.shape  # 获取输入形状

    # 为了能够按周期重排成二维图像，先将序列长度补齐到 periodicity 的整数倍
    pad_left = 0
    if context_len % periodicity != 0:
        pad_left = periodicity - context_len % periodicity

    # 将时间序列维度放到最后一维，方便后续按周期切分
    x_enc = einops.rearrange(x_enc, 'b s n -> b n s')

    # 左侧复制填充，避免引入无意义的零值
    x_pad = F.pad(x_enc, (pad_left, 0), mode='replicate')
    
    # 按照 periodicity 切成二维块：频率维 f 与周期维 p
    x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=periodicity)
    
    # 双线性插值缩放到统一图像尺寸
    x_resized_2d = F.interpolate(x_2d, size=(image_size, image_size), mode='bilinear', align_corners=False)

    # 复制为 3 通道，形成伪 RGB 图像
    images = einops.repeat(x_resized_2d, 'b 1 h w -> b c h w', c=3)  # [B * nvars, 3, H, W]

    # 恢复批与变量维度，并在变量维度上取平均以融合多变量信息
    images = einops.rearrange(images, '(b n) c h w -> b n c h w', b=B, n=nvars)  # [B, nvars, 3, H, W]
    images = images.mean(dim=1)  # Average over nvars to get [B, 3, H, W]
    
    return images


def time_series_to_image_with_fft_and_wavelet(x_enc, image_size, context_len, periodicity):
    """
    Convert time series data into 3-channel image tensors using FFT and Wavelet transforms.
    
    Args:
        x_enc (torch.Tensor): Input time series data of shape [B, seq_len, nvars].
        image_size (int): Size of the output image (height and width).
        context_len (int): Length of the time series sequence.
        periodicity (int): Periodicity used to reshape the time series into 2D.
        
    Returns:
        torch.Tensor: Image tensors of shape [B, 3, H, W].
    """
    def _apply_fourier_transform(x_2d):
        """
        Apply Fourier transform to the input 2D time series data.
        """
        # 在最后一维进行 FFT，捕获周期维的频谱特征
        x_fft = torch.fft.fft(x_2d, dim=-1)
        # 取幅值作为频谱强度，丢弃相位信息
        x_fft_abs = torch.abs(x_fft)  # Take the magnitude part of the Fourier transform
        return x_fft_abs

    def _apply_wavelet_transform(x_2d):
        """
        Apply wavelet transform to the input 2D time series data.
        """
        # Haar 小波分解：J=1 表示一层分解
        dwt = DWTForward(J=1, wave='haar')
        # cA: Low-frequency components, cD: High-frequency components
        cA, cD = dwt(x_2d)  # [B * nvars, 1, f, p]
        # 高频分量列表取第一层，并去掉多余的通道维
        cD_reshaped = cD[0].squeeze(1)  # [B * nvars, 3, f, p]
        # 拼接低频与高频特征，形成更丰富的局部变化表示
        wavelet_result = torch.cat([cA, cD_reshaped], dim=1)  # [B * nvars, 4, f, p]
        # 在通道维上取平均，压缩回单通道表示
        wavelet_result = wavelet_result.mean(dim=1, keepdim=True)  # [B * nvars, 1, f, p]
        return wavelet_result
    
    # 解析输入张量的三维语义：批大小、序列长度、变量数量
    B, seq_len, nvars = x_enc.shape  # 获取输入形状

    # 为了能够按周期重排成二维图像，先将序列长度补齐到 periodicity 的整数倍
    pad_left = 0
    if context_len % periodicity != 0:
        pad_left = periodicity - context_len % periodicity

    # 将时间序列维度放到最后一维，方便后续按周期切分
    x_enc = einops.rearrange(x_enc, 'b s n -> b n s')

    # 左侧复制填充，避免引入无意义的零值
    x_pad = F.pad(x_enc, (pad_left, 0), mode='replicate')
    
    # 按照 periodicity 切成二维块：频率维 f 与周期维 p
    x_2d = einops.rearrange(x_pad, 'b n (p f) -> (b n) 1 f p', f=periodicity)
    
    # 双线性插值缩放到统一图像尺寸，作为时间域通道输入
    x_resized_2d = F.interpolate(x_2d, size=(image_size, image_size), mode='bilinear', align_corners=False)

    # 生成频域与小波域特征
    x_fft = _apply_fourier_transform(x_2d)
    x_wavelet = _apply_wavelet_transform(x_2d)
    # 将频域与小波域特征插值到统一尺寸
    x_resized_fft = F.interpolate(x_fft, size=(image_size, image_size), mode='bilinear', align_corners=False)
    x_resized_wavelet = F.interpolate(x_wavelet, size=(image_size, image_size), mode='bilinear', align_corners=False)
    # 拼接三个通道：时间域 + 频域 + 小波域
    images = torch.concat([x_resized_2d, x_resized_fft, x_resized_wavelet], dim=1)  # [B * nvars, 3, H, W]

    # 恢复批与变量维度，并在变量维度上取平均以融合多变量信息
    images = einops.rearrange(images, '(b n) c h w -> b n c h w', b=B, n=nvars)  # [B, nvars, 3, H, W]
    images = images.mean(dim=1)  # Average over nvars to get [B, 3, H, W]
    
    return images


import torch
import torch.nn.functional as F


def time_series_to_simple_image_hot(x_enc, image_size, context_len, periodicity):
    """
    将“短时窗口多车道时序”转换为 3 通道图像：
      - Channel 0: raw   （原始强度/热力图）
      - Channel 1: delta （时间差分，反映变化幅度）
      - Channel 2: trend （时间平滑后的趋势，反映慢变化）

    最终输出是一个 Time×Lane 的热力图（本实现里：height=Lane, width=Time），并插值到指定 image_size。

    Args:
        x_enc:
            - torch.Tensor, shape [B, T, D] 或 [T, D]
            - B: batch 大小
            - T: 时间长度（context window 内采样点数）
            - D: 变量/车道数（如 8 个车道/方向）
        image_size:
            - int：输出图像为 (image_size, image_size)
            - 或 (H, W)：输出图像为指定高宽
        context_len:
            - 期望的时间长度 T'
            - 若输入 T != T'：
                * T > T'：从左侧裁剪（保留最近的 T' 个时刻）
                * T < T'：左侧补齐（用第一个时刻重复 pad，避免补 0 引入“假信号”）
        periodicity:
            - 用来估计 trend 平滑窗口的“周期性尺度”（例如 60s 周期、2s 采样 => p=30）
            - 若为 None，则使用默认窗口 k=3

    Returns:
        img:
            torch.Tensor, shape [B, 3, H, W]
            dtype=float32，数值范围 [0, 1]
            其中 H,W 为 image_size 指定的输出大小
    """

    # =========================
    # 1) 输入清洗 / 形状统一
    # =========================

    # 如果不是 Tensor（比如 numpy/list），先转成 float32 Tensor
    if not torch.is_tensor(x_enc):
        x = torch.tensor(x_enc, dtype=torch.float32)
    else:
        x = x_enc

    # 如果是 [T, D]（单样本），补一个 batch 维度 => [1, T, D]
    if x.dim() == 2:
        x = x.unsqueeze(0)

    # 最终必须是 3 维：[B, T, D]
    if x.dim() != 3:
        raise ValueError(f"x_enc must be [B,T,D] or [T,D], got {tuple(x.shape)}")

    # 统一 dtype，避免后续 conv / interpolate 出现类型不一致
    x = x.to(dtype=torch.float32)

    # 读出维度语义
    B, T, D = x.shape

    # =========================
    # 2) 对齐时间长度到 context_len
    #    - 保留最近片段（右对齐）
    #    - 不足则在左侧重复首帧补齐
    # =========================
    if context_len is not None:
        context_len = int(context_len)

        if T > context_len:
            # T 太长：保留最近的 context_len 个点（右边那段）
            x = x[:, -context_len:, :]
            T = context_len

        elif T < context_len:
            # T 太短：左侧补齐 pad 个点
            pad = context_len - T

            # 用“第一个时间步”重复 pad 次作为左侧填充
            # 好处：避免补 0 造成突变（0 可能完全不符合真实量纲）
            pad_vals = x[:, :1, :].repeat(1, pad, 1)  # [B, pad, D]
            x = torch.cat([pad_vals, x], dim=1)       # [B, context_len, D]
            T = context_len

    # 数值稳定项，避免 log/除法等出现 NaN
    eps = 1e-6

    # =========================
    # 3) 一些工具函数：压缩尺度、鲁棒标准化、映射到 [0,1]
    # =========================

    def signed_log1p(z: torch.Tensor) -> torch.Tensor:
        """
        带符号的 log(1+|z|)：
          - 对大值做压缩（减少极端值支配）
          - 允许 z 有负值（差分可能出现负）
        """
        return torch.sign(z) * torch.log1p(torch.abs(z))

    def robust_z(z: torch.Tensor, clip: float = 3.0) -> torch.Tensor:
        """
        鲁棒 z-score 标准化（按 sample 做）：
          1) 将 [B,T,D] 展平为 [B, T*D]
          2) 对每个样本用 median 作为中心
          3) 用 MAD (median absolute deviation) 做尺度
          4) 转回原形状，并做 clip 防止极端值

        说明：
          - MAD 乘 1.4826 可在正态分布下近似等价于标准差
          - 相比 mean/std 对异常值更不敏感，更适合交通流这种可能尖峰的序列
        """
        flat = z.reshape(z.shape[0], -1)  # [B, T*D]
        med = flat.median(dim=1, keepdim=True).values  # [B,1]
        mad = (flat - med).abs().median(dim=1, keepdim=True).values + eps  # [B,1]
        z0 = ((flat - med) / (1.4826 * mad)).view_as(z).clamp(-clip, clip)  # [B,T,D]
        return z0

    def to_01(z: torch.Tensor, clip: float = 3.0) -> torch.Tensor:
        """
        将 [-clip, clip] 线性映射到 [0,1]：
            z=-clip -> 0
            z= 0    -> 0.5
            z= clip -> 1
        """
        return (z + clip) / (2.0 * clip)

    # =========================
    # 4) Channel 0: raw（原始强度热力图）
    #    流程：log 压缩 -> 鲁棒标准化 -> 映射到 [0,1]
    # =========================
    raw = to_01(robust_z(signed_log1p(x)))

    # =========================
    # 5) Channel 1: delta（时间差分）
    #    反映相邻时刻变化：x[t]-x[t-1]
    # =========================
    delta = torch.zeros_like(x)          # [B,T,D]
    delta[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]  # 从 t=1 开始做差分
    # t=0 的差分保持为 0（也可以考虑复制 t=1 的差分，这里选择 0 更简单）
    delta = to_01(robust_z(delta))

    # =========================
    # 6) Channel 2: trend（趋势 / 平滑）
    #    用 moving average 做时间平滑（每个车道独立平滑）
    # =========================

    # 选择平滑窗口 k（尽量是“适中”的奇数，便于 symmetric padding）
    if periodicity is None:
        # 不给周期信息：最小的平滑窗口
        k = 3
    else:
        # 给了周期：用周期的一半（或更小）作为平滑窗口上限
        p = max(1, int(periodicity))
        k = max(3, min(T, p // 2))

    # 确保 k 为奇数（这样 padding=k//2 才能对称对齐）
    if k % 2 == 0:
        k += 1

    # 再确保 k 不超过 T，且尽量保持 odd
    # - 如果 T 是奇数，k<=T 即可
    # - 如果 T 是偶数，T-1 是奇数，避免出现 k=T 且 T 偶数导致对齐不“居中”
    k = min(k, T if T % 2 == 1 else max(1, T - 1))

    # conv1d 期望输入形状 [N, C, L]
    # 我们希望对“每个车道(变量)”独立做时间平滑：
    #   - 把 x 从 [B,T,D] 转为 [B,D,T]，让 D 变成通道 C
    x_bt = x.permute(0, 2, 1)  # [B, D, T]

    # 构造 moving average 卷积核：
    #   - groups=D => 每个车道用自己的 1 个通道独立卷积（互不混合）
    #   - weight 形状 [out_channels, in_channels/groups, kernel_size]
    #   - 这里 out_channels=D, in_channels/groups=1，所以 weight 为 [D,1,k]
    w = torch.ones(D, 1, k, device=x.device, dtype=x.dtype) / float(k)

    # padding=k//2 做“近似 same”长度输出（注意：边界会受 padding 影响）
    trend = F.conv1d(x_bt, w, padding=k // 2, groups=D)  # [B, D, T]
    trend = trend.permute(0, 2, 1)                       # [B, T, D]

    # 同样做鲁棒标准化并映射到 [0,1]
    trend = to_01(robust_z(trend))

    # =========================
    # 7) 打包成图像张量 [B, 3, H, W]
    #    这里先得到 [B,3,D,T]，再插值到目标大小
    # =========================

    # 为了让“竖轴=Lane，横轴=Time”，我们把 [B,T,D] 转为 [B,D,T]
    # 然后把三个通道 stack 在 dim=1
    img = torch.stack(
        [
            raw.permute(0, 2, 1),    # [B, D, T]
            delta.permute(0, 2, 1),  # [B, D, T]
            trend.permute(0, 2, 1),  # [B, D, T]
        ],
        dim=1
    )  # [B, 3, D, T]

    # 解析目标输出分辨率
    out_hw = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)

    # 双线性插值 resize 到 [B,3,H,W]
    # align_corners=False 是更常用、更稳定的设置
    img = F.interpolate(img, size=out_hw, mode="bilinear", align_corners=False)

    # 最后保险起见 clamp 到 [0,1]
    img = img.clamp(0.0, 1.0)

    return img
