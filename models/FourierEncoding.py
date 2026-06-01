"""
随机Fourier特征映射
将低维空间坐标映射到高维Fourier空间，帮助KAN学习高频地质结构

参考: Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains", NeurIPS 2020
"""

import torch
import torch.nn as nn


class FourierEncoding(nn.Module):
    """随机Fourier特征映射

    将输入 x ∈ R^d 映射为:
        gamma(x) = [sin(2*pi*B*x), cos(2*pi*B*x)]

    其中 B ∈ R^(m x d) 是随机高斯矩阵，m 是频率数

    Args:
        input_dim: 输入维度
        num_frequencies: 频率数 (输出维度 = 2 * num_frequencies)
        scale: 高斯分布的标准差 (控制频率范围)
        learnable: B 是否可学习
    """

    def __init__(self, input_dim, num_frequencies=64, scale=10.0, learnable=False):
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.scale = scale

        # 随机频率矩阵 B ~ N(0, scale^2)
        B = torch.randn(input_dim, num_frequencies) * scale

        if learnable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer('B', B)

    @property
    def output_dim(self):
        """输出维度 = 2 * num_frequencies (sin + cos)"""
        return 2 * self.num_frequencies

    def forward(self, x):
        """
        Args:
            x: (batch, input_dim) 输入坐标

        Returns:
            features: (batch, 2 * num_frequencies) Fourier特征
        """
        # x @ B: (batch, input_dim) @ (input_dim, num_frequencies) -> (batch, num_frequencies)
        x_proj = 2 * torch.pi * x @ self.B

        # 拼接 sin 和 cos
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class PositionalEncoding(nn.Module):
    """正弦位置编码（用于Transformer风格的编码）

    gamma(p) = [sin(2^0 * pi * p), cos(2^0 * pi * p),
                sin(2^1 * pi * p), cos(2^1 * pi * p),
                ...
                sin(2^(L-1) * pi * p), cos(2^(L-1) * pi * p)]

    Args:
        input_dim: 输入维度
        num_frequencies: 频率数 L
    """

    def __init__(self, input_dim, num_frequencies=10):
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        # 频率: 2^0, 2^1, ..., 2^(L-1)
        freqs = 2.0 ** torch.arange(num_frequencies).float()
        self.register_buffer('freqs', freqs)

    @property
    def output_dim(self):
        return 2 * self.input_dim * self.num_frequencies

    def forward(self, x):
        """
        Args:
            x: (batch, input_dim)

        Returns:
            encoded: (batch, 2 * input_dim * num_frequencies)
        """
        # x: (batch, input_dim) -> (batch, input_dim, 1)
        x = x.unsqueeze(-1)
        # freqs: (L,) -> (1, 1, L)
        freqs = self.freqs.view(1, 1, -1)
        # x_proj: (batch, input_dim, L)
        x_proj = 2 * torch.pi * x * freqs
        # 拼接 sin 和 cos: (batch, input_dim, 2*L) -> (batch, 2*input_dim*L)
        encoded = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return encoded.reshape(encoded.shape[0], -1)


if __name__ == '__main__':
    # 测试 FourierEncoding
    print("=" * 50)
    print("Test: FourierEncoding")
    enc = FourierEncoding(input_dim=2, num_frequencies=64, scale=10.0)
    x = torch.randn(4, 2)  # (batch=4, 2D空间坐标)
    y = enc(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape} (expected: [4, 128])")
    print(f"  Params: {sum(p.numel() for p in enc.parameters())}")

    # 测试梯度
    x.requires_grad_(True)
    y = enc(x)
    grad = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    print(f"  Grad:   {grad.shape}")
    print(f"  Grad finite: {torch.isfinite(grad).all().item()}")

    # 测试 PositionalEncoding
    print("\n" + "=" * 50)
    print("Test: PositionalEncoding")
    pos_enc = PositionalEncoding(input_dim=2, num_frequencies=10)
    y = pos_enc(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape} (expected: [4, 40])")

    # 测试 learnable
    print("\n" + "=" * 50)
    print("Test: Learnable FourierEncoding")
    enc_learn = FourierEncoding(input_dim=2, num_frequencies=32, learnable=True)
    print(f"  B is parameter: {isinstance(enc_learn.B, nn.Parameter)}")
    print(f"  B shape: {enc_learn.B.shape}")
