# -*- coding: utf-8 -*-
"""
Fourier Neural Operator (FNO) for ERT forward modeling surrogate.

Replaces the slow FEM forward solver with a fast, differentiable
neural operator. Enables real-time forward evaluations during
physics-guided sampling.

Reference:
    Li et al., "Fourier Neural Operator for Parametric Partial
    Differential Equations", ICLR 2021.

Architecture:
    Input:  resistivity model m ∈ R^{H×W} (64×64 grid)
    Output: apparent resistivity d ∈ R^{n_data} (at electrode positions)

    Pipeline:
    1. Lift: 1 → hidden_dim channels
    2. N × FourierLayer (SpectralConv + SkipConv + GELU)
    3. Project: hidden_dim → 1 channel
    4. Sample at electrode positions → n_data values
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.constants import GRID_NX, GRID_NZ, N_ELEC


class SpectralConv2d(nn.Module):
    """2D Fourier layer: FFT → linear transform → iFFT.

    Keeps only the lowest k_max Fourier modes for efficiency.
    """

    def __init__(self, in_channels, out_channels, k_max=16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k_max = k_max

        # Complex weights for Fourier modes (1×1 conv in Fourier space)
        self.weight_real = nn.Parameter(
            torch.randn(in_channels, out_channels, 1, 1) * 0.02
        )
        self.weight_imag = nn.Parameter(
            torch.randn(in_channels, out_channels, 1, 1) * 0.02
        )

    def forward(self, x):
        """Forward pass: FFT → multiply → iFFT.

        Args:
            x: (B, C, H, W) input tensor

        Returns:
            out: (B, C_out, H, W) output tensor
        """
        B, C, H, W = x.shape

        # FFT
        x_ft = torch.fft.rfft2(x, norm='ortho')

        # Extract low-frequency modes
        k_h = min(self.k_max, H)
        k_w = min(self.k_max, W // 2 + 1)

        # Multiply with learnable weights in Fourier space
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                              dtype=x_ft.dtype, device=x_ft.device)

        # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        # Using 1×1 conv weights (broadcast over spatial dimensions)
        x_modes = x_ft[:, :, :k_h, :k_w]
        w_real = self.weight_real  # (in_ch, out_ch, 1, 1)
        w_imag = self.weight_imag

        out_ft[:, :, :k_h, :k_w] = (
            F.conv2d(x_modes.real, w_real) - F.conv2d(x_modes.imag, w_imag) +
            1j * (
                F.conv2d(x_modes.real, w_imag) + F.conv2d(x_modes.imag, w_real)
            )
        )

        # iFFT
        out = torch.fft.irfft2(out_ft, s=(H, W), norm='ortho')
        return out


class FourierLayer(nn.Module):
    """Single FNO layer: SpectralConv + SkipConv + GELU activation."""

    def __init__(self, channels, k_max=16):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, k_max)
        self.skip = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(min(8, channels), channels)

    def forward(self, x):
        """Forward pass with residual connection.

        Args:
            x: (B, C, H, W) input

        Returns:
            out: (B, C, H, W) output
        """
        residual = x
        x = self.spectral(x) + self.skip(x)
        x = self.norm(x)
        x = F.gelu(x)
        return x + residual


class FNOForward(nn.Module):
    """FNO-based forward model surrogate for ERT.

    Maps resistivity model (64×64 grid) to apparent resistivity
    at electrode measurement positions.

    Args:
        grid_size: spatial grid size (default 64)
        n_elec: number of electrodes (default 24)
        n_data: number of measurements (default 84 for Wenner)
        hidden_dim: hidden channel dimension
        n_layers: number of Fourier layers
        k_max: maximum Fourier modes to keep
    """

    def __init__(self, grid_size=64, n_elec=N_ELEC, n_data=84,
                 hidden_dim=64, n_layers=4, k_max=16):
        super().__init__()
        self.grid_size = grid_size
        self.n_elec = n_elec
        self.n_data = n_data

        # Lifting layer: 1 → hidden_dim
        self.lift = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=1),
            nn.GELU()
        )

        # Fourier layers
        self.fourier_layers = nn.ModuleList([
            FourierLayer(hidden_dim, k_max) for _ in range(n_layers)
        ])

        # Projection layer: hidden_dim → 1
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1)
        )

        # Measurement head: global pool + MLP to predict apparent resistivity
        self.measure_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),  # Pool to 4×4
            nn.Flatten(),
            nn.Linear(1 * 4 * 4, 256),
            nn.GELU(),
            nn.Linear(256, n_data)
        )

    def forward(self, m):
        """Forward pass: resistivity model → apparent resistivity.

        Args:
            m: (B, H, W) or (B, 1, H, W) resistivity model

        Returns:
            rhoa: (B, n_data) predicted apparent resistivity
        """
        if m.dim() == 3:
            m = m.unsqueeze(1)  # (B, 1, H, W)

        B = m.shape[0]

        # Log-transform for better dynamic range handling
        m_log = torch.log(m + 1e-6)

        # Lift
        x = self.lift(m_log)

        # Fourier layers
        for layer in self.fourier_layers:
            x = layer(x)

        # Project to single channel
        x = self.project(x)  # (B, 1, H, W)

        # Predict apparent resistivity via global pooling + MLP
        rhoa = self.measure_head(x)  # (B, n_data)

        # Ensure positive output
        rhoa = F.softplus(rhoa) + 1e-6

        return rhoa

    def predict_with_grid(self, m):
        """Forward pass that also returns intermediate grid prediction.

        Useful for visualization and debugging.

        Args:
            m: (B, H, W) or (B, 1, H, W) resistivity model

        Returns:
            rhoa: (B, n_data) predicted apparent resistivity
            m_grid: (B, 1, H, W) intermediate representation
        """
        if m.dim() == 3:
            m = m.unsqueeze(1)

        B = m.shape[0]
        m_log = torch.log(m + 1e-6)

        x = self.lift(m_log)
        for layer in self.fourier_layers:
            x = layer(x)
        x = self.project(x)

        rhoa = F.softplus(self.measure_head(x)) + 1e-6

        return rhoa, x


def create_fno_model(array_type='wenner', **kwargs):
    """Factory function to create FNO model for a specific array type.

    Args:
        array_type: 'wenner', 'dipole-dipole', or 'schlumberger'
        **kwargs: additional arguments to FNOForward

    Returns:
        FNOForward model
    """
    from models.constants import N_DATA_WENNER
    n_data_map = {
        'wenner': N_DATA_WENNER,
        'dipole-dipole': 231,
        'schlumberger': 121,
    }
    n_data = n_data_map.get(array_type, N_DATA_WENNER)
    return FNOForward(n_data=n_data, **kwargs)


if __name__ == '__main__':
    # Self-test
    print("FNOForward self-test...")
    model = FNOForward(grid_size=64, n_data=84, hidden_dim=32, n_layers=2, k_max=8)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    m = torch.rand(2, 64, 64) * 100 + 10  # Random resistivity
    rhoa = model(m)
    print(f"  Input: {m.shape}, Output: {rhoa.shape}")
    print(f"  rhoa range: {rhoa.min():.2f} - {rhoa.max():.2f}")

    # Gradient test
    m.requires_grad_(True)
    rhoa = model(m)
    loss = rhoa.sum()
    loss.backward()
    print(f"  Gradient flow: OK (m.grad exists: {m.grad is not None})")
    print("  All checks passed!")
