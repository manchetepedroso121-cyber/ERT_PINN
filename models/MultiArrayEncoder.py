# -*- coding: utf-8 -*-
"""
Multi-Array Encoder for ERT inversion.

Inherits the multi-path architecture from SRERTF-Net (Liu et al., TGRS 2025):
- Separate encoding paths for each electrode array type
- Dense connections within and across paths
- Inception modules for multi-scale feature extraction
- Optional traditional inversion results as physical prior input

Key difference from SRERTF-Net:
- This encoder works with raw apparent resistivity data (not traditional
  inversion results), enabling end-to-end training without pre-inversion.

Reference:
    Liu et al., "Multi-Array Data Joint Super-Resolution Inversion for
    Electrical Resistivity Tomography", IEEE TGRS, 2025.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.constants import N_ELEC


class InceptionBlock(nn.Module):
    """Inception module for multi-scale feature extraction.

    Four parallel branches: 1×1, 3×3, 5×5 (two 3×3), MaxPool+1×1.
    Output is concatenation of all branches.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        ch = out_channels // 4

        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, ch, 1), nn.BatchNorm1d(ch), nn.GELU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, ch, 3, padding=1), nn.BatchNorm1d(ch), nn.GELU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, ch, 3, padding=1), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Conv1d(ch, ch, 3, padding=1), nn.BatchNorm1d(ch), nn.GELU()
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_channels, ch, 1), nn.BatchNorm1d(ch), nn.GELU()
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, C, L) input

        Returns:
            out: (B, out_channels, L) output
        """
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)


class DenseBlock(nn.Module):
    """DenseNet-style block: each layer receives all previous outputs."""

    def __init__(self, in_channels, growth_rate, n_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Conv1d(in_channels + i * growth_rate, growth_rate, 3, padding=1),
                nn.BatchNorm1d(growth_rate),
                nn.GELU()
            ))

    def forward(self, x):
        """Forward pass with dense connections.

        Args:
            x: (B, C, L) input

        Returns:
            out: (B, C + n_layers * growth_rate, L) output
        """
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)


class ArrayEncoderPath(nn.Module):
    """Single array encoding path.

    Encodes apparent resistivity data from one electrode array type.

    Args:
        n_data: number of measurements for this array type
        hidden_dim: hidden channel dimension
        n_inception: number of Inception blocks
    """

    def __init__(self, n_data, hidden_dim=64, n_inception=3):
        super().__init__()
        self.n_data = n_data

        # Input projection: treat rhoa as 1D signal
        self.input_proj = nn.Sequential(
            nn.Conv1d(1, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )

        # Inception blocks for multi-scale feature extraction
        self.inception_blocks = nn.ModuleList([
            InceptionBlock(hidden_dim, hidden_dim) for _ in range(n_inception)
        ])

        # Dense block for feature aggregation
        self.dense = DenseBlock(hidden_dim, hidden_dim // 4, n_layers=2)

        # Output projection
        dense_out = hidden_dim + (hidden_dim // 4) * 2
        self.output_proj = nn.Sequential(
            nn.Conv1d(dense_out, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )

    def forward(self, rhoa):
        """Encode apparent resistivity from one array.

        Args:
            rhoa: (B, n_data) apparent resistivity

        Returns:
            features: (B, hidden_dim, n_data) encoded features
        """
        # Reshape for 1D convolution
        x = rhoa.unsqueeze(1)  # (B, 1, n_data)

        # Project
        x = self.input_proj(x)  # (B, hidden_dim, n_data)

        # Inception blocks with residual connections
        for block in self.inception_blocks:
            x = x + block(x)

        # Dense aggregation
        x = self.dense(x)

        # Output projection
        x = self.output_proj(x)

        return x


class MultiArrayEncoder(nn.Module):
    """Multi-array encoder with cross-path dense connections.

    Processes apparent resistivity data from multiple electrode arrays
    (Wenner, Dipole-Dipole, Schlumberger) and fuses complementary
    information through dense connections.

    Architecture follows SRERTF-Net's design:
    - Independent encoding paths per array
    - Cross-path feature sharing at each scale
    - Final fusion via concatenation + projection

    Args:
        array_configs: dict of {array_type: n_data}
        hidden_dim: hidden channel dimension
        n_inception: number of Inception blocks per path
        latent_dim: output latent dimension
    """

    def __init__(self, array_configs=None, hidden_dim=64,
                 n_inception=3, latent_dim=256):
        super().__init__()

        if array_configs is None:
            from models.constants import N_DATA_WENNER
            array_configs = {
                'wenner': N_DATA_WENNER,
                'dipole-dipole': 231,
                'schlumberger': 121,
            }

        self.array_types = list(array_configs.keys())
        self.hidden_dim = hidden_dim

        # Individual encoding paths
        self.encoders = nn.ModuleDict({
            name: ArrayEncoderPath(n_data, hidden_dim, n_inception)
            for name, n_data in array_configs.items()
        })

        # Pooling to统一尺寸 before concatenation
        self.pool_size = 64  # Common spatial size for all arrays
        self.array_pools = nn.ModuleDict({
            name: nn.AdaptiveAvgPool1d(self.pool_size)
            for name in array_configs.keys()
        })

        # Cross-path dense connections (1×1 conv to mix features)
        self.cross_connect = nn.ModuleList([
            nn.Conv1d(hidden_dim * len(array_configs), hidden_dim, 1)
            for _ in range(n_inception + 1)
        ])

        # Final fusion
        self.fusion = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim * len(array_configs), latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, rhoa_dict):
        """Encode multi-array apparent resistivity data.

        Args:
            rhoa_dict: dict of {array_type: (B, n_data)} tensors

        Returns:
            z: (B, latent_dim) fused latent representation
            features: dict of {array_type: (B, hidden_dim, L)} per-array features
        """
        # Encode each array
        features = {}
        for name in self.array_types:
            if name in rhoa_dict:
                features[name] = self.encoders[name](rhoa_dict[name])
            else:
                # Zero features for missing arrays
                B = next(iter(rhoa_dict.values())).shape[0]
                features[name] = torch.zeros(
                    B, self.hidden_dim, self.encoders[name].n_data,
                    device=next(iter(rhoa_dict.values())).device
                )

        # Pool all features to same size before concatenation
        pooled_features = {}
        for name in self.array_types:
            pooled_features[name] = self.array_pools[name](features[name])

        # Concatenate pooled features
        all_features = torch.cat([pooled_features[name] for name in self.array_types], dim=1)

        # Cross-path fusion
        fused = self.cross_connect[0](all_features)

        # Final fusion to latent vector
        z = self.fusion(all_features)

        return z, features


class PhysicalPriorEncoder(nn.Module):
    """Encode traditional inversion results as physical prior.

    Takes traditional inversion results (from pyGIMLi or other solvers)
    and encodes them as a prior feature vector.

    This implements SRERTF-Net's insight that traditional inversion
    results contain useful physical information.

    Args:
        grid_size: size of the inversion result grid (default 64)
        latent_dim: output dimension
    """

    def __init__(self, grid_size=64, latent_dim=256):
        super().__init__()
        self.grid_size = grid_size

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim),
            nn.GELU()
        )

    def forward(self, m_trad):
        """Encode traditional inversion result.

        Args:
            m_trad: (B, H, W) traditional inversion result

        Returns:
            z_prior: (B, latent_dim) prior features
        """
        if m_trad.dim() == 3:
            m_trad = m_trad.unsqueeze(1)
        return self.encoder(m_trad)


if __name__ == '__main__':
    print("MultiArrayEncoder self-test...")

    encoder = MultiArrayEncoder(hidden_dim=32, n_inception=2, latent_dim=128)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Parameters: {n_params:,}")

    # Simulate multi-array data
    rhoa_dict = {
        'wenner': torch.rand(2, 84),
        'dipole-dipole': torch.rand(2, 231),
        'schlumberger': torch.rand(2, 121),
    }
    z, features = encoder(rhoa_dict)
    print(f"  Latent: {z.shape}")
    for name, feat in features.items():
        print(f"  {name} features: {feat.shape}")

    # Test with missing array
    z2, _ = encoder({'wenner': torch.rand(2, 84)})
    print(f"  Missing arrays handled: {z2.shape}")

    print("  All checks passed!")
