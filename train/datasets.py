# -*- coding: utf-8 -*-
"""
Shared dataset utilities for ERT inversion training.

Provides:
- InversionDataset: PyTorch Dataset for loading .npz training samples
- load_samples: Helper to convert Dataset to list of (rho, rhoa) tuples
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class InversionDataset(Dataset):
    """Dataset for ERT inversion training.

    Each sample contains:
    - rho: true resistivity model (n_cells,)
    - rhoa_obs: observed apparent resistivity (n_data,)

    Args:
        data_dir: directory containing .npz files
        noise_level: 'noisy' (default) or 'clean'
    """

    def __init__(self, data_dir, noise_level='noisy'):
        self.data_dir = data_dir
        self.noise_level = noise_level
        self.files = sorted([
            f for f in os.listdir(data_dir) if f.endswith('.npz')
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.data_dir, self.files[idx]))
        rho = data['rho'].astype(np.float32)
        rho_matrix = data['rho_matrix'].astype(np.float32)
        rhoa_key = 'rhoa_noisy' if self.noise_level == 'noisy' else 'rhoa_clean'
        rhoa = data[rhoa_key].astype(np.float32)
        return torch.FloatTensor(rho), torch.FloatTensor(rhoa), torch.FloatTensor(rho_matrix)


def load_samples(dataset):
    """Convert a Dataset to a list of (rho, rhoa) tuples.

    Args:
        dataset: InversionDataset or compatible Dataset

    Returns:
        list of (rho_tensor, rhoa_tensor) tuples
    """
    return [(dataset[i][0], dataset[i][1]) for i in range(len(dataset))]


def load_eval_samples(dataset):
    """Convert a Dataset to a list of (rho, rhoa, rho_matrix) tuples.

    Includes the 64x64 grid resistivity for proper grid-based evaluation.

    Args:
        dataset: InversionDataset

    Returns:
        list of (rho_tensor, rhoa_tensor, rho_matrix_tensor) tuples
    """
    return [(dataset[i][0], dataset[i][1], dataset[i][2]) for i in range(len(dataset))]
