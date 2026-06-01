"""
共享测试配置和 fixtures
"""

import os
import sys
import pytest
import numpy as np
import torch

# 添加项目根目录到 path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def n_data():
    """默认测量数据维度"""
    return 84


@pytest.fixture
def n_elec():
    """默认电极数量"""
    return 24


@pytest.fixture
def embed_dim():
    """默认嵌入维度"""
    return 64  # 测试用较小维度，加快速度


@pytest.fixture
def batch_size():
    """默认批大小"""
    return 4


@pytest.fixture
def sample_rhoa(batch_size, n_data):
    """模拟视电阻率数据"""
    return torch.randn(batch_size, n_data)


@pytest.fixture
def sample_elec_pos(batch_size, n_elec):
    """模拟电极位置"""
    x = torch.linspace(-9.6, 9.6, n_elec).unsqueeze(0).expand(batch_size, -1)
    y = torch.zeros(batch_size, n_elec)
    return torch.stack([x, y], dim=-1)


@pytest.fixture
def sample_geometry(batch_size, n_data, n_elec):
    """模拟四点索引"""
    geo = torch.zeros(batch_size, n_data, 4, dtype=torch.long)
    for i in range(batch_size):
        for j in range(n_data):
            indices = torch.randperm(n_elec)[:4]
            geo[i, j] = indices
    return geo
