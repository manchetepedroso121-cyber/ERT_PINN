"""
对比学习损失 - InfoNCE Loss
用于自监督预训练：学习区分不同样本的表示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InfoNCELoss(nn.Module):
    """InfoNCE对比学习损失

    通过正样本对和负样本对的学习，使编码器产生有区分性的表示

    正样本：同一样本的不同增强视图
    负样本：不同样本的表示
    """

    def __init__(self, temperature=0.07):
        """
        Args:
            temperature: 温度参数，控制对比学习的难度
                        较小的温度使分布更尖锐（更难）
                        较大的温度使分布更平滑（更容易）
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, features_i, features_j):
        """
        Args:
            features_i: 样本i的编码特征 (batch, n_data, embed_dim)
            features_j: 样本j的编码特征（同一样本的不同增强）(batch, n_data, embed_dim)

        Returns:
            loss: InfoNCE损失
        """
        batch_size = features_i.shape[0]

        # 在序列维度(dim=1)上平均池化，得到每个样本的全局表示
        # features_i shape: (batch, n_data, embed_dim) -> (batch, embed_dim)
        z_i = features_i.mean(dim=1)
        z_j = features_j.mean(dim=1)

        # L2归一化 (eps prevents NaN for zero vectors)
        z_i = F.normalize(z_i, dim=1, eps=1e-8)
        z_j = F.normalize(z_j, dim=1, eps=1e-8)

        # 拼接所有表示
        z = torch.cat([z_i, z_j], dim=0)  # (2*batch, embed_dim)

        # 计算相似度矩阵
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # (2*batch, 2*batch)

        # 创建掩码：排除自身相似度
        mask = torch.eye(2 * batch_size, device=sim_matrix.device).bool()

        # 正样本对：(i, j) 和 (j, i)
        # 正样本索引：对角偏移batch_size
        pos_indices = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size)
        ]).to(sim_matrix.device)

        # 对角线置为极小值（排除自身）
        sim_matrix.masked_fill_(mask, -1e9)

        # InfoNCE损失
        pos_sim = sim_matrix[torch.arange(2 * batch_size), pos_indices]
        log_prob = pos_sim - torch.logsumexp(sim_matrix, dim=1)

        loss = -log_prob.mean()
        return loss


class ContrastiveAugmentation:
    """ERT数据增强（用于对比学习）

    对同一视电阻率数据生成两个不同的增强视图。
    注意: 与 data.augmentation.ERTAugmentation 不同，本类操作 tensor 而非 dict。
    """

    def __init__(self, noise_std_range=(0.005, 0.02),
                 scale_range=(0.95, 1.05),
                 shift_range=(-0.05, 0.05)):
        """
        Args:
            noise_std_range: 高斯噪声标准差范围
            scale_range: 缩放范围
            shift_range: 平移范围
        """
        self.noise_std_range = noise_std_range
        self.scale_range = scale_range
        self.shift_range = shift_range

    def __call__(self, rhoa):
        """
        Args:
            rhoa: 视电阻率 tensor (n_data,) 或 (batch, n_data)

        Returns:
            view1, view2: 两个增强视图
        """
        view1 = self._augment(rhoa)
        view2 = self._augment(rhoa)
        return view1, view2

    def _augment(self, rhoa):
        """单次增强"""
        x = rhoa.clone()

        # 高斯噪声
        noise_std = np.random.uniform(*self.noise_std_range)
        x = x + torch.randn_like(x) * noise_std

        # 缩放
        scale = np.random.uniform(*self.scale_range)
        x = x * scale

        # 平移
        shift = np.random.uniform(*self.shift_range)
        x = x + shift

        return x


class ContrastivePretrainDataset(torch.utils.data.Dataset):
    """对比学习预训练数据集

    每次获取同一样本的两个不同增强视图
    """

    def __init__(self, rhoa_data, augmentation=None):
        """
        Args:
            rhoa_data: 视电阻率数据 (n_samples, n_data)
            augmentation: 数据增强
        """
        self.rhoa = rhoa_data
        self.augmentation = augmentation or ContrastiveAugmentation()

    def __len__(self):
        return len(self.rhoa)

    def __getitem__(self, idx):
        rhoa = torch.FloatTensor(self.rhoa[idx])
        view1, view2 = self.augmentation(rhoa)
        return view1, view2


if __name__ == '__main__':
    # 测试InfoNCE损失
    batch_size = 8
    n_data = 84
    embed_dim = 128

    loss_fn = InfoNCELoss(temperature=0.07)

    features_i = torch.randn(batch_size, n_data, embed_dim)
    features_j = torch.randn(batch_size, n_data, embed_dim)

    loss = loss_fn(features_i, features_j)
    print(f"InfoNCE Loss: {loss.item():.4f}")

    # 测试数据增强
    aug = ContrastiveAugmentation()
    rhoa = torch.randn(batch_size, n_data)
    view1, view2 = aug(rhoa)
    print(f"View1 shape: {view1.shape}, View2 shape: {view2.shape}")
