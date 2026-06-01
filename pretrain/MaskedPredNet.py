"""
掩码电极预测网络 - Encoder-Decoder 架构
用于自监督预训练：遮盖部分电极对数据，训练网络预测被遮盖的部分
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometryEncoder(nn.Module):
    """几何信息编码器

    将电极位置和四点索引编码为嵌入偏置，注入到主编码器中
    """

    def __init__(self, n_elec=24, embed_dim=128, hidden_dim=64):
        """
        Args:
            n_elec: 电极数量
            embed_dim: 输出嵌入维度（与主编码器一致）
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        # 电极位置编码: (n_elec, 2) -> (n_elec, hidden_dim)
        self.elec_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # 四点索引嵌入: 4个电极索引 -> 4个嵌入向量
        self.idx_embedding = nn.Embedding(n_elec, hidden_dim)
        # 融合层: 电极位置特征 + 四点索引特征 -> 序列级偏置
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 5, embed_dim),  # 1个全局位置特征 + 4个索引嵌入
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, elec_pos, geometry):
        """
        Args:
            elec_pos: 电极位置 (batch, n_elec, 2)
            geometry: 四点索引 (batch, n_data, 4)

        Returns:
            bias: 嵌入偏置 (batch, 1, embed_dim)，广播到所有序列位置
        """
        # 电极位置全局特征: 平均池化 -> (batch, hidden_dim)
        elec_feat = self.elec_proj(elec_pos)  # (batch, n_elec, hidden_dim)
        elec_global = elec_feat.mean(dim=1)  # (batch, hidden_dim)

        # 四点索引嵌取: 对 n_data 维度取平均得到全局表示
        # geometry shape: (batch, n_data, 4) -> (batch, 4)
        geo_flat = geometry.float().mean(dim=1).long()  # 先转float取均值，再转回long查表
        geo_emb = self.idx_embedding(geo_flat)  # (batch, 4, hidden_dim)
        geo_feat = geo_emb.reshape(geo_emb.shape[0], -1)  # (batch, 4*hidden_dim)

        # 融合
        combined = torch.cat([elec_global, geo_feat], dim=-1)  # (batch, 5*hidden_dim)
        bias = self.fusion(combined).unsqueeze(1)  # (batch, 1, embed_dim)

        return bias


class ERTEncoder(nn.Module):
    """ERT数据编码器

    将视电阻率序列编码为潜在特征表示
    支持可选的几何信息条件注入
    """

    def __init__(self, n_data=84, embed_dim=128, n_heads=4, n_layers=3, dropout=0.1,
                 geometry_dim=0, n_elec=24):
        """
        Args:
            n_data: 输入数据维度（电极对数量，84个测量值）
            embed_dim: 嵌入维度
            n_heads: Transformer注意力头数
            n_layers: Transformer层数
            dropout: Dropout率
            geometry_dim: 几何编码维度（0=不使用几何信息，>0=启用GeometryEncoder）
            n_elec: 电极数量（仅在 geometry_dim>0 时使用）
        """
        super().__init__()

        # 输入嵌入：将1D视电阻率映射到嵌入空间
        self.input_proj = nn.Linear(1, embed_dim)

        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, n_data, embed_dim) * 0.02)

        # 几何信息编码器（可选）
        self.use_geometry = geometry_dim > 0
        if self.use_geometry:
            self.geometry_encoder = GeometryEncoder(n_elec, embed_dim, geometry_dim)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.n_data = n_data
        self.embed_dim = embed_dim

    def forward(self, x, mask=None, elec_pos=None, geometry=None):
        """
        Args:
            x: 视电阻率 (batch, n_data) 或 (batch, n_data, 1)
            mask: 注意力mask (batch, n_data)，1=被遮盖，0=保留
            elec_pos: 电极位置 (batch, n_elec, 2)，仅 geometry_dim>0 时使用
            geometry: 四点索引 (batch, n_data, 4)，仅 geometry_dim>0 时使用

        Returns:
            features: 编码特征 (batch, n_data, embed_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch, n_data, 1)

        # 输入嵌入 + 位置编码
        x = self.input_proj(x) + self.pos_encoding  # (batch, n_data, embed_dim)

        # 几何条件注入（加性偏置）
        if self.use_geometry and elec_pos is not None and geometry is not None:
            geo_bias = self.geometry_encoder(elec_pos, geometry)  # (batch, 1, embed_dim)
            x = x + geo_bias

        # Transformer编码
        x = self.transformer(x, src_key_padding_mask=mask)

        return x


class ERTDecoder(nn.Module):
    """ERT数据解码器

    从编码特征重建视电阻率数据
    """

    def __init__(self, n_data=84, embed_dim=128, n_heads=4, n_layers=2, dropout=0.1):
        """
        Args:
            n_data: 输出数据维度
            embed_dim: 嵌入维度
            n_heads: 注意力头数
            n_layers: 解码器层数
            dropout: Dropout率
        """
        super().__init__()

        # Transformer解码器
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )

        self.n_data = n_data

    def forward(self, features):
        """
        Args:
            features: 编码特征 (batch, n_data, embed_dim)

        Returns:
            reconstructed: 重建的视电阻率 (batch, n_data)
        """
        x = self.transformer(features)
        x = self.output_proj(x).squeeze(-1)  # (batch, n_data)
        return x


class MaskedPredictionNetwork(nn.Module):
    """掩码预测网络

    完整的掩码预测模型：编码器 + 解码器
    用于自监督预训练
    """

    def __init__(self, n_data=84, embed_dim=128, n_heads=4,
                 enc_layers=3, dec_layers=2, dropout=0.1,
                 geometry_dim=0, n_elec=24):
        """
        Args:
            geometry_dim: 几何编码维度（0=不使用，>0=启用几何条件注入）
            n_elec: 电极数量
        """
        super().__init__()

        self.encoder = ERTEncoder(n_data, embed_dim, n_heads, enc_layers, dropout,
                                  geometry_dim=geometry_dim, n_elec=n_elec)
        self.decoder = ERTDecoder(n_data, embed_dim, n_heads, dec_layers, dropout)

        self.n_data = n_data
        self.embed_dim = embed_dim

    def forward(self, x, mask=None, elec_pos=None, geometry=None):
        """
        Args:
            x: 视电阻率 (batch, n_data)
            mask: 遮盖掩码 (batch, n_data)，1=被遮盖，0=保留
            elec_pos: 电极位置 (batch, n_elec, 2)，可选
            geometry: 四点索引 (batch, n_data, 4)，可选

        Returns:
            reconstructed: 重建的视电阻率 (batch, n_data)
            features: 编码特征 (batch, n_data, embed_dim)
        """
        features = self.encoder(x, mask=mask, elec_pos=elec_pos, geometry=geometry)
        reconstructed = self.decoder(features)
        return reconstructed, features

    def get_encoder(self):
        """获取编码器（用于下游任务）"""
        return self.encoder

    def encode(self, x, elec_pos=None, geometry=None):
        """直接编码输入，支持可选的几何信息"""
        return self.encoder(x, elec_pos=elec_pos, geometry=geometry)


def create_mask(n_data, mask_ratio=0.2, batch_size=1):
    """创建随机遮盖掩码

    Args:
        n_data: 数据维度
        mask_ratio: 遮盖比例
        batch_size: 批大小

    Returns:
        mask: (batch_size, n_data)，1=被遮盖，0=保留
    """
    n_mask = int(n_data * mask_ratio)
    mask = torch.zeros(batch_size, n_data)
    for i in range(batch_size):
        indices = torch.randperm(n_data)[:n_mask]
        mask[i, indices] = 1.0
    return mask


def masked_mse_loss(pred, target, mask):
    """计算被遮盖位置的MSE损失

    Args:
        pred: 预测值 (batch, n_data)
        target: 目标值 (batch, n_data)
        mask: 遮盖掩码 (batch, n_data)，1=被遮盖

    Returns:
        loss: 标量损失
    """
    # 只计算被遮盖位置的损失
    masked_pred = pred[mask.bool()]
    masked_target = target[mask.bool()]

    if len(masked_pred) == 0:
        return torch.tensor(0.0, device=pred.device)

    loss = F.mse_loss(masked_pred, masked_target)
    return loss


if __name__ == '__main__':
    # 测试网络
    batch_size = 4
    n_data = 84
    n_elec = 24
    embed_dim = 128

    # 测试1：无几何信息（默认模式，向后兼容）
    print("=" * 50)
    print("Test 1: Without geometry (default)")
    model = MaskedPredictionNetwork(n_data=n_data, embed_dim=embed_dim)

    x = torch.randn(batch_size, n_data)
    mask = create_mask(n_data, mask_ratio=0.2, batch_size=batch_size)

    reconstructed, features = model(x, mask=mask)

    print(f"Input: {x.shape}")
    print(f"Mask: {mask.shape}, masked ratio: {mask.mean():.2f}")
    print(f"Reconstructed: {reconstructed.shape}")
    print(f"Features: {features.shape}")

    loss = masked_mse_loss(reconstructed, x, mask)
    print(f"Loss: {loss.item():.4f}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # 测试2：带几何信息
    print("\n" + "=" * 50)
    print("Test 2: With geometry conditioning")
    model_geo = MaskedPredictionNetwork(
        n_data=n_data, embed_dim=embed_dim, geometry_dim=64, n_elec=n_elec
    )

    elec_pos = torch.randn(batch_size, n_elec, 2)
    geometry = torch.randint(0, n_elec, (batch_size, n_data, 4))

    reconstructed_geo, features_geo = model_geo(x, mask=mask,
                                                 elec_pos=elec_pos, geometry=geometry)

    print(f"Reconstructed (geo): {reconstructed_geo.shape}")
    print(f"Features (geo): {features_geo.shape}")

    total_params_geo = sum(p.numel() for p in model_geo.parameters())
    print(f"Total parameters (geo): {total_params_geo:,}")
