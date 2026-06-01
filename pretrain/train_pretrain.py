"""
自监督预训练脚本
结合掩码预测和对比学习两种预训练任务
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pretrain.MaskedPredNet import MaskedPredictionNetwork, create_mask, masked_mse_loss
from pretrain.ContrastiveLoss import InfoNCELoss, ContrastiveAugmentation, ContrastivePretrainDataset


# ============================================================
# 配置
# ============================================================
CONFIG = {
    # 网络参数
    'n_data': 84,
    'embed_dim': 256,
    'n_heads': 8,
    'enc_layers': 6,
    'dec_layers': 4,
    'dropout': 0.1,

    # 掩码预测参数
    'mask_ratio': 0.2,

    # 对比学习参数
    'temperature': 0.07,

    # 训练参数
    'batch_size': 16,
    'lr': 1e-4,
    'epochs': 500,
    'weight_decay': 1e-4,
    'grad_clip': 1.0,

    # 损失权重
    'lambda_mask': 1.0,
    'lambda_contrast': 1.0,

    # 数据
    'pretrain_data_path': 'data/pretrain/pretrain_rhoa.npy',

    # 输出
    'output_dir': 'pretrain/checkpoints',
}


def load_pretrain_data(data_path):
    """加载预训练数据"""
    rhoa = np.load(data_path)
    print(f"Loaded pretrain data: {rhoa.shape}")
    return rhoa


def evaluate(model, data, config, device, n_samples=100):
    """评估掩码预测效果"""
    model.eval()
    indices = np.random.choice(len(data), min(n_samples, len(data)), replace=False)
    rhoa = torch.FloatTensor(data[indices]).to(device)

    with torch.no_grad():
        mask = create_mask(config['n_data'], config['mask_ratio'], rhoa.shape[0]).to(device)
        reconstructed, features = model(rhoa, mask=mask)

    # 计算被遮盖位置的MAE
    masked_true = rhoa[mask.bool()].cpu().numpy()
    masked_pred = reconstructed[mask.bool()].cpu().numpy()
    mae = np.mean(np.abs(masked_true - masked_pred))
    rel_error = np.mean(np.abs(masked_true - masked_pred) / (np.abs(masked_true) + 1e-8))

    # 全量MAE
    full_mae = torch.mean(torch.abs(rhoa - reconstructed)).item()

    return {
        'masked_mae': mae,
        'masked_rel_error': rel_error,
        'full_mae': full_mae,
    }


def visualize_results(model, data, config, device, save_path):
    """可视化掩码预测结果"""
    model.eval()

    # 取一个样本
    idx = np.random.randint(len(data))
    rhoa = torch.FloatTensor(data[idx:idx+1]).to(device)

    with torch.no_grad():
        mask = create_mask(config['n_data'], config['mask_ratio'], 1).to(device)
        reconstructed, features = model(rhoa, mask=mask)

    rhoa_np = rhoa[0].cpu().numpy()
    recon_np = reconstructed[0].cpu().numpy()
    mask_np = mask[0].cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 原始数据
    axes[0, 0].plot(rhoa_np, 'b-', alpha=0.7, label='Original')
    axes[0, 0].set_title('Original Apparent Resistivity')
    axes[0, 0].set_xlabel('Measurement Index')
    axes[0, 0].set_ylabel('ρa')
    axes[0, 0].legend()

    # 掩码数据
    masked_rhoa = rhoa_np.copy()
    masked_rhoa[mask_np == 1] = np.nan
    axes[0, 1].plot(masked_rhoa, 'b-', alpha=0.7, label='Visible')
    masked_idx = np.where(mask_np == 1)[0]
    axes[0, 1].scatter(masked_idx, rhoa_np[mask_np == 1], c='red', s=20, label='Masked (GT)')
    axes[0, 1].set_title('Masked Input')
    axes[0, 1].legend()

    # 重建结果
    axes[1, 0].plot(rhoa_np, 'b-', alpha=0.5, label='Original')
    axes[1, 0].plot(recon_np, 'r--', alpha=0.7, label='Reconstructed')
    axes[1, 0].scatter(masked_idx, recon_np[mask_np == 1], c='green', s=20, label='Reconstructed (masked)')
    axes[1, 0].set_title('Reconstruction')
    axes[1, 0].legend()

    # 被遮盖位置的误差
    error = np.abs(rhoa_np[mask_np == 1] - recon_np[mask_np == 1])
    axes[1, 1].bar(range(len(error)), error, color='coral')
    axes[1, 1].set_title(f'Masked Position Error (MAE={np.mean(error):.4f})')
    axes[1, 1].set_xlabel('Masked Index')
    axes[1, 1].set_ylabel('Absolute Error')

    fig.suptitle('Masked Prediction Results', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {save_path}")


def save_checkpoint(model, optimizer, epoch, losses, save_path):
    """保存检查点"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
    }, save_path)


def plot_loss_curve(losses, save_path):
    """绘制损失曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(losses, 'b-', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Pretraining Loss Curve')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved loss curve to {save_path}")


def main():
    """主训练函数"""
    config = CONFIG
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 加载数据
    data_path = os.path.join(project_root, config['pretrain_data_path'])
    rhoa = load_pretrain_data(data_path)

    # 对数变换 + 归一化（稳定训练）
    rhoa = np.log(np.maximum(rhoa, 1e-6))
    rhoa_mean = rhoa.mean()
    rhoa_std = rhoa.std() + 1e-8
    rhoa = (rhoa - rhoa_mean) / rhoa_std
    print(f"Normalized: mean={rhoa_mean:.2f}, std={rhoa_std:.2f}")

    # 创建数据集和加载器（dataloader 已内置增强，返回两个增强视图）
    augmentation = ContrastiveAugmentation()
    dataset = ContrastivePretrainDataset(rhoa, augmentation)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)

    # 损失函数（在循环外创建，避免重复实例化）
    contrast_loss_fn = InfoNCELoss(config['temperature'])

    # 创建模型
    model = MaskedPredictionNetwork(
        n_data=config['n_data'],
        embed_dim=config['embed_dim'],
        n_heads=config['n_heads'],
        enc_layers=config['enc_layers'],
        dec_layers=config['dec_layers'],
        dropout=config['dropout'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    # 学习率调度（warmup + cosine）
    warmup_epochs = 20
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (config['epochs'] - warmup_epochs)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 创建输出目录
    output_dir = os.path.join(project_root, config['output_dir'])
    os.makedirs(output_dir, exist_ok=True)

    # 训练
    print(f"\nStarting pretraining for {config['epochs']} epochs...")
    losses = []
    best_loss = float('inf')

    for epoch in range(config['epochs']):
        # 掩码预测训练
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            view1, view2 = batch
            view1 = view1.to(device)
            view2 = view2.to(device)

            # 掩码预测：用 view1 作为输入（已有轻度增强）
            mask = create_mask(config['n_data'], config['mask_ratio'], view1.shape[0]).to(device)
            reconstructed, features = model(view1, mask=mask)
            loss_mask = masked_mse_loss(reconstructed, view1, mask)

            # 对比学习：直接用 dataloader 返回的两个增强视图（不再二次增强）
            feat1 = model.encode(view1)
            feat2 = model.encode(view2)
            loss_contrast = contrast_loss_fn(feat1, feat2)

            # 总损失
            loss = config['lambda_mask'] * loss_mask + config['lambda_contrast'] * loss_contrast

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        scheduler.step()

        # 保存最优模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                model, optimizer, epoch, losses,
                os.path.join(output_dir, 'best_model.pth')
            )

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {avg_loss:.4f}, Best: {best_loss:.4f}")

            # 评估
            metrics = evaluate(model, rhoa, config, device)
            print(f"  Masked MAE: {metrics['masked_mae']:.4f}, Rel Error: {metrics['masked_rel_error']:.4f}")

    # 保存最终模型
    save_checkpoint(
        model, optimizer, config['epochs'], losses,
        os.path.join(output_dir, 'final_model.pth')
    )

    # 保存编码器权重（用于下游任务）
    torch.save(
        model.encoder.state_dict(),
        os.path.join(output_dir, 'pretrained_encoder.pth')
    )

    # 绘制损失曲线
    plot_loss_curve(losses, os.path.join(output_dir, 'loss_curve.png'))

    # 可视化结果
    visualize_results(
        model, rhoa, config, device,
        os.path.join(output_dir, 'prediction_results.png')
    )

    print(f"\nPretraining complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
