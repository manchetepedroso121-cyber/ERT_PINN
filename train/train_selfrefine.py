# -*- coding: utf-8 -*-
"""
Training script for the self-refining inversion model.

Trains the MultiArrayEncoder + SelfRefineDecoder using the
physics-consistent self-refining loss (no labels required).

Loss = L_data + α·L_selfconsist + β·L_reg
  L_data       = ||FNO(G(d)) - d||²              (data fit)
  L_selfconsist = ||G(FNO(G(d))) - G(d)||²       (fixed-point consistency)
  L_reg        = TV(m) + λ·Smoothness(m)          (regularization)

Usage:
    python train/train_selfrefine.py --array wenner --epochs 500
    python train/train_selfrefine.py --quick  # Quick test
"""

import os
import sys
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.FNOForward import FNOForward, create_fno_model
from models.MultiArrayEncoder import MultiArrayEncoder
from models.SelfRefineDecoder import SelfRefineDecoder
from models.constants import N_ELEC, GRID_NX, GRID_NZ, N_DATA_WENNER
from train.datasets import InversionDataset, load_eval_samples
from train.engine import set_seed, safe_json_dump
from eval.evaluate import compute_metrics


def tv_loss(img):
    """Total variation loss for spatial smoothness."""
    diff_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
    diff_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
    return diff_h.mean() + diff_w.mean()


def smoothness_loss(img):
    """Second-order smoothness loss."""
    if img.dim() == 3:
        img = img.unsqueeze(1)
    lap_h = img[:, :, 2:, :] - 2 * img[:, :, 1:-1, :] + img[:, :, :-2, :]
    lap_w = img[:, :, :, 2:] - 2 * img[:, :, :, 1:-1] + img[:, :, :, :-2]
    return lap_h.pow(2).mean() + lap_w.pow(2).mean()


class SelfRefineLoss(nn.Module):
    """Physics-consistent self-refining loss.

    Combines:
    1. Data fit: ||FNO(G(d)) - d||²
    2. Self-consistency: ||G(FNO(G(d))) - G(d)||²
    3. Regularization: TV + smoothness
    """

    def __init__(self, lambda_data=1.0, lambda_selfconsist=0.5,
                 lambda_tv=0.01, lambda_smooth=0.01):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_selfconsist = lambda_selfconsist
        self.lambda_tv = lambda_tv
        self.lambda_smooth = lambda_smooth

    def forward(self, m_pred, rhoa_pred, rhoa_obs, m_refined=None):
        """Compute self-refining loss.

        Args:
            m_pred: (B, H, W) predicted resistivity model
            rhoa_pred: (B, n_data) FNO-predicted apparent resistivity
            rhoa_obs: (B, n_data) observed apparent resistivity
            m_refined: (B, H, W) refined model after one iteration (for self-consistency)

        Returns:
            loss: total loss
            loss_dict: dict of individual loss components
        """
        # Data fit (log-space for better dynamic range)
        log_pred = torch.log(rhoa_pred + 1e-6)
        log_obs = torch.log(rhoa_obs + 1e-6)
        l_data = F.mse_loss(log_pred, log_obs)

        # Self-consistency
        l_selfconsist = torch.tensor(0.0, device=m_pred.device)
        if m_refined is not None:
            # m_refined should be close to m_pred (fixed-point condition)
            l_selfconsist = F.mse_loss(
                torch.log(m_refined + 1e-6),
                torch.log(m_pred.detach() + 1e-6)
            )

        # Regularization
        l_tv = tv_loss(m_pred.unsqueeze(1) if m_pred.dim() == 3 else m_pred)
        l_smooth = smoothness_loss(m_pred)

        # Total loss
        loss = (
            self.lambda_data * l_data +
            self.lambda_selfconsist * l_selfconsist +
            self.lambda_tv * l_tv +
            self.lambda_smooth * l_smooth
        )

        return loss, {
            'total': loss.item(),
            'data': l_data.item(),
            'selfconsist': l_selfconsist.item(),
            'tv': l_tv.item(),
            'smooth': l_smooth.item(),
        }


def train_epoch(encoder, decoder, fno, rhoa_batch, criterion, optimizer, device):
    """Train one step of self-refining inversion.

    The training loop:
    1. Encode: z = Encoder(rhoa_obs)
    2. Predict: m_0 = Decoder(z)
    3. Forward: d̂ = FNO(m_0)
    4. Refine: m_1 = RefineBlock(m_0, d̂ - d_obs)
    5. Loss: L_data + L_selfconsist + L_reg
    """
    encoder.train()
    decoder.train()

    rhoa_obs = rhoa_batch.to(device)

    # Encode
    rhoa_dict = {'wenner': rhoa_obs}  # Expand for multi-array later
    z, _ = encoder(rhoa_dict)

    # Initial prediction
    m_0 = decoder.initial_prediction(z)

    # Forward model prediction
    rhoa_pred = fno(m_0)

    # One refinement step
    residual = torch.log(rhoa_pred.detach() + 1e-6) - torch.log(rhoa_obs + 1e-6)
    m_1, _ = decoder.refine_block(m_0, residual)

    # Compute loss
    loss, loss_dict = criterion(m_0, rhoa_pred, rhoa_obs, m_refined=m_1)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(encoder.parameters()) + list(decoder.parameters()), 1.0
    )
    optimizer.step()

    return loss_dict


def main():
    parser = argparse.ArgumentParser(description='Train self-refining inversion')
    parser.add_argument('--array', type=str, default='wenner')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_refine', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--fno_path', type=str, default=None,
                        help='Path to trained FNO checkpoint')
    parser.add_argument('--fno_hidden_dim', type=int, default=32)
    parser.add_argument('--fno_n_layers', type=int, default=3)
    parser.add_argument('--fno_k_max', type=int, default=8)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.quick:
        args.epochs = 20

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load data
    from torch.utils.data import DataLoader

    data_dir = os.path.join(project_root, 'data', 'training', args.array, 'train')
    if not os.path.exists(data_dir):
        print(f"Data not found: {data_dir}")
        return

    dataset = InversionDataset(data_dir)

    def collate_fixed_size(batch):
        """Custom collate: only stack rhoa and rho_matrix (fixed size), drop rho."""
        rhoa = torch.stack([item[1] for item in batch])
        rho_matrix = torch.stack([item[2] for item in batch])
        return None, rhoa, rho_matrix

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         collate_fn=collate_fixed_size)
    print(f"Training data: {len(dataset)} samples")

    # Load or create FNO
    n_data_map = {'wenner': 84, 'dipole-dipole': 231, 'schlumberger': 121}
    n_data = n_data_map.get(args.array, 84)

    fno = create_fno_model(args.array, hidden_dim=args.fno_hidden_dim,
                            n_layers=args.fno_n_layers, k_max=args.fno_k_max).to(device)
    if args.fno_path and os.path.exists(args.fno_path):
        fno.load_state_dict(torch.load(args.fno_path, map_location=device, weights_only=True))
        print(f"Loaded FNO from {args.fno_path}")
    else:
        print("Using untrained FNO (for quick test only)")
    fno.eval()  # FNO is fixed during inversion training

    # Create encoder and decoder
    encoder = MultiArrayEncoder(
        array_configs={args.array: n_data},
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim
    ).to(device)

    decoder = SelfRefineDecoder(
        latent_dim=args.latent_dim,
        grid_size=GRID_NX,
        n_data=n_data,
        hidden_dim=args.hidden_dim,
        n_refine=args.n_refine
    ).to(device)

    n_params_enc = sum(p.numel() for p in encoder.parameters())
    n_params_dec = sum(p.numel() for p in decoder.parameters())
    print(f"Encoder: {n_params_enc:,} params, Decoder: {n_params_dec:,} params")

    # Training setup
    criterion = SelfRefineLoss()
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    save_dir = os.path.join(project_root, 'results', 'selfrefine', args.array)
    os.makedirs(save_dir, exist_ok=True)

    history = {'loss': [], 'data': [], 'selfconsist': []}
    best_loss = float('inf')

    print(f"\nTraining self-refining inversion ({args.epochs} epochs)...")
    t_start = time.time()

    for epoch in range(args.epochs):
        epoch_losses = []
        for batch in loader:
            # batch is a tuple of (rho, rhoa, rho_matrix) tensors
            # rho has variable size, rhoa and rho_matrix have fixed size
            if isinstance(batch, (list, tuple)):
                rhoa_batch = batch[1]  # (B, n_data)
            else:
                rhoa_batch = batch['rhoa']
            loss_dict = train_epoch(
                encoder, decoder, fno, rhoa_batch, criterion, optimizer, device
            )
            epoch_losses.append(loss_dict)

        scheduler.step()

        # Aggregate
        avg_loss = np.mean([d['total'] for d in epoch_losses])
        avg_data = np.mean([d['data'] for d in epoch_losses])
        avg_sc = np.mean([d['selfconsist'] for d in epoch_losses])
        history['loss'].append(avg_loss)
        history['data'].append(avg_data)
        history['selfconsist'].append(avg_sc)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(encoder.state_dict(), os.path.join(save_dir, 'best_encoder.pth'))
            torch.save(decoder.state_dict(), os.path.join(save_dir, 'best_decoder.pth'))

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            print(f"  Epoch {epoch+1:4d}: loss={avg_loss:.6f}  "
                  f"data={avg_data:.6f}  selfconsist={avg_sc:.6f}")

    t_total = time.time() - t_start
    print(f"\nTraining time: {t_total:.1f}s")

    # Save
    safe_json_dump(history, os.path.join(save_dir, 'training_history.json'))
    safe_json_dump({
        'best_loss': best_loss,
        'training_time': t_total,
        'n_params_encoder': n_params_enc,
        'n_params_decoder': n_params_dec,
        'config': vars(args),
    }, os.path.join(save_dir, 'config.json'))


if __name__ == '__main__':
    main()
