# -*- coding: utf-8 -*-
"""
Training script for FNO forward surrogate.

Trains a Fourier Neural Operator to approximate the FEM forward solver:
    F_θ(m) ≈ FEM(m)

This enables fast, differentiable forward evaluations during
physics-guided sampling in the self-refining decoder.

Usage:
    python train/train_fno.py --array wenner --epochs 200
    python train/train_fno.py --array all --epochs 500
    python train/train_fno.py --quick  # Quick test (5 epochs)
"""

import os
import sys
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.FNOForward import FNOForward, create_fno_model
from models.constants import N_ELEC, ELEC_SPACING, GRID_NX, GRID_NZ
from train.engine import set_seed, safe_json_dump
from eval.evaluate import compute_metrics


class FNOTrainingDataset(Dataset):
    """Dataset for FNO training: (rho_matrix, rhoa) pairs.

    Loads from data/training/{array}/{split}/ .npz files.
    Each file contains rho_matrix (64×64) and rhoa_clean (n_data,).
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.data_dir, self.files[idx]))
        rho_matrix = data['rho_matrix'].astype(np.float32)
        rhoa = data['rhoa_clean'].astype(np.float32)
        return torch.FloatTensor(rho_matrix), torch.FloatTensor(rhoa)


def train_epoch(model, loader, criterion, optimizer, device):
    """Train FNO for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for rho_matrix, rhoa in loader:
        rho_matrix = rho_matrix.to(device)
        rhoa = rhoa.to(device)

        # Forward
        rhoa_pred = model(rho_matrix)
        loss = criterion(rhoa_pred, rhoa)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def validate(model, loader, device):
    """Validate FNO model."""
    model.eval()
    all_mse = []
    all_rel_error = []

    with torch.no_grad():
        for rho_matrix, rhoa in loader:
            rho_matrix = rho_matrix.to(device)
            rhoa = rhoa.to(device)

            rhoa_pred = model(rho_matrix)

            for i in range(rhoa.shape[0]):
                pred = rhoa_pred[i].cpu().numpy()
                true = rhoa[i].cpu().numpy()
                mse = np.mean((pred - true) ** 2)
                rel = np.mean(np.abs(pred - true) / (np.abs(true) + 1e-6))
                all_mse.append(mse)
                all_rel_error.append(rel)

    return {
        'mse': float(np.mean(all_mse)),
        'relative_error': float(np.mean(all_rel_error)),
    }


def main():
    parser = argparse.ArgumentParser(description='Train FNO forward surrogate')
    parser.add_argument('--array', type=str, default='wenner',
                        choices=['wenner', 'dipole-dipole', 'schlumberger', 'all'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--k_max', type=int, default=16)
    parser.add_argument('--quick', action='store_true', help='Quick test (5 epochs)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.quick:
        args.epochs = 5

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data', 'training')

    arrays = ['wenner', 'dipole-dipole', 'schlumberger'] if args.array == 'all' else [args.array]

    for array_type in arrays:
        print(f"\n{'='*60}")
        print(f"Training FNO forward surrogate: {array_type}")
        print(f"{'='*60}")

        # Load data
        train_dir = os.path.join(data_dir, array_type, 'train')
        val_dir = os.path.join(data_dir, array_type, 'val')

        if not os.path.exists(train_dir):
            print(f"  Data not found: {train_dir}")
            continue

        train_dataset = FNOTrainingDataset(train_dir)
        val_dataset = FNOTrainingDataset(val_dir)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

        # Create model
        model = create_fno_model(
            array_type,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            k_max=args.k_max
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model: {n_params:,} parameters")

        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        # Training loop
        save_dir = os.path.join(project_root, 'results', 'fno', array_type)
        os.makedirs(save_dir, exist_ok=True)

        best_val_mse = float('inf')
        history = {'train_loss': [], 'val_mse': [], 'val_rel_error': []}

        t_start = time.time()
        for epoch in range(args.epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            scheduler.step()
            history['train_loss'].append(train_loss)

            if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
                val_metrics = validate(model, val_loader, device)
                history['val_mse'].append(val_metrics['mse'])
                history['val_rel_error'].append(val_metrics['relative_error'])

                if val_metrics['mse'] < best_val_mse:
                    best_val_mse = val_metrics['mse']
                    torch.save(model.state_dict(), os.path.join(save_dir, 'best_fno.pth'))

                print(f"  Epoch {epoch+1:4d}: train={train_loss:.6f}  "
                      f"val_MSE={val_metrics['mse']:.4f}  "
                      f"val_RelErr={val_metrics['relative_error']:.4f}")

        t_total = time.time() - t_start
        print(f"  Training time: {t_total:.1f}s")

        # Save final model and metrics
        torch.save(model.state_dict(), os.path.join(save_dir, 'final_fno.pth'))
        final_metrics = validate(model, val_loader, device)
        final_metrics['training_time'] = t_total
        final_metrics['n_params'] = n_params
        safe_json_dump(final_metrics, os.path.join(save_dir, 'fno_metrics.json'))
        safe_json_dump(history, os.path.join(save_dir, 'training_history.json'))

        print(f"  Final: MSE={final_metrics['mse']:.4f}, RelErr={final_metrics['relative_error']:.4f}")


if __name__ == '__main__':
    main()
