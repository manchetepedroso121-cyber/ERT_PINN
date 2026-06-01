# -*- coding: utf-8 -*-
"""
Baseline comparison scripts for MA-PC-SRI paper.

Runs all baseline methods and compares with MA-PC-SRI:
1. pyGIMLi traditional inversion (Marquardt-Levenberg)
2. MLP-PINN (deterministic DL)
3. SRERTF-Net style (U-Net, supervised)
4. MA-PC-SRI (our method)

Usage:
    python train/run_baselines.py --array wenner --n_samples 20
    python train/run_baselines.py --array all --n_samples 50
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.constants import N_ELEC, ELEC_SPACING, GRID_NX, GRID_NZ, N_DATA_WENNER
from models.FourierEncoding import FourierEncoding
from train.datasets import InversionDataset, load_eval_samples
from train.engine import set_seed, safe_json_dump
from eval.evaluate import compute_metrics
from eval.plot_residuals import (
    plot_resistivity_comparison,
    plot_residual_comparison,
    plot_forward_verification,
)


# ============================================================
# Baseline Model Definitions
# ============================================================

class MLPInversion(nn.Module):
    """MLP-based inversion baseline (no KAN, no self-refine).

    Fourier encoding → MLP → resistivity model.
    Deterministic, supervised, single-pass.
    """

    def __init__(self, fourier_freq=32, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        self.fourier = FourierEncoding(2, fourier_freq, 10.0)
        fourier_out = fourier_freq * 2

        layers = []
        in_dim = fourier_out
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.SiLU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        """Predict resistivity at given coordinates.

        Args:
            coords: (N, 2) normalized coordinates

        Returns:
            rho: (N, 1) predicted resistivity
        """
        feat = self.fourier(coords)
        rho = F.softplus(self.net(feat)) + 1.0
        return rho


class UNetInversion(nn.Module):
    """U-Net style inversion baseline (SRERTF-Net inspired).

    Takes apparent resistivity as input, outputs resistivity model.
    Supervised, single-pass, deterministic.
    """

    def __init__(self, n_data=84, grid_size=64):
        super().__init__()
        self.grid_size = grid_size

        # Encode rhoa to feature map
        self.encoder = nn.Sequential(
            nn.Linear(n_data, 256), nn.GELU(),
            nn.Linear(256, 512), nn.GELU(),
            nn.Linear(512, grid_size * grid_size // 4),
            nn.GELU()
        )

        # Decode to resistivity map (32x32 -> 64x64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Softplus()
        )

    def forward(self, rhoa):
        """Predict resistivity model from apparent resistivity.

        Args:
            rhoa: (B, n_data) apparent resistivity

        Returns:
            m: (B, H, W) resistivity model
        """
        B = rhoa.shape[0]
        feat = self.encoder(rhoa)
        feat = feat.view(B, 1, self.grid_size // 2, self.grid_size // 2)
        m = self.decoder(feat).squeeze(1) + 1.0
        return m


# ============================================================
# Baseline Training
# ============================================================

def train_baseline_mlp(model, train_samples, val_samples, epochs, device, lr=1e-3):
    """Train MLP baseline with supervised loss."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    # Prepare grid coordinates
    x = np.linspace(-1, 1, GRID_NX).astype(np.float32)
    z = np.linspace(-1, 1, GRID_NZ).astype(np.float32)
    X, Z = np.meshgrid(x, z)
    coords = torch.FloatTensor(np.column_stack([X.ravel(), Z.ravel()])).to(device)

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        # Use first sample for simplicity (could batch)
        rho_true = train_samples[0][2].to(device)  # rho_matrix
        rho_pred = model(coords).squeeze().reshape(GRID_NX, GRID_NZ)
        loss = criterion(torch.log(rho_pred + 1e-6), torch.log(rho_true + 1e-6))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()

    return best_loss


def train_baseline_unet(model, train_samples, val_samples, epochs, device, lr=1e-3):
    """Train U-Net baseline with supervised loss."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for sample in train_samples[:5]:  # Use first 5 samples
            rhoa = sample[1].unsqueeze(0).to(device)
            rho_true = sample[2].unsqueeze(0).to(device)

            rho_pred = model(rhoa)
            loss = criterion(torch.log(rho_pred + 1e-6), torch.log(rho_true + 1e-6))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        if loss.item() < best_loss:
            best_loss = loss.item()

    return best_loss


# ============================================================
# Evaluation
# ============================================================

def evaluate_baseline(model, test_samples, model_type, device):
    """Evaluate a baseline model on test data.

    Args:
        model: trained model
        test_samples: list of (rho, rhoa, rho_matrix) tuples
        model_type: 'mlp' (coordinate-based) or 'unet' (data-based)
        device: torch device

    Returns:
        metrics dict
    """
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for sample in test_samples:
            rho_true = sample[2].numpy()
            rhoa = sample[1]

            if model_type == 'mlp':
                x = np.linspace(-1, 1, GRID_NX).astype(np.float32)
                z = np.linspace(-1, 1, GRID_NZ).astype(np.float32)
                X, Z = np.meshgrid(x, z)
                coords = torch.FloatTensor(np.column_stack([X.ravel(), Z.ravel()])).to(device)
                rho_pred = model(coords).squeeze().cpu().numpy().reshape(GRID_NX, GRID_NZ)
            elif model_type == 'unet':
                rhoa_input = rhoa.unsqueeze(0).to(device)
                rho_pred = model(rhoa_input).squeeze().cpu().numpy()
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            metrics = compute_metrics(rho_true.astype(np.float64),
                                       rho_pred.astype(np.float64))
            all_metrics.append(metrics)

    # Aggregate
    agg = {}
    for key in all_metrics[0].keys():
        vals = [m[key] for m in all_metrics if not np.isnan(m.get(key, float('nan')))]
        if vals:
            agg[key] = float(np.mean(vals))
            agg[f'{key}_std'] = float(np.std(vals))
        else:
            agg[key] = float('nan')
            agg[f'{key}_std'] = float('nan')

    return agg


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Run baseline comparisons')
    parser.add_argument('--array', type=str, default='wenner',
                        choices=['wenner', 'dipole-dipole', 'schlumberger', 'all'])
    parser.add_argument('--n_samples', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--fno_hidden_dim', type=int, default=32)
    parser.add_argument('--fno_n_layers', type=int, default=3)
    parser.add_argument('--fno_k_max', type=int, default=8)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.quick:
        args.epochs = 50
        args.n_samples = 5

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data', 'training')

    arrays = ['wenner', 'dipole-dipole', 'schlumberger'] if args.array == 'all' else [args.array]

    all_results = {}

    for array_type in arrays:
        print(f"\n{'='*60}")
        print(f"Baseline comparison: {array_type}")
        print(f"{'='*60}")

        n_data_map = {'wenner': 84, 'dipole-dipole': 231, 'schlumberger': 121}
        n_data = n_data_map.get(array_type, 84)

        # Load data
        train_dir = os.path.join(data_dir, array_type, 'train')
        val_dir = os.path.join(data_dir, array_type, 'val')
        test_dir = os.path.join(data_dir, array_type, 'test')

        if not os.path.exists(test_dir):
            print(f"  No test data for {array_type}")
            continue

        train_samples = load_eval_samples(InversionDataset(train_dir))
        val_samples = load_eval_samples(InversionDataset(val_dir))
        test_samples = load_eval_samples(InversionDataset(test_dir))[:args.n_samples]

        print(f"  Train: {len(train_samples)}, Test: {len(test_samples)}")

        array_results = {}

        # 1. MLP baseline
        print("  Training MLP baseline...")
        mlp = MLPInversion(fourier_freq=32).to(device)
        t0 = time.time()
        train_baseline_mlp(mlp, train_samples, val_samples, args.epochs, device)
        mlp_time = time.time() - t0
        mlp_metrics = evaluate_baseline(mlp, test_samples, 'mlp', device)
        array_results['MLP'] = mlp_metrics
        print(f"    MSE={mlp_metrics['mse']:.2f} Corr={mlp_metrics['correlation']:.4f} ({mlp_time:.1f}s)")

        # 2. U-Net baseline
        print("  Training U-Net baseline...")
        unet = UNetInversion(n_data=n_data, grid_size=GRID_NX).to(device)
        t0 = time.time()
        train_baseline_unet(unet, train_samples, val_samples, args.epochs, device)
        unet_time = time.time() - t0
        unet_metrics = evaluate_baseline(unet, test_samples, 'unet', device)
        array_results['UNet'] = unet_metrics
        print(f"    MSE={unet_metrics['mse']:.2f} Corr={unet_metrics['correlation']:.4f} ({unet_time:.1f}s)")

        # 3. Load MA-PC-SRI results if available
        sr_dir = os.path.join(project_root, 'results', 'selfrefine', array_type)
        if os.path.exists(os.path.join(sr_dir, 'best_encoder.pth')):
            try:
                print("  Loading MA-PC-SRI results...")
                from models.MultiArrayEncoder import MultiArrayEncoder
                from models.SelfRefineDecoder import SelfRefineDecoder
                from models.FNOForward import create_fno_model

                encoder = MultiArrayEncoder(
                    array_configs={array_type: n_data},
                    hidden_dim=args.hidden_dim, latent_dim=args.latent_dim
                ).to(device)
                decoder = SelfRefineDecoder(
                    latent_dim=args.latent_dim, grid_size=GRID_NX, n_data=n_data,
                    hidden_dim=args.hidden_dim, n_refine=5
                ).to(device)

                encoder.load_state_dict(torch.load(
                    os.path.join(sr_dir, 'best_encoder.pth'),
                    map_location=device, weights_only=True
                ))
                decoder.load_state_dict(torch.load(
                    os.path.join(sr_dir, 'best_decoder.pth'),
                    map_location=device, weights_only=True
                ))

                # Load FNO
                fno_dir = os.path.join(project_root, 'results', 'fno', array_type)
                fno = create_fno_model(array_type, hidden_dim=args.fno_hidden_dim,
                                        n_layers=args.fno_n_layers, k_max=args.fno_k_max).to(device)
                fno_path = os.path.join(fno_dir, 'best_fno.pth')
                if os.path.exists(fno_path):
                    fno.load_state_dict(torch.load(fno_path, map_location=device, weights_only=True))
                fno.eval()

                # Evaluate MA-PC-SRI
                encoder.eval()
                decoder.eval()
                sr_metrics_list = []
                with torch.no_grad():
                    for sample in test_samples:
                        rhoa = sample[1].unsqueeze(0).to(device)
                        rho_true = sample[2].numpy()
                        rhoa_dict = {array_type: rhoa}
                        z, _ = encoder(rhoa_dict)
                        m_final, _, _ = decoder(z, rhoa, fno_model=fno)
                        rho_pred = m_final.squeeze().cpu().numpy()
                        m = compute_metrics(rho_true, rho_pred)
                        sr_metrics_list.append(m)

                sr_agg = {}
                for key in sr_metrics_list[0].keys():
                    vals = [m[key] for m in sr_metrics_list if not np.isnan(m.get(key, float('nan')))]
                    if vals:
                        sr_agg[key] = float(np.mean(vals))
                array_results['MA-PC-SRI'] = sr_agg
                print(f"    MSE={sr_agg.get('mse', float('nan')):.2f} Corr={sr_agg.get('correlation', float('nan')):.4f}")
            except Exception as e:
                print(f"  MA-PC-SRI loading failed: {e}")

        all_results[array_type] = array_results

    # Save results
    save_dir = os.path.join(project_root, 'results', 'paper')
    os.makedirs(save_dir, exist_ok=True)
    safe_json_dump(all_results, os.path.join(save_dir, 'baseline_comparison.json'))

    # Print comparison table
    print(f"\n{'='*60}")
    print("BASELINE COMPARISON SUMMARY")
    print(f"{'='*60}")
    for array_type, results in all_results.items():
        print(f"\n{array_type}:")
        print(f"  {'Method':<15} {'MSE':<12} {'Corr':<12} {'PSNR':<12} {'SSIM':<12}")
        print(f"  {'-'*63}")
        for method, m in results.items():
            print(f"  {method:<15} {m.get('mse',0):<12.2f} {m.get('correlation',0):<12.4f} "
                  f"{m.get('psnr',0):<12.2f} {m.get('ssim',0):<12.4f}")


if __name__ == '__main__':
    main()
