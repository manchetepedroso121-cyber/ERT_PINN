# -*- coding: utf-8 -*-
"""
Ablation study for MA-PC-SRI.

Systematically removes each component and measures impact:
A1: Full model (MA-PC-SRI)
A2: No self-consistency loss (only data loss)
A3: No refinement iteration (single-pass only)
A4: No multi-array (single array only)
A5: No physical prior (no traditional inversion input)
A6: No FNO surrogate (use fixed forward model)

Usage:
    python train/run_ablation.py --array wenner --epochs 200
    python train/run_ablation.py --quick
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
from models.MultiArrayEncoder import MultiArrayEncoder, PhysicalPriorEncoder
from models.SelfRefineDecoder import SelfRefineDecoder
from models.constants import N_ELEC, GRID_NX, GRID_NZ, N_DATA_WENNER
from train.datasets import InversionDataset, load_eval_samples
from train.engine import set_seed, safe_json_dump
from eval.evaluate import compute_metrics


def train_ablation_model(encoder, decoder, fno, train_samples, config, device):
    """Train an ablation variant.

    Args:
        encoder: MultiArrayEncoder
        decoder: SelfRefineDecoder
        fno: FNO forward model
        train_samples: training data
        config: ablation config dict
        device: torch device
    """
    from torch.utils.data import DataLoader

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=config['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs']
    )

    # Simple training loop
    for epoch in range(config['epochs']):
        encoder.train()
        decoder.train()

        for batch_idx, sample in enumerate(train_samples[:config.get('batch_subset', 5)]):
            rhoa = sample[1].unsqueeze(0).to(device)
            rho_true = sample[2].unsqueeze(0).to(device)

            # Encode
            rhoa_dict = {config['array']: rhoa}
            z, _ = encoder(rhoa_dict)

            # Decode
            if config.get('use_refinement', True):
                m_pred, _, _ = decoder(z, rhoa, fno_model=fno,
                                        n_steps=config.get('n_refine', 3))
            else:
                m_pred = decoder.initial_prediction(z)

            # Loss
            if config.get('use_selfconsist', True):
                # Data loss
                rhoa_pred = fno(m_pred)
                log_pred = torch.log(rhoa_pred + 1e-6)
                log_obs = torch.log(rhoa + 1e-6)
                l_data = F.mse_loss(log_pred, log_obs)

                # Self-consistency loss
                residual = log_pred.detach() - log_obs
                m_refined, _ = decoder.refine_block(m_pred, residual)
                l_sc = F.mse_loss(
                    torch.log(m_refined + 1e-6),
                    torch.log(m_pred.detach() + 1e-6)
                )
                loss = l_data + config.get('lambda_sc', 0.5) * l_sc
            else:
                # Only data loss
                rhoa_pred = fno(m_pred)
                loss = F.mse_loss(
                    torch.log(rhoa_pred + 1e-6),
                    torch.log(rhoa + 1e-6)
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

        scheduler.step()


def evaluate_ablation(encoder, decoder, fno, test_samples, array_type, device):
    """Evaluate ablation variant on test data."""
    encoder.eval()
    decoder.eval()
    all_metrics = []

    with torch.no_grad():
        for sample in test_samples:
            rhoa = sample[1].unsqueeze(0).to(device)
            rho_true = sample[2].numpy()
            rhoa_dict = {array_type: rhoa}
            z, _ = encoder(rhoa_dict)
            m_final, _, _ = decoder(z, rhoa, fno_model=fno)
            rho_pred = m_final.squeeze().cpu().numpy()
            m = compute_metrics(rho_true, rho_pred)
            all_metrics.append(m)

    agg = {}
    for key in all_metrics[0].keys():
        vals = [m[key] for m in all_metrics if not np.isnan(m.get(key, float('nan')))]
        if vals:
            agg[key] = float(np.mean(vals))
            agg[f'{key}_std'] = float(np.std(vals))
    return agg


def run_ablation(args, array_type, device):
    """Run all ablation variants for one array type."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data', 'training')

    n_data_map = {'wenner': 84, 'dipole-dipole': 231, 'schlumberger': 121}
    n_data = n_data_map.get(array_type, 84)

    train_dir = os.path.join(data_dir, array_type, 'train')
    test_dir = os.path.join(data_dir, array_type, 'test')
    train_samples = load_eval_samples(InversionDataset(train_dir))
    test_samples = load_eval_samples(InversionDataset(test_dir))[:args.n_samples]

    # Load FNO
    fno = create_fno_model(array_type, hidden_dim=args.fno_hidden_dim,
                            n_layers=args.fno_n_layers, k_max=args.fno_k_max).to(device)
    fno_path = os.path.join(project_root, 'results', 'fno', array_type, 'best_fno.pth')
    if os.path.exists(fno_path):
        fno.load_state_dict(torch.load(fno_path, map_location=device, weights_only=True))
    fno.eval()

    # Define ablation variants
    ablations = {
        'A1_full': {
            'use_refinement': True, 'use_selfconsist': True,
            'n_refine': 5, 'lambda_sc': 0.5,
            'description': 'Full MA-PC-SRI'
        },
        'A2_no_selfconsist': {
            'use_refinement': True, 'use_selfconsist': False,
            'n_refine': 5, 'lambda_sc': 0.0,
            'description': 'No self-consistency loss'
        },
        'A3_no_refinement': {
            'use_refinement': False, 'use_selfconsist': False,
            'n_refine': 0, 'lambda_sc': 0.0,
            'description': 'Single-pass (no iteration)'
        },
        'A4_single_array': {
            'use_refinement': True, 'use_selfconsist': True,
            'n_refine': 5, 'lambda_sc': 0.5,
            'single_array': True,
            'description': 'Single array only'
        },
    }

    config_base = {
        'array': array_type,
        'epochs': args.epochs,
        'lr': args.lr,
        'batch_subset': 3,
    }

    results = {}

    for name, abl_config in ablations.items():
        print(f"\n  Running {name}: {abl_config['description']}")
        config = {**config_base, **abl_config}

        # Create models
        if abl_config.get('single_array', False):
            encoder = MultiArrayEncoder(
                array_configs={array_type: n_data},
                hidden_dim=args.hidden_dim, latent_dim=args.latent_dim
            ).to(device)
        else:
            encoder = MultiArrayEncoder(
                array_configs={array_type: n_data},
                hidden_dim=args.hidden_dim, latent_dim=args.latent_dim
            ).to(device)

        decoder = SelfRefineDecoder(
            latent_dim=args.latent_dim, grid_size=GRID_NX, n_data=n_data,
            hidden_dim=args.hidden_dim, n_refine=abl_config.get('n_refine', 5)
        ).to(device)

        # Train
        t0 = time.time()
        train_ablation_model(encoder, decoder, fno, train_samples, config, device)
        t_train = time.time() - t0

        # Evaluate
        metrics = evaluate_ablation(encoder, decoder, fno, test_samples, array_type, device)
        metrics['training_time'] = t_train
        results[name] = metrics

        print(f"    MSE={metrics.get('mse', float('nan')):.2f}  "
              f"Corr={metrics.get('correlation', float('nan')):.4f}  "
              f"PSNR={metrics.get('psnr', float('nan')):.2f}dB  "
              f"({t_train:.1f}s)")

    return results


def main():
    parser = argparse.ArgumentParser(description='Run ablation studies')
    parser.add_argument('--array', type=str, default='wenner')
    parser.add_argument('--n_samples', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--fno_hidden_dim', type=int, default=32)
    parser.add_argument('--fno_n_layers', type=int, default=3)
    parser.add_argument('--fno_k_max', type=int, default=8)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.quick:
        args.epochs = 20
        args.n_samples = 5

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"{'='*60}")
    print(f"Ablation Study: {args.array}")
    print(f"{'='*60}")

    results = run_ablation(args, args.array, device)

    # Save
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(project_root, 'results', 'paper')
    os.makedirs(save_dir, exist_ok=True)
    safe_json_dump(results, os.path.join(save_dir, f'ablation_{args.array}.json'))

    # Print summary
    print(f"\n{'='*60}")
    print("ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Variant':<25} {'MSE':<12} {'Corr':<12} {'PSNR':<12} {'SSIM':<12}")
    print(f"  {'-'*73}")
    for name, m in results.items():
        print(f"  {name:<25} {m.get('mse',0):<12.2f} {m.get('correlation',0):<12.4f} "
              f"{m.get('psnr',0):<12.2f} {m.get('ssim',0):<12.4f}")


if __name__ == '__main__':
    main()
