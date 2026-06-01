# -*- coding: utf-8 -*-
"""
Complete evaluation pipeline for MA-PC-SRI.

Loads trained models, evaluates on test set, generates report
with all metrics and visualizations.

Usage:
    python train/evaluate_model.py --array wenner
    python train/evaluate_model.py --array all --n_samples 50
"""

import os
import sys
import json
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.FNOForward import create_fno_model
from models.MultiArrayEncoder import MultiArrayEncoder
from models.SelfRefineDecoder import SelfRefineDecoder
from models.constants import N_ELEC, GRID_NX, GRID_NZ
from train.datasets import InversionDataset, load_eval_samples
from train.engine import set_seed, safe_json_dump
from eval.evaluate import compute_metrics, forward_verify
from eval.plot_residuals import (
    plot_resistivity_comparison,
    plot_residual_comparison,
    plot_forward_verification,
)


def load_trained_models(array_type, device, project_root,
                         hidden_dim=32, latent_dim=128,
                         fno_hidden_dim=32, fno_n_layers=3, fno_k_max=8):
    """Load all trained models for a given array type.

    Returns:
        encoder, decoder, fno or None if not found
    """
    n_data_map = {'wenner': 84, 'dipole-dipole': 231, 'schlumberger': 121}
    n_data = n_data_map.get(array_type, 84)

    # Load encoder
    encoder_path = os.path.join(project_root, 'results', 'selfrefine',
                                 array_type, 'best_encoder.pth')
    if not os.path.exists(encoder_path):
        print(f"  Encoder not found: {encoder_path}")
        return None, None, None

    encoder = MultiArrayEncoder(
        array_configs={array_type: n_data},
        hidden_dim=hidden_dim, latent_dim=latent_dim
    ).to(device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
    encoder.eval()

    # Load decoder
    decoder_path = os.path.join(project_root, 'results', 'selfrefine',
                                 array_type, 'best_decoder.pth')
    decoder = SelfRefineDecoder(
        latent_dim=latent_dim, grid_size=GRID_NX, n_data=n_data,
        hidden_dim=hidden_dim, n_refine=5
    ).to(device)
    decoder.load_state_dict(torch.load(decoder_path, map_location=device, weights_only=True))
    decoder.eval()

    # Load FNO
    fno = create_fno_model(array_type, hidden_dim=fno_hidden_dim,
                             n_layers=fno_n_layers, k_max=fno_k_max).to(device)
    fno_path = os.path.join(project_root, 'results', 'fno', array_type, 'best_fno.pth')
    if os.path.exists(fno_path):
        fno.load_state_dict(torch.load(fno_path, map_location=device, weights_only=True))
    fno.eval()

    return encoder, decoder, fno


def evaluate_single_sample(encoder, decoder, fno, rhoa, rho_true, array_type,
                            device, n_uq_samples=5):
    """Evaluate a single test sample.

    Returns:
        metrics: dict of evaluation metrics
        rho_pred: predicted resistivity model
        samples: list of UQ samples
        uncertainty: pixel-wise std
    """
    rhoa_input = rhoa.unsqueeze(0).to(device)
    rhoa_dict = {array_type: rhoa_input}

    with torch.no_grad():
        # Encode
        z, _ = encoder(rhoa_dict)

        # Single prediction
        m_final, history, residuals = decoder(z, rhoa_input, fno_model=fno)
        rho_pred = m_final.squeeze().cpu().numpy()

        # UQ sampling
        samples, uncertainty = decoder.sample_diverse(
            z, rhoa_input, fno_model=fno, n_samples=n_uq_samples
        )
        samples_np = [s.squeeze().cpu().numpy() for s in samples]
        uncertainty_np = uncertainty.squeeze().cpu().numpy()

    # Compute metrics
    metrics = compute_metrics(rho_true.astype(np.float64), rho_pred.astype(np.float64))

    # Forward verification
    if fno is not None:
        with torch.no_grad():
            rhoa_pred_tensor = fno(m_final)
        rhoa_pred = rhoa_pred_tensor.squeeze().cpu().numpy()
        rhoa_obs = rhoa.cpu().numpy()
        log_pred = np.log(rhoa_pred + 1e-6)
        log_obs = np.log(rhoa_obs + 1e-6)
        metrics['data_misfit'] = float(np.sqrt(np.mean((log_pred - log_obs) ** 2)))

    metrics['refinement_steps'] = len(history) - 1
    metrics['final_residual'] = residuals[-1] if residuals else 0.0

    return metrics, rho_pred, samples_np, uncertainty_np


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained MA-PC-SRI model')
    parser.add_argument('--array', type=str, default='wenner',
                        choices=['wenner', 'dipole-dipole', 'schlumberger', 'all'])
    parser.add_argument('--n_samples', type=int, default=20)
    parser.add_argument('--n_uq', type=int, default=10, help='UQ samples per test')
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--fno_hidden_dim', type=int, default=32)
    parser.add_argument('--fno_n_layers', type=int, default=3)
    parser.add_argument('--fno_k_max', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_plots', action='store_true', default=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data', 'training')

    arrays = ['wenner', 'dipole-dipole', 'schlumberger'] if args.array == 'all' else [args.array]

    all_results = {}

    for array_type in arrays:
        print(f"\n{'='*60}")
        print(f"Evaluating MA-PC-SRI: {array_type}")
        print(f"{'='*60}")

        # Load models
        encoder, decoder, fno = load_trained_models(
            array_type, device, project_root,
            hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
            fno_hidden_dim=args.fno_hidden_dim,
            fno_n_layers=args.fno_n_layers, fno_k_max=args.fno_k_max
        )
        if encoder is None:
            print(f"  Skipping {array_type} (models not found)")
            continue

        # Load test data
        test_dir = os.path.join(data_dir, array_type, 'test')
        if not os.path.exists(test_dir):
            print(f"  No test data for {array_type}")
            continue

        test_samples = load_eval_samples(InversionDataset(test_dir))
        n_eval = min(len(test_samples), args.n_samples)
        print(f"  Evaluating {n_eval} test samples...")

        # Evaluate
        all_metrics = []
        for i in range(n_eval):
            rhoa = test_samples[i][1]
            rho_true = test_samples[i][2].numpy()

            metrics, rho_pred, samples, uncertainty = evaluate_single_sample(
                encoder, decoder, fno, rhoa, rho_true, array_type,
                device, n_uq_samples=args.n_uq
            )
            all_metrics.append(metrics)

            # Save plots for first few samples
            if args.save_plots and i < 3:
                fig_dir = os.path.join(project_root, 'results', 'paper',
                                        'figures', array_type)
                os.makedirs(fig_dir, exist_ok=True)

                # Resistivity comparison
                plot_resistivity_comparison(
                    rho_true, [rho_pred],
                    ['MA-PC-SRI'],
                    os.path.join(fig_dir, f'sample_{i}_resistivity.png'),
                    title=f'{array_type} Sample {i}'
                )

                # Uncertainty map
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                vmin = min(rho_true.min(), rho_pred.min())
                vmax = max(rho_true.max(), rho_pred.max())

                im0 = axes[0].imshow(rho_true, cmap='jet', vmin=vmin, vmax=vmax,
                                      aspect='auto', origin='upper')
                axes[0].set_title('Ground Truth')
                plt.colorbar(im0, ax=axes[0])

                im1 = axes[1].imshow(rho_pred, cmap='jet', vmin=vmin, vmax=vmax,
                                      aspect='auto', origin='upper')
                axes[1].set_title(f'Prediction (Corr={metrics["correlation"]:.3f})')
                plt.colorbar(im1, ax=axes[1])

                im2 = axes[2].imshow(uncertainty, cmap='hot', aspect='auto', origin='upper')
                axes[2].set_title('Uncertainty (Std)')
                plt.colorbar(im2, ax=axes[2])

                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, f'sample_{i}_uq.png'),
                            dpi=300, bbox_inches='tight')
                plt.close()

            if (i + 1) % 5 == 0:
                print(f"    {i+1}/{n_eval} samples evaluated")

        # Aggregate metrics
        agg = {}
        for key in all_metrics[0].keys():
            vals = [m[key] for m in all_metrics
                    if isinstance(m.get(key), (int, float)) and not np.isnan(m.get(key, float('nan')))]
            if vals:
                agg[key] = float(np.mean(vals))
                agg[f'{key}_std'] = float(np.std(vals))

        all_results[array_type] = agg

        print(f"\n  Results ({array_type}):")
        for k, v in agg.items():
            if not k.endswith('_std') and isinstance(v, float):
                print(f"    {k}: {v:.4f}")

    # Save results
    save_dir = os.path.join(project_root, 'results', 'paper')
    os.makedirs(save_dir, exist_ok=True)
    safe_json_dump(all_results, os.path.join(save_dir, 'evaluation_results.json'))
    print(f"\nResults saved to {save_dir}")


if __name__ == '__main__':
    main()
