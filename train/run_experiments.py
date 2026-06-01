# -*- coding: utf-8 -*-
"""
Experiment runner for MA-PC-SRI.

Runs all experiments for the paper:
1. FNO forward surrogate training
2. Self-refining inversion training
3. Baseline comparisons
4. Ablation studies
5. Uncertainty quantification visualization

Usage:
    python train/run_experiments.py --full
    python train/run_experiments.py --quick  # Quick test
"""

import os
import sys
import json
import argparse
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.constants import N_ELEC, GRID_NX, GRID_NZ
from train.engine import set_seed, safe_json_dump
from eval.evaluate import compute_metrics
from eval.plot_residuals import (
    plot_residual_comparison,
    plot_resistivity_comparison,
    plot_forward_verification,
)


def run_fno_experiment(args, array_type, device):
    """Run FNO forward surrogate experiment."""
    from train.train_fno import FNOTrainingDataset, train_epoch, validate
    from models.FNOForward import create_fno_model
    from torch.utils.data import DataLoader

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data', 'training', array_type)

    train_loader = DataLoader(
        FNOTrainingDataset(os.path.join(data_dir, 'train')),
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        FNOTrainingDataset(os.path.join(data_dir, 'val')),
        batch_size=args.batch_size, shuffle=False
    )

    model = create_fno_model(array_type, hidden_dim=64, n_layers=4, k_max=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.fno_epochs)
    criterion = torch.nn.MSELoss()

    import torch.nn as nn
    best_mse = float('inf')
    for epoch in range(args.fno_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        if (epoch + 1) % 20 == 0:
            val_metrics = validate(model, val_loader, device)
            if val_metrics['mse'] < best_mse:
                best_mse = val_metrics['mse']
                save_dir = os.path.join(project_root, 'results', 'fno', array_type)
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_fno.pth'))

    return model, best_mse


def run_selfrefine_experiment(args, array_type, fno_model, device):
    """Run self-refining inversion experiment."""
    from models.MultiArrayEncoder import MultiArrayEncoder
    from models.SelfRefineDecoder import SelfRefineDecoder
    from train.train_selfrefine import SelfRefineLoss, train_epoch
    from torch.utils.data import DataLoader
    from train.datasets import InversionDataset

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data', 'training', array_type, 'train')

    n_data_map = {'wenner': 84, 'dipole-dipole': 231, 'schlumberger': 121}
    n_data = n_data_map.get(array_type, 84)

    dataset = InversionDataset(data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    encoder = MultiArrayEncoder(
        array_configs={array_type: n_data},
        hidden_dim=args.hidden_dim, latent_dim=args.latent_dim
    ).to(device)

    decoder = SelfRefineDecoder(
        latent_dim=args.latent_dim, grid_size=GRID_NX, n_data=n_data,
        hidden_dim=args.hidden_dim, n_refine=args.n_refine
    ).to(device)

    criterion = SelfRefineLoss()
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float('inf')
    for epoch in range(args.epochs):
        epoch_losses = []
        for batch in loader:
            rhoa_batch = batch[1] if isinstance(batch, (list, tuple)) else batch
            loss_dict = train_epoch(
                encoder, decoder, fno_model, rhoa_batch, criterion, optimizer, device
            )
            epoch_losses.append(loss_dict)
        scheduler.step()

        avg_loss = np.mean([d['total'] for d in epoch_losses])
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_dir = os.path.join(project_root, 'results', 'selfrefine', array_type)
            os.makedirs(save_dir, exist_ok=True)
            torch.save(encoder.state_dict(), os.path.join(save_dir, 'best_encoder.pth'))
            torch.save(decoder.state_dict(), os.path.join(save_dir, 'best_decoder.pth'))

    return encoder, decoder, best_loss


def main():
    parser = argparse.ArgumentParser(description='Run MA-PC-SRI experiments')
    parser.add_argument('--array', type=str, default='wenner',
                        choices=['wenner', 'dipole-dipole', 'schlumberger'])
    parser.add_argument('--fno_epochs', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--n_refine', type=int, default=5)
    parser.add_argument('--n_samples_uq', type=int, default=10)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.quick:
        args.fno_epochs = 5
        args.epochs = 10
        args.n_samples_uq = 3

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_root, 'results', 'paper')
    os.makedirs(results_dir, exist_ok=True)

    all_results = {}

    # Step 1: Train FNO
    print(f"\n{'='*60}")
    print(f"Step 1: Training FNO forward surrogate ({args.array})")
    print(f"{'='*60}")
    fno_model, fno_mse = run_fno_experiment(args, args.array, device)
    all_results['fno_mse'] = fno_mse
    print(f"  FNO MSE: {fno_mse:.4f}")

    # Step 2: Train self-refining inversion
    print(f"\n{'='*60}")
    print(f"Step 2: Training self-refining inversion ({args.array})")
    print(f"{'='*60}")
    encoder, decoder, sr_loss = run_selfrefine_experiment(
        args, args.array, fno_model, device
    )
    all_results['selfrefine_loss'] = sr_loss
    print(f"  Best loss: {sr_loss:.6f}")

    # Step 3: Uncertainty quantification
    print(f"\n{'='*60}")
    print(f"Step 3: Uncertainty quantification")
    print(f"{'='*60}")
    from train.datasets import InversionDataset, load_eval_samples
    test_dir = os.path.join(project_root, 'data', 'training', args.array, 'test')
    if os.path.exists(test_dir):
        test_samples = load_eval_samples(InversionDataset(test_dir))
        rhoa_test = test_samples[0][1].unsqueeze(0).to(device)

        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            rhoa_dict = {args.array: rhoa_test}
            z, _ = encoder(rhoa_dict)
            samples, uncertainties = decoder.sample_diverse(
                z, rhoa_test, fno_model, n_samples=args.n_samples_uq
            )

        mean_model = torch.stack(samples).mean(dim=0).squeeze().cpu().numpy()
        uncertainty = uncertainties.squeeze().cpu().numpy()
        rho_true = test_samples[0][2].numpy()

        # Plot
        fig_dir = os.path.join(results_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)

        plot_resistivity_comparison(
            rho_true, [mean_model],
            ['MA-PC-SRI Mean'],
            os.path.join(fig_dir, 'uq_mean_prediction.png'),
            title='Posterior Mean Prediction'
        )

        # Save uncertainty map
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(uncertainty, cmap='hot', aspect='auto', origin='upper')
        ax.set_title('Prediction Uncertainty (Std)')
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'uncertainty_map.png'), dpi=300)
        plt.close()

        all_results['mean_mse'] = float(np.mean((mean_model - rho_true) ** 2))
        all_results['mean_correlation'] = float(np.corrcoef(
            mean_model.ravel(), rho_true.ravel()
        )[0, 1])

        print(f"  Mean MSE: {all_results['mean_mse']:.2f}")
        print(f"  Mean Correlation: {all_results['mean_correlation']:.4f}")

    # Save results
    safe_json_dump(all_results, os.path.join(results_dir, 'experiment_results.json'))
    print(f"\nResults saved to {results_dir}")


if __name__ == '__main__':
    main()
