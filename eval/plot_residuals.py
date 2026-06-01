# -*- coding: utf-8 -*-
"""
Visualization functions for ERT inversion quality assessment.

Provides publication-quality plots matching the SRERTF-Net paper style:
- Residual comparison (predicted vs observed rhoa)
- Single-trace method comparison
- Side-by-side resistivity heatmaps
- Forward verification scatter plot

Usage:
    from eval.plot_residuals import (
        plot_residual_comparison,
        plot_single_trace_comparison,
        plot_resistivity_comparison,
        plot_forward_verification,
    )
"""

import os
import numpy as np


def plot_residual_comparison(rhoa_pred, rhoa_obs, save_path, title="Residual Comparison"):
    """Plot predicted vs observed apparent resistivity with residuals.

    Creates a 2-panel figure:
    - Top: predicted vs observed rhoa (semilog)
    - Bottom: relative error bar chart

    Args:
        rhoa_pred: (n_data,) predicted apparent resistivity
        rhoa_obs: (n_data,) observed apparent resistivity
        save_path: output file path (.png)
        title: figure title
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})

    idx = np.arange(len(rhoa_obs))

    # Top panel: overlay
    axes[0].semilogy(idx, rhoa_obs, 'b-', alpha=0.7, label='Observed', linewidth=1.5)
    axes[0].semilogy(idx, rhoa_pred, 'r--', alpha=0.7, label='Predicted', linewidth=1.5)
    axes[0].set_ylabel('Apparent Resistivity (Ohm*m)')
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Bottom panel: relative error
    residual = (rhoa_pred - rhoa_obs) / (np.abs(rhoa_obs) + 1e-6) * 100
    axes[1].bar(idx, residual, color='steelblue', alpha=0.7, width=1.0)
    axes[1].axhline(y=0, color='k', linewidth=0.5)
    axes[1].axhline(y=5, color='r', linewidth=0.5, linestyle='--', alpha=0.5, label='+/- 5%')
    axes[1].axhline(y=-5, color='r', linewidth=0.5, linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Measurement Index')
    axes[1].set_ylabel('Relative Error (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_single_trace_comparison(rhoa_pred_list, rhoa_obs, labels, save_path,
                                  title="Method Comparison"):
    """Plot comparison of multiple methods on a single trace.

    Args:
        rhoa_pred_list: list of (n_data,) predicted rhoa arrays from different methods
        rhoa_obs: (n_data,) observed rhoa array
        labels: list of method names
        save_path: output file path (.png)
        title: figure title
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    idx = np.arange(len(rhoa_obs))

    ax.semilogy(idx, rhoa_obs, 'k-', alpha=0.8, label='Observed', linewidth=2)

    colors = ['steelblue', 'coral', 'forestgreen', 'purple', 'orange', 'brown']
    for i, (pred, label) in enumerate(zip(rhoa_pred_list, labels)):
        ax.semilogy(idx, pred, '--', color=colors[i % len(colors)],
                     alpha=0.7, label=label, linewidth=1.5)

    ax.set_xlabel('Measurement Index')
    ax.set_ylabel('Apparent Resistivity (Ohm*m)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_resistivity_comparison(rho_true_grid, rho_pred_grids, labels, save_path,
                                 title="Resistivity Model Comparison"):
    """Plot side-by-side resistivity model comparison.

    Creates a row of heatmaps: ground truth | method1 | method2 | ...
    Matches the SRERTF-Net paper figure style.

    Args:
        rho_true_grid: (H, W) ground truth resistivity
        rho_pred_grids: list of (H, W) predicted resistivity arrays
        labels: list of method names
        save_path: output file path (.png)
        title: figure title
    """
    import matplotlib.pyplot as plt

    n_panels = 1 + len(rho_pred_grids)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    if n_panels == 1:
        axes = [axes]

    # Common color scale
    vmin = min(rho_true_grid.min(), min(p.min() for p in rho_pred_grids))
    vmax = max(rho_true_grid.max(), max(p.max() for p in rho_pred_grids))

    im = axes[0].imshow(rho_true_grid, cmap='jet', aspect='auto',
                         vmin=vmin, vmax=vmax, origin='upper')
    axes[0].set_title('Ground Truth')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Z')
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    for i, (pred, label) in enumerate(zip(rho_pred_grids, labels)):
        im = axes[i + 1].imshow(pred, cmap='jet', aspect='auto',
                                 vmin=vmin, vmax=vmax, origin='upper')
        axes[i + 1].set_title(label)
        axes[i + 1].set_xlabel('X')
        plt.colorbar(im, ax=axes[i + 1], shrink=0.8)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_forward_verification(rhoa_pred, rhoa_obs, save_path,
                               title="Forward Verification"):
    """Plot forward verification: observed vs predicted rhoa scatter + misfit.

    Creates a 2-panel figure:
    - Left: scatter plot of observed vs predicted rhoa (log-log)
    - Right: misfit per measurement

    Args:
        rhoa_pred: (n_data,) predicted apparent resistivity (from forward model)
        rhoa_obs: (n_data,) observed apparent resistivity
        save_path: output file path (.png)
        title: figure title
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: scatter plot
    ax = axes[0]
    ax.loglog(rhoa_obs, rhoa_pred, 'o', markersize=3, alpha=0.5, color='steelblue')

    # Perfect fit line
    lims = [min(rhoa_obs.min(), rhoa_pred.min()),
            max(rhoa_obs.max(), rhoa_pred.max())]
    ax.loglog(lims, lims, 'r--', linewidth=1.5, label='Perfect fit')

    ax.set_xlabel('Observed rhoa (Ohm*m)')
    ax.set_ylabel('Predicted rhoa (Ohm*m)')
    ax.set_title(f'{title}\nCorrelation: {np.corrcoef(np.log(rhoa_obs + 1e-6), np.log(rhoa_pred + 1e-6))[0, 1]:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Right: misfit
    ax = axes[1]
    misfit = (np.log(rhoa_pred + 1e-6) - np.log(rhoa_obs + 1e-6)) ** 2
    idx = np.arange(len(misfit))
    ax.bar(idx, misfit, color='coral', alpha=0.7, width=1.0)
    ax.axhline(y=np.mean(misfit), color='k', linestyle='--', linewidth=1,
               label=f'RMSE: {np.sqrt(np.mean(misfit)):.4f}')
    ax.set_xlabel('Measurement Index')
    ax.set_ylabel('Log-space squared error')
    ax.set_title('Data Misfit')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


if __name__ == '__main__':
    # Quick self-test with synthetic data
    np.random.seed(42)
    n = 84
    rhoa_obs = np.random.uniform(50, 200, n).astype(np.float32)
    rhoa_pred = rhoa_obs * (1 + np.random.normal(0, 0.1, n))

    os.makedirs('figures/test', exist_ok=True)

    plot_residual_comparison(rhoa_pred, rhoa_obs, 'figures/test/residual.png', 'Test')
    plot_single_trace_comparison(
        [rhoa_pred * 0.9, rhoa_pred * 1.1], rhoa_obs,
        ['Method A', 'Method B'], 'figures/test/trace.png'
    )
    rho_true = np.random.uniform(50, 200, (64, 64)).astype(np.float32)
    plot_resistivity_comparison(
        rho_true, [rho_true + np.random.randn(64, 64) * 10],
        ['Prediction'], 'figures/test/resistivity.png'
    )
    plot_forward_verification(rhoa_pred, rhoa_obs, 'figures/test/forward.png')
    print("All plots generated successfully!")
