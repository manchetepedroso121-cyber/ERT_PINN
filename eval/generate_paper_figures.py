# -*- coding: utf-8 -*-
"""
Generate all figures for the MA-PC-SRI paper.

Creates publication-quality figures:
1. Method architecture diagram (described, not drawn)
2. Multi-array fusion comparison
3. Self-refinement convergence
4. Uncertainty quantification maps
5. Baseline comparison heatmaps
6. Ablation study bar charts
7. Noise robustness curves
8. Training loss curves

Usage:
    python eval/generate_paper_figures.py
    python eval/generate_paper_figures.py --array wenner
"""

import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.constants import GRID_NX, GRID_NZ


def load_results(project_root, filename):
    """Load JSON results file."""
    path = os.path.join(project_root, 'results', 'paper', filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def plot_method_comparison(results, save_dir):
    """Plot bar chart comparing MA-PC-SRI with baselines."""
    import matplotlib.pyplot as plt

    if not results:
        print("  No baseline results found")
        return

    for array_type, data in results.items():
        methods = list(data.keys())
        metrics = ['mse', 'correlation', 'psnr', 'ssim']
        available = [m for m in metrics if m in list(data.values())[0]]

        fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5))
        if len(available) == 1:
            axes = [axes]

        for idx, metric in enumerate(available):
            values = [data[m].get(metric, 0) for m in methods]
            colors = ['steelblue'] * len(methods)
            if 'MA-PC-SRI' in methods:
                colors[methods.index('MA-PC-SRI')] = 'coral'

            axes[idx].bar(range(len(methods)), values, color=colors, alpha=0.8)
            axes[idx].set_xticks(range(len(methods)))
            axes[idx].set_xticklabels(methods, rotation=45, ha='right')
            axes[idx].set_title(metric.upper())
            axes[idx].grid(True, alpha=0.3, axis='y')

        fig.suptitle(f'Method Comparison: {array_type}', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'comparison_{array_type}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: comparison_{array_type}.png")


def plot_ablation_results(results, save_dir):
    """Plot ablation study results."""
    import matplotlib.pyplot as plt

    if not results:
        print("  No ablation results found")
        return

    for array_type, data in results.items():
        variants = list(data.keys())
        metrics = ['mse', 'correlation', 'psnr']
        available = [m for m in metrics if m in list(data.values())[0]]

        fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5))
        if len(available) == 1:
            axes = [axes]

        for idx, metric in enumerate(available):
            values = [data[v].get(metric, 0) for v in variants]
            colors = ['coral' if 'full' in v.lower() else 'steelblue' for v in variants]

            axes[idx].bar(range(len(variants)), values, color=colors, alpha=0.8)
            axes[idx].set_xticks(range(len(variants)))
            axes[idx].set_xticklabels(variants, rotation=45, ha='right', fontsize=8)
            axes[idx].set_title(metric.upper())
            axes[idx].grid(True, alpha=0.3, axis='y')

        fig.suptitle(f'Ablation Study: {array_type}', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'ablation_{array_type}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: ablation_{array_type}.png")


def plot_training_curves(history, save_dir, name='training'):
    """Plot training loss curves."""
    import matplotlib.pyplot as plt

    if not history:
        print(f"  No {name} history found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Training loss
    if 'loss' in history:
        axes[0].plot(history['loss'], label='Total Loss', linewidth=2)
    if 'data' in history:
        axes[0].plot(history['data'], label='Data Loss', linewidth=1.5, alpha=0.7)
    if 'selfconsist' in history:
        axes[0].plot(history['selfconsist'], label='Self-Consistency', linewidth=1.5, alpha=0.7)
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    # Validation metrics
    if 'val_mse' in history:
        axes[1].plot(history['val_mse'], label='Val MSE', linewidth=2, color='coral')
    if 'val_psnr' in history:
        ax2 = axes[1].twinx()
        ax2.plot(history['val_psnr'], label='Val PSNR', linewidth=2, color='steelblue')
        ax2.set_ylabel('PSNR (dB)')
        ax2.legend(loc='upper left')
    if 'val_rel_error' in history:
        axes[1].plot(history['val_rel_error'], label='Val Rel Error', linewidth=1.5, alpha=0.7)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Metric')
    axes[1].set_title('Validation Metrics')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{name}_curves.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {name}_curves.png")


def plot_metrics_summary(all_results, save_dir):
    """Create a summary table figure."""
    import matplotlib.pyplot as plt

    if not all_results:
        print("  No results for summary")
        return

    arrays = list(all_results.keys())
    metrics = ['mse', 'correlation', 'psnr', 'ssim', 'gmsd']

    # Create table data
    fig, ax = plt.subplots(figsize=(12, 3 + len(arrays)))
    ax.axis('off')

    cell_text = []
    for arr in arrays:
        row = [arr]
        for m in metrics:
            val = all_results[arr].get(m, float('nan'))
            if np.isnan(val):
                row.append('N/A')
            elif m in ['mse', 'gmsd']:
                row.append(f'{val:.2f}')
            else:
                row.append(f'{val:.4f}')
        cell_text.append(row)

    col_labels = ['Array'] + [m.upper() for m in metrics]
    table = ax.table(cellText=cell_text, colLabels=col_labels,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)

    # Color header
    for j, label in enumerate(col_labels):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    plt.title('MA-PC-SRI Evaluation Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(os.path.join(save_dir, 'metrics_summary.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: metrics_summary.png")


def main():
    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('--array', type=str, default='wenner')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(project_root, 'results', 'paper', 'figures')
    os.makedirs(save_dir, exist_ok=True)

    print("Generating paper figures...")

    # 1. Baseline comparison
    baselines = load_results(project_root, 'baseline_comparison.json')
    if baselines:
        plot_method_comparison(baselines, save_dir)

    # 2. Ablation study
    ablation = load_results(project_root, f'ablation_{args.array}.json')
    if ablation:
        plot_ablation_results({args.array: ablation}, save_dir)

    # 3. Training curves (FNO)
    fno_history_path = os.path.join(project_root, 'results', 'fno',
                                      args.array, 'training_history.json')
    if os.path.exists(fno_history_path):
        with open(fno_history_path) as f:
            fno_history = json.load(f)
        plot_training_curves(fno_history, save_dir, name='fno')

    # 4. Training curves (SelfRefine)
    sr_history_path = os.path.join(project_root, 'results', 'selfrefine',
                                     args.array, 'training_history.json')
    if os.path.exists(sr_history_path):
        with open(sr_history_path) as f:
            sr_history = json.load(f)
        plot_training_curves(sr_history, save_dir, name='selfrefine')

    # 5. Evaluation summary
    eval_results = load_results(project_root, 'evaluation_results.json')
    if eval_results:
        plot_metrics_summary(eval_results, save_dir)

    # 6. Experiment results
    exp_results = load_results(project_root, 'experiment_results.json')
    if exp_results:
        print(f"\n  Experiment results:")
        for k, v in exp_results.items():
            if isinstance(v, (int, float)):
                print(f"    {k}: {v:.4f}")

    print(f"\nAll figures saved to {save_dir}")


if __name__ == '__main__':
    main()
