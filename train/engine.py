# -*- coding: utf-8 -*-
"""
Shared training engine for MA-PC-SRI.

Centralizes:
- Seed setting with CUDA reproducibility
- Metrics aggregation across seeds
- Safe JSON serialization (handles NaN/Inf)
- Default configuration
- Checkpoint utilities
"""

import os
import json
import math
import numpy as np
import torch


# ============================================================
# Default Configuration (single source of truth)
# ============================================================

DEFAULT_CONFIG = {
    # FNO training
    'fno_epochs': 200,
    'fno_hidden_dim': 64,
    'fno_n_layers': 4,
    'fno_k_max': 16,
    'fno_lr': 1e-3,

    # Self-refine training
    'sr_epochs': 500,
    'sr_hidden_dim': 64,
    'sr_latent_dim': 256,
    'sr_n_refine': 5,
    'sr_lr': 1e-4,
    'sr_lambda_data': 1.0,
    'sr_lambda_selfconsist': 0.5,
    'sr_lambda_tv': 0.01,
    'sr_lambda_smooth': 0.01,

    # General
    'batch_size': 8,
    'patience': 300,
    'checkpoint_freq': 100,
    'grad_clip_norm': 1.0,
    'seed': 42,
}


# ============================================================
# Seed Setting
# ============================================================

def set_seed(seed):
    """Set all random seeds for reproducibility.

    Args:
        seed: integer seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Metrics Aggregation
# ============================================================

def aggregate_metrics(metrics_list):
    """Aggregate metrics across multiple seeds/samples.

    Computes mean and std for each metric.

    Args:
        metrics_list: list of dicts

    Returns:
        aggregated: dict with mean and _std keys
    """
    agg = {}
    for key in metrics_list[0].keys():
        if not key.endswith('_std'):
            vals = [m[key] for m in metrics_list
                    if isinstance(m.get(key), (int, float)) and not np.isnan(m.get(key, float('nan')))]
            if vals:
                agg[key] = float(np.mean(vals))
                agg[f'{key}_std'] = float(np.std(vals))
            else:
                agg[key] = float('nan')
                agg[f'{key}_std'] = float('nan')
    return agg


# ============================================================
# Safe JSON Serialization
# ============================================================

def safe_json_dump(data, filepath, indent=2):
    """JSON dump that converts NaN/Inf to None (valid JSON null).

    Python's json.dump produces invalid JSON when encountering NaN/Inf.
    This function sanitizes the data first.

    Args:
        data: dict/list to serialize
        filepath: output file path
        indent: JSON indentation
    """
    def sanitize(obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        elif isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(sanitize(data), f, indent=indent)


# ============================================================
# Convergence Detection
# ============================================================

def check_convergence(loss_history, window=100, threshold=0.003):
    """Check if training has converged.

    Convergence: relative loss decrease over last `window` epochs < threshold.

    Args:
        loss_history: list of loss values
        window: number of epochs to check
        threshold: minimum relative decrease

    Returns:
        converged: bool
    """
    if len(loss_history) < window:
        return False
    recent = loss_history[-window:]
    if recent[0] == 0:
        return True
    decrease = (recent[0] - recent[-1]) / recent[0]
    return decrease < threshold


# ============================================================
# Checkpoint Utilities
# ============================================================

def _find_latest_checkpoint(save_dir):
    """Find the latest checkpoint file in save_dir.

    Args:
        save_dir: directory containing checkpoints

    Returns:
        path to latest checkpoint, or None if no checkpoint found
    """
    import glob

    if not save_dir or not os.path.isdir(save_dir):
        return None

    candidates = []
    for pattern in ['best_*.pth', 'final_*.pth', 'checkpoint_epoch_*.pth']:
        candidates.extend(glob.glob(os.path.join(save_dir, pattern)))

    if not candidates:
        return None

    return max(candidates, key=os.path.getmtime)
