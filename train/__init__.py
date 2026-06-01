# -*- coding: utf-8 -*-
"""Training modules for MA-PC-SRI.

Provides:
- datasets: Shared dataset classes
- engine: Training engine utilities
"""

from train.datasets import InversionDataset, load_samples, load_eval_samples
from train.engine import set_seed, aggregate_metrics, safe_json_dump, DEFAULT_CONFIG
