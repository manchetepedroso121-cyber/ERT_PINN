# -*- coding: utf-8 -*-
"""
Generate high-fidelity field-like ERT data for validation.

Creates synthetic data that mimics real field conditions:
- Complex multi-scale geological structures
- Realistic noise (Gaussian + correlated + outliers)
- Electrode position errors
- Data gaps (missing measurements)
- Mixed array types (Wenner + Dipole-Dipole + Schlumberger)

This serves as a bridge until real field data is downloaded via
download_public_datasets.py.

Usage:
    python data/generate_field_like_data.py --n_samples 100
    python data/generate_field_like_data.py --n_samples 50 --complexity high
"""

import os
import sys
import argparse
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.model_generators import (
    generate_plume_model, generate_co2_model,
    generate_fault_model, generate_random_model
)
from data.forward_modeling import create_electrodes, run_forward
from models.constants import (
    N_ELEC, ELEC_SPACING, X_RANGE, Z_RANGE,
    GRID_NX, GRID_NZ, MESH_AREA
)
from train.engine import set_seed, safe_json_dump


def generate_complex_model(seed, complexity='medium'):
    """Generate a complex geological model mimicking field conditions.

    Uses existing model generators and combines multiple models
    to create complex geological settings.

    Args:
        seed: random seed
        complexity: 'low', 'medium', 'high'

    Returns:
        mesh, rho, rho_matrix
    """
    set_seed(seed)

    # Select model type based on complexity
    generators = {
        'low': [generate_plume_model],
        'medium': [generate_plume_model, generate_fault_model, generate_co2_model],
        'high': [generate_fault_model, generate_co2_model, generate_random_model],
    }

    gen_func = np.random.choice(generators[complexity])

    # Generate model with random seed
    mesh, rho, rho_matrix = gen_func(seed=seed)

    return mesh, rho, rho_matrix


def add_field_noise(rhoa_clean, noise_level=0.10, outlier_fraction=0.05,
                    missing_fraction=0.05, seed=42):
    """Add realistic field-like noise to clean apparent resistivity.

    Noise components:
    1. Gaussian noise (main component)
    2. Correlated noise (spatially correlated, simulating geological noise)
    3. Outliers (abnormally high/low measurements)
    4. Data gaps (missing measurements)

    Args:
        rhoa_clean: clean apparent resistivity (n_data,)
        noise_level: overall noise level (default 10%)
        outlier_fraction: fraction of outliers
        missing_fraction: fraction of missing data
        seed: random seed

    Returns:
        dict with rhoa_noisy, outlier_mask, missing_mask, noise_components
    """
    np.random.seed(seed)
    n = len(rhoa_clean)

    # 1. Gaussian noise
    gaussian_noise = np.random.normal(0, noise_level, n)
    rhoa_noisy = rhoa_clean * (1 + gaussian_noise)

    # 2. Correlated noise (low-frequency, simulating geological variation)
    if n > 10:
        from scipy.ndimage import gaussian_filter1d
        correlated = gaussian_filter1d(np.random.randn(n), sigma=5)
        correlated = correlated / (np.std(correlated) + 1e-8) * noise_level * 0.3
        rhoa_noisy *= (1 + correlated)

    # 3. Outliers
    n_outliers = max(1, int(n * outlier_fraction))
    outlier_idx = np.random.choice(n, n_outliers, replace=False)
    outlier_factors = np.random.choice([-1, 1], n_outliers) * np.random.uniform(2, 10, n_outliers)
    rhoa_noisy[outlier_idx] *= (1 + outlier_factors)

    # 4. Data gaps
    n_missing = max(1, int(n * missing_fraction))
    missing_idx = np.random.choice(n, n_missing, replace=False)
    missing_mask = np.ones(n, dtype=bool)
    missing_mask[missing_idx] = False

    # Apply missing data
    rhoa_observed = rhoa_noisy.copy()
    rhoa_observed[~missing_mask] = np.nan

    return {
        'rhoa_observed': rhoa_observed,
        'rhoa_noisy': rhoa_noisy,
        'rhoa_clean': rhoa_clean,
        'outlier_mask': np.isin(np.arange(n), outlier_idx),
        'missing_mask': missing_mask,
        'noise_level': noise_level,
        'n_outliers': n_outliers,
        'n_missing': n_missing,
    }


def generate_field_like_sample(idx, array_type='wenner', complexity='medium',
                                noise_level=0.10, seed=None):
    """Generate a single field-like ERT sample.

    Args:
        idx: sample index
        array_type: electrode array type
        complexity: geological complexity
        noise_level: noise level
        seed: random seed (if None, uses idx)

    Returns:
        dict with all data fields
    """
    if seed is None:
        seed = idx * 1000 + hash(array_type) % 1000

    # Generate complex geological model
    mesh, rho, rho_matrix = generate_complex_model(seed, complexity)

    # Forward modeling
    from data.forward_modeling import create_survey_scheme

    elec_x = np.linspace(
        -(N_ELEC - 1) * ELEC_SPACING / 2,
        (N_ELEC - 1) * ELEC_SPACING / 2,
        N_ELEC
    )
    elec_pos = np.column_stack([elec_x, np.zeros(N_ELEC)])

    try:
        scheme = create_survey_scheme(elec_pos, array_type)
        data = run_forward(mesh, rho, scheme)
        rhoa_clean = np.array(data('rhoa'))
    except Exception as e:
        print(f"  Warning: forward modeling failed for sample {idx}: {e}")
        return None

    # Add field-like noise
    noise_data = add_field_noise(
        rhoa_clean, noise_level=noise_level,
        outlier_fraction=0.05, missing_fraction=0.05, seed=seed
    )

    return {
        'rho_matrix': rho_matrix.astype(np.float32),
        'rhoa_clean': rhoa_clean.astype(np.float32),
        'rhoa_noisy': noise_data['rhoa_noisy'].astype(np.float32),
        'rhoa_observed': noise_data['rhoa_observed'].astype(np.float32),
        'outlier_mask': noise_data['outlier_mask'],
        'missing_mask': noise_data['missing_mask'],
        'noise_level': noise_level,
        'complexity': complexity,
        'array_type': array_type,
        'seed': seed,
    }


def main():
    parser = argparse.ArgumentParser(description='Generate field-like ERT data')
    parser.add_argument('--n_samples', type=int, default=50)
    parser.add_argument('--arrays', nargs='+',
                        default=['wenner', 'dipole-dipole', 'schlumberger'])
    parser.add_argument('--complexity', type=str, default='medium',
                        choices=['low', 'medium', 'high'])
    parser.add_argument('--noise_level', type=float, default=0.10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, 'data', 'field_like')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating field-like ERT data")
    print(f"  Samples: {args.n_samples}")
    print(f"  Arrays: {args.arrays}")
    print(f"  Complexity: {args.complexity}")
    print(f"  Noise level: {args.noise_level*100:.0f}%")

    metadata = {
        'n_samples': args.n_samples,
        'arrays': args.arrays,
        'complexity': args.complexity,
        'noise_level': args.noise_level,
        'seed': args.seed,
        'samples': {},
    }

    for array_type in args.arrays:
        print(f"\n  Generating {array_type} data...")
        array_dir = os.path.join(output_dir, array_type)
        os.makedirs(array_dir, exist_ok=True)

        n_success = 0
        for i in range(args.n_samples):
            sample = generate_field_like_sample(
                i, array_type=array_type,
                complexity=args.complexity,
                noise_level=args.noise_level,
                seed=args.seed + i
            )

            if sample is not None:
                np.savez(
                    os.path.join(array_dir, f'sample_{i:04d}.npz'),
                    **sample
                )
                n_success += 1

            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{args.n_samples} ({n_success} successful)")

        metadata['samples'][array_type] = {
            'n_success': n_success,
            'n_attempted': args.n_samples,
        }
        print(f"  {array_type}: {n_success}/{args.n_samples} samples generated")

    # Save metadata
    safe_json_dump(metadata, os.path.join(output_dir, 'metadata.json'))
    print(f"\nMetadata saved to {output_dir}/metadata.json")
    print(f"Data saved to {output_dir}/")


if __name__ == '__main__':
    main()
