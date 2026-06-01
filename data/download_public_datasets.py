# -*- coding: utf-8 -*-
"""
Download and process public ERT field datasets for validation.

Run this script when network is available to download real field data.
The downloaded data will be saved to data/field_public/ in a format
compatible with the project's evaluation pipeline.

Usage:
    python data/download_public_datasets.py --source all
    python data/download_public_datasets.py --source eki
    python data/download_public_datasets.py --source timelapse
"""

import os
import sys
import argparse
import subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download_eki_dataset(save_dir):
    """Download EKI Geophysics 2020 dataset (Tso et al., GJI 2021).

    Contains real field ERT data from multiple sites including a Chinese
    karstic hillslope (2D_chenqi-new) directly relevant to this project.

    URL: https://github.com/cmtso/EKI_geophysics_2020
    """
    target = os.path.join(save_dir, 'EKI_geophysics_2020')
    if os.path.exists(target):
        print(f"  Already downloaded: {target}")
        return target

    print("  Downloading EKI Geophysics 2020 dataset...")
    try:
        subprocess.run([
            'git', 'clone', '--depth', '1',
            'https://github.com/cmtso/EKI_geophysics_2020.git',
            target
        ], check=True, timeout=120)
        print(f"  Downloaded to {target}")
        return target
    except Exception as e:
        print(f"  Download failed: {e}")
        return None


def download_timelapse_dataset(save_dir):
    """Download pyGIMLi/BERT timelapse ERT repository.

    Contains multiple well-documented field datasets:
    - Hillslope (rain infiltration, Germany)
    - ALERT (tracer migration, UK)
    - Infiltration (3D, 392 electrodes)
    - SAMOS (saltwater monitoring)

    URL: https://github.com/gimli-org/timelapseERT
    """
    target = os.path.join(save_dir, 'timelapseERT')
    if os.path.exists(target):
        print(f"  Already downloaded: {target}")
        return target

    print("  Downloading timelapse ERT dataset...")
    try:
        subprocess.run([
            'git', 'clone', '--depth', '1',
            'https://github.com/gimli-org/timelapseERT.git',
            target
        ], check=True, timeout=120)
        print(f"  Downloaded to {target}")
        return target
    except Exception as e:
        print(f"  Download failed: {e}")
        return None


def download_stratigraphy_dataset(save_dir):
    """Download ERT field tests for stratigraphy interpretation.

    Three benchmark profiles: Fold, Horizontal, Inclined.

    URL: https://github.com/StratigraphyInterpretation/ERT-field-tests
    """
    target = os.path.join(save_dir, 'ERT-field-tests')
    if os.path.exists(target):
        print(f"  Already downloaded: {target}")
        return target

    print("  Downloading stratigraphy ERT field tests...")
    try:
        subprocess.run([
            'git', 'clone', '--depth', '1',
            'https://github.com/StratigraphyInterpretation/ERT-field-tests.git',
            target
        ], check=True, timeout=120)
        print(f"  Downloaded to {target}")
        return target
    except Exception as e:
        print(f"  Download failed: {e}")
        return None


def process_eki_chenqi(data_dir, output_dir):
    """Process the Chinese karstic hillslope dataset from EKI repo.

    This dataset is from Chenqi, China - a karstic hillslope survey.
    It's directly relevant as a Chinese geological setting.

    Args:
        data_dir: path to EKI_geophysics_2020/2D_chenqi-new
        output_dir: where to save processed data
    """
    from pygimli.physics import ert

    protocol_file = os.path.join(data_dir, 'protocol.dat')
    electrode_file = os.path.join(data_dir, 'electrodes.dat')

    if not os.path.exists(protocol_file):
        print(f"  Warning: {protocol_file} not found")
        return None

    # Load electrode positions
    elec = np.loadtxt(electrode_file, skiprows=1)
    print(f"  Electrodes: {elec.shape[0]} positions")

    # Load protocol (A, B, M, N indices + resistance)
    proto = np.loadtxt(protocol_file, skiprows=1)
    print(f"  Measurements: {proto.shape[0]}")

    # Save in project format
    os.makedirs(output_dir, exist_ok=True)
    np.savez(os.path.join(output_dir, 'chenqi_field.npz'),
             electrodes=elec,
             protocol=proto)
    print(f"  Saved to {output_dir}/chenqi_field.npz")
    return proto


def process_timelapse_hillslope(data_dir, output_dir):
    """Process the Hillslope timelapse dataset.

    2D ERT monitoring of rain infiltration in Mulde, Germany.
    Combined Wenner + dipole-dipole array.

    Args:
        data_dir: path to timelapseERT/Hillslope
        output_dir: where to save processed data
    """
    rhoa_file = os.path.join(data_dir, 'rhoa.shm')
    if not os.path.exists(rhoa_file):
        # Try alternative file names
        for f in os.listdir(data_dir):
            print(f"    Found: {f}")
        return None

    print(f"  Processing Hillslope dataset...")
    # Use pyGIMLi to load
    try:
        from pygimli.physics import ert
        data = ert.load(rhoa_file)
        rhoa = np.array(data('rhoa'))
        sensors = np.array([list(s) for s in data.sensors()])

        os.makedirs(output_dir, exist_ok=True)
        np.savez(os.path.join(output_dir, 'hillslope_field.npz'),
                 rhoa=rhoa,
                 sensors=sensors,
                 n_sensors=data.sensorCount(),
                 n_data=data.size())
        print(f"  Saved: {data.size()} measurements, {data.sensorCount()} electrodes")
        return data
    except Exception as e:
        print(f"  Processing failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Download public ERT datasets')
    parser.add_argument('--source', type=str, default='all',
                        choices=['all', 'eki', 'timelapse', 'stratigraphy'],
                        help='Which dataset to download')
    parser.add_argument('--process', action='store_true',
                        help='Process downloaded data into project format')
    args = parser.parse_args()

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'field_public')
    os.makedirs(save_dir, exist_ok=True)

    print(f"Downloading public ERT datasets to {save_dir}")

    results = {}
    if args.source in ['all', 'eki']:
        results['eki'] = download_eki_dataset(save_dir)
    if args.source in ['all', 'timelapse']:
        results['timelapse'] = download_timelapse_dataset(save_dir)
    if args.source in ['all', 'stratigraphy']:
        results['stratigraphy'] = download_stratigraphy_dataset(save_dir)

    if args.process:
        print("\nProcessing downloaded data...")
        output_dir = os.path.join(save_dir, 'processed')
        os.makedirs(output_dir, exist_ok=True)

        if results.get('eki'):
            chenqi_dir = os.path.join(results['eki'], '2D_chenqi-new')
            if os.path.exists(chenqi_dir):
                process_eki_chenqi(chenqi_dir, output_dir)

        if results.get('timelapse'):
            hillslope_dir = os.path.join(results['timelapse'], 'Hillslope')
            if os.path.exists(hillslope_dir):
                process_timelapse_hillslope(hillslope_dir, output_dir)

    print("\nDone! Summary:")
    for name, path in results.items():
        status = "OK" if path else "FAILED"
        print(f"  {name}: {status}")


if __name__ == '__main__':
    main()
