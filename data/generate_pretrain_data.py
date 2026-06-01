"""
预训练数据生成脚本 - 生成1000个随机地质模型（无标签）
用于自监督预训练（掩码预测任务）
用法: cd SelfSup-KAN-ERT && python data/generate_pretrain_data.py
"""

import os
import sys
import numpy as np
import pickle
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.forward_modeling import (
    create_electrodes, create_survey_scheme, run_forward,
    get_data_array
    
)
from data.model_generators import generate_random_model


CONFIG = {
    'n_samples': 3000,  # 提升至3000，增强预训练多样性
    'n_elec': 24,
    'elec_spacing': 0.8,
    'array_types': ['wenner', 'dipole-dipole', 'schlumberger'],  # 多种装置
    'output_dir': 'data/pretrain',
}


def generate_pretrain_sample(idx, elecs, scheme):
    """生成单个预训练样本（仅视电阻率，无标签）"""
    try:
        elec_x = elecs[:, 0]
        mesh, rho, rho_matrix = generate_random_model(seed=idx * 7 + 42, elec_x=elec_x)
        data = run_forward(mesh, rho, scheme, noise_level=0.02)
        rhoa = get_data_array(data)

        return {
            'index': idx,
            'rhoa': rhoa,           # 含少量噪声的视电阻率
            'rho_matrix': rho_matrix,  # 电阻率模型（用于验证，不用于训练）
        }
    except Exception as e:
        return None


def main():
    print("=" * 60)
    print("SelfSup-KAN-ERT 预训练数据生成")
    print("=" * 60)

    config = CONFIG
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, config['output_dir'])
    os.makedirs(output_dir, exist_ok=True)

    elecs = create_electrodes(config['n_elec'], config['elec_spacing'])
    array_types = config['array_types']
    n_per_array = config['n_samples'] // len(array_types)

    all_samples = []
    all_failed = 0

    for array_type in array_types:
        print(f"\nGenerating {n_per_array} samples for {array_type}...")
        scheme = create_survey_scheme(elecs, array_type)

        for idx in tqdm(range(n_per_array), desc=f"  {array_type}"):
            # 用 array_type 的 hash 偏移种子，避免不同装置生成相同模型
            global_idx = idx + hash(array_type) % 100000
            sample = generate_pretrain_sample(global_idx, elecs, scheme)
            if sample is not None:
                sample['array_type'] = array_type
                all_samples.append(sample)
            else:
                all_failed += 1

    print(f"\nGenerated {len(all_samples)}/{config['n_samples']} samples ({all_failed} failed)")

    # 按装置类型分组保存（不同装置的测量数不同，不能直接 stack）
    from collections import defaultdict
    grouped = defaultdict(list)
    for s in all_samples:
        grouped[s['array_type']].append(s)

    for array_type, samples in grouped.items():
        rhoa_arr = np.stack([s['rhoa'] for s in samples])

        # 保存 pkl
        filepath = os.path.join(output_dir, f'pretrain_{array_type}.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump({
                'rhoa': rhoa_arr,
                'n_samples': len(samples),
                'array_type': array_type,
                'config': config,
            }, f)

        # 保存 numpy
        np.save(os.path.join(output_dir, f'pretrain_{array_type}.npy'), rhoa_arr)

        print(f"  {array_type}: {rhoa_arr.shape}, range [{rhoa_arr.min():.1f}, {rhoa_arr.max():.1f}]")

    # 保存汇总（以最小维度的装置为准，用于向后兼容）
    min_n_data = min(len(s['rhoa']) for s in all_samples)
    rhoa_padded = []
    for s in all_samples:
        r = s['rhoa']
        if len(r) >= min_n_data:
            rhoa_padded.append(r[:min_n_data])
        else:
            rhoa_padded.append(np.pad(r, (0, min_n_data - len(r))))
    rhoa_all = np.stack(rhoa_padded)
    np.save(os.path.join(output_dir, 'pretrain_rhoa.npy'), rhoa_all)

    print(f"\nSaved to {output_dir}")
    print(f"Combined rhoa shape: {rhoa_all.shape}")


if __name__ == '__main__':
    main()
