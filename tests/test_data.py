"""
数据模块单元测试
覆盖: forward_modeling, model_generators 接口
"""

import pytest
import numpy as np
import torch


class TestForwardModeling:
    """正演模拟接口测试"""

    def test_create_electrodes(self):
        from data.forward_modeling import create_electrodes
        elecs = create_electrodes(n_elec=24, spacing=1.0)
        assert elecs.shape == (24, 2)
        # 应该居中
        assert abs(elecs[:, 0].mean()) < 0.1

    def test_create_electrodes_custom_start(self):
        from data.forward_modeling import create_electrodes
        elecs = create_electrodes(n_elec=10, spacing=0.5, x_start=0)
        assert elecs.shape == (10, 2)
        assert elecs[0, 0] == 0.0

    def test_scheme_map_schlumberger(self):
        """验证 schlumberger 拼写变体"""
        from data.forward_modeling import create_survey_scheme, create_electrodes
        elecs = create_electrodes(n_elec=24, spacing=1.0)
        # 不应抛出异常
        scheme = create_survey_scheme(elecs, 'schlumberger')
        assert scheme is not None


class TestModelGenerators:
    """地质模型生成器测试"""

    def test_generate_plume_model(self):
        from data.model_generators import generate_plume_model
        mesh, rho, rho_matrix = generate_plume_model(seed=42)
        assert rho_matrix.shape == (64, 64)
        assert np.all(rho_matrix > 0), "Resistivity should be positive"

    def test_generate_co2_model(self):
        from data.model_generators import generate_co2_model
        mesh, rho, rho_matrix = generate_co2_model(seed=42)
        assert rho_matrix.shape == (64, 64)
        assert np.all(rho_matrix > 0)

    def test_generate_fault_model(self):
        from data.model_generators import generate_fault_model
        mesh, rho, rho_matrix = generate_fault_model(seed=42)
        assert rho_matrix.shape == (64, 64)
        assert np.all(rho_matrix > 0)

    def test_generate_random_model(self):
        from data.model_generators import generate_random_model
        mesh, rho, rho_matrix = generate_random_model(seed=42)
        assert rho_matrix.shape == (64, 64)
        assert np.all(rho_matrix > 0)

    def test_reproducibility(self):
        """相同种子应生成相同结果"""
        from data.model_generators import generate_plume_model
        _, _, m1 = generate_plume_model(seed=123)
        _, _, m2 = generate_plume_model(seed=123)
        np.testing.assert_array_equal(m1, m2)
