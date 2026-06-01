# -*- coding: utf-8 -*-
"""
Tests for MA-PC-SRI core models.

Tests:
- FNOForward: forward shape, gradient flow, positivity
- MultiArrayEncoder: multi-array encoding, missing arrays
- SelfRefineDecoder: initial prediction, refinement, diversity
"""

import pytest
import torch
import numpy as np


class TestFNOForward:
    """Tests for FNO forward surrogate."""

    def test_forward_shape(self):
        from models.FNOForward import FNOForward
        model = FNOForward(grid_size=64, n_data=84, hidden_dim=16, n_layers=2, k_max=8)
        m = torch.rand(2, 64, 64) * 100 + 10
        rhoa = model(m)
        assert rhoa.shape == (2, 84)

    def test_forward_4d_input(self):
        from models.FNOForward import FNOForward
        model = FNOForward(grid_size=64, n_data=84, hidden_dim=16, n_layers=2, k_max=8)
        m = torch.rand(2, 1, 64, 64) * 100 + 10
        rhoa = model(m)
        assert rhoa.shape == (2, 84)

    def test_output_positive(self):
        from models.FNOForward import FNOForward
        model = FNOForward(grid_size=64, n_data=84, hidden_dim=16, n_layers=2, k_max=8)
        m = torch.rand(2, 64, 64) * 100 + 10
        rhoa = model(m)
        assert (rhoa > 0).all()

    def test_gradient_flow(self):
        from models.FNOForward import FNOForward
        model = FNOForward(grid_size=64, n_data=84, hidden_dim=16, n_layers=2, k_max=8)
        m = torch.rand(2, 64, 64) * 100 + 10
        m.requires_grad_(True)
        rhoa = model(m)
        loss = rhoa.sum()
        loss.backward()
        assert m.grad is not None
        assert not torch.isnan(m.grad).any()

    def test_create_fno_model(self):
        from models.FNOForward import create_fno_model
        model = create_fno_model('wenner', hidden_dim=16, n_layers=2, k_max=8)
        assert model.n_data == 84


class TestMultiArrayEncoder:
    """Tests for multi-array encoder."""

    def test_single_array(self):
        from models.MultiArrayEncoder import MultiArrayEncoder
        encoder = MultiArrayEncoder(
            array_configs={'wenner': 84},
            hidden_dim=16, n_inception=2, latent_dim=64
        )
        z, features = encoder({'wenner': torch.rand(2, 84)})
        assert z.shape == (2, 64)
        assert 'wenner' in features

    def test_multi_array(self):
        from models.MultiArrayEncoder import MultiArrayEncoder
        encoder = MultiArrayEncoder(
            array_configs={'wenner': 84, 'dipole-dipole': 231},
            hidden_dim=16, n_inception=2, latent_dim=64
        )
        z, features = encoder({
            'wenner': torch.rand(2, 84),
            'dipole-dipole': torch.rand(2, 231)
        })
        assert z.shape == (2, 64)
        assert len(features) == 2

    def test_missing_array(self):
        from models.MultiArrayEncoder import MultiArrayEncoder
        encoder = MultiArrayEncoder(
            array_configs={'wenner': 84, 'dipole-dipole': 231},
            hidden_dim=16, n_inception=2, latent_dim=64
        )
        # Only provide wenner, missing dipole-dipole
        z, features = encoder({'wenner': torch.rand(2, 84)})
        assert z.shape == (2, 64)

    def test_gradient_flow(self):
        from models.MultiArrayEncoder import MultiArrayEncoder
        encoder = MultiArrayEncoder(
            array_configs={'wenner': 84},
            hidden_dim=16, n_inception=2, latent_dim=64
        )
        rhoa = torch.rand(2, 84, requires_grad=True)
        z, _ = encoder({'wenner': rhoa})
        z.sum().backward()
        assert rhoa.grad is not None


class TestSelfRefineDecoder:
    """Tests for self-refining decoder."""

    def test_initial_prediction(self):
        from models.SelfRefineDecoder import SelfRefineDecoder
        decoder = SelfRefineDecoder(
            latent_dim=64, grid_size=64, n_data=84,
            hidden_dim=16, n_refine=3
        )
        z = torch.rand(2, 64)
        m_0 = decoder.initial_prediction(z)
        assert m_0.shape == (2, 64, 64)
        assert (m_0 > 0).all()

    def test_refine_step(self):
        from models.SelfRefineDecoder import SelfRefineDecoder
        decoder = SelfRefineDecoder(
            latent_dim=64, grid_size=64, n_data=84,
            hidden_dim=16, n_refine=3
        )
        m_k = torch.rand(2, 64, 64) * 100 + 10
        rhoa_pred = torch.rand(2, 84) * 100 + 10
        rhoa_obs = torch.rand(2, 84) * 100 + 10
        m_next, correction = decoder.refine_block(m_k, rhoa_pred - rhoa_obs)
        assert m_next.shape == (2, 64, 64)

    def test_forward_without_fno(self):
        from models.SelfRefineDecoder import SelfRefineDecoder
        decoder = SelfRefineDecoder(
            latent_dim=64, grid_size=64, n_data=84,
            hidden_dim=16, n_refine=3
        )
        z = torch.rand(2, 64)
        rhoa_obs = torch.rand(2, 84) * 100 + 10
        m_final, history, residuals = decoder(z, rhoa_obs, fno_model=None)
        assert m_final.shape == (2, 64, 64)
        assert len(history) == 1  # Only initial prediction

    def test_forward_with_fno(self):
        from models.SelfRefineDecoder import SelfRefineDecoder
        decoder = SelfRefineDecoder(
            latent_dim=64, grid_size=64, n_data=84,
            hidden_dim=16, n_refine=3
        )

        class DummyFNO(torch.nn.Module):
            def forward(self, m):
                return torch.rand(m.shape[0], 84) * 100 + 10

        z = torch.rand(2, 64)
        rhoa_obs = torch.rand(2, 84) * 100 + 10
        m_final, history, residuals = decoder(z, rhoa_obs, fno_model=DummyFNO(), n_steps=3)
        assert m_final.shape == (2, 64, 64)
        assert len(history) == 4  # initial + 3 refinement
        assert len(residuals) == 3

    def test_diverse_sampling(self):
        from models.SelfRefineDecoder import SelfRefineDecoder
        decoder = SelfRefineDecoder(
            latent_dim=64, grid_size=64, n_data=84,
            hidden_dim=16, n_refine=2
        )

        class DummyFNO(torch.nn.Module):
            def forward(self, m):
                return torch.rand(m.shape[0], 84) * 100 + 10

        z = torch.rand(2, 64)
        rhoa_obs = torch.rand(2, 84) * 100 + 10
        samples, uncertainties = decoder.sample_diverse(
            z, rhoa_obs, fno_model=DummyFNO(), n_samples=3
        )
        assert len(samples) == 3
        assert uncertainties.shape == (2, 64, 64)
        assert (uncertainties >= 0).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
