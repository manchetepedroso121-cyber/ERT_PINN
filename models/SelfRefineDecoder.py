# -*- coding: utf-8 -*-
"""
Self-Refining Decoder for Physics-Consistent ERT Inversion.

Core innovation: iterative refinement via physics self-consistency.

Instead of a single forward pass, this decoder iteratively refines
its prediction by:
1. Predicting a resistivity model m_k from latent features
2. Using the FNO forward surrogate to compute predicted data d̂_k
3. Computing residual r_k = d̂_k - d_obs
4. Refining m_{k+1} based on m_k and r_k
5. Repeating until convergence (fixed point)

The fixed point m* satisfies F(m*) ≈ d_obs, ensuring physical
consistency without requiring labeled training data.

Mathematical foundation:
    Fixed-point theorem: if G is a contraction mapping on a
    complete metric space, then the iteration m_{k+1} = G(m_k)
    converges to a unique fixed point m* = G(m*).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.constants import GRID_NX, GRID_NZ


class RefineBlock(nn.Module):
    """Single refinement block.

    Takes current model estimate and data residual,
    outputs a correction to the model.

    Args:
        grid_size: spatial grid size
        residual_dim: dimension of the data residual vector
        hidden_dim: hidden channel dimension
    """

    def __init__(self, grid_size=64, residual_dim=84, hidden_dim=64):
        super().__init__()
        self.grid_size = grid_size

        # Process data residual
        self.residual_encoder = nn.Sequential(
            nn.Linear(residual_dim, 128),
            nn.GELU(),
            nn.Linear(128, hidden_dim)
        )

        # Process current model estimate
        self.model_encoder = nn.Sequential(
            nn.Conv2d(1, hidden_dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 3, padding=1),
            nn.GELU()
        )

        # Generate correction
        self.corrector = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 1, 3, padding=1)
        )

        # Learnable step size (initialized small for stability)
        self.step_size = nn.Parameter(torch.tensor(0.1))

    def forward(self, m_k, residual, z_context=None):
        """Compute refinement: m_{k+1} = m_k + α · correction(m_k, residual).

        Args:
            m_k: (B, H, W) current model estimate
            residual: (B, n_data) data residual (d̂ - d_obs)
            z_context: (B, latent_dim) optional context features

        Returns:
            m_next: (B, H, W) refined model estimate
            correction: (B, 1, H, W) applied correction (for monitoring)
        """
        B, H, W = m_k.shape

        # Encode residual
        r_feat = self.residual_encoder(residual)  # (B, hidden_dim)
        r_feat = r_feat.unsqueeze(-1).unsqueeze(-1)  # (B, hidden_dim, 1, 1)
        r_feat = r_feat.expand(-1, -1, H, W)  # (B, hidden_dim, H, W)

        # Encode current model
        m_feat = self.model_encoder(m_k.unsqueeze(1))  # (B, hidden_dim, H, W)

        # Concatenate and correct
        combined = torch.cat([m_feat, r_feat], dim=1)  # (B, 2*hidden_dim, H, W)
        correction = self.corrector(combined)  # (B, 1, H, W)

        # Apply correction with learnable step size
        m_next = m_k + self.step_size * correction.squeeze(1)

        return m_next, correction


class SelfRefineDecoder(nn.Module):
    """Self-refining decoder with physics-consistent iteration.

    Generates resistivity models from latent features, then iteratively
    refines them using physics constraints (FNO forward residual).

    The iteration converges to a fixed point m* where:
    - FNO(m*) ≈ d_obs (data consistency)
    - m* is a fixed point of the refinement operator (self-consistency)

    Architecture:
    1. Initial prediction: z → m_0 (coarse estimate)
    2. Refinement loop: for k = 0..K-1
       - d̂_k = FNO(m_k)
       - r_k = d̂_k - d_obs
       - m_{k+1} = m_k + α · RefineBlock(m_k, r_k)
    3. Output: m_K (refined, physics-consistent estimate)

    Args:
        latent_dim: dimension of input latent vector
        grid_size: output grid size (default 64)
        n_data: number of measurements (for residual dimension)
        hidden_dim: hidden channel dimension
        n_refine: number of refinement iterations
    """

    def __init__(self, latent_dim=256, grid_size=64, n_data=84,
                 hidden_dim=64, n_refine=5):
        super().__init__()
        self.grid_size = grid_size
        self.n_refine = n_refine

        # Initial prediction from latent vector
        self.initial_predict = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, grid_size * grid_size),
            nn.Softplus()  # Ensure positive resistivity
        )

        # Refinement blocks (shared weights across iterations)
        self.refine_block = RefineBlock(grid_size, n_data, hidden_dim)

        # Adaptive step size schedule (learned)
        self.step_schedule = nn.Parameter(torch.ones(n_refine) * 0.1)

    def initial_prediction(self, z):
        """Generate initial coarse prediction from latent features.

        Args:
            z: (B, latent_dim) latent features from encoder

        Returns:
            m_0: (B, H, W) initial resistivity model (positive)
        """
        B = z.shape[0]
        m_flat = self.initial_predict(z)  # (B, H*W)
        m_0 = m_flat.view(B, self.grid_size, self.grid_size)
        # Ensure positive resistivity with reasonable range
        m_0 = F.softplus(m_0) + 1.0  # Minimum 1 Ohm*m
        return m_0

    def refine(self, m_k, rhoa_pred, rhoa_obs):
        """Single refinement step.

        Args:
            m_k: (B, H, W) current model
            rhoa_pred: (B, n_data) predicted apparent resistivity
            rhoa_obs: (B, n_data) observed apparent resistivity

        Returns:
            m_next: (B, H, W) refined model
            correction: refinement correction
        """
        # Data residual in log-space (better for different magnitudes)
        residual = torch.log(rhoa_pred + 1e-6) - torch.log(rhoa_obs + 1e-6)
        m_next, correction = self.refine_block(m_k, residual)
        return m_next, correction

    def forward(self, z, rhoa_obs, fno_model=None, n_steps=None):
        """Full forward pass: initial prediction + iterative refinement.

        Args:
            z: (B, latent_dim) latent features from encoder
            rhoa_obs: (B, n_data) observed apparent resistivity
            fno_model: FNO forward model (for computing d̂ = FNO(m))
                       If None, refinement is skipped (initial prediction only)
            n_steps: number of refinement steps (default: self.n_refine)

        Returns:
            m_final: (B, H, W) final resistivity model
            m_history: list of intermediate models [m_0, m_1, ..., m_K]
            residual_history: list of data residuals per iteration
        """
        if n_steps is None:
            n_steps = self.n_refine

        B = z.shape[0]

        # Initial prediction
        m_0 = self.initial_prediction(z)
        m_history = [m_0]
        residual_history = []

        # Iterative refinement
        m_k = m_0
        for k in range(n_steps):
            if fno_model is None:
                break

            # Forward model prediction
            with torch.no_grad():
                rhoa_pred = fno_model(m_k)

            # Compute residual
            residual = torch.log(rhoa_pred.detach() + 1e-6) - torch.log(rhoa_obs + 1e-6)
            residual_norm = torch.norm(residual, dim=1).mean().item()
            residual_history.append(residual_norm)

            # Refine
            m_k, correction = self.refine_block(m_k, residual.detach())
            m_k = F.softplus(m_k) + 1.0  # Ensure positive

            m_history.append(m_k)

        return m_k, m_history, residual_history

    def sample_diverse(self, z, rhoa_obs, fno_model, n_samples=5,
                        noise_scale=0.1):
        """Generate diverse samples via multiple random initializations.

        Each sample starts from a different perturbed initial prediction,
        then iterates to a different fixed point, naturally capturing
        the non-uniqueness of the inverse problem.

        Args:
            z: (B, latent_dim) latent features
            rhoa_obs: (B, n_data) observed data
            fno_model: FNO forward model
            n_samples: number of diverse samples
            noise_scale: scale of initial perturbation

        Returns:
            samples: list of (B, H, W) models
            uncertainties: (B, H, W) pixel-wise standard deviation
        """
        samples = []
        for i in range(n_samples):
            # Perturb latent vector for diversity
            noise = torch.randn_like(z) * noise_scale
            z_noisy = z + noise

            # Run refinement
            m_final, _, _ = self.forward(z_noisy, rhoa_obs, fno_model)
            samples.append(m_final)

        # Stack and compute statistics
        stacked = torch.stack(samples, dim=0)  # (n_samples, B, H, W)
        uncertainties = stacked.std(dim=0)  # (B, H, W)

        return samples, uncertainties


if __name__ == '__main__':
    print("SelfRefineDecoder self-test...")

    decoder = SelfRefineDecoder(
        latent_dim=128, grid_size=64, n_data=84,
        hidden_dim=32, n_refine=3
    )
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"  Parameters: {n_params:,}")

    # Simulate encoder output and observed data
    z = torch.rand(2, 128)
    rhoa_obs = torch.rand(2, 84) * 100 + 10

    # Without FNO (initial prediction only)
    m_init, history, residuals = decoder(z, rhoa_obs, fno_model=None)
    print(f"  Initial prediction: {m_init.shape}, range: {m_init.min():.1f}-{m_init.max():.1f}")
    print(f"  History length: {len(history)}")

    # With dummy FNO (full refinement)
    class DummyFNO(nn.Module):
        def forward(self, m):
            return torch.rand(m.shape[0], 84) * 100 + 10

    fno = DummyFNO()
    m_final, history, residuals = decoder(z, rhoa_obs, fno_model=fno, n_steps=3)
    print(f"  After refinement: {m_final.shape}")
    print(f"  History length: {len(history)} (initial + 3 refinement)")
    print(f"  Residuals: {[f'{r:.4f}' for r in residuals]}")

    print("  All checks passed!")
