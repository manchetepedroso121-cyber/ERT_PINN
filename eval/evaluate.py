# -*- coding: utf-8 -*-
"""
Unified evaluation module for ERT inversion quality assessment.
All scripts should import metrics from here to ensure consistency.

Usage:
    from eval.evaluate import compute_metrics, denormalize_prediction
"""

import numpy as np


def denormalize_prediction(rho_pred_norm, rhoa_mean, rhoa_std):
    """Convert normalized prediction back to original resistivity scale.

    Args:
        rho_pred_norm: (H, W) predicted resistivity in normalized log-space
        rhoa_mean: mean of log(rhoa) used for normalization
        rhoa_std: std of log(rhoa) used for normalization

    Returns:
        rho_pred: (H, W) predicted resistivity in Ohm*m
    """
    return np.exp(rho_pred_norm * rhoa_std + rhoa_mean)


def compute_metrics(rho_true, rho_pred):
    """Compute all evaluation metrics.

    Args:
        rho_true: (H, W) true resistivity in Ohm*m
        rho_pred: (H, W) predicted resistivity in Ohm*m

    Returns:
        metrics: dict with keys: mse, rmse, mae, correlation, ssim,
                 relative_error, psnr, gmsd
    """
    mse = float(np.mean((rho_true - rho_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(rho_true - rho_pred)))
    corr = float(np.corrcoef(rho_true.ravel(), rho_pred.ravel())[0, 1])
    ssim = float(compute_ssim(rho_true, rho_pred))
    rel_error = float(np.mean(np.abs(rho_true - rho_pred) / (np.abs(rho_true) + 1e-6)))
    psnr = compute_psnr(rho_true, rho_pred)
    gmsd = compute_gmsd(rho_true, rho_pred)

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'correlation': corr,
        'ssim': ssim,
        'relative_error': rel_error,
        'psnr': psnr,
        'gmsd': gmsd,
    }


def compute_ssim(img1, img2, k1=0.01, k2=0.03, L=None, window_size=None):
    """Compute Structural Similarity Index (SSIM).

    Args:
        img1, img2: (H, W) images
        k1, k2: SSIM stability constants
        L: dynamic range of pixel values. If None, auto-computed from data range.
        window_size: if provided, compute windowed SSIM and average (standard approach).
                     If None, compute global SSIM (simplified).

    Returns:
        ssim: float in [-1, 1]
    """
    if L is None:
        L = max(float(img1.max()) - float(img1.min()),
                float(img2.max()) - float(img2.min()))
        if L < 1e-10:
            L = 1.0  # fallback for constant images

    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2

    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = img1.var()
    sigma2_sq = img2.var()
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)

    return numerator / denominator


def compute_boundary_error(rho_true, rho_pred, margin=5):
    """Compute error in boundary region.

    Args:
        rho_true, rho_pred: (H, W) resistivity
        margin: number of boundary pixels

    Returns:
        boundary_mae: float
    """
    h, w = rho_true.shape
    boundary_mask = np.zeros_like(rho_true, dtype=bool)
    boundary_mask[:margin, :] = True
    boundary_mask[-margin:, :] = True
    boundary_mask[:, :margin] = True
    boundary_mask[:, -margin:] = True

    return float(np.mean(np.abs(rho_true[boundary_mask] - rho_pred[boundary_mask])))


def compute_psnr(rho_true, rho_pred, max_val=None):
    """Compute Peak Signal-to-Noise Ratio (PSNR).

    PSNR = 10 * log10(MAX^2 / MSE)

    Args:
        rho_true: (H, W) true resistivity
        rho_pred: (H, W) predicted resistivity
        max_val: maximum value for dynamic range. If None, uses max of true.

    Returns:
        psnr: float in dB (higher is better)
    """
    mse = np.mean((rho_true - rho_pred) ** 2)
    if max_val is None:
        max_val = float(rho_true.max())
    if mse < 1e-10:
        return 100.0  # cap at 100 dB for near-perfect reconstruction
    return float(10 * np.log10(max_val ** 2 / mse))


def compute_gmsd(rho_true, rho_pred, c=0.001):
    """Compute Gradient Magnitude Similarity Deviation (GMSD).

    GMSD measures reconstruction quality based on gradient magnitude
    similarity. Lower is better (0 = perfect).

    Reference: Xue et al., "Gradient Magnitude Similarity Deviation:
    A Highly Efficient Perceptual Image Quality Index", IEEE TIP 2014

    Args:
        rho_true: (H, W) true resistivity
        rho_pred: (H, W) predicted resistivity
        c: stability constant

    Returns:
        gmsd: float (lower is better, 0 = perfect)
    """
    from scipy.ndimage import sobel

    def gradient_magnitude(img):
        gx = sobel(img, axis=1)
        gy = sobel(img, axis=0)
        return np.sqrt(gx ** 2 + gy ** 2)

    gm_true = gradient_magnitude(rho_true.astype(np.float64))
    gm_pred = gradient_magnitude(rho_pred.astype(np.float64))

    # Gradient magnitude similarity
    gms = (2 * gm_true * gm_pred + c) / (gm_true ** 2 + gm_pred ** 2 + c)

    return float(np.std(gms))


def forward_verify(model, mesh, solver, rhoa_obs, device, grid_size=64):
    """Forward verification: predict rho, run FEM forward, compare rhoa.

    Validates physical consistency by checking whether the predicted
    resistivity model reproduces the observed apparent resistivity
    when passed through the forward solver.

    Args:
        model: trained inversion model
        mesh: pyGIMLi mesh
        solver: ERTForwardSolver
        rhoa_obs: observed apparent resistivity (n_data,)
        device: torch device
        grid_size: evaluation grid size

    Returns:
        dict with: rhoa_pred, rhoa_obs, data_misfit, relative_misfit, rho_pred
    """
    import torch

    model.eval()
    cell_centers = solver.get_cell_centers(mesh)
    coords_norm = solver.normalize_coords(cell_centers)
    coords_tensor = torch.FloatTensor(coords_norm).to(device)

    with torch.no_grad():
        rho_pred, _ = model(coords_tensor)
        rho_pred_np = rho_pred.squeeze().cpu().numpy()

    # Forward modeling with predicted model
    rhoa_pred = solver.forward(mesh, rho_pred_np)

    # Data misfit (log-space RMSE)
    log_pred = np.log(rhoa_pred + 1e-6)
    log_obs = np.log(rhoa_obs + 1e-6)
    data_misfit = float(np.sqrt(np.mean((log_pred - log_obs) ** 2)))

    # Relative misfit
    relative_misfit = float(np.mean(np.abs(rhoa_pred - rhoa_obs) / (np.abs(rhoa_obs) + 1e-6)))

    return {
        'rhoa_pred': rhoa_pred,
        'rhoa_obs': rhoa_obs,
        'data_misfit': data_misfit,
        'relative_misfit': relative_misfit,
        'rho_pred': rho_pred_np,
    }


def evaluate_model_grid(model, rhoa_tensor, rho_true, rhoa_mean, rhoa_std, device, grid_size=64):
    """Evaluate model on full grid, returning both metrics and predicted resistivity.

    This is the single source of truth for model evaluation across all scripts.

    Args:
        model: trained model with forward(coords, rhoa) -> (rho, phi)
        rhoa_tensor: (1, n_data) normalized apparent resistivity
        rho_true: (H, W) true resistivity in Ohm*m
        rhoa_mean: normalization mean
        rhoa_std: normalization std
        device: torch device
        grid_size: evaluation grid resolution

    Returns:
        metrics: dict with evaluation metrics
        rho_pred: (H, W) predicted resistivity in Ohm*m
    """
    import torch

    model.eval()
    with torch.no_grad():
        x_norm = np.linspace(-1, 1, grid_size).astype(np.float32)
        z_norm = np.linspace(-1, 1, grid_size).astype(np.float32)
        X, Z = np.meshgrid(x_norm, z_norm)
        coords = torch.FloatTensor(np.column_stack([X.ravel(), Z.ravel()])).to(device)
        rhoa_batch = rhoa_tensor.expand(coords.shape[0], -1)
        rho_pred_norm, _ = model(coords, rhoa_batch)
        rho_pred_norm = rho_pred_norm.cpu().numpy().reshape(grid_size, grid_size)

    rho_pred = denormalize_prediction(rho_pred_norm, rhoa_mean, rhoa_std)
    metrics = compute_metrics(rho_true, rho_pred)

    return metrics, rho_pred


if __name__ == '__main__':
    # Self-test
    np.random.seed(42)
    rho_true = np.random.uniform(50, 200, (64, 64)).astype(np.float32)
    rho_pred = rho_true + np.random.randn(64, 64).astype(np.float32) * 10

    metrics = compute_metrics(rho_true, rho_pred)
    print("Evaluation metrics self-test:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    boundary_mae = compute_boundary_error(rho_true, rho_pred)
    print(f"  boundary_mae: {boundary_mae:.4f}")
