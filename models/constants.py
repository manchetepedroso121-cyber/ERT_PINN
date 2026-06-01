# -*- coding: utf-8 -*-
"""
Shared constants for the SelfSup-KAN-ERT project.

Centralizes magic numbers used across multiple modules to improve
maintainability and consistency.
"""


# ============================================================
# Numerical Stability
# ============================================================

EPSILON = 1e-6          # Floor for log() and division to prevent NaN/Inf
EPSILON_LOOSE = 1e-8    # Looser epsilon for gradient projection denominators


# ============================================================
# Domain Geometry
# ============================================================

X_RANGE = (-10, 10)     # Horizontal extent (meters)
Z_RANGE = (-10, 0)      # Vertical extent (meters, negative = depth)
DEPTH = -10             # Domain depth (meters)


# ============================================================
# Electrode Configuration
# ============================================================

N_ELEC = 24             # Number of electrodes
ELEC_SPACING = 0.8      # Electrode spacing (meters)


# ============================================================
# Grid Dimensions
# ============================================================

GRID_NX = 64            # Grid cells in x-direction
GRID_NZ = 64            # Grid cells in z-direction


# ============================================================
# Measurement Counts (Wenner array with 24 electrodes)
# ============================================================

N_DATA_WENNER = 84      # Number of measurements for Wenner array


# ============================================================
# Training Defaults
# ============================================================

GRAD_CLIP_NORM = 1.0    # Default gradient clipping norm
MASK_RATIO = 0.2        # Default mask ratio for self-supervised pretraining
INFO_NCE_TEMPERATURE = 0.07  # InfoNCE temperature parameter
RANDOM_SEED = 42        # Default random seed for reproducibility


# ============================================================
# Mesh Generation
# ============================================================

MESH_AREA = 0.3         # Default mesh element area for pyGIMLi
MESH_AREA_COARSE = 0.5  # Coarser mesh element area
