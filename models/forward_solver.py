# -*- coding: utf-8 -*-
"""
ERT Forward Solver - pyGIMLi integration for real forward modeling.

Provides physics-accurate forward modeling for ERT inversion:
    sigma(x,z) -> rhoa (apparent resistivity at electrode positions)

All forward calls use pyGIMLi's finite-element solver.

Reference: Rucker et al., "pyGIMLi: An open-source library for modelling
and inversion in geophysics", Computers & Geosciences, 2017
"""

import os
import sys
import numpy as np
import pygimli as pg
from pygimli.physics import ert

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.forward_modeling import (
    create_electrodes, create_survey_scheme, run_forward,
    get_data_array, get_geometry_info
)
from data.model_generators import (
    generate_plume_model, generate_co2_model, generate_fault_model,
    _mesh_to_grid
)
from models.constants import N_ELEC, ELEC_SPACING, X_RANGE, Z_RANGE, GRID_NX, GRID_NZ


class ERTForwardSolver:
    """ERT forward solver wrapping pyGIMLi.

    Manages electrodes, measurement scheme, mesh, and forward modeling
    for a single electrode array type.

    Args:
        n_elec: number of electrodes
        elec_spacing: electrode spacing (m)
        array_type: 'wenner', 'dipole-dipole', 'schlumberger'
    """

    def __init__(self, n_elec=N_ELEC, elec_spacing=ELEC_SPACING, array_type='wenner'):
        self.n_elec = n_elec
        self.elec_spacing = elec_spacing
        self.array_type = array_type

        # Create electrode positions
        self.elec_x = np.linspace(
            -(n_elec - 1) * elec_spacing / 2,
            (n_elec - 1) * elec_spacing / 2,
            n_elec
        )
        self.elec_pos = np.column_stack([self.elec_x, np.zeros(n_elec)])

        # Create measurement scheme
        self.scheme = create_survey_scheme(self.elec_pos, array_type)
        self.n_data = self.scheme.size()

        # Geometry (four-point indices)
        self.geometry = get_geometry_info(self.scheme)

    def generate_model(self, model_type='plume', seed=None):
        """Generate a geological model with mesh.

        Args:
            model_type: 'plume', 'co2', or 'fault'
            seed: random seed for reproducibility

        Returns:
            mesh: pyGIMLi mesh object
            rho: resistivity vector (n_cells,) as numpy array
            rho_matrix: resistivity on regular grid (64, 64)
        """
        generators = {
            'plume': generate_plume_model,
            'co2': generate_co2_model,
            'fault': generate_fault_model,
        }
        gen = generators[model_type]
        mesh, rho, rho_matrix = gen(
            x_range=(-10, 10), depth=-10, seed=seed, elec_x=self.elec_x
        )
        return mesh, np.array(rho), rho_matrix

    def forward(self, mesh, rho_model):
        """Run forward modeling: resistivity model -> apparent resistivity.

        Uses pyGIMLi's finite-element ERT simulator.

        Args:
            mesh: pyGIMLi mesh (from generate_model)
            rho_model: resistivity values per mesh cell (Ohm*m)

        Returns:
            rhoa: apparent resistivity array (n_data,)
        """
        if isinstance(rho_model, np.ndarray):
            # Validate input to prevent silent pyGIMLi failures
            if np.any(np.isnan(rho_model)) or np.any(np.isinf(rho_model)):
                raise ValueError("rho_model contains NaN or Inf values")
            if np.any(rho_model <= 0):
                raise ValueError("rho_model contains non-positive values")
            rho_model = pg.Vector(rho_model.tolist())
        data = run_forward(mesh, rho_model, self.scheme, noise_level=0.0)
        return get_data_array(data)

    def forward_with_noise(self, mesh, rho_model, noise_level=0.05):
        """Forward modeling with measurement noise.

        Args:
            mesh: pyGIMLi mesh
            rho_model: resistivity values per mesh cell
            noise_level: relative noise (0.05 = 5%)

        Returns:
            rhoa: noisy apparent resistivity
        """
        if isinstance(rho_model, np.ndarray):
            rho_model = pg.Vector(rho_model.tolist())
        data = run_forward(mesh, rho_model, self.scheme, noise_level=noise_level)
        return get_data_array(data)

    def get_cell_centers(self, mesh):
        """Get mesh cell center coordinates.

        Args:
            mesh: pyGIMLi mesh

        Returns:
            centers: (n_cells, 2) array of [x, z] coordinates
        """
        n_cells = mesh.cellCount()
        centers = np.zeros((n_cells, 2))
        for i in range(n_cells):
            c = mesh.cell(i).center()
            centers[i, 0] = c.x()
            centers[i, 1] = c.y()
        return centers

    def normalize_coords(self, coords, x_range=X_RANGE, z_range=Z_RANGE):
        """Normalize coordinates to [-1, 1].

        Args:
            coords: (N, 2) coordinates
            x_range: (x_min, x_max)
            z_range: (z_min, z_max)

        Returns:
            coords_norm: (N, 2) normalized coordinates
        """
        coords_norm = coords.copy()
        coords_norm[:, 0] = 2 * (coords[:, 0] - x_range[0]) / (x_range[1] - x_range[0]) - 1
        coords_norm[:, 1] = 2 * (coords[:, 1] - z_range[0]) / (z_range[1] - z_range[0]) - 1
        return coords_norm

    def denormalize_rho(self, rho_norm, rho_mean, rho_std):
        """Convert normalized log-resistivity back to Ohm*m.

        Args:
            rho_norm: normalized log-resistivity
            rho_mean: mean of log(rho)
            rho_std: std of log(rho)

        Returns:
            rho: resistivity in Ohm*m
        """
        return np.exp(rho_norm * rho_std + rho_mean)

    def interpolate_to_grid(self, mesh, values, nx=GRID_NX, nz=GRID_NZ):
        """Interpolate mesh values to regular grid for visualization.

        Args:
            mesh: pyGIMLi mesh
            values: values per mesh cell
            nx, nz: grid dimensions

        Returns:
            grid: (nz, nx) interpolated values
        """
        return _mesh_to_grid(mesh, values, X_RANGE, Z_RANGE[0], nx, nz)


class MultiArrayForwardSolver:
    """Multi-array ERT forward solver.

    Manages forward modeling for multiple electrode array types,
    enabling multi-array data fusion for improved inversion.

    Args:
        n_elec: number of electrodes
        elec_spacing: electrode spacing
        array_types: list of array type names
    """

    def __init__(self, n_elec=N_ELEC, elec_spacing=ELEC_SPACING,
                 array_types=None):
        if array_types is None:
            array_types = ['wenner', 'dipole-dipole', 'schlumberger']

        self.solvers = {}
        for arr in array_types:
            self.solvers[arr] = ERTForwardSolver(
                n_elec=n_elec,
                elec_spacing=elec_spacing,
                array_type=arr
            )

    def forward(self, mesh, rho_model, array_type):
        """Forward modeling for a specific array type."""
        return self.solvers[array_type].forward(mesh, rho_model)

    def forward_all(self, mesh, rho_model):
        """Forward modeling for all array types.

        Returns:
            dict of {array_type: rhoa}
        """
        return {
            arr: solver.forward(mesh, rho_model)
            for arr, solver in self.solvers.items()
        }

    def generate_model(self, model_type='plume', seed=None):
        """Generate model using the first solver's mesh."""
        first = list(self.solvers.values())[0]
        return first.generate_model(model_type, seed)


if __name__ == '__main__':
    print("=" * 50)
    print("Test: ERTForwardSolver")
    solver = ERTForwardSolver(n_elec=24, array_type='wenner')
    print(f"  Array: {solver.array_type}, {solver.n_data} measurements")

    mesh, rho, rho_matrix = solver.generate_model('plume', seed=42)
    print(f"  Mesh: {mesh.cellCount()} cells")

    rhoa = solver.forward(mesh, rho)
    print(f"  Forward: rhoa range [{rhoa.min():.1f}, {rhoa.max():.1f}]")

    rhoa_noisy = solver.forward_with_noise(mesh, rho, 0.05)
    print(f"  Noisy: rhoa range [{rhoa_noisy.min():.1f}, {rhoa_noisy.max():.1f}]")

    centers = solver.get_cell_centers(mesh)
    print(f"  Cell centers: {centers.shape}")

    coords_norm = solver.normalize_coords(centers)
    print(f"  Normalized coords range: [{coords_norm.min():.2f}, {coords_norm.max():.2f}]")

    grid = solver.interpolate_to_grid(mesh, rho)
    print(f"  Interpolated grid: {grid.shape}")

    print("\nTest: MultiArrayForwardSolver")
    multi = MultiArrayForwardSolver(n_elec=24)
    rho_all = multi.forward_all(mesh, rho)
    for arr, rhoa in rho_all.items():
        print(f"  {arr}: {len(rhoa)} measurements")

    print("\nAll tests passed!")
