# -*- coding: utf-8 -*-
"""MA-PC-SRI model modules.

Multi-Array Physics-Consistent Self-Refining Inversion for ERT.

Modules:
- FourierEncoding: Random Fourier feature mapping
- FNOForward: Fourier Neural Operator forward surrogate
- MultiArrayEncoder: Multi-array encoder (inherited from SRERTF-Net)
- SelfRefineDecoder: Self-refining decoder (core innovation)
- PCGrad: Gradient surgery for multi-task learning
- forward_solver: pyGIMLi ERT forward modeling (ground truth)
- constants: Centralized configuration constants
"""

from models.FourierEncoding import FourierEncoding
from models.FNOForward import FNOForward, create_fno_model
from models.MultiArrayEncoder import (
    MultiArrayEncoder, PhysicalPriorEncoder,
    InceptionBlock, DenseBlock, ArrayEncoderPath
)
from models.SelfRefineDecoder import SelfRefineDecoder, RefineBlock
from models.PCGrad import PCGrad
from models.forward_solver import ERTForwardSolver
from models import constants

__all__ = [
    # Core architecture
    'FNOForward', 'create_fno_model',
    'MultiArrayEncoder', 'PhysicalPriorEncoder',
    'SelfRefineDecoder', 'RefineBlock',
    # Building blocks
    'FourierEncoding', 'InceptionBlock', 'DenseBlock', 'ArrayEncoderPath',
    # Utilities
    'PCGrad', 'ERTForwardSolver', 'constants',
]
