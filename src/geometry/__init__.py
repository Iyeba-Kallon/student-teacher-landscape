"""Loss-landscape geometry metrics.

All measurements are performed in fp32 with BatchNorm frozen. See the module
docstrings for exact definitions:

- ``bn_utils.prepare_model_for_geometry`` — mandatory model preparation
- ``sharpness.adaptive_sharpness``        — headline metric (ASAM-style m-sharpness)
- ``hessian.hessian_metrics``             — Hessian trace + top eigenvalue (PyHessian)
"""

from .bn_utils import (assert_fp32_no_autocast, bn_is_frozen,
                       prepare_model_for_geometry)
from .sharpness import DEFAULT_ETA, DEFAULT_RHO, adaptive_sharpness
from .hessian import hessian_metrics  # lazy-imports pyhessian only when called

__all__ = [
    "prepare_model_for_geometry",
    "assert_fp32_no_autocast",
    "bn_is_frozen",
    "adaptive_sharpness",
    "hessian_metrics",
    "DEFAULT_RHO",
    "DEFAULT_ETA",
]
