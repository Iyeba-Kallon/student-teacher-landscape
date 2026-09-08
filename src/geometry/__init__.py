"""Loss-landscape geometry metrics, all measured in fp32 with BatchNorm frozen.

- prepare_model_for_geometry: required setup for any measurement
- adaptive_sharpness: the headline flatness metric (ASAM-style m-sharpness)
- hessian_metrics: Hessian trace and top eigenvalue (PyHessian)
"""

from .bn_utils import (assert_fp32_no_autocast, bn_is_frozen,
                       prepare_model_for_geometry)
from .hessian import hessian_metrics
from .sharpness import DEFAULT_ETA, DEFAULT_RHO, adaptive_sharpness

__all__ = [
    "prepare_model_for_geometry",
    "assert_fp32_no_autocast",
    "bn_is_frozen",
    "adaptive_sharpness",
    "hessian_metrics",
    "DEFAULT_RHO",
    "DEFAULT_ETA",
]
