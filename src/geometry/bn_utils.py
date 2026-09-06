"""Model preparation for loss-landscape geometry measurements.

Every sharpness / Hessian measurement MUST call
:func:`prepare_model_for_geometry` first. It enforces the three rules from the
pilot spec:

1. ``model.eval()`` so BatchNorm uses its *running* statistics, not per-batch
   statistics.
2. BatchNorm running stats are **frozen** — ``momentum`` is set to 0 and the
   layers are individually put in eval mode, so no forward pass (even an
   accidental one in train mode) can move them. Contaminated BN curvature is a
   known failure mode for these estimates.
3. The whole model is cast to **fp32**. Geometry is always measured in full
   precision, even for checkpoints that were *trained* with AMP.

There is deliberately no ``autocast`` anywhere in ``src/geometry`` — call
:func:`assert_fp32_no_autocast` at the top of every measurement entry point.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_BN = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)


def prepare_model_for_geometry(model: nn.Module) -> nn.Module:
    """Put ``model`` in eval mode, freeze BN running stats, cast to fp32.

    Mutates ``model`` in place and also returns it. Intended to be called on a
    freshly loaded checkpoint whose only remaining use is geometry measurement.
    """
    model.eval()
    model.float()

    n_bn = 0
    for m in model.modules():
        if isinstance(m, _BN):
            m.eval()
            m.momentum = 0.0            # no EMA update even if forced to train()
            if m.running_mean is not None:
                m.running_mean = m.running_mean.float()
                m.running_var = m.running_var.float()
            n_bn += 1

    # Sharpness and Hessian both need gradients w.r.t. every parameter.
    for p in model.parameters():
        p.requires_grad_(True)

    return model


def assert_fp32_no_autocast(model: nn.Module | None = None) -> None:
    """Guard: geometry code must never run under autocast, and the model
    (if given) must be fp32."""
    if torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled():
        raise RuntimeError(
            "Geometry measurements must not run under torch.autocast. "
            "Precision is a training-time variable only."
        )
    if model is not None:
        dtypes = {p.dtype for p in model.parameters()}
        if dtypes and dtypes != {torch.float32}:
            raise RuntimeError(f"model parameters are not all fp32: {dtypes}")


def bn_is_frozen(model: nn.Module) -> bool:
    """True iff every BN layer is in eval mode with momentum 0 (for tests)."""
    for m in model.modules():
        if isinstance(m, _BN):
            if m.training or (m.momentum not in (0, 0.0)):
                return False
    return True
