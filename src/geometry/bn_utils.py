"""Getting a model ready for a geometry measurement.

Call prepare_model_for_geometry() before measuring sharpness or the Hessian. It
does three things:

1. model.eval(), so BatchNorm uses its running statistics.
2. Freezes those statistics (momentum 0, every BN layer in eval mode) so no
   forward pass can move them. Curvature measured while BN stats drift is wrong.
3. Casts the model to fp32. Geometry is always measured in fp32, even for a
   model that was trained with AMP.

Nothing in this package uses autocast. assert_fp32_no_autocast() is the guard.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_BN = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)


def prepare_model_for_geometry(model: nn.Module) -> nn.Module:
    """Eval mode, frozen BN, fp32. Mutates model in place and returns it."""
    model.eval()
    model.float()

    for m in model.modules():
        if isinstance(m, _BN):
            m.eval()
            m.momentum = 0.0
            if m.running_mean is not None:
                m.running_mean = m.running_mean.float()
                m.running_var = m.running_var.float()

    for p in model.parameters():
        p.requires_grad_(True)      # sharpness and the Hessian both need grads

    return model


def assert_fp32_no_autocast(model: nn.Module | None = None) -> None:
    """Raise if autocast is active, or if the model has non-fp32 parameters."""
    cuda_ac = torch.is_autocast_enabled()
    try:
        cpu_ac = torch.is_autocast_cpu_enabled()
    except AttributeError:                      # renamed in newer torch
        cpu_ac = torch.is_autocast_enabled("cpu")
    if cuda_ac or cpu_ac:
        raise RuntimeError("geometry must not run under torch.autocast")
    if model is not None:
        dtypes = {p.dtype for p in model.parameters()}
        if dtypes and dtypes != {torch.float32}:
            raise RuntimeError(f"model parameters are not all fp32: {dtypes}")


def bn_is_frozen(model: nn.Module) -> bool:
    """True if every BN layer is in eval mode with momentum 0. Used by tests."""
    return all(
        not m.training and m.momentum in (0, 0.0)
        for m in model.modules() if isinstance(m, _BN)
    )
