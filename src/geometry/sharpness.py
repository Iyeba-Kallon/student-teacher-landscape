"""Adaptive m-sharpness.

For micro-batches B of size m from the fixed geometry subset, we estimate

    S_rho(w) = mean_B [ max over ||inv(T_w) eps|| <= rho of  L_B(w + eps) - L_B(w) ]

where L_B is the mean cross-entropy on B (not the KD loss, so every model is
measured against the same objective). The inner max is the standard single SAM
ascent step: linearizing L_B(w + eps) and substituting u = inv(T_w) eps gives

    eps* = rho * T_w^2 g / ||T_w g||        with g = grad L_B(w)

and a global L2 norm over all parameters. T_w is the ASAM adaptive operator
(Kwon et al., 2021): T_w = |w| + eta for weight tensors (ndim >= 2) and T_w = 1
for biases and BatchNorm gamma/beta. That makes the metric invariant to
node-wise rescaling, which matters across different widths. adaptive=False sets
T_w = 1 everywhere (plain m-sharpness / SAM).

rho is fixed (0.05) and reported with every result. Everything runs in fp32 with
BN frozen; the caller must have run prepare_model_for_geometry first.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .bn_utils import assert_fp32_no_autocast

DEFAULT_RHO = 0.05
DEFAULT_ETA = 0.01
_EPS = 1e-12


def _scales(params: list[torch.Tensor], eta: float, adaptive: bool) -> list[torch.Tensor]:
    out = []
    for p in params:
        if adaptive and p.dim() >= 2:
            out.append(p.detach().abs() + eta)
        else:
            out.append(torch.ones_like(p))
    return out


@torch.enable_grad()
def _sharpness_one_batch(
    model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor,
    params: list[torch.Tensor], rho: float, eta: float, adaptive: bool,
) -> float:
    model.zero_grad(set_to_none=True)
    clean_loss = F.cross_entropy(model(inputs), targets)
    clean_loss.backward()
    clean = float(clean_loss.detach())

    grads = [p.grad.detach() for p in params]
    scales = _scales(params, eta, adaptive)
    tg_norm = torch.sqrt(sum(((s * g) ** 2).sum() for s, g in zip(scales, grads))) + _EPS

    eps = []
    with torch.no_grad():
        for p, s, g in zip(params, scales, grads):
            e = rho * (s ** 2) * g / tg_norm
            p.add_(e)
            eps.append(e)
        perturbed = float(F.cross_entropy(model(inputs), targets).detach())
        for p, e in zip(params, eps):
            p.sub_(e)

    model.zero_grad(set_to_none=True)
    return perturbed - clean


def adaptive_sharpness(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    rho: float = DEFAULT_RHO,
    eta: float = DEFAULT_ETA,
    adaptive: bool = True,
    n_batches: int | None = None,
) -> dict[str, float]:
    """Estimate rho-sharpness (see the module docstring).

    data_loader should already yield micro-batches of the desired size m, i.e.
    build_geometry_loader(..., batch_size=m).
    """
    assert_fp32_no_autocast(model)
    was_training = model.training
    model.eval()

    params = [p for p in model.parameters() if p.requires_grad]
    saved_grads = [p.grad for p in params]
    for p in params:
        p.grad = None

    values: list[float] = []
    for i, (inputs, targets) in enumerate(data_loader):
        if n_batches is not None and i >= n_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        values.append(_sharpness_one_batch(model, inputs, targets, params,
                                           rho, eta, adaptive))

    for p, g in zip(params, saved_grads):
        p.grad = g
    if was_training:
        model.train()

    t = torch.tensor(values)
    return {
        "adaptive_sharpness": float(t.mean()),
        "sharpness_std": float(t.std(unbiased=False)) if len(t) > 1 else 0.0,
        "sharpness_min": float(t.min()),
        "sharpness_max": float(t.max()),
        "rho": rho,
        "eta": eta,
        "adaptive": adaptive,
        "n_micro_batches": len(values),
    }
