"""Adaptive worst-case sharpness (m-sharpness estimator).

Definition used in this project — this is the number reported in Section 5.2
and it is stated here in full so it is unambiguous.

We estimate, over micro-batches ``B`` of size ``m`` drawn from the fixed
2,000-example geometry subset:

    S_rho(w) = mean_B [  max_{|| T_w^{-1} eps ||_2 <= rho}  L_B(w + eps) - L_B(w)  ]

- ``L_B`` is the mean cross-entropy on micro-batch ``B`` (NOT the KD loss, so
  teacher and students are measured on the same objective).
- The inner maximization is approximated by the standard single normalized
  ascent step. With the linearization ``L_B(w+eps) ~ L_B(w) + g . eps`` and the
  substitution ``u = T_w^{-1} eps`` the maximizer is

      eps* = rho * T_w^2 g / || T_w g ||_2 ,     g = grad_w L_B(w)

  (global L2 norm over all parameters).
- ``T_w`` is the **adaptive** elementwise operator (Kwon et al., 2021, ASAM):
    * weight tensors (ndim >= 2):        T_w = |w| + eta      (eta = 0.01)
    * biases and BatchNorm gamma/beta:   T_w = 1              (no rescaling)
  This makes the metric invariant to node-wise weight re-scaling, which matters
  when comparing networks of different width. Setting ``adaptive=False`` uses
  ``T_w = 1`` everywhere, recovering plain (non-adaptive) m-sharpness / SAM.
- ``rho`` is fixed (default 0.05) and is reported alongside every value.

All forward/backward passes run in fp32 with BatchNorm frozen (the caller must
have run :func:`src.geometry.bn_utils.prepare_model_for_geometry`).
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


def _adaptive_scales(params: list[torch.Tensor], eta: float,
                     adaptive: bool) -> list[torch.Tensor]:
    scales = []
    for p in params:
        if adaptive and p.dim() >= 2:
            scales.append(p.detach().abs() + eta)
        else:
            scales.append(torch.ones_like(p))
    return scales


@torch.enable_grad()
def _sharpness_one_micro_batch(
    model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor,
    params: list[torch.Tensor], rho: float, eta: float, adaptive: bool,
) -> float:
    model.zero_grad(set_to_none=True)
    clean_loss = F.cross_entropy(model(inputs), targets)
    clean_loss.backward()
    clean = float(clean_loss.detach())

    grads = [p.grad.detach() for p in params]
    scales = _adaptive_scales(params, eta, adaptive)

    # || T_w g ||_2  over all parameters.
    tg_norm = torch.sqrt(
        sum(((s * g) ** 2).sum() for s, g in zip(scales, grads))
    ) + _EPS

    # eps* = rho * T_w^2 g / || T_w g ||  ; apply, measure, revert.
    eps_list = []
    with torch.no_grad():
        for p, s, g in zip(params, scales, grads):
            e = rho * (s ** 2) * g / tg_norm
            p.add_(e)
            eps_list.append(e)
        perturbed = float(F.cross_entropy(model(inputs), targets).detach())
        for p, e in zip(params, eps_list):
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
    """Estimate rho-sharpness. See the module docstring for the exact definition.

    ``data_loader`` should already yield micro-batches of the desired size ``m``
    (``src.data.build_geometry_loader(..., batch_size=m)``).
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
        values.append(_sharpness_one_micro_batch(
            model, inputs, targets, params, rho, eta, adaptive
        ))

    # restore any grads the caller had
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
