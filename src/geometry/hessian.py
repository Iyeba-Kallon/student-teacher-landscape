"""Hessian-based curvature metrics via PyHessian.

- **Hessian trace** — Hutchinson's estimator (mean of Rademacher quadratic
  forms), ``pyhessian.hessian.trace()``.
- **Top Hessian eigenvalue** ``lambda_max`` — power iteration,
  ``pyhessian.hessian.eigenvalues(top_n=1)``.

Protocol (matches the sharpness measurement):
- The model must already be prepared with
  :func:`src.geometry.bn_utils.prepare_model_for_geometry` (eval, BN frozen, fp32).
- Curvature is measured over the fixed, seeded 2,000-example geometry subset,
  passed as a ``DataLoader`` so PyHessian averages Hessian-vector products over
  its mini-batches.
- The loss is plain cross-entropy (``nn.CrossEntropyLoss``), never the KD loss.
- No autocast anywhere.
- The RNG is seeded immediately before the estimators so the Hutchinson probes
  and the power-iteration init are reproducible.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..utils import set_seed
from .bn_utils import assert_fp32_no_autocast


def hessian_metrics(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
    seed: int = 0,
    trace_max_iter: int = 100,
    trace_tol: float = 1e-3,
    eig_max_iter: int = 100,
    eig_tol: float = 1e-3,
    top_n: int = 1,
) -> dict[str, float]:
    """Return ``hessian_trace``, ``hessian_top_eigenvalue`` (+ diagnostics)."""
    assert_fp32_no_autocast(model)
    try:
        from pyhessian import hessian
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "pyhessian is required for Hessian metrics: pip install pyhessian"
        ) from e

    criterion = criterion or nn.CrossEntropyLoss()
    cuda = device.type == "cuda"

    was_training = model.training
    model.eval()
    set_seed(seed)  # reproducible Hutchinson probes / power-iteration init

    hc = hessian(model, criterion, dataloader=data_loader, cuda=cuda)

    eigvals, _ = hc.eigenvalues(maxIter=eig_max_iter, tol=eig_tol, top_n=top_n)
    trace_list = hc.trace(maxIter=trace_max_iter, tol=trace_tol)

    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()

    trace_arr = np.asarray(trace_list, dtype=np.float64)
    return {
        "hessian_top_eigenvalue": float(eigvals[0]),
        "hessian_eigenvalues_top_n": [float(v) for v in eigvals],
        "hessian_trace": float(trace_arr.mean()),
        "hessian_trace_std": float(trace_arr.std()),
        "hessian_trace_n_iter": int(trace_arr.size),
    }
