"""Hessian trace and top eigenvalue, via PyHessian.

The trace comes from Hutchinson's estimator, the top eigenvalue from power
iteration. Both are measured over the geometry subset (passed as a DataLoader so
PyHessian averages Hessian-vector products over the mini-batches), with plain
cross-entropy and no autocast. The model must already be through
prepare_model_for_geometry. The RNG is seeded just before the estimators so the
probes are reproducible.
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
    assert_fp32_no_autocast(model)
    try:
        from pyhessian import hessian
    except ImportError as e:
        raise RuntimeError("pip install pyhessian") from e

    criterion = criterion or nn.CrossEntropyLoss()

    was_training = model.training
    model.eval()
    set_seed(seed)

    hc = hessian(model, criterion, dataloader=data_loader, cuda=device.type == "cuda")
    eigvals, _ = hc.eigenvalues(maxIter=eig_max_iter, tol=eig_tol, top_n=top_n)
    trace_list = hc.trace(maxIter=trace_max_iter, tol=trace_tol)

    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()

    trace = np.asarray(trace_list, dtype=np.float64)
    return {
        "hessian_top_eigenvalue": float(eigvals[0]),
        "hessian_eigenvalues_top_n": [float(v) for v in eigvals],
        "hessian_trace": float(trace.mean()),
        "hessian_trace_std": float(trace.std()),
        "hessian_trace_n_iter": int(trace.size),
    }
