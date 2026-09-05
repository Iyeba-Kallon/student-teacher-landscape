"""Hessian-based curvature metrics via PyHessian.

STATUS: placeholder — implementation lands in the next step.

Metrics:
- Hessian trace           -> Hutchinson estimator (PyHessian `.trace()`)
- Top Hessian eigenvalue  -> power iteration (PyHessian `.eigenvalues(top_n=1)`)

Protocol:
- Model prepared via prepare_model_for_geometry (eval, BN frozen, fp32).
- Curvature is measured on a fixed, seeded subset of the CIFAR-10 train set
  (2,000 examples, identical for every model) so numbers are comparable.
- Loss is standard cross-entropy (NOT the KD loss) so teacher and students are
  measured on the same objective.
- No autocast anywhere.
"""

from __future__ import annotations


def hessian_metrics(model, criterion, inputs, targets, cuda: bool = True):
    """Return dict with 'hessian_trace' and 'hessian_top_eigenvalue'."""
    raise NotImplementedError("Implemented in the next step.")
