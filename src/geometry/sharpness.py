"""Adaptive sharpness (m-sharpness formulation).

STATUS: placeholder — implementation lands in the next step.

Definition used in this project (documented so it is unambiguous):

We report *adaptive* worst-case sharpness following Kwon et al. (2021),
"ASAM: Adaptive Sharpness-Aware Minimization", using the m-sharpness
estimator from Foret et al. (2021), "Sharpness-Aware Minimization":

    S_rho(w) = E_{B ~ mini-batches of size m} [
                   max_{ || T_w^{-1} eps ||_2 <= rho }  L_B(w + eps) - L_B(w)
               ]

- T_w is the elementwise adaptive scaling operator, T_w = diag(|w|) (with the
  usual +1 on normalization-layer / bias params), which makes the metric
  invariant to parameter re-scaling.
- The inner max is approximated with a single normalized gradient-ascent step
  (the standard SAM ascent step), computed per micro-batch of size m.
- rho is fixed (default 0.05) and reported alongside every number.
- Estimated over the fixed, seeded 2,000-example geometry subset, micro-batched
  into chunks of size m (default 128).
- All forward/backward passes run in fp32 with BN frozen (see bn_utils).
- This is the HEADLINE geometry metric for the paper (Hessian trace is secondary).

The non-adaptive variant (plain m-sharpness, T_w = I) is available via a flag
for comparison.
"""

from __future__ import annotations


def adaptive_sharpness(model, loss_fn, data_loader, rho: float = 0.05,
                       m: int = 128, adaptive: bool = True, n_batches: int | None = None):
    """Estimate rho-sharpness. See module docstring for the exact definition."""
    raise NotImplementedError("Implemented in the next step.")
