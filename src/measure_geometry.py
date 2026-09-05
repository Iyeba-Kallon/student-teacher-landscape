"""Measure loss landscape geometry for a checkpoint.

STATUS: placeholder — implementation lands in the next step.

Usage:

    python -m src.measure_geometry --checkpoint results/student_w0.5_amp_s0/checkpoints/best.pt

Produces (all in fp32, BN frozen, no autocast):
- adaptive_sharpness (rho reported)
- hessian_trace
- hessian_top_eigenvalue

Notes:
- The same seeded data subset is used for every checkpoint so numbers compare.
- Works on both fp32- and AMP-trained checkpoints; the checkpoint is loaded and
  cast to fp32 before any measurement.
- Appends results to the run's metrics.csv / summary.json.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("Implemented in the next step.")


if __name__ == "__main__":
    main()
