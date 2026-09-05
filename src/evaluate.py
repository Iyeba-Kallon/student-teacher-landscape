"""Evaluate a checkpoint on CIFAR-10 (ID) and CIFAR-10-C (OOD).

STATUS: placeholder — implementation lands in the next step.

Usage:

    python -m src.evaluate --checkpoint results/teacher_fp32_s0/checkpoints/best.pt

Reports:
- ID: top-1 accuracy / error on the CIFAR-10 test set.
- OOD: per-corruption, per-severity top-1 accuracy on CIFAR-10-C, plus the
  mean over the 15 corruptions x 5 severities (and mCE relative to the teacher
  when a teacher baseline is supplied).
- Evaluation always runs in fp32 with model.eval(); AMP is a training-only knob.
- Appends results to the run's metrics.csv / summary.json.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("Implemented in the next step.")


if __name__ == "__main__":
    main()
