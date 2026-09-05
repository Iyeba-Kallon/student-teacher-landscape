"""Train a teacher, or distill a student, from a single config file.

STATUS: placeholder — implementation lands in the next step.

Usage (see README for the full pilot sequence):

    python -m src.train --config configs/teacher_fp32.yaml
    python -m src.train --config configs/student_w0.5_fp32.yaml

Behavior:
- `mode: teacher`   -> plain cross-entropy training.
- `mode: student`   -> KD training; requires `teacher_checkpoint` in the config.
- `precision: fp32` -> no autocast.
- `precision: amp`  -> torch.cuda.amp autocast + GradScaler for the TRAINING
  loop only. Checkpoints always store fp32 master weights.
- Writes checkpoints + metrics.csv + summary.json under results/<run_name>/.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("Implemented in the next step.")


if __name__ == "__main__":
    main()
