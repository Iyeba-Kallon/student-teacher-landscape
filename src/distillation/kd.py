"""Standard response-based knowledge distillation (Hinton et al., 2015).

STATUS: placeholder — implementation lands in the next step.

Loss:
    L = alpha * T^2 * KL(softmax(z_s / T) || softmax(z_t / T))
        + (1 - alpha) * CE(z_s, y)

- Teacher logits z_t are computed with the teacher in eval() mode and under
  torch.no_grad(); the teacher is never updated.
- T (temperature) and alpha are read from the config.
- The T^2 factor keeps gradient magnitudes comparable across temperatures.
"""

from __future__ import annotations


def kd_loss(student_logits, teacher_logits, targets, temperature: float, alpha: float):
    """Combined KD + cross-entropy loss. See module docstring for the formula."""
    raise NotImplementedError("Implemented in the next step.")
