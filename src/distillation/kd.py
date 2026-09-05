"""Standard response-based knowledge distillation (Hinton et al., 2015).

Loss:

    L = alpha * T^2 * KL( softmax(z_t / T) || softmax(z_s / T) )
        + (1 - alpha) * CE(z_s, y)

- ``z_t`` are teacher logits, computed with the teacher in ``eval()`` mode and
  under ``torch.no_grad()`` — the teacher is frozen and never updated.
- ``T`` (temperature) softens both distributions; the ``T^2`` factor keeps the
  soft-loss gradient magnitude comparable across temperatures (Hinton et al.).
- ``alpha`` is the weight on the soft (distillation) term; ``1 - alpha`` weights
  the hard cross-entropy term.

Numerical note: the loss casts logits to fp32 before the softmax/KL/CE math so
the objective is stable even when the surrounding training step runs under AMP
autocast. (This is a *training-time* stability measure and is unrelated to the
"geometry must be fp32" rule.)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class KDLossOutput:
    total: torch.Tensor          # differentiable scalar used for .backward()
    kd: float                    # detached soft-term value (already * T^2)
    ce: float                    # detached hard-term value
    alpha: float
    temperature: float


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
    alpha: float,
) -> KDLossOutput:
    """Combined KD + cross-entropy loss. See module docstring for the formula."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    T = float(temperature)
    s = student_logits.float()
    t = teacher_logits.float().detach()   # teacher is frozen; no grad path

    # KL(teacher || student) with both distributions softened by T.
    log_p_student = F.log_softmax(s / T, dim=1)
    p_teacher = F.softmax(t / T, dim=1)
    kd_term = F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (T * T)

    ce_term = F.cross_entropy(s, targets)

    total = alpha * kd_term + (1.0 - alpha) * ce_term
    return KDLossOutput(
        total=total,
        kd=kd_term.detach().item(),
        ce=ce_term.detach().item(),
        alpha=alpha,
        temperature=T,
    )


@torch.no_grad()
def freeze_teacher(teacher: torch.nn.Module) -> torch.nn.Module:
    """Put the teacher in eval mode and disable grads on all its parameters."""
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher
