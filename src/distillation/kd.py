"""Response-based knowledge distillation (Hinton et al., 2015).

    L = alpha * T^2 * KL(softmax(z_t / T) || softmax(z_s / T))  +  (1 - alpha) * CE(z_s, y)

T softens both distributions; the T^2 factor keeps the soft term's gradient on
the same scale across temperatures. alpha weights the soft term against the hard
cross-entropy term.

The logits are cast to fp32 inside the loss so the softmax/KL/CE stay stable when
the training step runs under AMP. That is a training-time choice and has nothing
to do with the fp32 rule for geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class KDLossOutput:
    total: torch.Tensor      # scalar to call .backward() on
    kd: float                # soft term, detached (already scaled by T^2)
    ce: float                # hard term, detached
    alpha: float
    temperature: float


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
    alpha: float,
) -> KDLossOutput:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    T = float(temperature)
    s = student_logits.float()
    t = teacher_logits.float().detach()

    kd_term = F.kl_div(
        F.log_softmax(s / T, dim=1), F.softmax(t / T, dim=1), reduction="batchmean"
    ) * (T * T)
    ce_term = F.cross_entropy(s, targets)
    total = alpha * kd_term + (1.0 - alpha) * ce_term

    return KDLossOutput(total, kd_term.detach().item(), ce_term.detach().item(),
                        alpha, T)


@torch.no_grad()
def freeze_teacher(teacher: torch.nn.Module) -> torch.nn.Module:
    """Eval mode, no gradients. Call once before the training loop."""
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher
