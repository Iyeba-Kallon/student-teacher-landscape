"""Distillation losses.

Only response-based (logit) KD is implemented for the pilot. Feature-based
distillation would be added here as a sibling module without touching the
training script (which only depends on the ``kd_loss`` interface).
"""

from .kd import KDLossOutput, freeze_teacher, kd_loss

__all__ = ["kd_loss", "freeze_teacher", "KDLossOutput"]
