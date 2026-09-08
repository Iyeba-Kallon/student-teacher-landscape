"""Distillation losses. Only logit KD for now; feature distillation would go here."""

from .kd import KDLossOutput, freeze_teacher, kd_loss

__all__ = ["kd_loss", "freeze_teacher", "KDLossOutput"]
