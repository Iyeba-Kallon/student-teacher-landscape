"""BatchNorm handling for geometry measurements.

STATUS: placeholder — implementation lands in the next step.

Every sharpness / Hessian measurement MUST:
1. Call model.eval() so BN uses running statistics (not batch statistics).
2. Freeze BN running stats (set momentum handling so no update happens even
   if a forward pass is run in a context that would otherwise update them).
3. Run entirely in fp32 — never inside torch.autocast — regardless of whether
   the model was trained with AMP.

`prepare_model_for_geometry` is the single entry point that enforces all three.
"""

from __future__ import annotations


def prepare_model_for_geometry(model):
    """Put model in eval mode, freeze BN running stats, and cast to fp32."""
    raise NotImplementedError("Implemented in the next step.")
