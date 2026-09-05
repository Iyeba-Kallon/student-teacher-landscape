"""Checkpoint save / load with reproducibility metadata.

Every checkpoint is a single dict that carries enough information to rebuild the
model without the original config file:

    model_state    : fp32 state_dict (AMP runs keep fp32 master weights, so this
                     is always fp32 regardless of training precision)
    arch           : model registry key
    width_mult     : channel width multiplier
    num_classes    : output dimension
    mode           : "teacher" | "student"
    precision      : "fp32" | "amp"  (how the model was TRAINED)
    seed           : run seed
    epoch          : epoch index this checkpoint was saved at
    metrics        : dict of metrics at save time (e.g. id_acc)
    config         : full nested config dict
    env            : software/hardware snapshot
    torch_version  : torch.__version__
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import Config, config_to_dict
from .models import build_model
from .utils import capture_env


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    cfg: Config,
    epoch: int,
    metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "arch": cfg.model.arch,
        "width_mult": cfg.model.width_mult,
        "num_classes": cfg.model.num_classes,
        "mode": cfg.mode,
        "precision": cfg.precision,
        "seed": cfg.seed,
        "epoch": epoch,
        "metrics": metrics,
        "config": config_to_dict(cfg),
        "env": capture_env(),
        "torch_version": torch.__version__,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    """Load the raw checkpoint dict."""
    return torch.load(path, map_location=map_location, weights_only=False)


def build_model_from_checkpoint(
    path: str | Path,
    map_location: str = "cpu",
    device: torch.device | str | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Rebuild the model described by a checkpoint and load its weights.

    Returns ``(model, checkpoint_dict)``. The model is returned in fp32 and in
    ``eval()`` mode; callers that need training mode must switch it themselves.
    """
    ckpt = load_checkpoint(path, map_location=map_location)
    model = build_model(
        arch=ckpt["arch"],
        num_classes=ckpt["num_classes"],
        width_mult=ckpt["width_mult"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.float()
    model.eval()
    if device is not None:
        model.to(device)
    return model, ckpt
