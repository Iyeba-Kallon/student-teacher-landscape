"""Saving and loading checkpoints.

A checkpoint is one dict holding the weights plus everything needed to rebuild
the model without the original config: arch, width_mult, num_classes, mode,
precision, seed, epoch, metrics, the full config, an environment snapshot and
the torch version. Weights are always fp32 (AMP keeps fp32 master weights).
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
    return torch.load(path, map_location=map_location, weights_only=False)


def build_model_from_checkpoint(
    path: str | Path,
    map_location: str = "cpu",
    device: torch.device | str | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Rebuild the model from a checkpoint and load its weights.

    The model comes back in fp32 and eval mode; switch it to train() yourself if
    you need to.
    """
    ckpt = load_checkpoint(path, map_location=map_location)
    model = build_model(ckpt["arch"], ckpt["num_classes"], ckpt["width_mult"])
    model.load_state_dict(ckpt["model_state"])
    model.float()
    model.eval()
    if device is not None:
        model.to(device)
    return model, ckpt
