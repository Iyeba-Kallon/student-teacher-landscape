"""Model registry. Add new architectures to _MODELS so configs can name them."""

from __future__ import annotations

import torch.nn as nn

from .resnet import ResNetCIFAR, resnet18_cifar

_MODELS = {
    "resnet18_cifar": resnet18_cifar,
}


def build_model(arch: str, num_classes: int = 10, width_mult: float = 1.0) -> nn.Module:
    if arch not in _MODELS:
        raise KeyError(f"unknown arch {arch!r}; known: {sorted(_MODELS)}")
    return _MODELS[arch](num_classes=num_classes, width_mult=width_mult)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["build_model", "count_parameters", "resnet18_cifar", "ResNetCIFAR"]
