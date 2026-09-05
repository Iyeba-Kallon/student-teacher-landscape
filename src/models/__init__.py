"""Model registry.

Add new architectures here so `build_model` can construct them from a config
without the training script needing to know the details.
"""

from __future__ import annotations

import torch.nn as nn

from .resnet import resnet18_cifar, ResNetCIFAR

_MODEL_REGISTRY = {
    "resnet18_cifar": resnet18_cifar,
}


def build_model(arch: str, num_classes: int = 10, width_mult: float = 1.0) -> nn.Module:
    """Construct a model by name.

    Parameters
    ----------
    arch:        key into the model registry, e.g. ``"resnet18_cifar"``.
    num_classes: number of output logits.
    width_mult:  channel width multiplier (1.0 = teacher, 0.5 / 0.25 = students).
    """
    if arch not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown arch '{arch}'. Known: {sorted(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[arch](num_classes=num_classes, width_mult=width_mult)


def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["build_model", "count_parameters", "resnet18_cifar", "ResNetCIFAR"]
