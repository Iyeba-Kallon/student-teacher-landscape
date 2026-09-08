"""Training and evaluation loops.

Separate from train.py so evaluate.py can reuse evaluate_classifier and so the
fp32/AMP branch only exists once.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .config import OptimConfig, ScheduleConfig
from .distillation.kd import kd_loss


def build_optimizer(model: nn.Module, cfg: OptimConfig) -> Optimizer:
    if cfg.name.lower() != "sgd":
        raise ValueError(f"only sgd is supported, got {cfg.name!r}")
    return torch.optim.SGD(
        model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
        weight_decay=cfg.weight_decay, nesterov=cfg.nesterov,
    )


def build_scheduler(optimizer: Optimizer, cfg: ScheduleConfig):
    """LR scheduler stepped once per epoch."""
    name = cfg.name.lower()
    warmup = max(0, cfg.warmup_epochs)

    if name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    if name == "cosine":
        main = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, cfg.epochs - warmup)
        )
    elif name == "multistep":
        main = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg.milestones, gamma=cfg.gamma
        )
    else:
        raise ValueError(f"unknown schedule {cfg.name!r}")

    if warmup == 0:
        return main
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, main], milestones=[warmup]
    )


def train_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    scaler: "torch.cuda.amp.GradScaler",
    amp: bool,
    teacher: nn.Module | None = None,
    kd_temperature: float = 4.0,
    kd_alpha: float = 0.9,
    grad_clip: float | None = None,
    log_every: int = 0,
    limit_batches: int = 0,
) -> dict[str, float]:
    """One pass over the loader.

    No teacher means plain cross-entropy (a teacher run). With a teacher (already
    frozen and in eval mode) it does KD; the teacher runs under no_grad in the
    same autocast context. When amp is False the autocast and GradScaler calls
    are no-ops, so both precisions run the same code.
    """
    model.train()
    if teacher is not None:
        teacher.eval()

    n = 0
    running = {"loss": 0.0, "acc": 0.0, "kd": 0.0, "ce": 0.0}

    for it, (inputs, targets) in enumerate(loader):
        if limit_batches and it >= limit_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        bs = targets.size(0)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(inputs)
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(inputs)
                out = kd_loss(logits, teacher_logits, targets,
                              kd_temperature, kd_alpha)
                loss, kd_val, ce_val = out.total, out.kd, out.ce
            else:
                loss = F.cross_entropy(logits, targets)
                kd_val, ce_val = 0.0, float(loss.detach())

        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            acc = (logits.argmax(1) == targets).float().mean().item()
        running["loss"] += float(loss.detach()) * bs
        running["acc"] += acc * bs
        running["kd"] += kd_val * bs
        running["ce"] += ce_val * bs
        n += bs

        if log_every and it % log_every == 0:
            print(f"  iter {it:4d}/{len(loader)}  loss={loss.item():.4f}  acc={acc:.4f}")

    stats = {k: v / max(1, n) for k, v in running.items()}
    stats["lr"] = optimizer.param_groups[0]["lr"]
    if teacher is None:
        stats.pop("kd")
    return stats


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    limit_batches: int = 0,
) -> dict[str, float]:
    """Top-1 accuracy, error and mean cross-entropy. Always fp32, no autocast."""
    was_training = model.training
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for it, (inputs, targets) in enumerate(loader):
        if limit_batches and it >= limit_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss_sum += F.cross_entropy(logits, targets, reduction="sum").item()
        correct += (logits.argmax(1) == targets).sum().item()
        total += targets.size(0)
    if was_training:
        model.train()
    acc = correct / max(1, total)
    return {"acc": acc, "error": 1.0 - acc, "loss": loss_sum / max(1, total), "n": total}
