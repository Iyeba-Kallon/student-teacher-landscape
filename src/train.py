"""Train a teacher, or distill a student, from a single config file.

Usage
-----
    # teacher (plain cross-entropy)
    python -m src.train --config configs/teacher_fp32.yaml --seed 0

    # student (knowledge distillation; config points at a teacher checkpoint)
    python -m src.train --config configs/student_w0.5_amp.yaml --seed 0

    # quick smoke test
    python -m src.train --config configs/teacher_fp32.yaml --set schedule.epochs=1 data.num_workers=0

Behavior
--------
- ``mode: teacher``   -> cross-entropy training.
- ``mode: student``   -> KD training; requires ``kd.teacher_checkpoint``.
- ``precision: fp32`` -> autocast + GradScaler are no-ops.
- ``precision: amp``  -> ``torch.cuda.amp`` autocast + GradScaler for the
  TRAINING loop only. Requires CUDA. Checkpoints still store fp32 weights.
- Validation runs in fp32 every epoch.
- Writes ``results/<run_name>/{metrics.csv, summary.json, checkpoints/*.pt}``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from .checkpoint import build_model_from_checkpoint, save_checkpoint
from .config import Config, load_config
from .data import build_cifar10_loaders
from .distillation.kd import freeze_teacher
from .engine import (build_optimizer, build_scheduler, evaluate_classifier,
                     train_one_epoch)
from .models import build_model, count_parameters
from .utils import RunLogger, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=str, help="path to a YAML config")
    p.add_argument("--seed", type=int, default=None,
                   help="override cfg.seed (so one config covers all seeds)")
    p.add_argument("--wandb", action="store_true",
                   help="enable wandb logging (default: off; CSV/JSON always on)")
    p.add_argument("--set", nargs="*", default=None, metavar="k.k=v",
                   help="dotted config overrides, e.g. schedule.epochs=1 optim.lr=0.05")
    p.add_argument("--device", type=str, default=None, help="cuda | cpu (auto if unset)")
    return p.parse_args()


def resolve_device(requested: str | None, precision: str) -> torch.device:
    if requested:
        dev = torch.device(requested)
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if precision == "amp" and dev.type != "cuda":
        raise RuntimeError(
            "precision: amp requires a CUDA device. Use precision: fp32 on CPU."
        )
    return dev


def load_teacher(cfg: Config, device: torch.device) -> torch.nn.Module:
    ckpt_path = Path(cfg.kd.teacher_checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"teacher checkpoint not found: {ckpt_path}")
    teacher, tckpt = build_model_from_checkpoint(ckpt_path, device=device)
    if abs(tckpt.get("width_mult", 1.0) - cfg.kd.teacher_width_mult) > 1e-9:
        print(f"[warn] teacher checkpoint width_mult={tckpt.get('width_mult')} "
              f"!= cfg.kd.teacher_width_mult={cfg.kd.teacher_width_mult}")
    freeze_teacher(teacher)
    print(f"[teacher] loaded {ckpt_path} "
          f"(width_mult={tckpt.get('width_mult')}, "
          f"id_acc={tckpt.get('metrics', {}).get('id_acc', 'n/a')})")
    return teacher


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.set)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.wandb:
        cfg.wandb.enabled = True
    cfg.validate()

    set_seed(cfg.seed, deterministic=cfg.deterministic)
    device = resolve_device(args.device, cfg.precision)
    amp = cfg.precision == "amp"

    run_dir = cfg.run_dir()
    ckpt_dir = run_dir / "checkpoints"
    logger = RunLogger(run_dir, config=cfg, use_wandb=cfg.wandb.enabled)
    print(f"[run] {cfg.resolve_run_name()}  ->  {run_dir}")
    print(f"[cfg] mode={cfg.mode} precision={cfg.precision} seed={cfg.seed} "
          f"device={device} epochs={cfg.schedule.epochs}")

    # --- data ---
    train_loader, test_loader = build_cifar10_loaders(
        cfg.data.data_dir, cfg.data.batch_size, cfg.data.num_workers, seed=cfg.seed,
    )

    # --- model ---
    model = build_model(cfg.model.arch, cfg.model.num_classes,
                        cfg.model.width_mult).to(device)
    n_params = count_parameters(model)
    print(f"[model] {cfg.model.arch} width_mult={cfg.model.width_mult} "
          f"params={n_params:,}")
    logger.update_summary(model_params=n_params)

    teacher = load_teacher(cfg, device) if cfg.mode == "student" else None

    # --- optim ---
    optimizer = build_optimizer(model, cfg.optim)
    scheduler = build_scheduler(optimizer, cfg.schedule)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # --- loop ---
    best_acc = 0.0
    best_epoch = -1
    t0 = time.time()
    for epoch in range(cfg.schedule.epochs):
        train_stats = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer, device=device,
            scaler=scaler, amp=amp, teacher=teacher,
            kd_temperature=cfg.kd.temperature, kd_alpha=cfg.kd.alpha,
            limit_batches=cfg.debug.limit_train_batches,
        )
        scheduler.step()
        val_stats = evaluate_classifier(
            model, test_loader, device,
            limit_batches=cfg.debug.limit_val_batches,
        )

        logger.log_metrics(epoch, "train", **train_stats)
        logger.log_metrics(epoch, "val", acc=val_stats["acc"],
                           error=val_stats["error"], loss=val_stats["loss"])
        print(f"epoch {epoch:3d}/{cfg.schedule.epochs}  "
              f"train_loss={train_stats['loss']:.4f}  "
              f"train_acc={train_stats['acc']:.4f}  "
              f"val_acc={val_stats['acc']:.4f}  lr={train_stats['lr']:.5f}")

        metrics = {"id_acc": val_stats["acc"], "id_error": val_stats["error"],
                   "train_loss": train_stats["loss"]}
        save_checkpoint(ckpt_dir / "last.pt", model, cfg, epoch, metrics)
        if val_stats["acc"] > best_acc:
            best_acc = val_stats["acc"]
            best_epoch = epoch
            save_checkpoint(ckpt_dir / "best.pt", model, cfg, epoch, metrics)

    elapsed = time.time() - t0
    logger.update_summary(
        best_id_acc=best_acc, best_epoch=best_epoch,
        final_id_acc=val_stats["acc"], train_seconds=elapsed,
    )
    logger.finish()
    print(f"[done] best val_acc={best_acc:.4f} @ epoch {best_epoch}  "
          f"({elapsed/60:.1f} min)  ->  {ckpt_dir/'best.pt'}")


if __name__ == "__main__":
    main()
