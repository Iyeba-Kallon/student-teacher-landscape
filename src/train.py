"""Train a teacher or distill a student from one config file.

    python -m src.train --config configs/teacher_fp32.yaml --seed 0
    python -m src.train --config configs/student_w0.5_fp32.yaml --seed 0

mode: student needs kd.teacher_checkpoint (which may contain {precision}/{seed}/
{results_dir} placeholders, filled in after --seed is applied). precision: amp
uses autocast + GradScaler for training only and needs CUDA; validation and the
saved weights are always fp32. Output goes to results/<run_name>/.

Resume: if results/<run_name>/checkpoints/last.pt already exists, training picks
up from the epoch after it instead of starting over. last.pt carries the
optimizer and GradScaler state alongside the weights; the LR schedule is rebuilt
by stepping the scheduler to the resume epoch, so it is correct even for
checkpoints written by older code. The one thing not restored is the data
loader's RNG/shuffle state, so a resumed run sees
a different batch order than an uninterrupted one would have; harmless for
training, just means a resumed run isn't bit-for-bit reproducible against a
from-scratch run with the same seed.
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Any

import torch

from .checkpoint import build_model_from_checkpoint, load_checkpoint, save_checkpoint
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
    p.add_argument("--config", required=True, help="path to a YAML config")
    p.add_argument("--seed", type=int, default=None, help="override cfg.seed")
    p.add_argument("--wandb", action="store_true", help="also log to wandb")
    p.add_argument("--set", nargs="*", default=None, metavar="key=value",
                   help="config overrides, e.g. schedule.epochs=1 optim.lr=0.05")
    p.add_argument("--device", default=None, help="cuda or cpu (auto if unset)")
    return p.parse_args()


def resolve_device(requested: str | None, precision: str) -> torch.device:
    dev = torch.device(requested) if requested else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    if precision == "amp" and dev.type != "cuda":
        raise RuntimeError("precision: amp needs a CUDA device; use fp32 on CPU")
    return dev


def load_teacher(cfg: Config, device: torch.device) -> torch.nn.Module:
    ckpt_path = Path(cfg.kd.teacher_checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"teacher checkpoint not found: {ckpt_path}")
    teacher, tckpt = build_model_from_checkpoint(ckpt_path, device=device)
    if abs(tckpt.get("width_mult", 1.0) - cfg.kd.teacher_width_mult) > 1e-9:
        print(f"[warn] teacher width_mult {tckpt.get('width_mult')} "
              f"!= config {cfg.kd.teacher_width_mult}")
    freeze_teacher(teacher)
    print(f"[teacher] {ckpt_path} "
          f"(width_mult={tckpt.get('width_mult')}, "
          f"id_acc={tckpt.get('metrics', {}).get('id_acc', 'n/a')})")
    return teacher


def load_resume_state(ckpt_dir: Path, device: torch.device) -> dict[str, Any] | None:
    """If checkpoints/last.pt exists, load it for a resume. Otherwise None."""
    last_path = ckpt_dir / "last.pt"
    if not last_path.exists():
        return None
    ckpt = load_checkpoint(last_path, map_location=str(device))
    best_path = ckpt_dir / "best.pt"
    if best_path.exists():
        best_ckpt = load_checkpoint(best_path, map_location="cpu")
        best_acc = best_ckpt["metrics"].get("id_acc", 0.0)
        best_epoch = best_ckpt["epoch"]
    else:  # last.pt without a best.pt should not happen, but do not crash on it
        best_acc, best_epoch = ckpt["metrics"].get("id_acc", 0.0), ckpt["epoch"]
    return {
        "model_state": ckpt["model_state"],
        "optimizer_state": ckpt.get("optimizer_state"),
        "scaler_state": ckpt.get("scaler_state"),
        "start_epoch": ckpt["epoch"] + 1,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.set)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.wandb:
        cfg.wandb.enabled = True

    # Fill in the teacher path after the seed override so --seed 1 picks up the
    # matching teacher.
    if cfg.mode == "student" and cfg.kd.teacher_checkpoint:
        cfg.kd.teacher_checkpoint = cfg.kd.teacher_checkpoint.format(
            precision=cfg.precision, seed=cfg.seed,
            width_mult=cfg.model.width_mult, results_dir=cfg.results_dir,
        )
    cfg.validate()

    set_seed(cfg.seed, deterministic=cfg.deterministic)
    device = resolve_device(args.device, cfg.precision)
    amp = cfg.precision == "amp"

    run_dir = cfg.run_dir()
    ckpt_dir = run_dir / "checkpoints"
    # append=True: a resumed run extends metrics.csv/summary.json instead of
    # wiping them; harmless for a fresh run, which has nothing to append to yet.
    logger = RunLogger(run_dir, config=cfg, use_wandb=cfg.wandb.enabled, append=True)
    print(f"[run] {cfg.resolve_run_name()}  ->  {run_dir}")
    print(f"[cfg] mode={cfg.mode} precision={cfg.precision} seed={cfg.seed} "
          f"device={device} epochs={cfg.schedule.epochs}")

    train_loader, test_loader = build_cifar10_loaders(
        cfg.data.data_dir, cfg.data.batch_size, cfg.data.num_workers, seed=cfg.seed,
    )

    model = build_model(cfg.model.arch, cfg.model.num_classes,
                        cfg.model.width_mult).to(device)
    n_params = count_parameters(model)
    print(f"[model] {cfg.model.arch} width_mult={cfg.model.width_mult} "
          f"params={n_params:,}")
    logger.update_summary(model_params=n_params)

    teacher = load_teacher(cfg, device) if cfg.mode == "student" else None

    optimizer = build_optimizer(model, cfg.optim)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_epoch, best_acc, best_epoch = 0, 0.0, -1
    resume = load_resume_state(ckpt_dir, device)
    if resume is not None:
        model.load_state_dict(resume["model_state"])
        if resume["optimizer_state"] is not None:
            optimizer.load_state_dict(resume["optimizer_state"])
        else:
            print("[resume] WARNING: checkpoint has no optimizer state (written by "
                  "older code); momentum buffers start from zero")
        if resume["scaler_state"] is not None:
            scaler.load_state_dict(resume["scaler_state"])
        start_epoch = resume["start_epoch"]
        best_acc, best_epoch = resume["best_acc"], resume["best_epoch"]

        # load_state_dict also restores the LR the optimizer had when it was saved
        # (or, from an earlier buggy resume, a wrong one). The recursive cosine
        # update decays from whatever LR it finds, so put the base LR back before the
        # scheduler is built. The schedule is a pure function of the epoch index, so
        # it is then rebuilt by stepping to the resume epoch, which also works for
        # checkpoints from older code that saved no state at all.
        for group in optimizer.param_groups:
            group["lr"] = cfg.optim.lr
            group.pop("initial_lr", None)

    scheduler = build_scheduler(optimizer, cfg.schedule)
    if resume is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # "scheduler.step() before optimizer.step()"
            for _ in range(start_epoch):
                scheduler.step()

        print(f"[resume] found checkpoints at epoch {start_epoch - 1}; "
              f"continuing from epoch {start_epoch} "
              f"(best so far: {best_acc:.4f} @ epoch {best_epoch}), "
              f"lr={optimizer.param_groups[0]['lr']:.5f}")
        if start_epoch >= cfg.schedule.epochs:
            print(f"[resume] already reached epochs={cfg.schedule.epochs}; nothing to do")

    t0 = time.time()
    for epoch in range(start_epoch, cfg.schedule.epochs):
        train_stats = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer, device=device,
            scaler=scaler, amp=amp, teacher=teacher,
            kd_temperature=cfg.kd.temperature, kd_alpha=cfg.kd.alpha,
            limit_batches=cfg.debug.limit_train_batches,
        )
        scheduler.step()
        val_stats = evaluate_classifier(
            model, test_loader, device, limit_batches=cfg.debug.limit_val_batches,
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
        # optimizer/scaler state only needs to live on last.pt: it is what a resume
        # loads from. best.pt stays a plain model checkpoint, since that is what
        # evaluate.py / measure_geometry.py / build_model_from_checkpoint consume.
        # The LR scheduler is not saved: it is rebuilt from the epoch index.
        save_checkpoint(ckpt_dir / "last.pt", model, cfg, epoch, metrics, extra={
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
        })
        if val_stats["acc"] > best_acc:
            best_acc, best_epoch = val_stats["acc"], epoch
            save_checkpoint(ckpt_dir / "best.pt", model, cfg, epoch, metrics)

    if cfg.schedule.name.lower() == "cosine":
        final_lr = optimizer.param_groups[0]["lr"]
        if final_lr > 1e-3:
            print(f"[WARN] cosine schedule finished at lr={final_lr:.5f}, not ~0: this "
                  f"run did not anneal (was it resumed with a reset schedule?). "
                  f"Delete {run_dir} and retrain before using it.")

    elapsed = time.time() - t0
    final_acc = val_stats["acc"] if start_epoch < cfg.schedule.epochs else best_acc
    logger.update_summary(best_id_acc=best_acc, best_epoch=best_epoch,
                          final_id_acc=final_acc, train_seconds=elapsed)
    logger.finish()
    print(f"[done] best val_acc={best_acc:.4f} @ epoch {best_epoch}  "
          f"({elapsed / 60:.1f} min)  ->  {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
