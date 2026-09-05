"""Shared utilities: seeding, environment capture, and offline CSV/JSON logging.

The config system lives in ``src/config.py``; this module re-exports
:func:`load_config` for convenience.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import Config, config_to_dict, load_config  # noqa: F401  (re-export)

__all__ = ["set_seed", "seed_worker", "make_generator", "capture_env",
           "RunLogger", "load_config", "accuracy"]


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed all RNGs and (optionally) request deterministic kernels.

    ``deterministic=True`` sets cuDNN to deterministic/non-benchmark mode and
    calls ``torch.use_deterministic_algorithms(True, warn_only=True)``. We use
    ``warn_only`` so ops without a deterministic implementation (e.g. some
    pooling backward kernels) warn instead of raising — full determinism is not
    achievable for every op, but training becomes near-reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Required for deterministic CUBLAS (matmul) on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # older torch without warn_only
            pass
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` for reproducible shuffling/augmentation."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """A CPU generator to pass as DataLoader ``generator=`` for reproducibility."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# --------------------------------------------------------------------------- #
# Environment capture (recorded in every summary.json)
# --------------------------------------------------------------------------- #
def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parent,
        )
        return out.decode().strip()
    except Exception:
        return None


def capture_env() -> dict[str, Any]:
    """Snapshot of the software/hardware stack for reproducibility."""
    info: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "git_commit": _git_commit(),
    }
    if torch.cuda.is_available():
        info.update(
            cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),
            gpu_name=torch.cuda.get_device_name(0),
            gpu_count=torch.cuda.device_count(),
        )
    return info


# --------------------------------------------------------------------------- #
# Metrics helper
# --------------------------------------------------------------------------- #
@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy in [0, 1] for a batch."""
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


# --------------------------------------------------------------------------- #
# Offline logging: metrics.csv + summary.json  (+ optional wandb)
# --------------------------------------------------------------------------- #
class RunLogger:
    """Append-only CSV of per-step metrics plus a single rolling summary.json.

    Everything is written to disk immediately and works with no network. wandb
    is only imported/used when ``use_wandb=True``.
    """

    def __init__(self, run_dir: str | Path, *, config: Config | None = None,
                 use_wandb: bool = False) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.csv"
        self.summary_path = self.run_dir / "summary.json"

        self._rows: list[dict[str, Any]] = []
        self._fieldnames: list[str] = []
        self.summary: dict[str, Any] = {"env": capture_env()}
        if config is not None:
            self.summary["config"] = config_to_dict(config)
            self.summary["run_name"] = config.resolve_run_name()
        self._flush_summary()

        self._wandb = None
        if use_wandb:
            self._init_wandb(config)

    # -- wandb (optional) --------------------------------------------------- #
    def _init_wandb(self, config: Config | None) -> None:
        try:
            import wandb  # noqa: PLC0415  (optional dependency)
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("use_wandb=True but wandb is not installed") from e
        wcfg = getattr(config, "wandb", None)
        self._wandb = wandb.init(
            project=getattr(wcfg, "project", "student-teacher-landscape"),
            entity=getattr(wcfg, "entity", None),
            name=self.summary.get("run_name"),
            config=self.summary.get("config"),
            dir=str(self.run_dir),
        )

    # -- per-step metrics ------------------------------------------------- #
    def log_metrics(self, step: int, split: str, **metrics: float) -> None:
        """Record one row: ``step, split, <metric>=<value>, ...``."""
        row: dict[str, Any] = {"step": step, "split": split}
        row.update({k: v for k, v in metrics.items()})
        self._rows.append(row)
        for k in row:
            if k not in self._fieldnames:
                self._fieldnames.append(k)
        self._flush_metrics()
        if self._wandb is not None:
            self._wandb.log({f"{split}/{k}": v for k, v in metrics.items()}, step=step)

    def _flush_metrics(self) -> None:
        with open(self.metrics_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._fieldnames)
            writer.writeheader()
            for r in self._rows:
                writer.writerow(r)

    # -- rolling summary ------------------------------------------------- #
    def update_summary(self, **kv: Any) -> None:
        self.summary.update(kv)
        self._flush_summary()

    def _flush_summary(self) -> None:
        with open(self.summary_path, "w", encoding="utf-8") as fh:
            json.dump(self.summary, fh, indent=2, default=str)

    # -- lifecycle ----------------------------------------------------- #
    def finish(self) -> None:
        self._flush_metrics()
        self._flush_summary()
        if self._wandb is not None:
            self._wandb.finish()
