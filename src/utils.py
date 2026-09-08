"""Seeding, environment capture, and file-based logging.

Re-exports load_config from src.config so callers only import one module.
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

from .config import Config, config_to_dict, load_config  # noqa: F401

__all__ = ["set_seed", "seed_worker", "make_generator", "capture_env",
           "RunLogger", "load_config", "accuracy"]


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and torch.

    With deterministic=True this also turns off cuDNN autotuning and asks torch
    for deterministic kernels. warn_only=True keeps it from raising on the few
    ops that have no deterministic implementation; those just fall back.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """worker_init_fn that reseeds NumPy/random inside each DataLoader worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


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
    """Software/hardware details recorded in every summary.json."""
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


@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == targets).float().mean().item()


class RunLogger:
    """Writes metrics.csv (one row per logged step) and summary.json.

    Both files are rewritten on every call, so a run is safe to inspect or kill
    at any point. With append=True the logger loads whatever is already on disk
    first, so evaluate.py and measure_geometry.py extend a training run's files
    rather than overwrite them. wandb is only touched when use_wandb=True.
    """

    def __init__(self, run_dir: str | Path, *, config: Config | None = None,
                 use_wandb: bool = False, append: bool = False) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.csv"
        self.summary_path = self.run_dir / "summary.json"

        self._rows: list[dict[str, Any]] = []
        self._fieldnames: list[str] = []
        self.summary: dict[str, Any] = {}

        if append and self.metrics_path.exists():
            with open(self.metrics_path, "r", newline="", encoding="utf-8") as fh:
                self._rows = list(csv.DictReader(fh))
            for row in self._rows:
                for k in row:
                    if k not in self._fieldnames:
                        self._fieldnames.append(k)
        if append and self.summary_path.exists():
            with open(self.summary_path, "r", encoding="utf-8") as fh:
                self.summary = json.load(fh)

        self.summary.setdefault("env", capture_env())
        if config is not None:
            self.summary["config"] = config_to_dict(config)
            self.summary["run_name"] = config.resolve_run_name()
        self._flush_summary()

        self._wandb = None
        if use_wandb:
            self._init_wandb(config)

    def _init_wandb(self, config: Config | None) -> None:
        try:
            import wandb
        except ImportError as e:
            raise RuntimeError("use_wandb=True but wandb is not installed") from e
        wcfg = getattr(config, "wandb", None)
        self._wandb = wandb.init(
            project=getattr(wcfg, "project", "student-teacher-landscape"),
            entity=getattr(wcfg, "entity", None),
            name=self.summary.get("run_name"),
            config=self.summary.get("config"),
            dir=str(self.run_dir),
        )

    def log_metrics(self, step: int, split: str, **metrics: float) -> None:
        row: dict[str, Any] = {"step": step, "split": split, **metrics}
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
            writer.writerows(self._rows)

    def update_summary(self, **kv: Any) -> None:
        self.summary.update(kv)
        self._flush_summary()

    def _flush_summary(self) -> None:
        with open(self.summary_path, "w", encoding="utf-8") as fh:
            json.dump(self.summary, fh, indent=2, default=str)

    def finish(self) -> None:
        self._flush_metrics()
        self._flush_summary()
        if self._wandb is not None:
            self._wandb.finish()
