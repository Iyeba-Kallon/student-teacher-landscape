"""Run configuration: nested dataclasses loaded from a YAML file.

One YAML file in configs/ describes a run. load_config() parses it into a Config,
applies any `key.subkey=value` overrides from the command line, and validates.
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class ModelConfig:
    arch: str = "resnet18_cifar"
    width_mult: float = 1.0
    num_classes: int = 10


@dataclass
class DataConfig:
    dataset: str = "cifar10"
    data_dir: str = "data"
    batch_size: int = 128
    num_workers: int = 4


@dataclass
class OptimConfig:
    name: str = "sgd"
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    nesterov: bool = False


@dataclass
class ScheduleConfig:
    name: str = "cosine"          # cosine, multistep, or constant
    epochs: int = 200
    warmup_epochs: int = 0
    milestones: list[int] = field(default_factory=lambda: [100, 150])
    gamma: float = 0.1            # multistep decay factor


@dataclass
class KDConfig:
    """Distillation settings. Only used when mode == 'student'."""
    temperature: float = 4.0
    alpha: float = 0.9            # weight on the soft (KD) term
    teacher_checkpoint: Optional[str] = None
    teacher_arch: str = "resnet18_cifar"
    teacher_width_mult: float = 1.0


@dataclass
class DebugConfig:
    """Batch limits for smoke tests. 0 means no limit."""
    limit_train_batches: int = 0
    limit_val_batches: int = 0


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "student-teacher-landscape"
    entity: Optional[str] = None


@dataclass
class Config:
    mode: str = "teacher"        # teacher or student
    precision: str = "fp32"      # fp32 or amp (training only)
    seed: int = 0

    run_name: Optional[str] = None
    results_dir: str = "results"
    deterministic: bool = True

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    kd: KDConfig = field(default_factory=KDConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    def resolve_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        if self.mode == "teacher":
            return f"teacher_{self.precision}_s{self.seed}"
        w = f"{self.model.width_mult:g}"          # 0.5 -> "0.5", 0.25 -> "0.25"
        return f"student_w{w}_{self.precision}_s{self.seed}"

    def run_dir(self) -> Path:
        return Path(self.results_dir) / self.resolve_run_name()

    def validate(self) -> None:
        if self.mode not in ("teacher", "student"):
            raise ValueError(f"mode must be 'teacher' or 'student', got {self.mode!r}")
        if self.precision not in ("fp32", "amp"):
            raise ValueError(f"precision must be 'fp32' or 'amp', got {self.precision!r}")
        if self.mode == "student" and not self.kd.teacher_checkpoint:
            raise ValueError("student mode needs kd.teacher_checkpoint")


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Build a (possibly nested) dataclass from a plain dict, rejecting unknown keys."""
    if not dataclasses.is_dataclass(cls):
        return data
    hints = typing.get_type_hints(cls)
    unknown = set(data) - {f.name for f in dataclasses.fields(cls)}
    if unknown:
        raise KeyError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")

    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        ftype = hints[f.name]
        value = data[f.name]
        if dataclasses.is_dataclass(ftype) and isinstance(value, dict):
            kwargs[f.name] = _from_dict(ftype, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def apply_overrides(cfg: Config, overrides: list[str] | None) -> Config:
    """Apply `a.b.c=value` overrides in place. The value is parsed as YAML."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not key=value")
        key, _, raw = item.partition("=")
        value = yaml.safe_load(raw)
        obj: Any = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        if not hasattr(obj, parts[-1]):
            raise KeyError(f"override targets unknown field: {key}")
        setattr(obj, parts[-1], value)
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = _from_dict(Config, raw)
    cfg = apply_overrides(cfg, overrides)
    cfg.validate()
    return cfg


def config_to_dict(cfg: Config) -> dict[str, Any]:
    return dataclasses.asdict(cfg)
