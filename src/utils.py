"""Shared utilities: seeding, config loading, CSV/JSON logging.

STATUS: placeholder — implementation lands in the next step.

- set_seed(seed): seeds python / numpy / torch, sets deterministic cuDNN flags
  and torch.use_deterministic_algorithms where it does not break the models used.
- load_config(path): loads a YAML file into a dataclass-backed config object,
  supports CLI overrides.
- Logger: appends rows to results/<run>/metrics.csv and mirrors a full
  results/<run>/summary.json. Works fully offline. wandb is only touched when
  cfg.wandb.enabled is True.
"""

from __future__ import annotations
