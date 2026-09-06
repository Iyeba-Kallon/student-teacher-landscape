"""Dataset builders for CIFAR-10 (ID) and CIFAR-10-C (OOD).

- CIFAR-10 (torchvision): random-crop + horizontal-flip augmentation for
  training; normalization only for evaluation.
- Geometry subset: a fixed, seeded 2,000-image slice of the CIFAR-10 *training*
  set with **no augmentation**, reused for every sharpness / Hessian
  measurement so curvature numbers are comparable across models.
- CIFAR-10-C: loaded from the unpacked Zenodo tarball
  (see ``data/download_cifar10c.py``). Each ``<corruption>.npy`` is a
  (50000, 32, 32, 3) uint8 array = 10000 images per severity level 1..5, in
  order. ``labels.npy`` (50000,) repeats the CIFAR-10 test labels five times.
  Only normalization is applied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .utils import make_generator, seed_worker

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# The 15 "test" corruptions from Hendrycks & Dietterich (2019). The extra 4
# ("speckle_noise", "gaussian_blur", "spatter", "saturate") are validation
# corruptions and are intentionally not used in the pilot.
CIFAR10C_CORRUPTIONS = (
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
)
CIFAR10C_SEVERITIES = (1, 2, 3, 4, 5)


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def _normalize() -> transforms.Normalize:
    return transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)


def train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        _normalize(),
    ])


def eval_transform() -> transforms.Compose:
    return transforms.Compose([transforms.ToTensor(), _normalize()])


# --------------------------------------------------------------------------- #
# CIFAR-10 (in-distribution)
# --------------------------------------------------------------------------- #
def build_cifar10_loaders(
    data_dir: str | Path,
    batch_size: int = 128,
    num_workers: int = 4,
    seed: int = 0,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Return ``(train_loader, test_loader)`` for CIFAR-10."""
    data_dir = str(data_dir)
    train_set = datasets.CIFAR10(data_dir, train=True, download=download,
                                 transform=train_transform())
    test_set = datasets.CIFAR10(data_dir, train=False, download=download,
                                transform=eval_transform())

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
        worker_init_fn=seed_worker, generator=make_generator(seed),
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


# --------------------------------------------------------------------------- #
# Fixed geometry subset (train split, no augmentation)
# --------------------------------------------------------------------------- #
def build_geometry_loader(
    data_dir: str | Path,
    n_examples: int = 2000,
    batch_size: int = 128,
    subset_seed: int = 1234,
    download: bool = True,
) -> DataLoader:
    """A DataLoader over a fixed, seeded subset of the CIFAR-10 training set.

    Used for ALL geometry measurements. ``subset_seed`` is deliberately a
    separate constant (not the run seed) so the *same* images are used for
    every model regardless of how it was trained.
    """
    base = datasets.CIFAR10(str(data_dir), train=True, download=download,
                            transform=eval_transform())
    rng = np.random.default_rng(subset_seed)
    indices = rng.choice(len(base), size=n_examples, replace=False)
    indices.sort()
    subset = Subset(base, indices.tolist())
    # No shuffling: deterministic iteration order for reproducible curvature.
    return DataLoader(subset, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=True)


# --------------------------------------------------------------------------- #
# CIFAR-10-C (out-of-distribution)
# --------------------------------------------------------------------------- #
class CIFAR10C(Dataset):
    """One (corruption, severity) split of CIFAR-10-C."""

    def __init__(self, root: str | Path, corruption: str, severity: int,
                 transform=None) -> None:
        if corruption not in CIFAR10C_CORRUPTIONS:
            raise ValueError(f"Unknown corruption {corruption!r}")
        if severity not in CIFAR10C_SEVERITIES:
            raise ValueError(f"severity must be 1..5, got {severity}")

        cdir = Path(root) / "CIFAR-10-C"
        images_path = cdir / f"{corruption}.npy"
        labels_path = cdir / "labels.npy"
        if not images_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                f"CIFAR-10-C not found under {cdir}. "
                f"Run `python data/download_cifar10c.py --dest {root}` first."
            )

        images = np.load(images_path)          # (N, 32, 32, 3) uint8, N = 5 * per
        labels = np.load(labels_path)          # (N,) int
        if len(images) % 5 != 0 or len(images) != len(labels):
            raise ValueError(
                f"malformed CIFAR-10-C arrays: images={images.shape} labels={labels.shape}"
            )
        per = len(images) // 5                 # 10000 for the official dataset
        lo = (severity - 1) * per
        hi = severity * per
        self.images = images[lo:hi]
        self.labels = labels[lo:hi].astype(np.int64)
        self.transform = transform or eval_transform()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.images[idx]                 # HWC uint8
        # transforms.ToTensor handles an HWC uint8 ndarray -> CHW float in [0,1].
        img = self.transform(img)
        return img, int(self.labels[idx])


def build_cifar10c_loader(
    data_dir: str | Path,
    corruption: str,
    severity: int,
    batch_size: int = 128,
    num_workers: int = 4,
) -> DataLoader:
    ds = CIFAR10C(data_dir, corruption, severity)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def iter_cifar10c_loaders(
    data_dir: str | Path,
    batch_size: int = 128,
    num_workers: int = 4,
    corruptions: tuple[str, ...] = CIFAR10C_CORRUPTIONS,
    severities: tuple[int, ...] = CIFAR10C_SEVERITIES,
) -> Iterator[tuple[str, int, DataLoader]]:
    """Yield ``(corruption, severity, loader)`` for the full CIFAR-10-C sweep."""
    for corruption in corruptions:
        for severity in severities:
            yield corruption, severity, build_cifar10c_loader(
                data_dir, corruption, severity, batch_size, num_workers
            )
