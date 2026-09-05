"""Dataset builders for CIFAR-10 (ID) and CIFAR-10-C (OOD).

STATUS: placeholder — implementation lands in the next step.

- CIFAR-10: torchvision, standard train augmentation (random crop + flip),
  normalization with the usual CIFAR-10 mean/std.
- CIFAR-10-C: loaded from the unpacked Zenodo tarball (see
  data/download_cifar10c.py). Each corruption is a (50000, 32, 32, 3) uint8
  array with 10000 images per severity level 1..5, labels shared with the
  CIFAR-10 test set. Only normalization is applied (no augmentation).
"""

from __future__ import annotations

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR10C_CORRUPTIONS = (
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
)
