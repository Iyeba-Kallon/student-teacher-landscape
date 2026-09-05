"""Download and unpack the official CIFAR-10-C dataset.

STATUS: placeholder — full implementation lands in the next step.

CIFAR-10-C is not in torchvision. It is distributed as a single tar archive on
Zenodo (Hendrycks & Dietterich, 2019, "Benchmarking Neural Network Robustness
to Common Corruptions and Perturbations"):

    Record : https://zenodo.org/record/2535967
    File   : CIFAR-10-C.tar   (~2.9 GB)
    MD5    : 56bf5dcef84df0e2308c6dcbcbbd8499

This script will:
1. Stream-download CIFAR-10-C.tar into data/ (resumable, with a progress bar).
2. Verify the MD5 checksum.
3. Extract it, producing data/CIFAR-10-C/<corruption>.npy (+ labels.npy).
4. Skip work that is already done (idempotent).

Usage:
    python data/download_cifar10c.py --dest data/
"""

from __future__ import annotations

ZENODO_URL = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1"
ARCHIVE_MD5 = "56bf5dcef84df0e2308c6dcbcbbd8499"


def main() -> None:
    raise NotImplementedError("Implemented in the next step.")


if __name__ == "__main__":
    main()
