"""Download and unpack the official CIFAR-10-C dataset.

CIFAR-10-C is not in torchvision. It is distributed as a single tar archive on
Zenodo (Hendrycks & Dietterich, 2019, "Benchmarking Neural Network Robustness
to Common Corruptions and Perturbations"):

    Record : https://zenodo.org/records/2535967
    File   : CIFAR-10-C.tar   (~2.9 GB)

This script:
1. Looks up the archive's MD5 from the Zenodo API (falls back to a pinned
   constant if the API is unreachable).
2. Stream-downloads CIFAR-10-C.tar into ``<dest>/`` with a progress bar and
   HTTP-range resume for interrupted downloads.
3. Verifies the MD5 checksum.
4. Extracts it (with a path-traversal guard), producing
   ``<dest>/CIFAR-10-C/<corruption>.npy`` (+ ``labels.npy``).
5. Is idempotent: if the extracted ``.npy`` files are already present it does
   nothing; a verified archive is not re-downloaded.

Usage:
    python data/download_cifar10c.py --dest data/
    python data/download_cifar10c.py --dest data/ --keep-archive
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tarfile
from pathlib import Path

import requests

ZENODO_RECORD = "2535967"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
ARCHIVE_NAME = "CIFAR-10-C.tar"
DEFAULT_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{ARCHIVE_NAME}?download=1"
# Fallback MD5 for CIFAR-10-C.tar (used only if the Zenodo API cannot be reached).
FALLBACK_MD5 = "56bf5dcef84df0e2308c6dcbcbbd8499"

# 19 corruption arrays + labels.npy. The first 15 are the "test" corruptions;
# the last 4 are validation corruptions (still shipped in the archive).
EXPECTED_NPY = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
    "speckle_noise", "gaussian_blur", "spatter", "saturate",
]
CHUNK = 1 << 20  # 1 MiB


# --------------------------------------------------------------------------- #
def _progress(done: int, total: int, prefix: str = "") -> None:
    if total <= 0:
        sys.stdout.write(f"\r{prefix} {done / 1e6:.1f} MB")
    else:
        pct = 100 * done / total
        bar = "#" * int(pct // 2)
        sys.stdout.write(f"\r{prefix} [{bar:<50}] {pct:5.1f}%  "
                         f"{done / 1e6:.0f}/{total / 1e6:.0f} MB")
    sys.stdout.flush()


def _md5_of(path: Path, prefix: str = "  md5") -> str:
    h = hashlib.md5()
    size = path.stat().st_size
    done = 0
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
            done += len(block)
            _progress(done, size, prefix)
    print()
    return h.hexdigest()


def _lookup_zenodo_md5() -> str | None:
    try:
        r = requests.get(ZENODO_API, timeout=15)
        r.raise_for_status()
        for f in r.json().get("files", []):
            if f.get("key") == ARCHIVE_NAME:
                checksum = f.get("checksum", "")
                return checksum.split(":", 1)[-1] or None
    except Exception as e:  # offline / API change -> use fallback
        print(f"[warn] could not query Zenodo API ({e}); using pinned MD5")
    return None


def _download(url: str, dest: Path, expected_md5: str) -> Path:
    archive = dest / ARCHIVE_NAME

    # Already downloaded and valid?
    if archive.exists():
        print(f"[archive] {archive} exists; verifying MD5 ...")
        if _md5_of(archive) == expected_md5:
            print("[archive] MD5 OK")
            return archive
        print("[archive] MD5 mismatch; re-downloading from scratch")
        archive.unlink()

    resume_from = 0
    if archive.exists():
        resume_from = archive.stat().st_size
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}

    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        mode = "ab" if r.status_code == 206 and resume_from else "wb"
        if mode == "wb":
            resume_from = 0
        total = int(r.headers.get("Content-Length", 0)) + resume_from
        done = resume_from
        with open(archive, mode) as fh:
            for chunk in r.iter_content(chunk_size=CHUNK):
                fh.write(chunk)
                done += len(chunk)
                _progress(done, total, "  download")
    print()

    print("[archive] verifying MD5 ...")
    actual = _md5_of(archive)
    if actual != expected_md5:
        raise RuntimeError(
            f"MD5 mismatch for {archive}\n  expected {expected_md5}\n  got      {actual}\n"
            "Delete the file and retry, or pass --skip-md5-check if you trust the source."
        )
    print("[archive] MD5 OK")
    return archive


def _safe_extract(archive: Path, dest: Path) -> None:
    base = os.path.abspath(dest)
    with tarfile.open(archive, "r:*") as tar:
        for member in tar.getmembers():
            target = os.path.abspath(os.path.join(dest, member.name))
            if target != base and not target.startswith(base + os.sep):
                raise RuntimeError(f"unsafe path in archive: {member.name!r}")
        # Python 3.12+ supports a vetted extraction filter; use it when present.
        if sys.version_info >= (3, 12):
            tar.extractall(dest, filter="data")
        else:
            tar.extractall(dest)


def _already_extracted(cdir: Path) -> bool:
    if not cdir.is_dir():
        return False
    if not (cdir / "labels.npy").exists():
        return False
    return all((cdir / f"{name}.npy").exists() for name in EXPECTED_NPY)


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", default="data", type=str,
                   help="directory to download into / extract under (default: data)")
    p.add_argument("--url", default=DEFAULT_URL, help="override the download URL")
    p.add_argument("--skip-md5-check", action="store_true",
                   help="do not verify the archive checksum (not recommended)")
    p.add_argument("--keep-archive", action="store_true",
                   help="keep CIFAR-10-C.tar after extraction (default: delete it)")
    args = p.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    cdir = dest / "CIFAR-10-C"

    if _already_extracted(cdir):
        print(f"[done] CIFAR-10-C already present at {cdir} — nothing to do.")
        return

    expected_md5 = (None if args.skip_md5_check
                    else (_lookup_zenodo_md5() or FALLBACK_MD5))

    if args.skip_md5_check:
        archive = dest / ARCHIVE_NAME
        if not archive.exists():
            with requests.get(args.url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                done = 0
                with open(archive, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=CHUNK):
                        fh.write(chunk)
                        done += len(chunk)
                        _progress(done, total, "  download")
            print()
    else:
        archive = _download(args.url, dest, expected_md5)

    print(f"[extract] unpacking {archive} -> {dest}/ ...")
    _safe_extract(archive, dest)

    if not _already_extracted(cdir):
        raise RuntimeError(
            f"extraction finished but expected files are missing under {cdir}"
        )
    print(f"[extract] OK — {len(EXPECTED_NPY)} corruption arrays + labels.npy")

    if not args.keep_archive:
        archive.unlink(missing_ok=True)
        print(f"[cleanup] removed {archive} (pass --keep-archive to retain)")

    print(f"[done] CIFAR-10-C ready at {cdir}")


if __name__ == "__main__":
    main()
