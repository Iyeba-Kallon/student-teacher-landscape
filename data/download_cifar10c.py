"""Download and unpack CIFAR-10-C.

CIFAR-10-C is not in torchvision. It ships as one tar archive on Zenodo (record
2535967, from Hendrycks & Dietterich, 2019). This script downloads it, checks
the MD5 (looked up from the Zenodo API, with a pinned fallback), extracts it to
<dest>/CIFAR-10-C/<corruption>.npy, and deletes the tar. It is idempotent: if
the .npy files are already there it does nothing.

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
FALLBACK_MD5 = "56bf5dcef84df0e2308c6dcbcbbd8499"

# 19 corruption arrays + labels.npy (15 test corruptions, then 4 validation ones).
EXPECTED_NPY = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
    "speckle_noise", "gaussian_blur", "spatter", "saturate",
]
CHUNK = 1 << 20


def _progress(done: int, total: int, prefix: str) -> None:
    if total <= 0:
        sys.stdout.write(f"\r{prefix} {done / 1e6:.1f} MB")
    else:
        pct = 100 * done / total
        sys.stdout.write(f"\r{prefix} [{'#' * int(pct // 2):<50}] {pct:5.1f}%  "
                         f"{done / 1e6:.0f}/{total / 1e6:.0f} MB")
    sys.stdout.flush()


def _md5(path: Path) -> str:
    h = hashlib.md5()
    size = path.stat().st_size
    done = 0
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
            done += len(block)
            _progress(done, size, "  md5")
    print()
    return h.hexdigest()


def _zenodo_md5() -> str | None:
    try:
        r = requests.get(ZENODO_API, timeout=15)
        r.raise_for_status()
        for f in r.json().get("files", []):
            if f.get("key") == ARCHIVE_NAME:
                return f.get("checksum", "").split(":", 1)[-1] or None
    except Exception as e:
        print(f"[warn] Zenodo API unreachable ({e}); using the pinned MD5")
    return None


def _download(url: str, archive: Path) -> None:
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(archive, "wb") as fh:
            for chunk in r.iter_content(chunk_size=CHUNK):
                fh.write(chunk)
                done += len(chunk)
                _progress(done, total, "  download")
    print()


def _safe_extract(archive: Path, dest: Path) -> None:
    base = os.path.abspath(dest)
    with tarfile.open(archive, "r:*") as tar:
        for member in tar.getmembers():
            target = os.path.abspath(os.path.join(dest, member.name))
            if target != base and not target.startswith(base + os.sep):
                raise RuntimeError(f"unsafe path in archive: {member.name!r}")
        if sys.version_info >= (3, 12):
            tar.extractall(dest, filter="data")
        else:
            tar.extractall(dest)


def _already_extracted(cdir: Path) -> bool:
    return (cdir.is_dir() and (cdir / "labels.npy").exists()
            and all((cdir / f"{name}.npy").exists() for name in EXPECTED_NPY))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", default="data", help="where to download and extract")
    p.add_argument("--url", default=DEFAULT_URL, help="override the download URL")
    p.add_argument("--skip-md5-check", action="store_true",
                   help="do not verify the checksum")
    p.add_argument("--keep-archive", action="store_true",
                   help="keep the tar after extracting (default: delete it)")
    args = p.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    cdir = dest / "CIFAR-10-C"
    archive = dest / ARCHIVE_NAME

    if _already_extracted(cdir):
        print(f"[done] CIFAR-10-C already present at {cdir}")
        return

    expected_md5 = None if args.skip_md5_check else (_zenodo_md5() or FALLBACK_MD5)

    need_download = True
    if archive.exists() and expected_md5:
        print(f"[archive] {archive} exists, checking MD5")
        need_download = _md5(archive) != expected_md5
        if need_download:
            print("[archive] MD5 mismatch, downloading again")
            archive.unlink()
    if need_download:
        print(f"[archive] downloading {ARCHIVE_NAME}")
        _download(args.url, archive)
        if expected_md5:
            actual = _md5(archive)
            if actual != expected_md5:
                raise RuntimeError(
                    f"MD5 mismatch for {archive}: expected {expected_md5}, got {actual}. "
                    "Delete it and retry, or pass --skip-md5-check."
                )
            print("[archive] MD5 OK")

    print(f"[extract] {archive} -> {dest}/")
    _safe_extract(archive, dest)
    if not _already_extracted(cdir):
        raise RuntimeError(f"extraction finished but files are missing under {cdir}")
    print(f"[extract] {len(EXPECTED_NPY)} corruption arrays + labels.npy")

    if not args.keep_archive:
        archive.unlink(missing_ok=True)
        print(f"[cleanup] removed {archive}")

    print(f"[done] CIFAR-10-C ready at {cdir}")


if __name__ == "__main__":
    main()
