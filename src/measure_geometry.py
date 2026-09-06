"""Measure loss-landscape geometry for a trained checkpoint.

All measurements run in **fp32** with **BatchNorm frozen** and **no autocast**,
regardless of whether the checkpoint was trained with fp32 or AMP.

Usage
-----
    # one checkpoint
    python -m src.measure_geometry --checkpoint results/student_w0.5_amp_s0/checkpoints/best.pt

    # every results/*/checkpoints/best.pt
    python -m src.measure_geometry --all --results-dir results/

Produces (per run)
------------------
- ``results/<run>/geometry.json``   full results + the exact measurement settings
- ``results/<run>/summary.json``     gains a ``"geometry"`` block
- ``results/<run>/metrics.csv``      gains a ``geometry`` row (append)

Metrics
-------
- ``adaptive_sharpness``      ASAM-style m-sharpness, rho reported (headline)
- ``hessian_trace``           Hutchinson estimator (PyHessian)
- ``hessian_top_eigenvalue``  power iteration (PyHessian)

All three are computed on the same fixed, seeded 2,000-example subset of the
CIFAR-10 *training* set, identical for every model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .checkpoint import build_model_from_checkpoint
from .data import build_geometry_loader
from .geometry import (DEFAULT_ETA, DEFAULT_RHO, adaptive_sharpness,
                       assert_fp32_no_autocast, hessian_metrics,
                       prepare_model_for_geometry)
from .utils import RunLogger, set_seed


def measure_one(ckpt_path: Path, args) -> dict[str, Any]:
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    model, ckpt = build_model_from_checkpoint(ckpt_path, device=device)
    run_dir = ckpt_path.parent.parent
    run_name = ckpt.get("config", {}).get("run_name") or run_dir.name
    print(f"\n=== geometry: {run_name}  ({ckpt_path}) ===")
    print(f"  arch={ckpt['arch']} width_mult={ckpt['width_mult']} "
          f"mode={ckpt['mode']} trained_precision={ckpt['precision']} "
          f"seed={ckpt['seed']}  |  measured in fp32, device={device}")

    # --- mandatory preparation: eval + frozen BN + fp32 ---
    prepare_model_for_geometry(model)
    assert_fp32_no_autocast(model)

    # --- fixed geometry subset; batch size == sharpness micro-batch m ---
    set_seed(args.seed, deterministic=True)
    geom_loader = build_geometry_loader(
        args.data_dir, n_examples=args.n_geom, batch_size=args.m,
        subset_seed=args.subset_seed, download=True,
    )

    settings = {
        "measured_precision": "fp32",
        "trained_precision": ckpt["precision"],
        "bn": "eval + frozen running stats (momentum=0)",
        "geometry_subset": {"split": "cifar10_train", "n": args.n_geom,
                            "subset_seed": args.subset_seed},
        "sharpness": {"kind": "adaptive_m_sharpness" if args.adaptive
                      else "m_sharpness", "rho": args.rho, "eta": DEFAULT_ETA,
                      "m": args.m, "ascent_steps": 1,
                      "n_batches": args.sharpness_batches},
        "hessian": {"library": "pyhessian", "trace_estimator": "hutchinson",
                    "trace_max_iter": args.hessian_trace_iter,
                    "eig_method": "power_iteration",
                    "eig_max_iter": args.hessian_eig_iter, "seed": args.seed},
    }

    result: dict[str, Any] = {
        "run_name": run_name, "checkpoint": str(ckpt_path),
        "arch": ckpt["arch"], "width_mult": ckpt["width_mult"],
        "mode": ckpt["mode"], "precision": ckpt["precision"], "seed": ckpt["seed"],
        "settings": settings,
    }

    # --- Hessian first (does not modify parameters) ---
    if not args.skip_hessian:
        print("  [hessian] trace + top eigenvalue (PyHessian) ...")
        h = hessian_metrics(
            model, geom_loader, device, seed=args.seed,
            trace_max_iter=args.hessian_trace_iter,
            eig_max_iter=args.hessian_eig_iter,
        )
        result.update(h)
        print(f"    trace={h['hessian_trace']:.4f} (+/-{h['hessian_trace_std']:.4f}, "
              f"{h['hessian_trace_n_iter']} it)  lambda_max={h['hessian_top_eigenvalue']:.4f}")

    # --- Sharpness (perturbs then restores parameters) ---
    if not args.skip_sharpness:
        print(f"  [sharpness] {'adaptive ' if args.adaptive else ''}m-sharpness "
              f"rho={args.rho} m={args.m} ...")
        s = adaptive_sharpness(
            model, geom_loader, device, rho=args.rho, eta=DEFAULT_ETA,
            adaptive=args.adaptive, n_batches=args.sharpness_batches,
        )
        result.update(s)
        print(f"    adaptive_sharpness={s['adaptive_sharpness']:.5f} "
              f"(+/-{s['sharpness_std']:.5f}, {s['n_micro_batches']} micro-batches)")

    _write_outputs(run_dir, result)
    return result


def _write_outputs(run_dir: Path, result: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "geometry.json").write_text(json.dumps(result, indent=2, default=str))

    geom_summary = {k: result[k] for k in (
        "adaptive_sharpness", "sharpness_std", "rho", "eta", "adaptive",
        "hessian_trace", "hessian_trace_std", "hessian_top_eigenvalue",
    ) if k in result}

    logger = RunLogger(run_dir, append=True)
    row = {k: result[k] for k in (
        "adaptive_sharpness", "hessian_trace", "hessian_top_eigenvalue",
    ) if k in result}
    logger.log_metrics(0, "geometry", **row)
    logger.update_summary(geometry=geom_summary)
    logger.finish()
    print(f"  wrote {run_dir/'geometry.json'}; updated summary.json + metrics.csv")


def _discover_checkpoints(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("*/checkpoints/best.pt"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checkpoint", type=str)
    mode.add_argument("--all", action="store_true")

    p.add_argument("--results-dir", default="results", type=str)
    p.add_argument("--data-dir", default="data", type=str)
    p.add_argument("--device", default=None, type=str, help="cuda | cpu (auto)")

    # geometry subset
    p.add_argument("--n-geom", default=2000, type=int,
                   help="size of the fixed CIFAR-10-train geometry subset")
    p.add_argument("--subset-seed", default=1234, type=int)
    p.add_argument("--seed", default=0, type=int,
                   help="seed for Hutchinson probes / power iteration")

    # sharpness
    p.add_argument("--rho", default=DEFAULT_RHO, type=float)
    p.add_argument("--m", default=128, type=int, help="sharpness micro-batch size")
    p.add_argument("--adaptive", dest="adaptive", action="store_true", default=True)
    p.add_argument("--no-adaptive", dest="adaptive", action="store_false",
                   help="use plain (non-adaptive) m-sharpness instead")
    p.add_argument("--sharpness-batches", default=None, type=int,
                   help="cap on the number of micro-batches (default: all)")

    # hessian
    p.add_argument("--hessian-trace-iter", default=100, type=int)
    p.add_argument("--hessian-eig-iter", default=100, type=int)
    p.add_argument("--skip-hessian", action="store_true")
    p.add_argument("--skip-sharpness", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.all:
        ckpts = _discover_checkpoints(Path(args.results_dir))
        if not ckpts:
            raise SystemExit(f"no */checkpoints/best.pt under {args.results_dir}")
        print(f"[all] measuring geometry for {len(ckpts)} checkpoint(s)")
        for c in ckpts:
            measure_one(c, args)
    else:
        measure_one(Path(args.checkpoint), args)


if __name__ == "__main__":
    main()
