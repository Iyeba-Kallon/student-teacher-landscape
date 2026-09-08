"""Measure loss-landscape geometry for a trained checkpoint.

Everything runs in fp32 with BatchNorm frozen and no autocast, whether the
checkpoint was trained in fp32 or with AMP.

    python -m src.measure_geometry --checkpoint results/teacher_fp32_s0/checkpoints/best.pt
    python -m src.measure_geometry --all --results-dir results/

Per run it writes geometry.json (results plus the exact settings used), adds a
"geometry" block to summary.json, and appends a geometry row to metrics.csv.
The metrics are adaptive_sharpness (headline), hessian_trace, and
hessian_top_eigenvalue, all on the same fixed 2,000-example training subset.
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

    prepare_model_for_geometry(model)
    assert_fp32_no_autocast(model)

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
        "sharpness": {"kind": "adaptive_m_sharpness" if args.adaptive else "m_sharpness",
                      "rho": args.rho, "eta": DEFAULT_ETA, "m": args.m,
                      "ascent_steps": 1, "n_batches": args.sharpness_batches},
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

    # Hessian first: it does not touch the parameters. Sharpness perturbs and
    # restores them, so any tiny residual drift lands after the Hessian is done.
    if not args.skip_hessian:
        print("  [hessian] trace + top eigenvalue (PyHessian) ...")
        h = hessian_metrics(model, geom_loader, device, seed=args.seed,
                            trace_max_iter=args.hessian_trace_iter,
                            eig_max_iter=args.hessian_eig_iter)
        result.update(h)
        print(f"    trace={h['hessian_trace']:.4f} (+/-{h['hessian_trace_std']:.4f}, "
              f"{h['hessian_trace_n_iter']} it)  lambda_max={h['hessian_top_eigenvalue']:.4f}")

    if not args.skip_sharpness:
        print(f"  [sharpness] {'adaptive ' if args.adaptive else ''}m-sharpness "
              f"rho={args.rho} m={args.m} ...")
        s = adaptive_sharpness(model, geom_loader, device, rho=args.rho,
                               eta=DEFAULT_ETA, adaptive=args.adaptive,
                               n_batches=args.sharpness_batches)
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
    print(f"  wrote {run_dir / 'geometry.json'}; updated summary.json + metrics.csv")


def _discover_checkpoints(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("*/checkpoints/best.pt"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--checkpoint")
    target.add_argument("--all", action="store_true")

    p.add_argument("--results-dir", default="results")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--device", default=None, help="cuda or cpu (auto if unset)")

    p.add_argument("--n-geom", type=int, default=2000, help="geometry subset size")
    p.add_argument("--subset-seed", type=int, default=1234)
    p.add_argument("--seed", type=int, default=0,
                   help="seed for the Hessian estimators")

    p.add_argument("--rho", type=float, default=DEFAULT_RHO)
    p.add_argument("--m", type=int, default=128, help="sharpness micro-batch size")
    p.add_argument("--adaptive", dest="adaptive", action="store_true", default=True)
    p.add_argument("--no-adaptive", dest="adaptive", action="store_false",
                   help="use plain m-sharpness instead")
    p.add_argument("--sharpness-batches", type=int, default=None,
                   help="cap the number of micro-batches (default: all)")

    p.add_argument("--hessian-trace-iter", type=int, default=100)
    p.add_argument("--hessian-eig-iter", type=int, default=100)
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
