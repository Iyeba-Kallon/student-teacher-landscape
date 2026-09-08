"""Evaluate a checkpoint on CIFAR-10 (ID) and CIFAR-10-C (OOD).

Always fp32, model.eval(), no autocast.

    python -m src.evaluate --checkpoint results/teacher_fp32_s0/checkpoints/best.pt
    python -m src.evaluate --all --results-dir results/
    python -m src.evaluate --checkpoint <student ckpt> --baseline results/teacher_fp32_s0
    python -m src.evaluate --aggregate --results-dir results/ --out results/pilot_summary.csv

Per run it writes eval.json (ID result plus the 15x5 OOD grid) and
eval_cifar10c.csv (one row per corruption/severity), adds an "evaluation" block
to summary.json, and appends eval_* rows to metrics.csv.

The headline OOD number is the unweighted mean top-1 accuracy over the 15 test
corruptions x 5 severities. With --baseline it also reports mCE relative to that
model: mean over corruptions of sum_s err[c,s] / sum_s err_baseline[c,s]. That is
the Hendrycks-Dietterich mCE with the reference swapped from AlexNet to the given
baseline, so it only compares within this project; below 1 means more robust.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from .checkpoint import build_model_from_checkpoint
from .data import (CIFAR10C_CORRUPTIONS, CIFAR10C_SEVERITIES,
                   build_cifar10_loaders, iter_cifar10c_loaders)
from .engine import evaluate_classifier
from .utils import RunLogger


def _device(requested: str | None) -> torch.device:
    return torch.device(requested) if requested else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")


def evaluate_id(model, data_dir, batch_size, num_workers, device) -> dict[str, float]:
    _, test_loader = build_cifar10_loaders(data_dir, batch_size, num_workers,
                                           download=True)
    return evaluate_classifier(model, test_loader, device)


def evaluate_ood(
    model, data_dir, batch_size, num_workers, device,
    corruptions: tuple[str, ...], severities: tuple[int, ...],
) -> dict[str, Any]:
    """Full accuracy grid plus per-corruption and per-severity means."""
    grid: dict[str, dict[int, dict[str, float]]] = {}
    for corruption, severity, loader in iter_cifar10c_loaders(
        data_dir, batch_size, num_workers, corruptions, severities
    ):
        stats = evaluate_classifier(model, loader, device)
        grid.setdefault(corruption, {})[severity] = {
            "acc": stats["acc"], "error": stats["error"],
            "loss": stats["loss"], "n": stats["n"],
        }
        print(f"  {corruption:<18} s{severity}  acc={stats['acc']:.4f}")

    cells = [grid[c][s] for c in corruptions for s in severities]
    by_corruption = {
        c: {"acc": sum(grid[c][s]["acc"] for s in severities) / len(severities),
            "error": sum(grid[c][s]["error"] for s in severities) / len(severities)}
        for c in corruptions
    }
    by_severity = {
        s: {"acc": sum(grid[c][s]["acc"] for c in corruptions) / len(corruptions),
            "error": sum(grid[c][s]["error"] for c in corruptions) / len(corruptions)}
        for s in severities
    }
    return {
        "grid": grid,
        "acc_mean": sum(x["acc"] for x in cells) / len(cells),
        "error_mean": sum(x["error"] for x in cells) / len(cells),
        "by_corruption": by_corruption,
        "by_severity": by_severity,
        "corruptions": list(corruptions),
        "severities": list(severities),
    }


def compute_mce(grid, baseline_grid, corruptions, severities) -> dict[str, Any]:
    """mCE relative to a baseline model (see the module docstring)."""
    def base_err(c, s):
        # baseline_grid came from JSON, so severity keys may be strings.
        cell = baseline_grid[c].get(str(s)) or baseline_grid[c][s]
        return cell["error"]

    per_c: dict[str, float] = {}
    for c in corruptions:
        num = sum(grid[c][s]["error"] for s in severities)
        den = sum(base_err(c, s) for s in severities)
        per_c[c] = num / den if den > 0 else float("nan")
    valid = [v for v in per_c.values() if v == v]
    return {"mce": sum(valid) / len(valid) if valid else float("nan"),
            "by_corruption": per_c}


def _load_baseline_grid(baseline: str | None) -> dict | None:
    if not baseline:
        return None
    path = Path(baseline)
    eval_json = path if path.suffix == ".json" else path / "eval.json"
    if not eval_json.exists():
        print(f"[warn] {eval_json} not found; skipping mCE (evaluate the baseline first)")
        return None
    return json.loads(eval_json.read_text())["ood"]["grid"]


def evaluate_one(ckpt_path: Path, args) -> dict[str, Any]:
    device = _device(args.device)
    model, ckpt = build_model_from_checkpoint(ckpt_path, device=device)
    run_dir = ckpt_path.parent.parent
    run_name = ckpt.get("config", {}).get("run_name") or run_dir.name
    print(f"\n=== evaluate {run_name}  ({ckpt_path}) ===")
    print(f"  arch={ckpt['arch']} width_mult={ckpt['width_mult']} "
          f"mode={ckpt['mode']} precision={ckpt['precision']} seed={ckpt['seed']}")

    result: dict[str, Any] = {
        "run_name": run_name, "checkpoint": str(ckpt_path),
        "arch": ckpt["arch"], "width_mult": ckpt["width_mult"],
        "mode": ckpt["mode"], "precision": ckpt["precision"], "seed": ckpt["seed"],
        "epoch": ckpt["epoch"],
    }

    id_stats = evaluate_id(model, args.data_dir, args.batch_size,
                           args.num_workers, device)
    result["id"] = {k: id_stats[k] for k in ("acc", "error", "loss", "n")}
    print(f"  ID  acc={id_stats['acc']:.4f}  error={id_stats['error']:.4f}")

    if not args.id_only:
        ood = evaluate_ood(model, args.data_dir, args.batch_size, args.num_workers,
                           device, tuple(args.corruptions), tuple(args.severities))
        result["ood"] = ood
        print(f"  OOD acc_mean={ood['acc_mean']:.4f}  error_mean={ood['error_mean']:.4f}")

        baseline_grid = _load_baseline_grid(args.baseline)
        if baseline_grid is not None:
            mce = compute_mce(ood["grid"], baseline_grid,
                              tuple(args.corruptions), tuple(args.severities))
            result["mce_vs_baseline"] = mce
            result["baseline"] = args.baseline
            print(f"  mCE_vs_baseline={mce['mce']:.4f}  (baseline={args.baseline})")

    _write_run_outputs(run_dir, result)
    return result


def _write_run_outputs(run_dir: Path, result: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "eval.json").write_text(json.dumps(result, indent=2, default=str))

    if "ood" in result:
        with open(run_dir / "eval_cifar10c.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["corruption", "severity", "acc", "error", "loss", "n"])
            for c, per_s in result["ood"]["grid"].items():
                for s, cell in per_s.items():
                    w.writerow([c, s, cell["acc"], cell["error"], cell["loss"], cell["n"]])

    logger = RunLogger(run_dir, append=True)
    step = int(result.get("epoch", 0) or 0)
    logger.log_metrics(step, "eval_id", acc=result["id"]["acc"],
                       error=result["id"]["error"], loss=result["id"]["loss"])

    summary_eval: dict[str, Any] = {"id_acc": result["id"]["acc"],
                                    "id_error": result["id"]["error"]}
    if "ood" in result:
        ood = result["ood"]
        sev_acc = {f"sev{s}": ood["by_severity"][s]["acc"] for s in ood["severities"]}
        logger.log_metrics(step, "eval_ood_mean", acc=ood["acc_mean"],
                           error=ood["error_mean"], **sev_acc)
        summary_eval.update(
            ood_acc_mean=ood["acc_mean"],
            ood_error_mean=ood["error_mean"],
            ood_acc_by_severity={s: ood["by_severity"][s]["acc"] for s in ood["severities"]},
            ood_acc_by_corruption={c: v["acc"] for c, v in ood["by_corruption"].items()},
        )
    if "mce_vs_baseline" in result:
        summary_eval["mce_vs_baseline"] = result["mce_vs_baseline"]["mce"]
    logger.update_summary(evaluation=summary_eval)
    logger.finish()
    print(f"  wrote eval.json / eval_cifar10c.csv, updated summary.json + metrics.csv")


def _discover_checkpoints(results_dir: Path) -> list[Path]:
    ckpts = sorted(results_dir.glob("*/checkpoints/best.pt"))
    # Teachers first, so a student can use its teacher as an mCE baseline.
    return sorted(ckpts, key=lambda p: (0 if "teacher" in p.parts[-3] else 1, str(p)))


def run_all(args) -> None:
    results_dir = Path(args.results_dir)
    ckpts = _discover_checkpoints(results_dir)
    if not ckpts:
        raise SystemExit(f"no */checkpoints/best.pt under {results_dir}")
    print(f"[all] evaluating {len(ckpts)} checkpoint(s)")
    for ckpt in ckpts:
        local = argparse.Namespace(**vars(args))
        if args.baseline is None and "student" in ckpt.parts[-3]:
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            cand = results_dir / f"teacher_{ck['precision']}_s{ck['seed']}"
            if (cand / "eval.json").exists():
                local.baseline = str(cand)
        evaluate_one(ckpt, local)


def run_aggregate(args) -> None:
    results_dir = Path(args.results_dir)
    rows: list[dict[str, Any]] = []
    for summ_path in sorted(results_dir.glob("*/summary.json")):
        s = json.loads(summ_path.read_text())
        cfg = s.get("config", {})
        ev = s.get("evaluation", {})
        geom = s.get("geometry") or {}
        rows.append({
            "run_name": s.get("run_name", summ_path.parent.name),
            "mode": cfg.get("mode"),
            "precision": cfg.get("precision"),
            "seed": cfg.get("seed"),
            "width_mult": cfg.get("model", {}).get("width_mult"),
            "model_params": s.get("model_params"),
            "best_id_acc_train": s.get("best_id_acc"),
            "id_acc": ev.get("id_acc"),
            "id_error": ev.get("id_error"),
            "ood_acc_mean": ev.get("ood_acc_mean"),
            "ood_error_mean": ev.get("ood_error_mean"),
            "mce_vs_baseline": ev.get("mce_vs_baseline"),
            **{f"ood_acc_sev{k}": v
               for k, v in (ev.get("ood_acc_by_severity") or {}).items()},
            "adaptive_sharpness": geom.get("adaptive_sharpness"),
            "hessian_trace": geom.get("hessian_trace"),
            "hessian_top_eigenvalue": geom.get("hessian_top_eigenvalue"),
        })
    if not rows:
        raise SystemExit(f"no summary.json under {results_dir}")

    fieldnames = sorted({k for r in rows for k in r})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[aggregate] wrote {len(rows)} rows -> {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--checkpoint", help="one .pt checkpoint")
    target.add_argument("--all", action="store_true",
                        help="every <results-dir>/*/checkpoints/best.pt")
    target.add_argument("--aggregate", action="store_true",
                        help="collect every summary.json into one CSV")

    p.add_argument("--results-dir", default="results")
    p.add_argument("--out", default="results/pilot_summary.csv",
                   help="output CSV for --aggregate")
    p.add_argument("--baseline", default=None,
                   help="run dir or eval.json of a reference model for mCE")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default=None, help="cuda or cpu (auto if unset)")
    p.add_argument("--id-only", action="store_true",
                   help="skip CIFAR-10-C (e.g. before it is downloaded)")
    p.add_argument("--corruptions", nargs="*", default=list(CIFAR10C_CORRUPTIONS))
    p.add_argument("--severities", nargs="*", type=int, default=list(CIFAR10C_SEVERITIES))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.aggregate:
        run_aggregate(args)
    elif args.all:
        run_all(args)
    else:
        evaluate_one(Path(args.checkpoint), args)


if __name__ == "__main__":
    main()
