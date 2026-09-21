"""Find runs whose logged LR schedule is broken, and optionally set them aside.

    python scripts/check_lr_schedules.py results "teacher_fp32_s*"
    python scripts/check_lr_schedules.py results "*" --quarantine

A cosine run's LR must fall monotonically. If it ever rises, a resume reset the
schedule, which once produced a teacher that never annealed (91.6% instead of
95.5%). A run that has logged all its epochs but still ends above 1e-3 is flagged
too. A run that is only part-way through is fine as long as its LR keeps falling.

Without --quarantine it prints the bad runs and exits 1. With --quarantine it
moves each one to <results>/_quarantine/<name>-<timestamp> (nothing is deleted)
and exits 0, so the next training pass retrains it from scratch. The quarantine
folder is two levels deep, so evaluate.py and measure_geometry.py, which glob
<results>/*/checkpoints/best.pt, never see it.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

RISE_TOL = 1e-9       # an LR that goes up by more than this counts as a reset
FINAL_LR_MAX = 1e-3   # a finished cosine run ends near zero


def diagnose(run_dir: Path) -> str | None:
    """Return a description of what is wrong with this run, or None if it is fine."""
    metrics = run_dir / "metrics.csv"
    if not metrics.exists():
        return None
    with open(metrics, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] == "train" and r.get("lr")]
    lrs = [float(r["lr"]) for r in rows]
    if not lrs:
        return None

    for i in range(1, len(lrs)):
        if lrs[i] > lrs[i - 1] + RISE_TOL:
            return (f"LR rises at logged epoch {i} ({lrs[i - 1]:.5f} -> {lrs[i]:.5f}); "
                    f"the schedule was reset")

    target = None
    summary = run_dir / "summary.json"
    if summary.exists():
        try:
            target = json.loads(summary.read_text())["config"]["schedule"]["epochs"]
        except (KeyError, ValueError):
            pass
    finished = target is not None and int(float(rows[-1]["step"])) >= target - 1
    if finished and lrs[-1] > FINAL_LR_MAX:
        return f"finished {target} epochs but ended at lr={lrs[-1]:.5f}"
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir")
    p.add_argument("pattern", help='run-directory glob under results_dir, e.g. "*"')
    p.add_argument("--quarantine", action="store_true",
                   help="move bad runs to <results_dir>/_quarantine/ instead of failing")
    args = p.parse_args()

    root = Path(args.results_dir)
    bad = []
    for d in sorted(glob.glob(str(root / args.pattern))):
        run = Path(d)
        if run.is_dir() and run.name != "_quarantine":
            problem = diagnose(run)
            if problem:
                bad.append((run, problem))

    if not bad:
        print("[check] all LR schedules are consistent")
        return 0

    if args.quarantine:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        (root / "_quarantine").mkdir(exist_ok=True)
        for run, problem in bad:
            dest = root / "_quarantine" / f"{run.name}-{stamp}"
            shutil.move(str(run), str(dest))
            print(f"[quarantine] {run.name}: {problem}\n             moved to {dest}")
        return 0

    print("BAD RUNS (move them aside or delete them, then re-run to retrain):")
    for run, problem in bad:
        print(f"  {run}: {problem}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
