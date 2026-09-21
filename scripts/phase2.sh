#!/usr/bin/env bash
# Phase 2: the powered fp32 run. 3 widths (teacher, w0.5, w0.25) x 5 seeds = 15
# training runs, then evaluate + geometry + aggregate. Meant for one Kaggle
# session (~6 h on a T4). The AMP arm is deliberately left for later.
#
# Guards against the two silent failures:
#   - aborts up front if any fp32 config is not at 200 epochs
#   - always invokes src.train for every (config, seed); src.train itself checks
#     checkpoints/last.pt and either trains fresh, resumes from the saved epoch
#     (optimizer/scheduler/scaler state included), or exits as a fast no-op if
#     that run already reached the target epoch. A run that was interrupted
#     mid-training is never mistaken for a finished one.
#
# Env overrides:
#   SEEDS="0 1 2 3 4"
#   RESULTS_DIR=results
#
# Usage: bash scripts/phase2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

SEEDS="${SEEDS:-0 1 2 3 4}"
RD="${RESULTS_DIR:-results}"
CONFIGS=(configs/teacher_fp32.yaml configs/student_w0.5_fp32.yaml configs/student_w0.25_fp32.yaml)
LOG="$RD/phase2.log"
mkdir -p "$RD"

# --- guard: every config must be at 200 epochs ---
for c in "${CONFIGS[@]}"; do
  ep=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['schedule']['epochs'])" "$c")
  if [ "$ep" != "200" ]; then
    echo "ABORT: $c is at epochs=$ep, expected 200. Fix the config before launching." >&2
    exit 1
  fi
done
echo "=== phase2 started $(date) ===" | tee -a "$LOG"
python -c "import torch;print('[env] torch',torch.__version__,'cuda',torch.cuda.is_available())" | tee -a "$LOG"

run_train () {  # $1 config, $2 seed, $3 run_name
  echo "[run] $3 @ $(date)" | tee -a "$LOG"
  python -u -m src.train --config "$1" --seed "$2" 2>&1 | tee -a "$LOG"
}

# Cosine LR must fall monotonically to ~0. A jump up means a resume reset the
# schedule (that produced a bad teacher once), so refuse to build on such a run.
check_schedules () {  # $1 = glob of run dirs under $RD
  python - "$RD" "$1" <<'PY' | tee -a "$LOG"
import csv, glob, sys
root, pattern = sys.argv[1], sys.argv[2]
bad = []
for f in sorted(glob.glob(f"{root}/{pattern}/metrics.csv")):
    with open(f, newline="") as fh:
        lrs = [float(r["lr"]) for r in csv.DictReader(fh) if r["split"] == "train" and r.get("lr")]
    if not lrs:
        continue
    jump = next((i for i in range(1, len(lrs)) if lrs[i] > lrs[i - 1] + 1e-9), None)
    if jump is not None:
        bad.append(f"{f}: LR jumps up at logged epoch {jump} ({lrs[jump-1]:.5f} -> {lrs[jump]:.5f}), schedule was reset")
    elif lrs[-1] > 1e-3:
        bad.append(f"{f}: ended at lr={lrs[-1]:.5f}, schedule did not finish")
if bad:
    print("BAD RUNS (delete the run folder and re-run to retrain):")
    for b in bad:
        print("  " + b)
    sys.exit(1)
print("[check] all LR schedules complete and monotone")
PY
}

for s in $SEEDS; do run_train configs/teacher_fp32.yaml      "$s" "teacher_fp32_s$s"; done
check_schedules "teacher_fp32_s*"   # do not distill from a defective teacher
for s in $SEEDS; do run_train configs/student_w0.5_fp32.yaml  "$s" "student_w0.5_fp32_s$s"; done
for s in $SEEDS; do run_train configs/student_w0.25_fp32.yaml "$s" "student_w0.25_fp32_s$s"; done

check_schedules "*"                 # every run, before spending time on eval/geometry

echo "[evaluate] @ $(date)" | tee -a "$LOG"
python -u -m src.evaluate --all --results-dir "$RD" 2>&1 | tee -a "$LOG"

echo "[geometry] @ $(date)" | tee -a "$LOG"
python -u -m src.measure_geometry --all --results-dir "$RD" 2>&1 | tee -a "$LOG"

echo "[aggregate] @ $(date)" | tee -a "$LOG"
python -u -m src.evaluate --aggregate --results-dir "$RD" --out "$RD/pilot_summary.csv" 2>&1 | tee -a "$LOG"

echo "=== phase2 done $(date) ===" | tee -a "$LOG"
echo "Summary -> $RD/pilot_summary.csv"
