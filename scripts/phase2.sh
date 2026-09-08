#!/usr/bin/env bash
# Phase 2: the powered fp32 run. 3 widths (teacher, w0.5, w0.25) x 5 seeds = 15
# training runs, then evaluate + geometry + aggregate. Meant for one Kaggle
# session (~6 h on a T4). The AMP arm is deliberately left for later.
#
# Guards against the two silent failures:
#   - aborts up front if any fp32 config is not at 200 epochs
#   - skips a training run only when its checkpoints/best.pt exists (not just the
#     directory), so a crash-restart resumes instead of skipping unfinished work
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
  if [ -f "$RD/$3/checkpoints/best.pt" ]; then
    echo "[skip] $3" | tee -a "$LOG"
  else
    echo "[train] $3 @ $(date)" | tee -a "$LOG"
    python -u -m src.train --config "$1" --seed "$2" 2>&1 | tee -a "$LOG"
  fi
}

for s in $SEEDS; do run_train configs/teacher_fp32.yaml      "$s" "teacher_fp32_s$s"; done
for s in $SEEDS; do run_train configs/student_w0.5_fp32.yaml  "$s" "student_w0.5_fp32_s$s"; done
for s in $SEEDS; do run_train configs/student_w0.25_fp32.yaml "$s" "student_w0.25_fp32_s$s"; done

echo "[evaluate] @ $(date)" | tee -a "$LOG"
python -u -m src.evaluate --all --results-dir "$RD" 2>&1 | tee -a "$LOG"

echo "[geometry] @ $(date)" | tee -a "$LOG"
python -u -m src.measure_geometry --all --results-dir "$RD" 2>&1 | tee -a "$LOG"

echo "[aggregate] @ $(date)" | tee -a "$LOG"
python -u -m src.evaluate --aggregate --results-dir "$RD" --out "$RD/pilot_summary.csv" 2>&1 | tee -a "$LOG"

echo "=== phase2 done $(date) ===" | tee -a "$LOG"
echo "Summary -> $RD/pilot_summary.csv"
