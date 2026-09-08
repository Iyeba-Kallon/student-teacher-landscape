#!/usr/bin/env bash
# Phase-1 signal check (CPU): teacher + w0.5 student, fp32, 30 epochs, seeds {0,1}.
# Reduced geometry settings so the Hessian step is hours, not half a day.
#
# Resumable: a training run whose checkpoints/best.pt already exists is skipped,
# so re-running after a crash picks up where it left off. Delete a run's folder
# to force a redo.
#
# Usage:  bash scripts/phase1.sh   (assumes CIFAR-10-C already downloaded)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

SEEDS="${SEEDS:-0 1}"
RD=results
LOG="$RD/phase1.log"
mkdir -p "$RD"
echo "=== phase1 started $(date) ===" | tee -a "$LOG"

run_train () {  # $1 = config, $2 = seed, $3 = run_name
  if [ -f "$RD/$3/checkpoints/best.pt" ]; then
    echo "[skip] $3 (best.pt exists)" | tee -a "$LOG"
  else
    echo "[train] $3 @ $(date)" | tee -a "$LOG"
    python -m src.train --config "$1" --seed "$2" 2>&1 | tee -a "$LOG"
  fi
}

for s in $SEEDS; do run_train configs/teacher_fp32.yaml     "$s" "teacher_fp32_s$s"; done
for s in $SEEDS; do run_train configs/student_w0.5_fp32.yaml "$s" "student_w0.5_fp32_s$s"; done

echo "[evaluate] @ $(date)" | tee -a "$LOG"
python -m src.evaluate --all --results-dir "$RD" 2>&1 | tee -a "$LOG"

echo "[geometry] @ $(date)" | tee -a "$LOG"
python -m src.measure_geometry --all --results-dir "$RD" \
  --n-geom 1000 --hessian-trace-iter 30 --hessian-eig-iter 30 2>&1 | tee -a "$LOG"

echo "[aggregate] @ $(date)" | tee -a "$LOG"
python -m src.evaluate --aggregate --results-dir "$RD" \
  --out "$RD/pilot_summary.csv" 2>&1 | tee -a "$LOG"

echo "=== phase1 done $(date) ===" | tee -a "$LOG"
echo "Summary -> $RD/pilot_summary.csv"
