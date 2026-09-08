#!/usr/bin/env bash
# Phase-1 signal check: teacher + w0.5 student, fp32, seeds {0,1}.
# (fp32 configs are currently at 30 epochs — see the note in configs/teacher_fp32.yaml.)
#
# Designed for a GPU (Colab/Kaggle T4): the whole thing is ~15-25 min there.
# On CPU it is not practical for width-1.0 ResNet-18 — use a GPU.
#
# Resumable: a training run whose checkpoints/best.pt already exists is skipped,
# so re-running after an interrupted session picks up where it left off.
#
# Env overrides:
#   SEEDS="0 1"           seeds to run
#   GEOM_ARGS="..."       extra flags for measure_geometry
#                         (e.g. "--n-geom 1000 --hessian-trace-iter 30" to speed up)
#
# Usage:  bash scripts/phase1.sh      (assumes CIFAR-10-C already downloaded)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

SEEDS="${SEEDS:-0 1}"
GEOM_ARGS="${GEOM_ARGS:-}"
RD=results
LOG="$RD/phase1.log"
mkdir -p "$RD"
echo "=== phase1 started $(date) ===" | tee -a "$LOG"
python -c "import torch;print('[env] torch',torch.__version__,'cuda',torch.cuda.is_available())" | tee -a "$LOG"

run_train () {  # $1 = config, $2 = seed, $3 = run_name
  if [ -f "$RD/$3/checkpoints/best.pt" ]; then
    echo "[skip] $3 (best.pt exists)" | tee -a "$LOG"
  else
    echo "[train] $3 @ $(date)" | tee -a "$LOG"
    python -u -m src.train --config "$1" --seed "$2" 2>&1 | tee -a "$LOG"
  fi
}

for s in $SEEDS; do run_train configs/teacher_fp32.yaml     "$s" "teacher_fp32_s$s"; done
for s in $SEEDS; do run_train configs/student_w0.5_fp32.yaml "$s" "student_w0.5_fp32_s$s"; done

echo "[evaluate] @ $(date)" | tee -a "$LOG"
python -u -m src.evaluate --all --results-dir "$RD" 2>&1 | tee -a "$LOG"

echo "[geometry] @ $(date)  (args: ${GEOM_ARGS:-<locked defaults>})" | tee -a "$LOG"
python -u -m src.measure_geometry --all --results-dir "$RD" $GEOM_ARGS 2>&1 | tee -a "$LOG"

echo "[aggregate] @ $(date)" | tee -a "$LOG"
python -u -m src.evaluate --aggregate --results-dir "$RD" \
  --out "$RD/pilot_summary.csv" 2>&1 | tee -a "$LOG"

echo "=== phase1 done $(date) ===" | tee -a "$LOG"
echo "Summary -> $RD/pilot_summary.csv"
