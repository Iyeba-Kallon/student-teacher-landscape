#!/usr/bin/env bash
# Full pilot: train -> evaluate (ID + CIFAR-10-C) -> geometry -> aggregate.
#
# 18 training runs: {teacher, student w0.5, student w0.25} x {fp32, amp} x seeds.
# The AMP configs need CUDA; set SKIP_AMP=1 on a CPU-only machine.
#
#   bash scripts/run_pilot.sh
#   SEEDS="0 1 2" SKIP_AMP=1 bash scripts/run_pilot.sh
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=.

SEEDS="${SEEDS:-0 1 2}"
RESULTS_DIR="${RESULTS_DIR:-results}"
SKIP_DATA="${SKIP_DATA:-0}"
SKIP_AMP="${SKIP_AMP:-0}"

if [ "$SKIP_AMP" = "1" ]; then PRECISIONS="fp32"; else PRECISIONS="fp32 amp"; fi
STUDENTS="w0.5 w0.25"

# Skip a run only when its checkpoints/best.pt exists, so a crash-restart resumes
# rather than skipping unfinished work (an empty results/<run>/ does not count).
run_train () {  # $1 config, $2 seed, $3 run_name
  if [ -f "$RESULTS_DIR/$3/checkpoints/best.pt" ]; then
    echo "-- skip $3 (done) --"
  else
    echo "-- $3 --"
    python -m src.train --config "$1" --seed "$2"
  fi
}

echo "=== STEP 0: data ==="
if [ "$SKIP_DATA" != "1" ]; then
  python data/download_cifar10c.py --dest data/
fi

echo; echo "=== STEP 1: teachers ==="
for p in $PRECISIONS; do
  for s in $SEEDS; do
    run_train "configs/teacher_${p}.yaml" "$s" "teacher_${p}_s${s}"
  done
done

echo; echo "=== STEP 2: students (distillation) ==="
for w in $STUDENTS; do
  for p in $PRECISIONS; do
    for s in $SEEDS; do
      run_train "configs/student_${w}_${p}.yaml" "$s" "student_${w}_${p}_s${s}"
    done
  done
done

echo; echo "=== STEP 3: evaluate (ID + CIFAR-10-C) ==="
python -m src.evaluate --all --results-dir "$RESULTS_DIR"

echo; echo "=== STEP 4: loss-landscape geometry ==="
python -m src.measure_geometry --all --results-dir "$RESULTS_DIR"

echo; echo "=== STEP 5: aggregate ==="
python -m src.evaluate --aggregate --results-dir "$RESULTS_DIR" \
  --out "$RESULTS_DIR/pilot_summary.csv"

echo; echo "Done. Summary -> $RESULTS_DIR/pilot_summary.csv"
echo "Analyze:  jupyter notebook notebooks/analysis.ipynb"
