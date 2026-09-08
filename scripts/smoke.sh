#!/usr/bin/env bash
# End-to-end sanity check: 1 epoch, a few batches, tiny geometry settings.
# A few minutes on CPU. Writes to results_smoke/ so it never touches real runs.
#
# Usage: bash scripts/smoke.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

RD=results_smoke
FAST=(--set schedule.epochs=1 debug.limit_train_batches=8 debug.limit_val_batches=4
      data.num_workers=0 "results_dir=$RD")

rm -rf "$RD"

echo "== train teacher =="
python -m src.train --config configs/teacher_fp32.yaml --seed 0 "${FAST[@]}"

echo "== distill student (w0.5) =="
python -m src.train --config configs/student_w0.5_fp32.yaml --seed 0 "${FAST[@]}"

echo "== evaluate (ID only; skip CIFAR-10-C) =="
python -m src.evaluate --all --results-dir "$RD" --id-only --num-workers 0

echo "== geometry (tiny) =="
python -m src.measure_geometry --all --results-dir "$RD" \
  --n-geom 128 --m 64 --hessian-trace-iter 4 --hessian-eig-iter 6

echo "== aggregate =="
python -m src.evaluate --aggregate --results-dir "$RD" --out "$RD/pilot_summary.csv"

echo; echo "smoke OK -> $RD/pilot_summary.csv"
