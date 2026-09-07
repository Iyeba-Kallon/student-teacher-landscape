<#
.SYNOPSIS
  Fast end-to-end sanity check (~2 min on CPU): 1 epoch, a few batches, tiny
  geometry settings. Writes to results_smoke/ so it never touches real runs.

.EXAMPLE
  ./scripts/smoke.ps1
#>
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$env:PYTHONPATH = "."

$RD = "results_smoke"
$Fast = @("--set", "schedule.epochs=1", "debug.limit_train_batches=8",
          "debug.limit_val_batches=4", "data.num_workers=0", "results_dir=$RD")

if (Test-Path $RD) { Remove-Item -Recurse -Force $RD }

Write-Host "== train teacher ==" -ForegroundColor Cyan
python -m src.train --config configs/teacher_fp32.yaml --seed 0 @Fast

Write-Host "== distill student (w0.5) ==" -ForegroundColor Cyan
python -m src.train --config configs/student_w0.5_fp32.yaml --seed 0 @Fast

Write-Host "== evaluate (ID only) ==" -ForegroundColor Cyan
python -m src.evaluate --all --results-dir $RD --id-only --num-workers 0

Write-Host "== geometry (tiny) ==" -ForegroundColor Cyan
python -m src.measure_geometry --all --results-dir $RD `
  --n-geom 128 --m 64 --hessian-trace-iter 4 --hessian-eig-iter 6

Write-Host "== aggregate ==" -ForegroundColor Cyan
python -m src.evaluate --aggregate --results-dir $RD --out "$RD/pilot_summary.csv"

Write-Host "`nsmoke OK -> $RD/pilot_summary.csv" -ForegroundColor Green
