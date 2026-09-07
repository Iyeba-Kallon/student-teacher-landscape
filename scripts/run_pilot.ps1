<#
.SYNOPSIS
  Run the full pilot end-to-end (Sections 5.1 + 5.2): train -> evaluate -> geometry -> aggregate.

.DESCRIPTION
  18 training runs = {teacher, student w0.5, student w0.25} x {fp32, amp} x {seeds}.
  AMP configs require a CUDA device; pass -SkipAmp on a CPU-only machine.

.EXAMPLE
  ./scripts/run_pilot.ps1
  ./scripts/run_pilot.ps1 -Seeds 0,1,2 -SkipAmp
#>
param(
  [int[]] $Seeds = @(0, 1, 2),
  [switch] $SkipAmp,
  [switch] $SkipData,
  [string] $ResultsDir = "results"
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$env:PYTHONPATH = "."

$precisions = if ($SkipAmp) { @("fp32") } else { @("fp32", "amp") }
$students   = @("w0.5", "w0.25")

Write-Host "=== STEP 0: data ===" -ForegroundColor Cyan
if (-not $SkipData) {
  python data/download_cifar10c.py --dest data/
}

Write-Host "`n=== STEP 1: teachers ===" -ForegroundColor Cyan
foreach ($p in $precisions) {
  foreach ($s in $Seeds) {
    Write-Host "-- teacher $p seed $s --" -ForegroundColor Yellow
    python -m src.train --config "configs/teacher_$p.yaml" --seed $s
  }
}

Write-Host "`n=== STEP 2: students (distillation) ===" -ForegroundColor Cyan
foreach ($w in $students) {
  foreach ($p in $precisions) {
    foreach ($s in $Seeds) {
      Write-Host "-- student $w $p seed $s --" -ForegroundColor Yellow
      python -m src.train --config "configs/student_${w}_${p}.yaml" --seed $s
    }
  }
}

Write-Host "`n=== STEP 3: evaluate (ID + CIFAR-10-C) ===" -ForegroundColor Cyan
python -m src.evaluate --all --results-dir $ResultsDir

Write-Host "`n=== STEP 4: loss-landscape geometry ===" -ForegroundColor Cyan
python -m src.measure_geometry --all --results-dir $ResultsDir

Write-Host "`n=== STEP 5: aggregate ===" -ForegroundColor Cyan
python -m src.evaluate --aggregate --results-dir $ResultsDir --out "$ResultsDir/pilot_summary.csv"

Write-Host "`nDone. Summary -> $ResultsDir/pilot_summary.csv" -ForegroundColor Green
Write-Host "Analyze:  jupyter notebook notebooks/analysis.ipynb"
