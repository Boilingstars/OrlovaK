# Run from project root: powershell -NoProfile -ExecutionPolicy Bypass -File .\runs\full_100_driver.ps1
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$py = if (Get-Command python -ErrorAction SilentlyContinue) { 'python' } else { 'py' }
& $py scripts/k_driver.py @args
exit $LASTEXITCODE
