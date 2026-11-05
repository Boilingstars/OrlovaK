#!/usr/bin/env bash
# Linux launch of k_driver.py: one lock, smoke_3 then n=4–14, resumable.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="/usr/local/bin:/opt/xtb-6.7.1/xtb-dist/bin:${PATH}"
export XTBHOME="/opt/xtb-6.7.1/xtb-dist"
export XTBPATH="${XTBHOME}/share/xtb"
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
PY="${ROOT}/.venv/bin/python"
if [ ! -x "$PY" ]; then PY="$(command -v python3)"; fi
cd "$ROOT"
exec "$PY" "${ROOT}/scripts/k_driver.py" --critfreq 125.0 --workers 5 --relax-workers 4 --threads 1 --scale 1.0 --cn-tol 0.15 "$@"
