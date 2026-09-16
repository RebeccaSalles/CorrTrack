#!/bin/bash
# Capture a reproducibility manifest for a CorrTrack run on Inria Abaca.
# Usage: abaca/capture_manifest.sh <output-dir>
# Writes <output-dir>/environment.txt. Run from the CorrTrack working tree,
# inside the OAR job, with the corrtrack conda env active.
set -uo pipefail

OUT_DIR="${1:?usage: capture_manifest.sh <output-dir>}"
mkdir -p "$OUT_DIR"
MANIFEST="$OUT_DIR/environment.txt"

{
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "oar_job_id=${OAR_JOB_ID:-none}"
    echo "hostname=$(hostname)"
    echo "user=$(whoami)"

    echo "=== OAR ENV ==="
    env | grep -E '^OAR' | sort || true

    echo "=== OS ==="
    cat /etc/os-release

    echo "=== KERNEL ==="
    uname -a

    echo "=== CPU ==="
    lscpu
    echo "nproc=$(nproc)"

    echo "=== MEMORY ==="
    free -h

    echo "=== LOCAL DISK ==="
    df -h /tmp
    lsblk 2>/dev/null || true

    echo "=== GIT ==="
    git rev-parse HEAD 2>/dev/null || echo "not a git tree (staged deploy)"
    git status --short 2>/dev/null || true

    echo "=== CONDA ENV ==="
    timeout 60 conda info --envs 2>/dev/null || echo "(conda info skipped/timed out)"
    echo "CONDA_PREFIX=${CONDA_PREFIX:-unset}"
    python --version
    which python

    echo "=== THREAD ENV ==="
    env | grep -E 'OMP|MKL|OPENBLAS|NUMEXPR|BLIS' | sort || true

    echo "=== PYTHON / NUMPY / BLAS ==="
    python - <<'PY'
import sys
import numpy as np
print(sys.version)
print("NumPy:", np.__version__)
np.show_config()
try:
    import threadpoolctl, json
    print("threadpool_info:")
    print(json.dumps(threadpoolctl.threadpool_info(), indent=1))
except Exception as e:
    print("threadpoolctl unavailable:", e)
PY

    echo "=== PIP FREEZE ==="
    python -m pip freeze 2>/dev/null || true

    echo "=== CONDA LIST ==="
    # The exact env is captured once in abaca/environment.abaca.explicit.txt;
    # here just record the prefix + a guard-limited listing (conda list has
    # been seen to hang on a cold compute node's NFS view of ~/miniforge3).
    timeout 60 conda list --explicit 2>/dev/null || echo "(conda list skipped/timed out - see abaca/environment.abaca.explicit.txt)"

    echo "=== COMPILED KERNELS ==="
    ls -la ./*.so 2>/dev/null || echo "no .so present"

} > "$MANIFEST" 2>&1

echo "manifest written to $MANIFEST"
