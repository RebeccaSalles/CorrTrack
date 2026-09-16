"""Report the BLAS backend numpy is actually using, and thread behaviour."""
import glob
import json
import os
import subprocess
import sys

import numpy as np

print("python:", sys.version.split()[0])
print("numpy:", np.__version__)

so = [f for f in glob.glob(os.path.join(os.path.dirname(np.__file__), "**", "*.so"), recursive=True)
      if "_multiarray_umath" in f]
print("multiarray so:", so[0] if so else "NOT FOUND")
if so:
    ldd = subprocess.run(["ldd", so[0]], capture_output=True, text=True).stdout
    for line in ldd.splitlines():
        if any(k in line.lower() for k in ("blas", "lapack", "mkl")):
            print("  linked:", line.strip())

# force some BLAS work so the library is loaded
a = np.random.default_rng(0).random((512, 512))
_ = a @ a

try:
    import threadpoolctl
    print("threadpoolctl:", threadpoolctl.__version__)
    print(json.dumps(threadpoolctl.threadpool_info(), indent=1))
except Exception as e:  # noqa: BLE001
    print("threadpoolctl error:", repr(e))

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    print(f"{v}={os.environ.get(v, '<unset>')}")
