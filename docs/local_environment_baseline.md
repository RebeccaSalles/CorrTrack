# Local (validated) environment baseline

Captured 2026-09-10 from the machine that produces the current CorrTrack
results, for reproduction on Inria Abaca (Grid'5000 Sophia). See
`docs/agent_context.md` equivalents and the memory notes on Abaca.

## Host / OS

- Ubuntu 24.04.2 LTS (WSL2), `Linux 6.18.33.2-microsoft-standard-WSL2`
- Python **3.12.3**, system interpreter `/usr/bin/python3` (no venv/conda)
- All Python packages are **Debian/Ubuntu distro builds** (`+dfsg`,
  `/build/numpy-...ubuntu1`), not PyPI wheels or conda-forge.

## Python packages (relevant subset)

| package | version |
|---|---|
| numpy | 1.26.4 |
| scipy | 1.11.4 |
| pandas | 2.1.4 |
| dask | 2023.12.1 |
| distributed | (present, version string `0+unknown`) |
| Cython | 3.0.8 |
| scikit-learn | 1.4.1.post1 |
| joblib | 1.3.2 |
| threadpoolctl | 3.1.0 |
| numexpr | 2.9.0 |
| Bottleneck | 1.3.5 |
| matplotlib | 3.6.3 |

No Numba anywhere in the stack.

## BLAS / LAPACK  (IMPORTANT for benchmarking)

- numpy links `libblas.so.3` / `liblapack.so.3` via Debian alternatives.
- The selected alternative is **reference (Netlib) BLAS**:
  `/usr/lib/x86_64-linux-gnu/blas/libblas.so.3.12.0` (package `libblas3`,
  priority 10). It is the only alternative installed.
- `threadpoolctl.threadpool_info()` returns `[]` - numpy sees no
  OpenBLAS/MKL threadpool. The brute-force Pearson baseline therefore runs
  on single-threaded, unoptimised reference BLAS on this machine.
- Consequence: a conda-forge / PyPI numpy on Abaca will link **OpenBLAS**
  (multithreaded, much faster GEMM). That alone would speed up the
  brute-force baseline and shift CorrTrack's measured speedup ratio without
  any algorithm change. To keep Abaca numbers comparable to existing
  results, the Abaca env must either also use reference BLAS, or the whole
  comparison (baseline + CorrTrack) must be re-measured on Abaca with BLAS
  treated as a controlled factor and threads pinned.

## Compiled extensions

Cython `.pyx` -> `.so`, built with `python3 setup_cython.py build_ext
--inplace`, currently `*.cpython-312-x86_64-linux-gnu.so`
(`candidate_kernels`, `monitor_kernels`, `partition_kernels`,
`sketch_kernels`). Must be rebuilt on Abaca for that env's Python.

## Git state

- remote `origin` = https://github.com/RebeccaSalles/CorrTrack.git
- HEAD = `83f43c6` ("Update README.md")
- The working tree has extensive uncommitted changes (recall-formula
  correction, memory-leak fixes, numeric correlated-pair representation,
  gamma offset, sweep scripts). HEAD does NOT reflect the code that
  produces current results. A freeze/commit is required before any Abaca
  campaign so one fixed commit can be pinned.
- Working copy is ~13 GB (mostly `tmp_artifacts/`, `synth_outputs/`,
  build products); `datasets/` is 478 MB; `.git` is 90 MB. Deploy
  selectively, do not clone the full working tree.
