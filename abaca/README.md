# Running CorrTrack on Inria Abaca (Grid'5000 Sophia)

Operational notes for running CorrTrack experiments on Abaca. Abaca is
Grid'5000-based: the scheduler is **OAR**, not Slurm.

## Account / access

- Grid'5000 user `rpontess`, group `zenith`, site `sophia`.
- SSH: `ssh sophia.g5k` (needs the `g5k` / `*.g5k` blocks in `~/.ssh/config`
  with `~/.ssh/id_rsa`, which is registered in Grid'5000).
- The Sophia frontend (`fsophia`) is for file management, env setup and job
  submission only. **Never run computation on the frontend.**
- `-q abaca` is accepted and routed by priority; this account currently
  lands on the **p2** queue (max walltime 96 h).

## Environment

Debian 13 / Python 3.13 is the platform default. CorrTrack is validated on
Python 3.12, so it runs from a pinned conda env, not the system interpreter.

- Miniforge: `~/miniforge3` (`conda` / `mamba`).
- Env `corrtrack`: `~/miniforge3/envs/corrtrack`, Python 3.12, numpy 1.26.4,
  scipy 1.11.4, pandas 2.1.4, dask 2023.12.1, cython 3.0.8,
  scikit-learn 1.4.1, numexpr 2.9.0, statsmodels 0.14.1 (exact matches to
  the local validated stack); bottleneck / matplotlib float (non-numerical).
  Core third-party deps are numpy/scipy/pandas/scikit-learn/matplotlib/
  statsmodels + optional dask - nothing else.
- Spec files, regenerated from the built env:
  `environment.abaca.yml`, `environment.abaca.explicit.txt`,
  `pip-freeze.abaca.txt`.
- Recreate:
  `mamba env create -n corrtrack -f abaca/environment.abaca.explicit.txt`
  (explicit list is exact; the `.yml` is for reference / other platforms).

### BLAS

Local numpy uses reference (Netlib) BLAS, single-threaded. The Abaca env
uses **OpenBLAS 0.3.25** (conda-forge default, `nomkl`). This is a
deliberate change: all baseline-vs-CorrTrack comparisons are to be
re-measured on Abaca, never compared across the two BLAS backends. Pin
threads explicitly for controlled runs (see below).

## Code deployment

The GitHub repo (`RebeccaSalles/CorrTrack`) is currently far behind the
working tree - the `.pyx` kernels, `setup_cython.py`, the test suite and the
sweep scripts are all untracked, so a `git clone` cannot build. Until the
tree is committed and a commit pinned, the Abaca copy at
`~/corrtrack_release_dev` is an **rsync deploy** of the local working tree
(generated files, `tmp_artifacts/`, `.git` excluded). Refresh with the same
rsync; replace with `git clone` + `git checkout <pinned-commit>` once the
tree is committed.

## Cython kernels

`setup_cython.py` compiles with `-march=native -ffast-math`, so the `.so`
files are specific to the CPU that built them. Always
`python setup_cython.py build_ext --inplace` inside the job, on the target
node, and never reuse `.so` across cluster / node types.

## Workflow

1. **Smoke test** (`abaca/smoke.oar`): builds kernels, runs the 119-test
   regression suite, runs a tiny end-to-end pipeline, writes a manifest.
   `cd ~/corrtrack_release_dev && oarsub -S ./abaca/smoke.oar`
2. **Timing-validation**: whole node on one homogeneous cluster, baseline +
   CorrTrack in the same allocation, threads pinned. (script TBD per campaign)
3. **Campaign**: only after 1 and 2, one pinned commit + one env for the
   whole run.

Controlled thread env for the Dask-style parallel path:
```
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

## Storage

- `~` (`/home/rpontess`): NFS, ~24 GB soft quota. Code + env live here.
- Zenith group storage `/srv/storage/zenith@storage1.sophia.grid5000.fr`:
  NFS, **~98 % full** - check free space before staging results here. Not
  backed up.
- Node-local `/tmp`: use for high-I/O working files inside a job; copy
  results back before the job ends.

## Job control quick reference

```
oarsub -S ./abaca/smoke.oar         # submit batch job
oarstat -u                          # my jobs
oarstat -f -j JOB_ID                # job detail (node, state, walltime)
oarsub -C JOB_ID                    # attach to a running job
oardel JOB_ID                       # cancel
oarwalltime JOB_ID +1:00            # request walltime extension
```

## Results layout

`~/corrtrack_abaca_results/<run-id>/` per run:
`environment.txt` (manifest), `build.log`, `pytest.log`, `pipeline.log`,
plus experiment output.
