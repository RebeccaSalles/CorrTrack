# Abaca (Grid'5000 Sophia) setup status

Separate track from the main CorrTrack work. Managed from a dedicated
session so it does not collide with edits to `implementation_log.md` /
`current_task.md`. Operational detail is in `abaca/README.md`; the validated
local stack is in `docs/local_environment_baseline.md`.

## Decisions taken (2026-09-10, with the user)

- **Scope: infra setup only** for now. No experiment campaign yet.
- **BLAS: OpenBLAS on Abaca, re-measure everything.** Local numpy uses
  single-threaded reference BLAS; the Abaca env uses OpenBLAS. All
  baseline-vs-CorrTrack comparisons will be re-run on Abaca; prior local
  timings are reference only, never compared across the two backends.
- **Env manager: Miniforge + conda/mamba**, pinned Python 3.12 (not the
  Debian 13 system Python 3.13).
- **Do not migrate CorrTrack to Python 3.13** (no runtime benefit for a
  Cython/NumPy/BLAS workload; forces a full re-benchmark; Numba is not a
  dependency so 3.13 is not otherwise blocked).

## Done

- SSH: `ssh sophia.g5k` works (`~/.ssh/config` `g5k` / `*.g5k` blocks,
  `~/.ssh/id_rsa` registered in Grid'5000).
- Frontend recon: Debian 13 trixie, system Python 3.13.5, group `zenith`,
  `-q abaca` routes to the **p2** queue (96 h max walltime). Home quota
  ~24 GB soft. Group storage `/srv/storage/zenith@...` is ~98 % full.
- Miniforge installed at `~/miniforge3`.
- Conda env `corrtrack` created: Python 3.12.14, numpy 1.26.4, scipy
  1.11.4, pandas 2.1.4, dask 2023.12.1, cython 3.0.8, scikit-learn
  1.4.1.post1, numexpr 2.9.0 (exact matches); bottleneck 1.6.0 /
  matplotlib 3.8.4 float (non-numerical). OpenBLAS 0.3.25 pthreads, nomkl.
  Spec exported to `abaca/environment.abaca.{yml,explicit.txt}` +
  `abaca/pip-freeze.abaca.txt`.
- Code: **git clone** of `RebeccaSalles/CorrTrack` (public) at
  `~/corrtrack_release_dev`, branch `dev`, pinned HEAD `da0f4f7`
  ("LSH candidate backend, streaming memory fixes, proxy-anchor hyperopt,
  experiment suite"). The user committed + pushed the working tree to `dev`
  on 2026-09-10; verified no real `.py`/`.pyx` diff between that local tree
  and `origin/dev`. The earlier rsync copy has been deleted.
- Helper scripts under `abaca/` (`smoke.oar`, `capture_manifest.sh`,
  `check_blas.py`, `README.md`, env spec files) are NOT in the repo -
  `.gitignore` excludes `/docs/` and `/tasks/`, and `abaca/` was never
  added. They are rsynced in as an overlay on top of the clone. Worth
  adding to the repo eventually.
- Repo carries stale `build/` + top-level `.so` (x86-64 cpython-312, built
  on the WSL box) despite `.gitignore /build/` - `smoke.oar` does
  `rm -rf build ./*.cpython-*.so` before compiling on the node.
- Local numeric-equivalence reference captured (commit 83f43c6 + working
  tree): regression suite 119/119; synth-demo deterministic metrics
  (n_bands 32, corr_w_bf 606765, tested_w_bf 4401085,
  candidate_search_index_candidates 3467298, recall 0.9456, precision 1.0,
  f1 0.9720).
- **Smoke test PASSED on Abaca**, twice: job 3092016 (rsync deploy) and
  job **3092200** (the pinned `git clone` at `da0f4f7`), both mercantour2-1,
  2026-09-10. Kernels build on the node, regression suite 119/119 (+241
  subtests), end-to-end pipeline completes. Every deterministic metric
  bit-identical to the local reference (n_bands 32, recall 0.9456,
  precision 1.0, f1 0.9720) - the OpenBLAS switch and `-ffast-math` kernels
  do not perturb results. Timings differ (`sk_time` 213 s local -> 507 s on
  this 2013 Ivy Bridge node; the synth-demo config is
  single-threaded-Python-bound, not BLAS - real timing runs need
  mercantour3+). Manifest + logs kept at
  `~/corrtrack_abaca_results/smoke_3092200/`.
- conda note: `conda activate` / `conda list` are very slow (seen wedged)
  on a cold compute node's NFS view of `~/miniforge3`; `capture_manifest.sh`
  guards them with `timeout`. For campaigns, stage the env to node-local
  `/tmp` (conda-pack) instead of importing from NFS.

## Git blocker: RESOLVED (2026-09-10)

The user committed + pushed the full working tree to branch `dev`
(`da0f4f7`), which now contains the `.pyx` kernels, `setup_cython.py`, the
test suite and the sweep scripts. Abaca now runs a real `git clone` of
`dev` pinned at `da0f4f7`. A campaign pins that commit; `git pull` only
between campaigns, never mid-campaign.

## Rollout stage 1 (smoke): DONE. Stages 2-3 not started.

## Next steps (all deferred until a campaign is actually wanted)

1. **Stage 2 (timing-validation)**: pin `-p mercantour3`, whole node,
   threads pinned, baseline + CorrTrack in one allocation. Needs a new OAR
   script (`abaca/timing.oar`, not written). First real deliverable would be
   the OpenBLAS-vs-reference-BLAS effect on brute-force Pearson timing.
2. Add env-staging-to-`/tmp` (conda-pack) to the job scripts so conda/NFS
   slowness on cold nodes stops mattering.
3. Consider adding `abaca/` (OAR scripts + env spec) to the repo so the
   whole setup is version-controlled, not an rsync overlay.
4. `git pull` on `~/corrtrack_release_dev` to advance the pinned commit only
   between campaigns; rebuild kernels after any pull.
