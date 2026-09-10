"""Tests for synth_corr_gen.make_density_targeted_dataset (density-deterministic
synthetic generator, 2026-09-10). See docs/implementation_log.md and
~/.claude/plans/misty-stargazing-liskov.md.

The generator targets a brute-force EFFECTIVE TUPLE DENSITY (bf_correlated / bf_tested)
and returns the exact ground-truth (pair, lag, window, signed_r) set by construction.
Each test call runs its own short brute-force pass internally to self-verify, so the
grid is deliberately small.
"""
import numpy as np
import pytest

from synth_corr_gen import make_density_targeted_dataset
from library_corrtrack_parallel import _canonicalize_rows, _rows_as_void_keys

_EVAL = dict(corr_threshold=0.7, window_size=128, window_step=16, n_lags=112,
             n_eval_steps=120)


def _keyset(rows):
    if len(rows) == 0:
        return set()
    return set(_rows_as_void_keys(_canonicalize_rows(np.asarray(rows, dtype=np.int64))).tolist())


@pytest.mark.parametrize("m,target,base_proc,corr_sign,duty", [
    (60, 2e-2, {"type": "ar1", "phi": 0.6}, "pos", 1.0),
    (60, 5e-3, {"type": "ar1", "phi": 0.6}, "both", 1.0),
    (60, 1e-2, {"type": "random_walk"}, "both", 1.0),
    (80, 2e-2, {"type": "ar1", "phi": 0.6}, "both", 0.5),
])
def test_density_hit_and_exact_ground_truth(m, target, base_proc, corr_sign, duty):
    pp = base_proc["type"] == "random_walk"
    out = make_density_targeted_dataset(
        m, 3000, target_density=target, base_proc=base_proc, preprocess=pp,
        corr_sign=corr_sign, duty=duty, n_epochs=4, lag_band=3, seed=7, **_EVAL)

    # 1. density within tolerance (analytic == what the sweep sees when it uses the GT)
    assert abs(out["analytic_density"] - target) / target < 0.20

    # 2. analytic GT set matches brute force on the determinate windows: precision 1.0,
    #    recall very high (a few near-threshold pairs may dip under by construction)
    assert out["gt_precision"] == pytest.approx(1.0, abs=1e-9)
    assert out["gt_recall"] > 0.95

    # 3. ground truth is signed and canonical
    rows, corrs = out["gt_rows"], out["gt_corrs"]
    assert rows.shape[1] == 5 and rows.dtype == np.int64
    assert len(rows) == len(corrs)
    assert np.array_equal(rows, _canonicalize_rows(rows))
    assert np.all(np.abs(corrs) >= _EVAL["corr_threshold"] - 0.05)
    assert np.all(np.abs(corrs) <= 0.99 + 1e-6)
    if corr_sign == "pos":
        assert np.all(corrs > 0)


def test_negative_sign_flip_is_scheduled_and_confirmed():
    out = make_density_targeted_dataset(
        60, 3200, target_density=1e-2, base_proc={"type": "ar1", "phi": 0.6},
        corr_sign="both", duty=1.0, n_epochs=4, lag_band=1, seed=3, **_EVAL)
    events = out["meta"]["events"]
    # at least one +<->- transition somewhere in the schedule
    assert any({e["from"], e["to"]} == {1, -1} for e in events)
    # both signs appear in the ground truth
    assert np.any(out["gt_corrs"] > 0) and np.any(out["gt_corrs"] < 0)


def test_scheduled_off_epochs_produce_onset_offset_events():
    out = make_density_targeted_dataset(
        60, 3200, target_density=5e-3, base_proc={"type": "ar1", "phi": 0.6},
        corr_sign="pos", duty=0.5, n_epochs=4, lag_band=2, seed=9, **_EVAL)
    sched = np.array(out["meta"]["schedule"])
    assert (sched == 0).any() and (sched != 0).any()          # some OFF, some ON
    assert len(out["meta"]["events"]) > 0                     # transitions logged


def test_determinism_byte_identical(tmp_path):
    kw = dict(target_density=6e-3, base_proc={"type": "ar1", "phi": 0.6},
              corr_sign="both", duty=0.75, n_epochs=4, lag_band=3, seed=21, **_EVAL)
    a = make_density_targeted_dataset(80, 3000, save_dir=str(tmp_path / "a"), **kw)
    b = make_density_targeted_dataset(80, 3000, save_dir=str(tmp_path / "b"), **kw)
    assert np.array_equal(a["data"], b["data"])
    assert np.array_equal(a["gt_rows"], b["gt_rows"])
    assert np.array_equal(a["gt_corrs"], b["gt_corrs"])
    assert a["meta"]["events"] == b["meta"]["events"]
    assert a["meta"]["schedule"] == b["meta"]["schedule"]
    assert (tmp_path / "a" / f"{a['stem']}.npz").read_bytes() == \
           (tmp_path / "b" / f"{b['stem']}.npz").read_bytes()


def test_diff_space_only_when_preprocess():
    # random-walk base, preprocess=True -> correlation lives in diff space and BF (which
    # also diffs) confirms it.
    out = make_density_targeted_dataset(
        60, 3000, target_density=1e-2, base_proc={"type": "random_walk"},
        preprocess=True, corr_sign="pos", duty=1.0, n_epochs=2, lag_band=2, seed=4, **_EVAL)
    assert out["gt_precision"] == pytest.approx(1.0, abs=1e-9)
    assert out["gt_recall"] > 0.95


def test_low_density_uses_disjoint_pair_regime():
    out = make_density_targeted_dataset(
        60, 3000, target_density=8e-4, base_proc={"type": "ar1", "phi": 0.6},
        corr_sign="pos", duty=1.0, n_epochs=2, lag_band=1, seed=1, **_EVAL)
    # very low density -> small groups (size 2), many of them
    assert out["group_size"] <= 3
    assert abs(out["analytic_density"] - 8e-4) / 8e-4 < 0.25
    assert out["gt_precision"] == pytest.approx(1.0, abs=1e-9)
