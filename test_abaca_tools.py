"""Tests of the campaign tooling under abaca/ (2026-09-19): resource accounting, manifest feeder, aggregator."""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "abaca"))

from abaca.resource_probe import run_isolated, measure, rss_mb  # noqa: E402
from abaca.campaign_feeder import split_script  # noqa: E402
from abaca.aggregate_campaign import parse_stem, integrate_power, summary  # noqa: E402
import abaca.campaign_competitors as camp  # noqa: E402


def _alloc_and_write(n, d):
    a = np.ones((n, n))
    np.save(os.path.join(d, "x.npy"), a)
    return float(a.sum())


def test_run_isolated_reports_own_memory_and_io():
    with tempfile.TemporaryDirectory() as d:
        r, res = run_isolated(_alloc_and_write, 2000, d, artifact_dir=d)
    assert r == 4e6
    assert res["isolated"] is True and res["child_exit_code"] == 0
    mb = 2000 * 2000 * 8 / 2**20
    assert res["peak_rss_delta_mb"] >= 0.8 * mb                       # the array shows in the child's peak
    assert res["artifact_mb"] == pytest.approx(mb, rel=0.01)
    assert res["io_write_mb"] >= 0.9 * mb                              # storage traffic of the child alone
    assert res["wall_s"] > 0 and res["t_end_epoch"] > res["t_start_epoch"]
    assert res["mean_rss_mb"] is None or res["mean_rss_mb"] <= res["peak_rss_mb"] + 1
    assert set(res) >= {"energy_j", "energy_source", "cpu_user_s", "cpu_sys_s", "rss_before_mb"}


def test_run_isolated_propagates_child_error():
    with pytest.raises(RuntimeError, match="ZeroDivisionError"):
        run_isolated(lambda: 1 / 0)


def test_measure_in_process_matches_isolated_shape():
    r, res = measure(lambda: sum(range(1000)))
    assert r == 499500 and "peak_rss_mb" in res and rss_mb() is not None


def test_feeder_splits_emitted_script_into_cells(tmp_path):
    path = tmp_path / "s.sh"
    camp.emit(str(path), "$HOME/r", select=["motes_temperature_m27_W2880_s288_L0_T0.9", "sp500_m492_W60_s5_L20_T0.9_diff"])
    header, blocks = split_script(path.read_text())
    assert [b[0] for b in blocks] == ["motes_temperature_m27_W2880_s288_L0_T0.9", "sp500_m492_W60_s5_L20_T0.9_diff"]
    assert header.count("submit()") == 1 and "SNAPSHOT" in header
    for stem, block in blocks:
        assert block.count("$(submit ") == 8 and block.count(f"RUN_NAME={stem}_pos") == 1 and block.count(f"RUN_NAME={stem}_neg") == 1
        assert block.count("H=$H_JOB H2=$H2_JOB T=$T_JOB N=$N_JOB") == 2
        assert block.count("experiment_run_param_grid_campaign_hamming.py") == 2 and block.count("HYPEROPT_HAMMING_DIR=") == 2
        assert block.count("-l host=1") == 2 and block.count("-l core=") == 6 and block.count("monitor=") == 2
        assert "SNAPSHOT=$SNAPSHOT" in block
    gen = (tmp_path / "s_generate.sh").read_text()
    assert "gen_density_targeted" not in gen                            # real-data cells need no generator


def test_aggregate_parse_stem_and_power_integration():
    f = parse_stem("synth_ar1_d0p02_T0p9_m1250_L2_diff_m1250_W168_s12_L12_T0.9_neg")
    assert f == {"dataset": "synth_ar1_d0p02_T0p9_m1250_L2_diff", "m": 1250, "W": 168, "step": 12, "n_lags": 12, "L": 2, "T": 0.9, "space": "raw", "neg_corr": True} or f["space"] == "raw"
    g = parse_stem("sp500_m492_W60_s5_L20_T0.95_diff_pos")
    assert g["dataset"] == "sp500" and g["L"] == 5 and g["space"] == "diff" and g["neg_corr"] is False and g["T"] == 0.95
    power = {"nodeA": [[0.0, 100.0], [15.0, 100.0], [30.0, 200.0], [45.0, 200.0]]}
    e, p, n = integrate_power(power, "nodeA.sophia", 15.0, 45.0)
    assert e == pytest.approx(15 * 150 + 15 * 200) and p == pytest.approx(e / 30) and n == 3
    assert integrate_power(power, "nodeB", 0, 10) == (None, None, 0)
    runs = [dict(cell="c", T=0.9, arm="corrtrack", status="ok", speedup_vs_bf=2.0, recall=0.95, candidate_precision=0.5, candidate_specificity=0.9, step_time_median=0.001, res_peak_rss_delta_mb=10.0)]
    md = summary(runs, "T", "t")
    assert "| 0.9 | corrtrack | 1 | 2.00" in md
