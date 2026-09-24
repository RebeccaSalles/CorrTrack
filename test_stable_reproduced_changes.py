import csv
import json
import math
import os
import resource
import sys
import tempfile
from pathlib import Path
import unittest
import unittest.mock

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import corrtrack_run_corrtrack
import corrtrack_run_bruteforce
import candidate_kernels
import library_corrtrack_parallel
from library_corrtrack_parallel import (
    CorrJoinDoubleFilterIndex,
    StatStreamGridIndex,
    ParCorrGridIndex,
    Candidates_BF_BRAID,
    Candidates_BF_TSUBASA,
    five_sums,
    pearson_from_five_sums,
    CSV_DELIMITER,
    COMPARISON_COLUMNS,
    Candidates,
    Candidates_BF_Incremental,
    Candidates_BF_FilCorr,
    CorrTrack,
    CorrTrackMultiWindow,
    CorrTrack_compare,
    CorrTrack_optimize,
    OPTIM_RESULT_COLUMNS,
    RUN_RESULT_COLUMNS,
    Sketches,
    _extract_feature_overrides,
    _normalize_bf_key,
    _canonicalize_rows,
    _rows_as_void_keys,
    _distance_correlation_1d,
    _dist_corr_fast,
    _kendall_tau,
    _spearman_rho,
    _metric_corr_and_dist,
    _is_near_constant_stats,
    _is_structurally_spiked_stats,
    _passes_corr_threshold,
    _resolve_candidate_backend,
    _write_correlated_codebook,
    run_and_log_corrtrack,
    run_and_log_bruteforce,
)


class StableReproducedChangesTest(unittest.TestCase):
    def test_compact_correlated_codebook_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "run_correlated.csv")
            with open(csv_path, "w", newline="") as file:
                writer = csv.writer(file, delimiter=CSV_DELIMITER)
                writer.writerow(["id1_code", "id2_code", "time1_idx", "time2_idx", "corr"])
                writer.writerow([0, 1, 0, 0, 0.91])
            _write_correlated_codebook(csv_path, {"alpha": 0, "beta": 1})

            compare = CorrTrack_compare.__new__(CorrTrack_compare)
            compare.window_size = 4
            rows = list(compare._iter_correlated_rows(csv_path))

        self.assertEqual(rows, [(("alpha", "beta", 0, 0, 4), 0.91)])

    def test_artifact_metrics_canonicalize_numeric_bf_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            bf_csv = os.path.join(tmp, "bf_correlated.csv")
            pred_csv = os.path.join(tmp, "pred_correlated.csv")
            for path in (bf_csv, pred_csv):
                with open(path, "w", newline="") as file:
                    writer = csv.writer(file, delimiter=CSV_DELIMITER)
                    writer.writerow(["id1_code", "id2_code", "time1_idx", "time2_idx", "corr"])
                    writer.writerow([0, 1, 16, 0, 0.91])
            _write_correlated_codebook(bf_csv, {"0": 0, "1": 1})
            _write_correlated_codebook(pred_csv, {"s1": 0, "s2": 1})

            compare = CorrTrack_compare.__new__(CorrTrack_compare)
            compare.window_size = 4
            compare.ids = ["s1", "s2"]
            metrics = compare._stream_window_metrics_from_artifacts(
                os.path.join(tmp, "bf"),
                os.path.join(tmp, "pred"), total_pairs_bf=1,
            )

        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1_score"], 1.0)

    def test_artifact_metrics_ignore_local_codebook_sort_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            bf_csv = os.path.join(tmp, "bf_correlated.csv")
            pred_csv = os.path.join(tmp, "pred_correlated.csv")
            with open(bf_csv, "w", newline="") as file:
                writer = csv.writer(file, delimiter=CSV_DELIMITER)
                writer.writerow(["id1_code", "id2_code", "time1_idx", "time2_idx", "corr"])
                writer.writerow([0, 1, 0, 0, 0.91])
                writer.writerow([0, 2, 0, 0, 0.92])
            with open(pred_csv, "w", newline="") as file:
                writer = csv.writer(file, delimiter=CSV_DELIMITER)
                writer.writerow(["id1_code", "id2_code", "time1_idx", "time2_idx", "corr"])
                writer.writerow([0, 1, 0, 0, 0.92])
                writer.writerow([0, 2, 0, 0, 0.91])
            _write_correlated_codebook(bf_csv, {"0": 0, "10": 1, "2": 2})
            _write_correlated_codebook(pred_csv, {"s1": 0, "s3": 1, "s11": 2})

            compare = CorrTrack_compare.__new__(CorrTrack_compare)
            compare.window_size = 4
            compare.ids = [f"s{i + 1}" for i in range(11)]
            metrics = compare._stream_window_metrics_from_artifacts(
                os.path.join(tmp, "bf"),
                os.path.join(tmp, "pred"), total_pairs_bf=2,
            )

        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1_score"], 1.0)

    def test_pairwise_chunk_merge_respects_low_fd_limit(self):
        old_limits = resource.getrlimit(resource.RLIMIT_NOFILE)
        with tempfile.TemporaryDirectory() as tmp:
            chunk_paths = []
            for idx in range(40):
                path = os.path.join(tmp, f"chunk_{idx:03d}.csv")
                with open(path, "w", newline="") as file:
                    writer = csv.writer(file, delimiter=CSV_DELIMITER)
                    writer.writerow(["k", "v"])
                    writer.writerow([idx, f"value_{idx}"])
                chunk_paths.append(path)

            output_csv = os.path.join(tmp, "merged.csv")
            try:
                hard = old_limits[1]
                new_soft = min(32, hard if hard != resource.RLIM_INFINITY else 32)
                resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
                corrtrack = CorrTrack.__new__(CorrTrack)
                corrtrack._merge_sorted_csv_chunks(
                    chunk_paths,
                    output_csv,
                    ["k", "v"], key_fn=lambda row: int(row[0]),
                )
            finally:
                resource.setrlimit(resource.RLIMIT_NOFILE, old_limits)

            with open(output_csv, newline="") as file:
                rows = list(csv.reader(file, delimiter=CSV_DELIMITER))

        self.assertEqual(rows[0], ["k", "v"])
        self.assertEqual([int(row[0]) for row in rows[1:]], list(range(40)))

    def test_unvalidated_negative_metrics_use_ground_truth_sign_for_tp(self):
        gt = {
            ("a", "b", 0, 0, 4): 0.9,
            ("a", "c", 0, 0, 4): -0.8,
        }
        pred = {
            ("a", "b", 0, 0, 4): 1.0,
            ("a", "c", 0, 0, 4): 1.0,
            ("b", "c", 0, 0, 4): 1.0,
        }
        metrics = CorrTrack.compute_metrics_bf(
            pred,
            gt, windows=True, use_ground_truth_sign_for_tp=True,
        )
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["precision_pos"], 1 / 2)
        self.assertAlmostEqual(metrics["precision_neg"], 1 / 2)
        self.assertAlmostEqual(metrics["recall_pos"], 1.0)
        self.assertAlmostEqual(metrics["recall_neg"], 1.0)

    def test_stable_csv_schemas_drop_fixed_and_bootstrap_columns(self):
        for columns in (RUN_RESULT_COLUMNS, OPTIM_RESULT_COLUMNS, COMPARISON_COLUMNS):
            self.assertIn("candidate_key_mode", columns)
            self.assertIn("candidate_key_seed", columns)
            self.assertNotIn("freq_threshold", columns)
            self.assertNotIn("proxy_eval_mode", columns)
            self.assertNotIn("optim_bootstrap_time", columns)
            self.assertFalse(any(col.startswith("bootstrap_") for col in columns))
            self.assertFalse(any(col.startswith("proxy_bootstrap_") for col in columns))
        self.assertIn("mem_w", OPTIM_RESULT_COLUMNS)
        for columns in (RUN_RESULT_COLUMNS, OPTIM_RESULT_COLUMNS, COMPARISON_COLUMNS):
            # (2026-07-27) candidate_block_size_steps/candidate_block_index_dims
            # were sorted_arrays_bs-only params; removed along with that
            # backend in the release-restructuring cleanup -- see
            # docs/implementation_log.md's 2026-07-27 entries.
            self.assertNotIn("candidate_block_size_steps", columns)
            self.assertNotIn("candidate_block_index_dims", columns)
            self.assertIn("candidate_similarity", columns)
            self.assertIn("candidate_cosine_threshold", columns)
            self.assertIn("candidate_search_index_candidates", columns)
            self.assertIn("candidate_search_unique_pre_dot_pairs", columns)
            self.assertIn("candidate_search_duplicate_pre_dot_pairs", columns)
            self.assertIn("candidate_search_after_similarity", columns)
            self.assertIn("candidate_search_dot_checks", columns)
            self.assertIn("candidate_search_distance_checks", columns)
            self.assertNotIn("candidate_n_pivots", columns)
            self.assertNotIn("candidate_hamming_hmax", columns)
            # (2026-07-05) candidate-search metrics consolidation: these are
            # pure aliases of other still-exposed columns in every backend
            # (unique_index_candidates == valid_index_candidates,
            # after_coord == valid_index_candidates, partial_checks ==
            # after_coord), dropped to reduce column bloat -- see
            # docs/implementation_log.md.
            self.assertNotIn("candidate_search_unique_index_candidates", columns)
            self.assertNotIn("candidate_search_after_coord", columns)
            self.assertNotIn("candidate_search_partial_checks", columns)
        self.assertIn("proxy_search_distance_checks", OPTIM_RESULT_COLUMNS)
        self.assertIn("proxy_search_unique_pre_dot_pairs", OPTIM_RESULT_COLUMNS)
        self.assertIn("proxy_search_duplicate_pre_dot_pairs", OPTIM_RESULT_COLUMNS)
        self.assertIn("proxy_search_similarity_work_rate", OPTIM_RESULT_COLUMNS)
        self.assertIn("proxy_search_time_total", OPTIM_RESULT_COLUMNS)
        self.assertIn("proxy_timing_repeats", OPTIM_RESULT_COLUMNS)
        self.assertNotIn("proxy_search_unique_index_candidates", OPTIM_RESULT_COLUMNS)
        self.assertNotIn("proxy_search_after_coord", OPTIM_RESULT_COLUMNS)
        self.assertNotIn("proxy_search_partial_checks", OPTIM_RESULT_COLUMNS)
        self.assertTrue(corrtrack_run_corrtrack.config_folder().startswith("corrtrack_release_dev/"))

        compare = CorrTrack_compare.__new__(CorrTrack_compare)
        metrics = {
            "precision_pos": 0.0,
            "recall_pos": 0.0,
            "f1_score_pos": 0.0,
            "precision_neg": 0.0,
            "recall_neg": 0.0,
            "f1_score_neg": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "specificity": 0.0,
            "recall_min": None,
            "f1_score": 0.0,
            "aucroc": float("nan"),
            "pr_auc": float("nan"),
        }
        row = compare._build_comparison_row(
            "dataset",
            {},
            {},
            metrics,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            (0.0, 0.0, 0.0, 0.0, 0.0),
        )
        self.assertEqual(len(row), len(COMPARISON_COLUMNS))

    def test_canonicalize_rows_matches_scalar_normalize_bf_key(self):
        # (2026-09-10) _canonicalize_rows must reproduce _normalize_bf_key exactly for a
        # batch, including the s1==s2 (autocorrelation) and t1==t2 (synchronous) branches.
        rng = np.random.default_rng(20260910)
        n = 12000
        s1 = rng.integers(0, 40, size=n)
        s2 = rng.integers(0, 40, size=n)
        # force a healthy fraction of s1==s2 and t1==t2 rows
        same_series = rng.random(n) < 0.25
        s2 = np.where(same_series, s1, s2)
        t1 = rng.integers(0, 5000, size=n) * 16
        t2 = rng.integers(0, 5000, size=n) * 16
        same_time = rng.random(n) < 0.25
        t2 = np.where(same_time, t1, t2)
        w = np.full(n, 256)
        rows = np.column_stack([s1, s2, t1, t2, w]).astype(np.int64)

        got = _canonicalize_rows(rows)
        want = np.array(
            [_normalize_bf_key((int(a), int(b), int(c), int(d), int(e)))
             for a, b, c, d, e in rows],
            dtype=np.int64,
        )
        np.testing.assert_array_equal(got, want)

        # void-key view: identical rows -> identical keys, set ops behave
        keys = _rows_as_void_keys(got)
        self.assertEqual(keys.shape, (n,))
        # a row present in itself
        self.assertTrue(np.isin(keys[:5], keys).all())

    def test_compute_metrics_bf_numeric_matches_dict_path(self):
        # (2026-09-10) The numeric (rows, corrs) metrics fast path must return the same
        # precision/recall/f1/pos/neg as the string-dict path for the same correlated sets.
        r = np.random.default_rng(99)

        def mk(n, seed):
            g = np.random.default_rng(seed)
            s1 = g.integers(0, 25, n); s2 = g.integers(0, 25, n)
            t1 = g.integers(0, 3000, n) * 16; t2 = g.integers(0, 3000, n) * 16
            rows = np.column_stack([s1, s2, t1, t2, np.full(n, 256)]).astype(np.int64)
            return rows, g.uniform(-1, 1, n)

        pr_rows, pr_c = mk(3000, 1)
        gt_rows, gt_c = mk(4000, 2)
        gt_rows = np.vstack([gt_rows, pr_rows[:1200]])
        gt_c = np.concatenate([gt_c, pr_c[:1200]])
        labels = [f"s{i}" for i in range(25)]

        def to_dict(rows, corrs):
            return {
                _normalize_bf_key((labels[a], labels[b], int(c), int(e), int(f))): float(v)
                for (a, b, c, e, f), v in zip(rows.tolist(), corrs.tolist())
            }

        mnum = CorrTrack.compute_metrics_bf(
            (pr_rows, pr_c), (gt_rows, gt_c), windows=True, total_pairs_bf=5_000_000,
        )
        mdict = CorrTrack.compute_metrics_bf(
            to_dict(pr_rows, pr_c), to_dict(gt_rows, gt_c), windows=True, total_pairs_bf=5_000_000,
        )
        for k in ("precision", "recall", "f1_score", "specificity",
                  "precision_pos", "recall_pos", "precision_neg", "recall_neg"):
            self.assertAlmostEqual(mnum[k], mdict[k], places=9, msg=k)

    def test_correlated_property_roundtrips_through_numeric_accumulator(self):
        # (2026-09-10) `correlated` is a lazy view over the numeric accumulator; appending
        # rows and reading back the dict must canonicalize on labels exactly as the old path.
        ct = CorrTrack.__new__(CorrTrack)
        ct._correlated_rows = None
        ct._correlated_corrs = None
        ct._correlated_count = 0
        ct._correlated_cap = 0
        ct._correlated_legacy_dict = {}
        ct._correlated_view_cache = None
        ct._correlated_view_cache_key = None
        ct.artifact_bookkeeping_time = 0.0
        ct.series_ids = {f"s{i}": i for i in range(5)}
        rows = np.array([[2, 0, 320, 160, 256], [0, 1, 160, 160, 256], [3, 3, 16, 48, 256]], dtype=np.int64)
        corrs = np.array([0.9, -0.8, 0.75])
        ct._append_correlated_numeric(rows, corrs)
        d = ct.correlated
        self.assertEqual(len(d), 3)
        # canonical on labels: later-start first, id-sorted on t1==t2 tie
        self.assertIn(_normalize_bf_key(("s2", "s0", 320, 160, 256)), d)
        self.assertIn(_normalize_bf_key(("s0", "s1", 160, 160, 256)), d)
        rr, cc = ct.correlated_rows()
        self.assertEqual(rr.shape, (3, 5))

    def test_candidate_cosine_threshold_offset_resolves_against_corr_threshold(self):
        # (2026-09-10) The retrieval gate can be set as a margin below corr_threshold
        # instead of an absolute cosine value; an explicit absolute still wins.
        common = dict(
            window_size=16, basic_window=4, window_step=4, n_vectors=16, n_lags=8,
            exec="sequential",
        )
        ct_off = CorrTrack(corr_threshold=0.85, candidate_cosine_threshold_offset=0.20, **common)
        self.assertAlmostEqual(ct_off.candidate_cosine_threshold, 0.65, places=6)

        ct_both = CorrTrack(
            corr_threshold=0.85, candidate_cosine_threshold=0.5,
            candidate_cosine_threshold_offset=0.20, **common,
        )
        self.assertAlmostEqual(ct_both.candidate_cosine_threshold, 0.5, places=6)

        ct_neither = CorrTrack(corr_threshold=0.85, **common)
        self.assertIsNotNone(ct_neither.candidate_cosine_threshold)

        overrides = _extract_feature_overrides({"candidate_cosine_threshold_offset": 0.15})
        self.assertEqual(overrides.get("candidate_cosine_threshold_offset"), 0.15)

    def test_legacy_comparison_row_keeps_metrics_after_candidate_search_columns(self):
        compare = CorrTrack_compare.__new__(CorrTrack_compare)
        compare.test_data = np.zeros((3, 8), dtype=float)
        compare.ids = ["a", "b"]
        compare.window_size = 4
        compare.window_step = 4
        compare.n_lags = 4
        compare.corr_threshold = 0.7
        compare.exec = "sequential"
        compare.parallel_sketch = False
        compare.parallel_candidates = False
        compare.parallel_validation = False
        compare.candidate_key_mode = "first"
        compare.candidate_key_seed = None
        compare.candidate_parallel_mode = "recent_shards"
        compare.candidate_bucket_width = None
        compare.candidate_block_size_steps = 1
        compare.candidate_block_index_dims = 1
        compare.candidate_similarity = "cosine"
        compare.candidate_cosine_threshold = 0.5
        compare.hybrid_validation = False
        compare.hybrid_validation_min_repeat_rate = 0.25
        compare.hybrid_validation_disable_rate = None
        compare.hybrid_validation_ema_alpha = 0.25
        compare.hybrid_validation_min_candidates = 256
        compare.numeric_rows = True
        compare.neg_corr = False
        compare.corr_val = True
        compare.recall_by_window = True
        compare.pair_min_dist_bf = None
        compare.candidate_time_bf = 1.0
        compare.validation_time_bf = 1.0
        compare.monitor_time_bf = 0.0
        compare.runtime_bf = 2.0
        compare.artifact_time_bf = 0.0
        compare.correlated_bf = 1
        compare.tested_bf = 1
        compare.total_bf = 1
        compare.baseline_mode = "bruteforce"

        def fake_mode_run(*_args, **_kwargs):
            compare.tested_w = 1
            compare.correlated_w = 1
            compare.total_w = 1
            compare.candidate_search_index_candidates = 7
            compare.candidate_search_after_similarity = 3
            compare.candidate_search_dot_checks = 2
            compare.candidate_search_distance_checks = 1
            return (0.1, 0.2, 0.3, 0.0), 1.0, 0.0, {("a", "b", 0, 0, 4): 1}

        compare._mode_run = fake_mode_run
        bst = {
            "nodes": 0,
            "seed": 1,
            "seed_toggle": 2,
            "n_vectors": 2,
            "grid_dimension": 2,
            "cell_size": 1.0,
            "grid_max": 1.0,
            "preprocess": False,
            "candidate_backend": "lsh_sign_dot",
            "candidate_similarity": "cosine",
            "candidate_cosine_threshold": 0.5,
        }
        with tempfile.TemporaryDirectory() as tmp:
            row = compare._parallel_mode_run(
                ("dataset", "main", "nD", tmp, "main_nD", bst, 2.0, {("a", "b", 0, 0, 4): 1})
            )
        values = dict(zip(COMPARISON_COLUMNS, row))
        self.assertEqual(values["candidate_search_index_candidates"], 7)
        self.assertEqual(values["candidate_search_after_similarity"], 3)
        self.assertEqual(values["candidate_search_dot_checks"], 2)
        self.assertEqual(values["candidate_search_distance_checks"], 1)
        self.assertEqual(values["precision"], "1.0000")
        self.assertEqual(values["recall"], "1.0000")
        self.assertEqual(values["f1"], "1.0000")

    def test_corrtrack_hybrid_validation_current_window_cache_matches_no_cache(self):
        # (2026-07-06) End-to-end regression test for "point 1"'s wiring
        # through CorrTrack (not just the low-level HybridValidationCache/
        # validate_corr_batch unit tests elsewhere in this file). An earlier
        # version of this wiring assumed CorrTrack had access to
        # Sketches._raw_window_sums -- it doesn't; that array belongs to a
        # structurally different object (a separate per-series buffer used
        # only for candidate-search sketching). This crashed with
        # AttributeError the first time it ran through the real
        # CorrTrack.run() streaming path (never exercised by a pure
        # HybridValidationCache unit test). Fixed by giving CorrTrack its own
        # _cw_raw_sums* cache, recomputed lazily (at most once per step, only
        # when there are pairs to validate) via
        # _recompute_current_window_raw_moments over CorrTrack's own
        # window_data.
        #
        # "point 1" was initially wired only into the hybrid_validation path,
        # which is off by default (experiment_run_param_grid.py); it was then
        # extended to the default (non-hybrid) path too -- first via
        # validate_corr_batch (needed Python-level per-pair bookkeeping in
        # _iter_validation_payloads that often cost more than it saved), then
        # rewritten to route through candidate_kernels.validate_corr_rows
        # instead (takes data+rows directly, computes "is this side current"
        # itself inside its own nogil loop, no Python-level bookkeeping
        # arrays needed at all -- the human's explicit ask once the
        # validate_corr_batch version's regression was found). This also
        # fixed CorrTrack._get_validated_corr's own inline sequential-mode
        # validation logic (a near-duplicate of _validate_pairs_standard) and
        # _get_validated_corr_numeric's hybrid/non-hybrid branches, which all
        # needed the same cache wired through once the payload shape
        # (rows/data/base_index, not x_batch/y_batch) changed. A
        # validation_current_window_cache toggle (default False -- opt-in,
        # not a universal win) lets this be disabled/A-B'd; it must never
        # change output.
        #
        # This test streams synthetic correlated/anti-correlated/independent
        # series through the real pipeline across every combination of exec
        # mode x hybrid_validation x validation_current_window_cache x
        # freq_threshold and checks the exact same correlated pairs and
        # correlation values are found in every case. See docs/
        # implementation_log.md.
        #
        # Was candidate_backend="flat" (freq_threshold=0.7 exercised the
        # legacy dict-based candidate path there, 0.0 the numeric-rows
        # path) -- "flat" was removed (the only backend left using the
        # dict-based path at all), so migrated to "brute_force" (needs no
        # cosine_threshold setup unlike "auto", confirmed directly:
        # swapping to "auto" without also setting candidate_cosine_
        # threshold silently found zero candidates). Both freq_threshold
        # values now exercise the numeric-rows path uniformly; the exec/
        # hybrid/cache-consistency guarantee this test exists for is
        # unaffected either way.
        rng = np.random.default_rng(20260706)
        n_steps = 200
        base = rng.normal(size=(2, n_steps))
        values = np.zeros((6, n_steps))
        values[0] = base[0]
        values[1] = base[0] * 0.95 + rng.normal(scale=0.05, size=n_steps)
        values[2] = base[1]
        values[3] = -base[1] * 0.9 + rng.normal(scale=0.1, size=n_steps)
        values[4] = rng.normal(size=n_steps)
        values[5] = rng.normal(size=n_steps)
        ids = np.array(["s0", "s1", "s2", "s3", "s4", "s5"])
        data = np.vstack([np.arange(n_steps), values])

        def run_once(exec_mode, hybrid, use_cache, freq_threshold):
            kwargs = dict(
                basic_window=4, window_step=4, n_vectors=8, n_lags=4, seed=11, seed_toggle=22, corr_threshold=0.7, neg_corr=True, preprocess=False, exec=exec_mode, max_workers=2, parallel_sketch=False, parallel_candidates=False, parallel_validation=False, candidate_backend="brute_force", hybrid_validation=hybrid, hybrid_validation_min_candidates=1, hybrid_validation_min_repeat_rate=0.0, validation_current_window_cache=use_cache, freq_threshold=freq_threshold,
            )
            corrtrack = CorrTrack(window_size=16, **kwargs)
            step = 4
            for start in range(0, n_steps - step + 1, step):
                corrtrack.run(data[:, start:start + step], ids, verbose=False, testing=False, corr_val=True, monitor=False)
            return corrtrack

        baseline = run_once("sequential", False, False, 0.7)
        self.assertGreater(len(baseline.correlated), 0)
        baseline_corr = {k: round(v, 6) for k, v in baseline.correlated.items()}

        for exec_mode in ("sequential", "parallel"):
            for hybrid in (False, True):
                for use_cache in (False, True):
                    for freq_threshold in (0.7, 0.0):
                        with self.subTest(exec_mode=exec_mode, hybrid=hybrid, use_cache=use_cache, freq_threshold=freq_threshold):
                            ct = run_once(exec_mode, hybrid, use_cache, freq_threshold)
                            ct_corr = {k: round(v, 6) for k, v in ct.correlated.items()}
                            self.assertEqual(set(ct_corr.keys()), set(baseline_corr.keys()))
                            for key, corr in baseline_corr.items():
                                self.assertAlmostEqual(corr, ct_corr[key], places=4)

    def test_validate_corr_rows_current_window_cache_matches_bruteforce(self):
        # (2026-07-06) "point 1", fully in Cython: candidate_kernels.validate_corr_rows
        # (the function backing the real, actually-exercised validation path
        # -- both CorrTrack's dict-based sequential/parallel branches and its
        # numeric-rows path all funnel through this) gained the same
        # current_window_sums* cache parameters as HybridValidationCache and
        # validate_corr_batch, but computes "is this side current" itself
        # inside its own nogil loop from s1/s2/start1/start2/w/n_cols (all
        # already available there), needing zero extra Python-level
        # bookkeeping arrays -- unlike validate_corr_batch, whose equivalent
        # Python-side bookkeeping in _iter_validation_payloads was found to
        # often cost more than it saved (see docs/implementation_log.md,
        # "'Point 1' regression found and fixed..."). Cross-checks both/
        # one-sided/neither-current cases against brute-force Pearson and the
        # no-cache path, directly on rows (the [s1,s2,t1,t2,w] int64 shape).
        rng = np.random.default_rng(20260706)
        n_series = 6
        n_cols = 50
        w = 32
        data = rng.normal(size=(n_series, n_cols))

        tail = data[:, -w:]
        raw_sums = tail.sum(axis=1)
        raw_sums_sq = (tail ** 2).sum(axis=1)
        raw_sums_cu = (tail ** 3).sum(axis=1)
        raw_sums_qu = (tail ** 4).sum(axis=1)

        cases = [
            (0, 1, n_cols - w, n_cols - w),
            (2, 3, n_cols - w, n_cols - w - 5),
            (4, 5, n_cols - w - 5, n_cols - w),
            (1, 4, n_cols - w - 3, n_cols - w - 3),
        ]
        rows = np.array([[s1, s2, t1, t2, w] for s1, s2, t1, t2 in cases], dtype=np.int64)
        expected = [
            float(np.corrcoef(data[s1, t1:t1 + w], data[s2, t2:t2 + w])[0, 1])
            for s1, s2, t1, t2 in cases
        ]

        _, corrs_no_cache, _, _, _ = candidate_kernels.validate_corr_rows(data, rows, 0, 0.0, False)
        _, corrs_with_cache, _, _, _ = candidate_kernels.validate_corr_rows(
            data, rows, 0, 0.0, False, current_window_sums=raw_sums, current_window_sums_sq=raw_sums_sq, current_window_sums_cu=raw_sums_cu, current_window_sums_qu=raw_sums_qu, current_window_size=w,
        )

        for exp, no_cache_corr, with_cache_corr in zip(expected, corrs_no_cache, corrs_with_cache):
            self.assertAlmostEqual(no_cache_corr, exp, places=9)
            self.assertAlmostEqual(with_cache_corr, exp, places=9)

    def test_validate_corr_batch_current_window_cache_matches_bruteforce(self):
        # (2026-07-06) "point 1", default (non-hybrid) validation path:
        # candidate_kernels.validate_corr_batch gained the same
        # current_window_sums* cache parameters as HybridValidationCache
        # (see test_hybrid_validation_current_window_cache_matches_bruteforce
        # below), keyed by per-row x_series/y_series + x_is_current/
        # y_is_current flags instead of a single (s1, s2, t1, t2) tuple per
        # row, since this function receives pre-sliced x/y batches with no
        # series/time metadata of its own. Cross-checks both/one-sided/
        # neither-current cases against brute-force Pearson and the no-cache
        # path. See docs/implementation_log.md.
        rng = np.random.default_rng(20260706)
        n_series = 6
        n_cols = 50
        w = 32
        data = rng.normal(size=(n_series, n_cols))

        tail = data[:, -w:]
        raw_sums = tail.sum(axis=1)
        raw_sums_sq = (tail ** 2).sum(axis=1)
        raw_sums_cu = (tail ** 3).sum(axis=1)
        raw_sums_qu = (tail ** 4).sum(axis=1)

        cases = [
            (0, 1, n_cols - w, n_cols - w),
            (2, 3, n_cols - w, n_cols - w - 5),
            (4, 5, n_cols - w - 5, n_cols - w),
            (1, 4, n_cols - w - 3, n_cols - w - 3),
        ]
        x_batch = np.array([data[s1, t1:t1 + w] for s1, s2, t1, t2 in cases])
        y_batch = np.array([data[s2, t2:t2 + w] for s1, s2, t1, t2 in cases])
        x_series = np.array([c[0] for c in cases], dtype=np.int64)
        y_series = np.array([c[1] for c in cases], dtype=np.int64)
        x_is_current = np.array([1 if c[2] + w == n_cols else 0 for c in cases], dtype=np.uint8)
        y_is_current = np.array([1 if c[3] + w == n_cols else 0 for c in cases], dtype=np.uint8)

        expected = [
            float(np.corrcoef(data[s1, t1:t1 + w], data[s2, t2:t2 + w])[0, 1])
            for s1, s2, t1, t2 in cases
        ]

        res_no_cache = candidate_kernels.validate_corr_batch(x_batch, y_batch, 0.0, False)
        res_with_cache = candidate_kernels.validate_corr_batch(
            x_batch, y_batch, 0.0, False, current_window_sums=raw_sums, current_window_sums_sq=raw_sums_sq, current_window_sums_cu=raw_sums_cu, current_window_sums_qu=raw_sums_qu, x_series=x_series, y_series=y_series, x_is_current=x_is_current, y_is_current=y_is_current,
        )

        for exp, no_cache_row, with_cache_row in zip(expected, res_no_cache, res_with_cache):
            self.assertAlmostEqual(no_cache_row[1], exp, places=9)
            self.assertAlmostEqual(with_cache_row[1], exp, places=9)

    def test_hybrid_validation_current_window_cache_matches_bruteforce(self):
        # (2026-07-06) "point 1": HybridValidationCache.validate_pairs' exact
        # (non-repeat) Pearson fallback can read a pair's "current window"
        # side (sx/sx2/sx3/sx4, or the sy equivalents) directly from the
        # per-series incrementally-maintained raw-window moments instead of
        # re-accumulating them in the O(w) loop, since every candidate pair
        # has at least one side anchored at the just-arrived window. This
        # cross-checks the new current_window_sums* cache parameters against
        # both a from-scratch brute-force Pearson computation and the
        # no-cache path, across: both sides current, only side1 current,
        # only side2 current, and neither side current (cache present but
        # unused). See docs/implementation_log.md.
        rng = np.random.default_rng(20260706)
        n_series = 6
        n_cols = 50
        w = 32
        data = rng.normal(size=(n_series, n_cols))

        tail = data[:, -w:]
        raw_sums = tail.sum(axis=1)
        raw_sums_sq = (tail ** 2).sum(axis=1)
        raw_sums_cu = (tail ** 3).sum(axis=1)
        raw_sums_qu = (tail ** 4).sum(axis=1)

        cases = [
            (0, 1, n_cols - w, n_cols - w),          # both sides current
            (2, 3, n_cols - w, n_cols - w - 5),       # side1 current only
            (4, 5, n_cols - w - 5, n_cols - w),       # side2 current only
            (1, 4, n_cols - w - 3, n_cols - w - 3),   # neither current
        ]
        series1 = np.array([c[0] for c in cases], dtype=np.int64)
        series2 = np.array([c[1] for c in cases], dtype=np.int64)
        curr_t1 = np.array([c[2] for c in cases], dtype=np.int64)
        curr_t2 = np.array([c[3] for c in cases], dtype=np.int64)
        window_sizes = np.full(len(cases), w, dtype=np.int64)

        expected = [
            float(np.corrcoef(data[s1, t1:t1 + w], data[s2, t2:t2 + w])[0, 1])
            for s1, s2, t1, t2 in cases
        ]

        cache_no = candidate_kernels.HybridValidationCache(1024)
        out_no_cache = cache_no.validate_pairs(
            data, series1, series2, curr_t1, curr_t2, window_sizes, 0, 8, 0.0, False,
        )

        cache_yes = candidate_kernels.HybridValidationCache(1024)
        out_with_cache = cache_yes.validate_pairs(
            data, series1, series2, curr_t1, curr_t2, window_sizes, 0, 8, 0.0, False, current_window_sums=raw_sums, current_window_sums_sq=raw_sums_sq, current_window_sums_cu=raw_sums_cu, current_window_sums_qu=raw_sums_qu, current_window_size=w,
        )

        for i, exp in enumerate(expected):
            self.assertAlmostEqual(out_no_cache["corr"][i], exp, places=9)
            self.assertAlmostEqual(out_with_cache["corr"][i], exp, places=9)

    def test_partition_sketches_full_vector_const_check_matches_per_row_norm(self):
        # (2026-07-06) Part 2.2: Sketches.partition_sketches' full_vector_candidates
        # branch used to call np.linalg.norm(vec) once per row inside a Python
        # loop to detect const/zero-norm sketch rows. Since self._sketch_matrix
        # is already unit-normalized upstream (in _sketches_from_scratch/
        # _incremental_sketches, via _normalize_sketch_matrix or the Cython
        # build_sketch_matrix/apply_orth_and_normalize kernels -- both write an
        # all-zero row exactly when the source row was const/zero-norm, and
        # divide by the row's own norm otherwise), a valid row's norm is always
        # exactly 1.0 post-normalization -- so this check can be computed once,
        # vectorized over all rows, instead of once per row. This test builds a
        # Sketches instance directly with a mix of unit-norm and all-zero rows
        # and checks the resulting is_const flags and vectors match exactly
        # what a naive per-row norm check would produce. See
        # docs/implementation_log.md.
        rng = np.random.default_rng(20260706)
        n_series = 6
        n_vectors = 8
        matrix = rng.normal(size=(n_series, n_vectors))
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix[2] = 0.0  # zero-norm row (e.g. a const source series)
        matrix[5] = 0.0

        sk = Sketches.__new__(Sketches)
        sk.series_ids = [f"s{i}" for i in range(n_series)]
        sk._sid_lookup = {sid: i for i, sid in enumerate(sk.series_ids)}
        sk._sketch_keys = [(sid, 0, 16) for sid in sk.series_ids]
        sk._sketch_matrix = matrix
        sk.sketches = dict(zip(sk._sketch_keys, matrix))
        sk.n_grids = 1
        sk.grid_dimensions = n_vectors
        sk.full_vector_candidates = True
        sk._const_flags = None
        sk.is_constant = {}

        sk.partition_sketches(16)

        self.assertEqual(len(sk.partitions), 1)
        partition = sk.partitions[0]
        for i, key in enumerate(sk._sketch_keys):
            expected_const = np.linalg.norm(matrix[i]) == 0.0
            vec, is_const, _ = partition[key]
            self.assertEqual(is_const, expected_const)
            self.assertTrue(np.array_equal(vec, matrix[i]))

    def test_candidate_key_mode_sampled_sketch_uses_sampled_projection(self):
        candidates = Candidates(
            n_lagged_windows=3, grid_dimension=2, cell_size=0.05, grid_max=1.0, freq_threshold=0, corr_threshold=0.7, n_vectors=2, sketch_std=1.0, n_grids=1, neg_corr=False, full_vector=True, candidate_backend="lsh_sign_dot", candidate_similarity="cosine", candidate_cosine_threshold=0.5, candidate_key_mode="sampled_sketch", candidate_key_seed=11, return_distances=False,
        )
        raw_vectors = np.array([[0.6, 0.8], [1.0, 0.0], [-0.6, -0.8]], dtype=np.float64)
        keys = candidates._candidate_key_values(raw_vectors, 2)
        self.assertEqual(keys.shape, (3, 2))
        self.assertAlmostEqual(keys[0, 0], 1.0)
        self.assertFalse(np.allclose(keys[:, 0], raw_vectors[:, 0]))

    def test_proxy_anchor_reads_numeric_candidate_rows(self):
        rng = np.random.default_rng(1)
        values = np.vstack(
            [
                np.sin(np.arange(16) / 2.0),
                np.sin(np.arange(16) / 2.0) + 0.01 * rng.normal(size=16),
                rng.normal(size=16),
            ]
        )
        train_data = np.vstack([np.arange(16), values])
        ids = np.array(["a", "b", "c"])
        optimizer = CorrTrack_optimize(
            train_data,
            ids, window_size=4, window_step=4, n_lags=4, corr_threshold=0.7, recall_by_window=True, alg="nD", neg_corr=False, corr_val=False, exec="sequential", parallel_sketch=False, parallel_candidates=False, parallel_validation=False, candidate_similarity="cosine", candidate_cosine_threshold=-1.0, proxy_config={"anchor_count": 1, "max_pair_rows": 100, "bootstrap_enabled": False},
        )
        param_combo = {
            "n_vectors": 4,
            "grid_dimension": 4,
            "cell_size": 1.0,
            "seed": 1,
            "seed_toggle": 2,
            "preprocess": False,
            "candidate_backend": "lsh_sign_dot",
            "candidate_similarity": "cosine",
            "candidate_cosine_threshold": -1.0,
        }
        optimizer._proxy_reference = optimizer._prepare_proxy_anchor_reference()
        corrtrack, feature_kwargs = optimizer._proxy_build_corrtrack_from_combo(param_combo)
        partitions = optimizer._proxy_partitions_for_corrtrack(corrtrack, optimizer._proxy_reference)
        candidate_keys = optimizer._proxy_candidate_keys_for_anchor(
            corrtrack,
            partitions,
            optimizer._proxy_reference["anchors"][0],
            feature_kwargs,
        )
        self.assertGreater(len(candidate_keys), 0)
        # (2026-09-20) numeric candidate rows and a numeric reference: every candidate row of this anchor is a
        # reference row (the proxy universe is the same candidate universe the search sees)
        self.assertEqual(np.asarray(candidate_keys).shape[1], 4)
        mask = optimizer._proxy_rows_to_candidate_mask(optimizer._proxy_reference, candidate_keys)
        self.assertGreater(int(mask.sum()), 0)
        record = optimizer._run_corrtrack_proxy_anchor((0, param_combo, "smoke"))
        self.assertEqual(record["status"], "success")
        self.assertGreater(record["cand_w"], 0)
        self.assertGreater(record["proxy_search_index_candidates"], 0)
        self.assertGreaterEqual(record["proxy_search_objective_rate"], record["proxy_candidate_rate"])

    def test_proxy_anchor_reference_labels_follow_preprocess(self):
        # (2026-09-19) the proxy hyperopt truth must live in the space the settings sketch: with
        # preprocess=True the labels are Pearson correlations of the DIFFERENCED windows (d[t] = x[t] -
        # x[t-1], d[0] = 0, the same rule as Sketches._preprocess_data with last_origin). Before the fix
        # the reference was always raw-space and a differenced grid was scored against the wrong truth.
        rng = np.random.default_rng(7)
        n = 64
        walk_a = np.cumsum(rng.normal(size=n)); walk_b = np.cumsum(rng.normal(size=n))
        # two random walks whose LEVELS are highly correlated (shared drift) but whose increments are not
        drift = np.linspace(0, 40, n)
        values = np.vstack([walk_a + drift, walk_b + drift, rng.normal(size=n)])
        train_data = np.vstack([np.arange(n), values])
        ids = np.array(["a", "b", "c"])
        optimizer = CorrTrack_optimize(
            train_data, ids, window_size=16, window_step=16, n_lags=0, corr_threshold=0.9, recall_by_window=True, alg="nD",
            neg_corr=False, corr_val=False, exec="sequential", parallel_sketch=False, parallel_candidates=False,
            parallel_validation=False, candidate_similarity="cosine", candidate_cosine_threshold=-1.0,
            proxy_config={"anchor_count": 4, "max_pair_rows": 1000, "bootstrap_enabled": False},
        )
        raw = optimizer._prepare_proxy_anchor_reference(preprocess=False)
        diff = optimizer._prepare_proxy_anchor_reference(preprocess=True)
        np.testing.assert_array_equal(raw["pair_rows"], diff["pair_rows"])
        d_values = np.diff(np.concatenate([values[:, :1], values], axis=1), axis=1)
        w = raw["window_size"]
        for space, ref, vals in (("raw", raw, values), ("diff", diff, d_values)):
            for (ia, ib, t_a, t_b), gt in zip(ref["pair_rows"].tolist(), ref["truth"]):
                xa = vals[ia, int(t_a): int(t_a) + int(w)]; xb = vals[ib, int(t_b): int(t_b) + int(w)]
                r = np.corrcoef(xa, xb)[0, 1]
                self.assertEqual(bool(gt), bool(np.isfinite(r) and r >= 0.9), msg=f"{space} {(ia, ib, t_a, t_b)} r={r:.3f}")
        # the sorted key view and the row matching: every reference row matches itself, a foreign row nothing
        mask = optimizer._proxy_rows_to_candidate_mask(raw, raw["pair_rows"][:, [1, 0, 3, 2]])   # swapped order canonicalizes back
        self.assertTrue(bool(mask.all()))
        self.assertFalse(optimizer._proxy_rows_to_candidate_mask(raw, np.array([[0, 1, 10**6, 10**6]])).any())
        # the shared drift makes the level truth denser than the increment truth
        self.assertGreater(int(raw["n_gt"]), int(diff["n_gt"]))

    def test_incremental_pair_validator_matches_validate_corr_rows(self):
        # (2026-09-19) candidate_kernels.IncrementalPairValidator: exact roll-forward of repeated candidate
        # pairs (sorted-merge state, leaving points from its own history when the buffer no longer holds
        # them). Over a simulated stream with L = 1 (history path) and L = 3 (buffer path), with repeated,
        # churned and duplicated rows, a row whose lag is not a multiple of the step, and forced refreshes,
        # every output must equal validate_corr_rows' (accepted, constants, spiked exactly; corr to 1e-9).
        import candidate_kernels as ck
        rng = np.random.default_rng(11)
        m, n = 40, 1200
        data_full = np.cumsum(rng.normal(size=(m, n)), axis=1)
        data_full[1] = data_full[0] + 0.05 * rng.normal(size=n)                 # a strongly correlated pair
        data_full[2, :] = 3.0                                                     # a constant series
        data_full[3, 600] += 200.0                                                # a spike
        for W, step, L in ((48, 6, 1), (48, 6, 3), (30, 5, 1)):
            v = ck.IncrementalPairValidator()
            buf_cols = W + (L - 1) * step
            keep = rng.choice(m * (m - 1) // 2, size=300, replace=False)
            pairs_all = np.array([(a, b) for a in range(m) for b in range(a + 1, m)])
            lag_of = rng.integers(0, L, size=len(pairs_all)) * step
            n_rolled_total = 0
            for s in range(0, 150):
                start = s * step
                if start + buf_cols > n:
                    break
                churn = rng.choice(len(pairs_all), size=40, replace=False)
                idx = np.concatenate([keep, churn, keep[:5]])                     # duplicates on purpose
                P = pairs_all[idx]
                t1 = np.full(len(P), start + (L - 1) * step); t2 = t1 - lag_of[idx]
                rows = np.column_stack([P[:, 0], P[:, 1], t1, t2, np.full(len(P), W)]).astype(np.int64)
                if L > 1:
                    rows[0, 3] = rows[0, 2] - 1                                    # lag not a multiple of the step
                rows = np.ascontiguousarray(rows)
                window = np.ascontiguousarray(data_full[:, start:start + buf_cols])
                a1, c1, d1, k1, p1 = ck.validate_corr_rows(window, rows, start, 0.8, False)
                a2, c2, d2, k2, p2, nr = v.validate_rows(window, rows, start, step, 0.8, False, 1e-3, 5.0, 4, 8, (L - 1) * step)
                np.testing.assert_array_equal(a1, a2); np.testing.assert_array_equal(k1, k2); np.testing.assert_array_equal(p1, p2)
                ok = np.isfinite(c1)
                np.testing.assert_allclose(c1[ok], c2[ok], atol=1e-9, rtol=0)
                np.testing.assert_allclose(d1[ok], d2[ok], atol=1e-7, rtol=0)
                n_rolled_total += nr
            self.assertGreater(n_rolled_total, 0, f"W={W} step={step} L={L}: nothing was rolled forward")
            self.assertGreater(v.rows_rolled / max(1, v.rows_seen), 0.5)
        # a hybrid_validation="auto" CorrTrack resolves by window size
        _resolve_hybrid_validation_flag = library_corrtrack_parallel._resolve_hybrid_validation_flag
        self.assertEqual(_resolve_hybrid_validation_flag("auto", 60), False)
        self.assertEqual(_resolve_hybrid_validation_flag("auto", 168), True)
        self.assertEqual(_resolve_hybrid_validation_flag("auto"), "auto")

    def test_proxy_series_subsample_aligns_sketches_with_reference(self):
        # (2026-09-20) with series_subsample_max_pairs binding, the reference is built on a stratified subset of
        # the series; the sketch/candidate side must be built on the SAME rows (reference["train_rows"]). Before
        # the fix the sketches covered all series under the subset's ids and the proxy recall collapsed.
        rng = np.random.default_rng(3)
        m, n = 60, 400
        base = np.cumsum(rng.normal(size=(6, n)), axis=1)
        values = np.vstack([base[i % 6] + 0.2 * np.cumsum(rng.normal(size=n)) for i in range(m)])
        train_data = np.vstack([np.arange(n), values])
        ids = np.array([f"s{i}" for i in range(m)])
        optimizer = CorrTrack_optimize(
            train_data, ids, window_size=40, window_step=8, n_lags=16, corr_threshold=0.9, recall_by_window=True, alg="nD",
            neg_corr=False, corr_val=False, exec="sequential", parallel_sketch=False, parallel_candidates=False,
            parallel_validation=False, candidate_similarity="cosine", candidate_cosine_threshold=0.7,
            proxy_config={"anchor_count": 4, "max_pair_rows": 200000, "bootstrap_enabled": False, "series_subsample_max_pairs": 800},
        )
        ref = optimizer._prepare_proxy_anchor_reference(preprocess=False)
        optimizer._proxy_reference = ref
        self.assertIsNotNone(ref["train_rows"])
        self.assertLess(len(ref["proxy_ids"]), m)
        self.assertEqual(len(ref["train_rows"]), len(ref["proxy_ids"]) + 1)
        combo = {"n_vectors": 64, "seed": 2468, "seed_toggle": 1357, "preprocess": False, "candidate_backend": "lsh_sign_dot",
                 "candidate_cosine_threshold_offset": 0.2, "candidate_lsh_target_occupancy": 3.0, "candidate_apply_hamming_filter": False,
                 "candidate_apply_dot_gamma_filter": True}
        ct, fk = optimizer._proxy_build_corrtrack_from_combo(combo)
        parts = optimizer._proxy_partitions_for_corrtrack(ct, ref)
        cand = np.concatenate([np.asarray(optimizer._proxy_candidate_keys_for_anchor(ct, parts, a, fk)).reshape((-1, 4)) for a in ref["anchors"]])
        self.assertGreater(cand.shape[0], 0)
        mask = optimizer._proxy_rows_to_candidate_mask(ref, cand)
        # every candidate row is a reference row (same universe, same series), and the truth is mostly recovered
        self.assertEqual(int(mask.sum()), int(np.unique(library_corrtrack_parallel._proxy_rows_as_keys(library_corrtrack_parallel._proxy_canonical_pair_rows(cand))).size))
        recall = float((mask & ref["truth"]).sum() / max(1, ref["truth"].sum()))
        self.assertGreater(recall, 0.8, f"proxy recall {recall:.3f} with a series subsample")

    def test_lagged_extension_of_tsubasa_and_corrjoin_matches_bruteforce(self):
        # (2026-09-23, user) TSUBASA and CorrJoin are synchronous as published; both are now run on
        # lagged cells as a disclosed extension (supports_lags="enabled_by_us"). Neither gains a new
        # mechanism, so both must reproduce the bruteforce LAGGED pair set exactly: TSUBASA because its
        # Lemma 1 holds between segments shifted by whole basic windows, CorrJoin because its two
        # Euclidean filters are false-negative-free and its own line 14 verifies every survivor.
        import tempfile
        rng = np.random.default_rng(19)
        m, n, W, step, n_lags, T = 24, 700, 48, 6, 18, 0.8
        base_sig = np.cumsum(rng.normal(size=n + n_lags))
        values = []
        for i in range(m):
            shift = (i % 4) * 6                                   # planted lags at 0, 6, 12, 18
            values.append(base_sig[n_lags - shift: n_lags - shift + n] + 0.35 * rng.normal(size=n))
        data = np.vstack([np.arange(n), np.vstack(values)])
        ids = np.array([f"s{i}" for i in range(m)])
        base = dict(window_size=W, window_step=step, basic_window=step, n_lags=n_lags, corr_threshold=T,
                    neg_corr=False, preprocess=False, exec="sequential", parallel_sketch=False,
                    parallel_candidates=False, parallel_validation=False, max_workers=0, monitor=False,
                    track_min_dist=True, artifact_mode="final", artifact_buffer_max_rows=250000,
                    artifact_merge_mode="merged", save_only_required_artifacts=True,
                    save_maxlag_artifacts=False, verbose=False, testing=False, validation_metric="pearson")
        with tempfile.TemporaryDirectory() as tmp:
            bf, _, bf_flags = library_corrtrack_parallel.run_and_log_bruteforce(
                "lagcheck", data, ids, dict(base, baseline_mode="bruteforce"), f"{tmp}/bf.csv",
                metadata={"nodes": 0}, recall_by_window=True, verbose=False, testing=False)
            self.assertGreater(int(bf["correlated"]), 0)
            bf_rows = {tuple(r) for r in np.asarray(bf_flags.correlated_rows()[0]).tolist()}
            self.assertGreater(len({r[2] - r[3] for r in bf_rows}), 1, "the fixture must contain lagged pairs")

            ts, _, ts_flags = library_corrtrack_parallel.run_and_log_bruteforce(
                "lagcheck", data, ids, dict(base, baseline_mode="tsubasa"), f"{tmp}/ts.csv",
                metadata={"nodes": 0}, recall_by_window=True, verbose=False, testing=False)
            self.assertEqual({tuple(r) for r in np.asarray(ts_flags.correlated_rows()[0]).tolist()}, bf_rows)
            self.assertEqual(ts["supports_lags"], "enabled_by_us")

            cj_params = dict(n_vectors=W // 4 + W // 2, seed=2468, seed_toggle=1357, preprocess=False,
                             data_representation="sketch_paa_svd", candidate_backend="corrjoin_double_filter",
                             corrjoin_ks=W // 4, corrjoin_ke=W // 2, corrjoin_kb=3)
            cj, _, cj_flags = library_corrtrack_parallel.run_and_log_corrtrack(
                "lagcheck", data, ids, base, cj_params, f"{tmp}/cj.csv", metadata={"nodes": 0, "alg": "corrjoin"},
                recall_by_window=True, corr_val=True, monitor=False, verbose=False, testing=False)
            self.assertEqual({tuple(r) for r in np.asarray(cj_flags.correlated_rows()[0]).tolist()}, bf_rows)
            self.assertEqual(cj["supports_lags"], "enabled_by_us")
            self.assertLess(int(cj["total_candidates"]), int(bf["total_candidates"]))   # it still prunes

        # the alignment guard: with window_step not a multiple of basic_window the lags would misalign
        with self.assertRaises(ValueError):
            library_corrtrack_parallel.Candidates_BF_TSUBASA(
                window_size=W, window_step=step, n_lags=n_lags, corr_threshold=T, basic_window=4)

    def test_kendall_tau_matches_scipy_exactly(self):
        # (2026-07-31) validation_metric="kendall" now routes through
        # candidate_kernels.kendall_tau_cy (Cython, no Python fallback) --
        # explicit user instruction: "I want no hotpaths in python
        # whatsoever." Verify bit-identical (not approximate) against
        # scipy.stats.kendalltau across heavy ties, monotonic-nonlinear,
        # independent, and constant-input data.
        import warnings
        from scipy.stats import kendalltau

        rng = np.random.default_rng(42)
        max_diff = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for trial in range(300):
                n = rng.integers(3, 60)
                mode = trial % 4
                if mode == 0:
                    x = rng.integers(0, 4, size=n).astype(np.float64)
                    y = rng.integers(0, 4, size=n).astype(np.float64)
                elif mode == 1:
                    x = rng.normal(size=n)
                    y = x * 0.7 + rng.normal(size=n) * 0.3
                elif mode == 2:
                    x = rng.normal(size=n)
                    y = rng.normal(size=n)
                else:
                    x = np.full(n, 1.0)
                    y = rng.normal(size=n)
                ours = _kendall_tau(x, y)
                ref = kendalltau(x, y).correlation
                if np.isnan(ours) or np.isnan(ref):
                    self.assertEqual(np.isnan(ours), np.isnan(ref), f"trial={trial}")
                    continue
                max_diff = max(max_diff, abs(ours - ref))
        self.assertLess(max_diff, 1e-9)

    def test_spearman_rho_matches_scipy_exactly(self):
        # (2026-07-31) Same rationale as test_kendall_tau_matches_scipy_
        # exactly, for validation_metric="spearman" ->
        # candidate_kernels.spearman_rho_cy.
        import warnings
        from scipy.stats import spearmanr

        rng = np.random.default_rng(43)
        max_diff = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for trial in range(300):
                n = rng.integers(3, 60)
                mode = trial % 4
                if mode == 0:
                    x = rng.integers(0, 4, size=n).astype(np.float64)
                    y = rng.integers(0, 4, size=n).astype(np.float64)
                elif mode == 1:
                    x = rng.normal(size=n)
                    y = x ** 2 + rng.normal(size=n) * 0.1
                elif mode == 2:
                    x = rng.normal(size=n)
                    y = rng.normal(size=n)
                else:
                    x = np.full(n, 1.0)
                    y = rng.normal(size=n)
                ours = _spearman_rho(x, y)
                ref = spearmanr(x, y).correlation
                if np.isnan(ours) or np.isnan(ref):
                    self.assertEqual(np.isnan(ours), np.isnan(ref), f"trial={trial}")
                    continue
                max_diff = max(max_diff, abs(ours - ref))
        self.assertLess(max_diff, 1e-9)

    def test_kendall_and_spearman_have_no_python_fallback(self):
        # (2026-07-31) Explicit user instruction: "I want no hotpaths in
        # python whatsoever" -- kendall/spearman validation must fail
        # loudly if the compiled Cython extension is ever unavailable, not
        # silently fall back to a (removed) Python/scipy implementation.
        # Matches this project's existing strict pattern for
        # distance_corr_sketch_proxy/candidate_backend="flat"/
        # "lsh_sign_dot".
        original_kendall = library_corrtrack_parallel._cy_kendall_tau
        original_spearman = library_corrtrack_parallel._cy_spearman_rho
        try:
            library_corrtrack_parallel._cy_kendall_tau = None
            library_corrtrack_parallel._cy_spearman_rho = None
            with self.assertRaises(RuntimeError):
                _kendall_tau(np.array([1.0, 2.0, 3.0]), np.array([3.0, 1.0, 2.0]))
            with self.assertRaises(RuntimeError):
                _spearman_rho(np.array([1.0, 2.0, 3.0]), np.array([3.0, 1.0, 2.0]))
        finally:
            library_corrtrack_parallel._cy_kendall_tau = original_kendall
            library_corrtrack_parallel._cy_spearman_rho = original_spearman

    def test_dist_corr_fast_matches_naive_and_python_reference(self):
        # (2026-07-31) dist_corr_algorithm="fast" ported to Cython
        # (candidate_kernels.distance_correlation_1d_fast_cy) -- same
        # rationale as Kendall/Spearman above: the pure-Python "fast"
        # version's cross-term sweep drove 4 pure-Python _FenwickTree
        # objects through an interpreted per-element loop. Verified against
        # BOTH the naive O(w^2) estimator and the pure-Python "fast"
        # reference across ties/linear/nonlinear data (the same 3 data
        # kinds test_huo_szekely_fast_matches_naive_distance_correlation_
        # exactly already covers) -- constant-input (0/0 boundary) cases
        # are deliberately excluded here since that instability is a
        # pre-existing property of the algorithm itself, already present
        # between the pure-Python "fast" version and naive, not something
        # this port introduces (see docs/implementation_log.md's 2026-07-31
        # entry for the direct A/B confirmation).
        from huo_szekely_distance_correlation import distance_correlation_1d_fast as py_fast

        rng = np.random.default_rng(0)
        max_diff_vs_py = 0.0
        max_diff_vs_naive = 0.0
        for trial in range(500):
            n = rng.integers(2, 60)
            kind = trial % 3
            if kind == 0:
                x = rng.integers(0, 5, size=n).astype(float)
                y = rng.integers(0, 5, size=n).astype(float)
            elif kind == 1:
                x = rng.normal(size=n)
                y = 0.5 * x + rng.normal(size=n) * 0.5
            else:
                x = rng.normal(size=n)
                y = x ** 2 + rng.normal(size=n) * 0.1
            ours = _dist_corr_fast(x, y)
            py = py_fast(x, y)
            slow = _distance_correlation_1d(x, y)
            self.assertEqual(np.isnan(ours), np.isnan(py), f"trial={trial}")
            if not np.isnan(ours):
                max_diff_vs_py = max(max_diff_vs_py, abs(ours - py))
            if not np.isnan(slow):
                max_diff_vs_naive = max(max_diff_vs_naive, abs(ours - slow))
        self.assertLess(max_diff_vs_py, 1e-9)
        self.assertLess(max_diff_vs_naive, 1e-9)

    def test_dist_corr_fast_edge_cases(self):
        self.assertTrue(np.isnan(_dist_corr_fast([1.0], [2.0])))
        self.assertTrue(np.isnan(_dist_corr_fast([5.0] * 10, [3.0] * 10)))
        self.assertAlmostEqual(_dist_corr_fast([1.0, 2.0], [3.0, 4.0]), 1.0, places=9)

    def test_dist_corr_fast_has_no_python_fallback(self):
        # (2026-07-31) Same "no hotpaths in python whatsoever" instruction
        # as Kendall/Spearman -- dist_corr_algorithm="fast" must fail
        # loudly if the compiled Cython extension is unavailable.
        original = library_corrtrack_parallel._cy_distance_correlation_1d_fast
        try:
            library_corrtrack_parallel._cy_distance_correlation_1d_fast = None
            with self.assertRaises(RuntimeError):
                _dist_corr_fast(np.array([1.0, 2.0, 3.0]), np.array([3.0, 1.0, 2.0]))
        finally:
            library_corrtrack_parallel._cy_distance_correlation_1d_fast = original

    def test_validate_corr_rows_nonlinear_bulk_matches_per_row_reference(self):
        # (2026-07-31) _validate_numeric_rows_nonlinear used to iterate
        # candidate rows via a Python for-loop, calling the (now-Cython)
        # metric function once per row -- Python-level loop/slicing/gate-
        # check overhead on every candidate. Replaced with ONE call to
        # candidate_kernels.validate_corr_rows_nonlinear, which handles the
        # whole rows array in a single nogil loop, matching Pearson's own
        # single-call design (_cy_validate_corr_rows). This test locks in
        # bit-exact (kendall/spearman) or floating-point-noise-level
        # (dist_corr naive/fast, different summation order) agreement
        # against the OLD per-row logic, replicated here standalone, across
        # all 4 metric/algorithm combinations -- including near-constant
        # and structurally-spiked series, to confirm the copied constant/
        # spike check formulas (verified algebraically identical to
        # _is_near_constant_stats/_is_structurally_spiked_stats before
        # this test was written) actually behave identically too.
        import candidate_kernels as ck

        def old_per_row(window_data, rows, window_start, metric, dist_corr_algorithm, threshold, neg_corr):
            n = rows.shape[0]
            accepted = np.zeros(n, dtype=np.uint8)
            corrs = np.full(n, np.nan)
            valid = np.zeros(n, dtype=np.uint8)
            for i in range(n):
                s1, s2, t1, t2, w = (int(v) for v in rows[i])
                t1_idx = t1 - window_start
                t2_idx = t2 - window_start
                if t1_idx < 0 or t2_idx < 0:
                    continue
                x = np.asarray(window_data[s1, t1_idx:t1_idx + w], dtype=np.float64)
                y = np.asarray(window_data[s2, t2_idx:t2_idx + w], dtype=np.float64)
                if x.shape[0] != w or y.shape[0] != w:
                    continue
                valid[i] = 1
                corr, dist, stats = _metric_corr_and_dist(
                    x, y, metric=metric, return_stats=True, dist_corr_algorithm=dist_corr_algorithm
                )
                n_, mean_x, mean_y, var_x, var_y = stats
                if _is_near_constant_stats(var_x, n_, std_thresh=1e-3) or _is_near_constant_stats(var_y, n_, std_thresh=1e-3):
                    continue
                if (
                    _is_structurally_spiked_stats(x, mean_x, var_x, n_, kurt_thresh=5.0)
                    or _is_structurally_spiked_stats(y, mean_y, var_y, n_, kurt_thresh=5.0)
                ):
                    continue
                corrs[i] = corr
                if _passes_corr_threshold(corr, threshold, neg_corr):
                    accepted[i] = 1
            return accepted, corrs, valid

        rng = np.random.default_rng(7)
        n_series = 12
        length = 500
        window_start = 1000
        data = rng.normal(size=(n_series, length))
        data[1] = data[0] * 0.8 + rng.normal(scale=0.3, size=length)
        t = np.linspace(-2, 2, length)
        data[3] = t
        data[4] = t ** 3
        data[5] = np.full(length, 3.0) + rng.normal(scale=1e-6, size=length)
        spike = rng.normal(scale=0.1, size=length)
        spike[::37] += 50
        data[6] = spike

        rows_list = []
        for w in (16, 32, 64):
            for s1 in range(n_series):
                for s2 in range(s1 + 1, n_series):
                    rows_list.append([s1, s2, window_start, window_start, w])
        rows = np.array(rows_list, dtype=np.int64)

        for metric, algo, metric_id in (
            ("spearman", "naive", 0),
            ("kendall", "naive", 1),
            ("dist_corr", "naive", 2),
            ("dist_corr", "fast", 3),
        ):
            old_acc, old_corr, old_valid = old_per_row(data, rows, window_start, metric, algo, 0.5, True)
            new_acc, new_corr, _new_dist, new_valid, new_const, new_spike = ck.validate_corr_rows_nonlinear(
                np.ascontiguousarray(data, dtype=np.float64),
                np.ascontiguousarray(rows, dtype=np.int64),
                window_start, metric_id, 0.5, True,
            )
            new_acc = np.asarray(new_acc)
            new_corr = np.asarray(new_corr)
            new_valid = np.asarray(new_valid)
            self.assertTrue(np.array_equal(old_valid, new_valid), f"{metric}/{algo}: valid mask mismatch")
            self.assertTrue(np.array_equal(old_acc, new_acc), f"{metric}/{algo}: accepted mismatch")
            both_finite = np.isfinite(old_corr) & np.isfinite(new_corr)
            self.assertTrue(
                np.array_equal(np.isnan(old_corr), np.isnan(new_corr)), f"{metric}/{algo}: nan mismatch"
            )
            if both_finite.any():
                max_diff = np.max(np.abs(old_corr[both_finite] - new_corr[both_finite]))
                self.assertLess(max_diff, 1e-9, f"{metric}/{algo}: corr diff {max_diff}")

        # near-constant (series 5) and structurally-spiked (series 6) rows
        # must actually be flagged, not just coincidentally excluded via
        # the threshold check.
        _acc, _corr, _dist, _valid, constants, spiked = ck.validate_corr_rows_nonlinear(
            np.ascontiguousarray(data, dtype=np.float64),
            np.ascontiguousarray(rows, dtype=np.int64),
            window_start, 1, 0.5, True,
        )
        constants = np.asarray(constants).astype(bool)
        spiked = np.asarray(spiked).astype(bool)
        self.assertTrue((rows[constants][:, :2] == 5).any())
        self.assertTrue((rows[spiked][:, :2] == 6).any())

    def test_validate_corr_rows_nonlinear_has_no_python_fallback(self):
        # (2026-07-31) Same "no hotpaths in python whatsoever" instruction
        # as Kendall/Spearman/dist_corr-fast -- the exact nonlinear
        # validation path must fail loudly if the compiled bulk kernel is
        # unavailable, not silently fall back to the (removed) per-row
        # Python loop.
        rng = np.random.default_rng(1)
        n_steps = 200
        values = rng.normal(size=(6, n_steps))
        values[1] = values[0] * 0.7 + rng.normal(scale=0.3, size=n_steps)
        data = np.vstack([np.arange(n_steps, dtype=np.float64), values])
        ids = [f"s{i}" for i in range(6)]
        ct = CorrTrack(
            window_size=32, basic_window=8, window_step=8, n_vectors=6, n_lags=16,
            corr_threshold=0.3, neg_corr=True, preprocess=False, exec="sequential",
            candidate_backend="brute_force", validation_metric="kendall",
        )
        original = library_corrtrack_parallel._cy_validate_corr_rows_nonlinear
        try:
            library_corrtrack_parallel._cy_validate_corr_rows_nonlinear = None
            step = 8
            with self.assertRaises(RuntimeError):
                for start in range(0, n_steps - step + 1, step):
                    ct.run(data[:, start:start + step], ids, verbose=False, testing=False, corr_val=True, monitor=False)
        finally:
            library_corrtrack_parallel._cy_validate_corr_rows_nonlinear = original

    def test_proxy_anchor_records_validation_metric(self):
        # (2026-07-31) Real gap found: validation_metric was never captured
        # in OPTIM_RESULT_COLUMNS/the hyperopt output record at all, so it
        # never survived into best_params_corrtrack.json -- meaning the
        # FINAL corrtrack run's own validation_metric came entirely from
        # experiment_run_exec_param.py's own default/CLI flag, completely
        # decoupled from whatever PARAM_GRID's validation_metric sweep
        # actually explored during tuning. Fixed by adding "validation_
        # metric" to OPTIM_RESULT_COLUMNS and recording it (from param_combo
        # initially, then from the actual constructed CorrTrack's own
        # resolved attribute, mirroring candidate_backend's exact pattern)
        # in _init_optim_record/_init_proxy_record_from_corrtrack/_run_
        # corrtrack_proxy_anchor. This test locks in that a proxy-anchor
        # trial's record reports the metric it actually validated with.
        self.assertIn("validation_metric", OPTIM_RESULT_COLUMNS)

        rng = np.random.default_rng(3)
        n_series, n_steps = 20, 300
        x0 = rng.normal(0, 1, n_steps)
        x1 = x0 * 0.9 + rng.normal(0, 0.2, n_steps)
        others = rng.normal(0, 1, (n_series - 2, n_steps))
        values = np.vstack([x0, x1, others])
        train_data = np.vstack([np.arange(n_steps), values])
        ids = np.array([f"s{i}" for i in range(n_series)])

        optimizer = CorrTrack_optimize(
            train_data, ids, window_size=64, window_step=16, n_lags=64, corr_threshold=0.7,
            recall_by_window=True, alg="nD", neg_corr=True, corr_val=False, exec="sequential",
            parallel_sketch=False, parallel_candidates=False, parallel_validation=False,
            proxy_config={"anchor_count": 3, "max_pair_rows": 2000, "bootstrap_enabled": False},
        )
        optimizer._proxy_reference = optimizer._prepare_proxy_anchor_reference()

        param_combo = {
            "n_vectors": 64, "seed": 1, "seed_toggle": 2, "preprocess": False,
            "candidate_backend": "lsh_approx", "data_representation": "sketch_concordance",
            "validation_metric": "kendall", "concordance_multichannel_gamma": 0.3,
            "concordance_target_dim": 738,
        }
        record = optimizer._run_corrtrack_proxy_anchor((0, param_combo, "diag"))
        self.assertEqual(record.get("validation_metric"), "kendall")

    def test_run_and_log_corrtrack_best_params_validation_metric_wins_over_base_config(self):
        # (2026-07-31) Direct consequence of the fix above: once best_
        # params_corrtrack.json carries validation_metric, {**base_config,
        # **run_params}'s existing merge order (run_params/best_params
        # already wins on key collision) should make the TUNED metric win
        # automatically over experiment_run_exec_param.py's own default --
        # no additional precedence code needed in corrtrack_run_corrtrack.py
        # itself. This test verifies that merge behaves as expected, not
        # just that the dict spread syntax looks right.
        n_series, n_steps = 4, 200
        rng = np.random.default_rng(0)
        values = rng.normal(size=(n_series, n_steps))
        data = np.vstack([np.arange(n_steps, dtype=np.float64), values])
        ids = [f"s{i}" for i in range(n_series)]

        base_config = dict(
            window_size=32, window_step=8, basic_window=None, n_lags=16,
            corr_threshold=0.5, neg_corr=True, exec="sequential",
            parallel_sketch=False, parallel_candidates=False, parallel_validation=False,
            max_workers=0, validation_metric="pearson",
        )
        run_params = dict(
            n_vectors=4, seed=1, seed_toggle=2, preprocess=False,
            candidate_backend="lsh_approx", validation_metric="kendall",
        )

        captured = {}
        original_init = CorrTrack.__init__

        def spy_init(self, *args, **kwargs):
            captured["validation_metric"] = kwargs.get("validation_metric")
            return original_init(self, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            with unittest.mock.patch.object(CorrTrack, "__init__", spy_init):
                run_and_log_corrtrack(
                    "diag", data, ids, base_config, run_params, os.path.join(tmp, "run.csv"),
                )
        self.assertEqual(captured.get("validation_metric"), "kendall")

    def test_bruteforce_loads_tuned_validation_metric_from_best_params(self):
        # (2026-07-31) corrtrack_run_bruteforce.py's own new best_params
        # reader: no explicit --validation-metric override, so the tuned
        # value (from the hyperparameter-search step's best_params_
        # corrtrack.json) should win over the exec-param/CLI default.
        # Falls back to None (caller keeps its own default) when the file
        # is absent, unparseable, or lacks the key.
        with tempfile.TemporaryDirectory() as tmp:
            optim_dir = os.path.join(tmp, "optim")
            os.makedirs(optim_dir)

            self.assertIsNone(corrtrack_run_bruteforce._load_tuned_validation_metric(optim_dir))

            with open(os.path.join(optim_dir, "best_params_corrtrack.json"), "w") as f:
                json.dump({"validation_metric": "kendall", "candidate_backend": "lsh_approx"}, f)
            self.assertEqual(
                corrtrack_run_bruteforce._load_tuned_validation_metric(optim_dir), "kendall"
            )

            with open(os.path.join(optim_dir, "best_params_corrtrack.json"), "w") as f:
                json.dump({"candidate_backend": "lsh_approx"}, f)
            self.assertIsNone(corrtrack_run_bruteforce._load_tuned_validation_metric(optim_dir))

    def test_run_corrtrack_experiment_brute_runs_after_param(self):
        # (2026-07-31) corrtrack_param_search.py is fully self-contained
        # (proxy-anchor tuning never reads bf_run.csv), while corrtrack_
        # run_bruteforce.py now reads validation_metric FROM best_params_
        # corrtrack.json -- so brute-force must run AFTER hyperparameter
        # search for that reading to ever find anything. Locks in the
        # pipeline reorder so it can't silently regress back.
        import run_corrtrack_experiment
        step_keys = [step_key for _, _, step_key, _ in run_corrtrack_experiment.STEPS]
        self.assertLess(step_keys.index("param"), step_keys.index("brute"))
        self.assertLess(step_keys.index("brute"), step_keys.index("corrtrack"))
        self.assertLess(step_keys.index("corrtrack"), step_keys.index("compare"))

    def test_proxy_anchor_concordance_multichannel_is_gamma_sensitive(self):
        # (2026-07-31) Real bug found via a live production run: data_
        # representation="sketch_concordance"/"sketch_multichannel" bypass
        # Candidates/grid_nodes entirely in the real run() (_run_
        # concordance_multichannel/_run_distance_corr_sketch_
        # multichannel), so the grid_nodes-based proxy-anchor path (built
        # for the Pearson/sketch_proj representation) could not proxy for
        # them at all -- their grid_nodes are only ever built with the
        # placeholder candidate_index_backend="brute_force" (never queried
        # by the real run), so every swept config reported ~100% candidate
        # rate regardless of candidate_lsh_n_bands or anything else. This
        # test locks in the fix: proxy-anchor evaluation must now be
        # genuinely sensitive to concordance_multichannel_gamma (the real
        # lever for this representation), producing DIFFERENT candidate
        # counts at different gamma values, not a constant "touch everyone".
        rng = np.random.default_rng(3)
        n_series, n_steps = 20, 300
        x0 = rng.normal(0, 1, n_steps)
        x1 = x0 * 0.9 + rng.normal(0, 0.2, n_steps)
        others = rng.normal(0, 1, (n_series - 2, n_steps))
        values = np.vstack([x0, x1, others])
        train_data = np.vstack([np.arange(n_steps), values])
        ids = np.array([f"s{i}" for i in range(n_series)])

        optimizer = CorrTrack_optimize(
            train_data, ids, window_size=64, window_step=16, n_lags=64, corr_threshold=0.7,
            recall_by_window=True, alg="nD", neg_corr=True, corr_val=False, exec="sequential",
            parallel_sketch=False, parallel_candidates=False, parallel_validation=False,
            proxy_config={"anchor_count": 3, "max_pair_rows": 2000, "bootstrap_enabled": False},
        )
        optimizer._proxy_reference = optimizer._prepare_proxy_anchor_reference()

        def total_candidates(gamma):
            param_combo = {
                "n_vectors": 64, "seed": 1, "seed_toggle": 2, "preprocess": False,
                "candidate_backend": "lsh_approx", "data_representation": "sketch_concordance",
                "validation_metric": "kendall", "concordance_multichannel_gamma": gamma,
                "concordance_target_dim": 738,
            }
            corrtrack, feature_kwargs = optimizer._proxy_build_corrtrack_from_combo(param_combo)
            self.assertTrue(corrtrack.concordance_multichannel_backend)
            self.assertAlmostEqual(corrtrack.concordance_multichannel_gamma, gamma, places=6)
            partitions = optimizer._proxy_partitions_for_corrtrack(corrtrack, optimizer._proxy_reference)
            total = 0
            for anchor in optimizer._proxy_reference["anchors"]:
                keys = optimizer._proxy_candidate_keys_for_anchor(corrtrack, partitions, anchor, feature_kwargs)
                total += len(keys)
            return total

        loose = total_candidates(0.15)
        tight = total_candidates(0.7)
        # Before the fix, both would have been ~identical (every pair,
        # regardless of gamma) -- a real gate must show a real gap.
        self.assertGreater(loose, tight)

    def test_proxy_anchor_distance_corr_sketch_multichannel_is_gamma_sensitive(self):
        # (2026-07-31) Same bug, same fix, for the dist_corr representation.
        rng = np.random.default_rng(4)
        n_series, n_steps = 20, 200
        x0 = rng.uniform(-1, 1, n_steps)
        x1 = x0 ** 2 + rng.normal(0, 0.05, n_steps)
        others = rng.normal(0, 1, (n_series - 2, n_steps))
        values = np.vstack([x0, x1, others])
        train_data = np.vstack([np.arange(n_steps), values])
        ids = np.array([f"s{i}" for i in range(n_series)])

        optimizer = CorrTrack_optimize(
            train_data, ids, window_size=64, window_step=16, n_lags=64, corr_threshold=0.4,
            recall_by_window=True, alg="nD", neg_corr=True, corr_val=False, exec="sequential",
            parallel_sketch=False, parallel_candidates=False, parallel_validation=False,
            proxy_config={"anchor_count": 3, "max_pair_rows": 2000, "bootstrap_enabled": False},
        )
        optimizer._proxy_reference = optimizer._prepare_proxy_anchor_reference()

        def total_candidates(gamma):
            param_combo = {
                "n_vectors": 64, "seed": 1, "seed_toggle": 2, "preprocess": False,
                "candidate_backend": "lsh_approx", "data_representation": "sketch_multichannel",
                "validation_metric": "dist_corr", "distance_corr_sketch_multichannel_gamma": gamma,
            }
            corrtrack, feature_kwargs = optimizer._proxy_build_corrtrack_from_combo(param_combo)
            self.assertTrue(corrtrack.distance_corr_sketch_multichannel_backend)
            partitions = optimizer._proxy_partitions_for_corrtrack(corrtrack, optimizer._proxy_reference)
            total = 0
            for anchor in optimizer._proxy_reference["anchors"]:
                keys = optimizer._proxy_candidate_keys_for_anchor(corrtrack, partitions, anchor, feature_kwargs)
                total += len(keys)
            return total

        loose = total_candidates(0.1)
        tight = total_candidates(0.5)
        self.assertGreater(loose, tight)

    def test_window_size_reduction_stays_incremental(self):
        rng = np.random.default_rng(123)
        ids = np.array(["a", "b", "c"])
        values = rng.normal(size=(3, 32))
        data = np.vstack([np.arange(32), values])
        kwargs = dict(
            basic_window=4, window_step=4, n_vectors=8, n_lags=8, seed=11, seed_toggle=22, corr_threshold=0.7, exec="sequential", parallel_sketch=False, parallel_candidates=False, parallel_validation=False, candidate_backend="brute_force",
        )
        corrtrack = CorrTrack(window_size=8, **kwargs)
        for start in range(0, 8, 4):
            corrtrack.run(
                data[:, start : start + 4],
                ids, verbose=False, testing=False, corr_val=False, monitor=False,
            )

        node = corrtrack.sketch_nodes[0]
        counts = {"scratch": 0}
        orig_scratch = node._sketches_from_scratch

        def scratch(*args, **kwargs):
            counts["scratch"] += 1
            return orig_scratch(*args, **kwargs)

        node._sketches_from_scratch = scratch

        corrtrack.run(data[:, 8:12], ids, verbose=False, testing=False, corr_val=False, monitor=False)
        corrtrack.update_window_size(4)
        corrtrack.run(data[:, 12:16], ids, verbose=False, testing=False, corr_val=False, monitor=False)
        self.assertEqual(counts["scratch"], 0)
        self.assertEqual(list(node.incrementable_index), [16.0])

        corrtrack.run(data[:, 16:20], ids, verbose=False, testing=False, corr_val=False, monitor=False)
        self.assertEqual(counts["scratch"], 0)
        self.assertEqual(list(node.incrementable_index), [20.0])

        fresh = CorrTrack(window_size=4, **kwargs)
        fresh.run(data[:, 16:20], ids, verbose=False, testing=False, corr_val=False, monitor=False)
        inc_sketch = {key: np.asarray(value) for key, value in node.sketches.items()}
        fresh_sketch = {key: np.asarray(value) for key, value in fresh.sketch_nodes[0].sketches.items()}
        self.assertEqual(sorted(inc_sketch), sorted(fresh_sketch))
        max_diff = max(float(np.max(np.abs(inc_sketch[key] - fresh_sketch[key]))) for key in inc_sketch)
        self.assertEqual(max_diff, 0.0)

    def _proxy_selection_row(self, cand_rate, time_total, recall=0.98, gt_support=100,
                              n_bands=None, recall_ub=None):
        recall_ub = recall if recall_ub is None else recall_ub
        return {
            "hyperopt_strategy": "proxy_anchor",
            "status": "success",
            "proxy_recall_lb": recall,
            "proxy_recall_ub": recall_ub,
            "proxy_recall_mean": recall,
            "proxy_recall_med": recall,
            "recall": recall,
            "candidate_lsh_n_bands": n_bands,
            "proxy_candidate_rate_mean": cand_rate,
            "proxy_candidate_rate_med": cand_rate,
            "proxy_candidate_rate": cand_rate,
            "proxy_search_objective_rate": cand_rate,
            "proxy_search_similarity_work_rate": cand_rate,
            "proxy_search_dot_work_rate": cand_rate,
            "proxy_search_index_rate": cand_rate,
            "proxy_search_unique_pre_dot_pairs": 100,
            "proxy_search_cascade_reject_rate": 0.5,
            "proxy_precision_mean": 0.9,
            "proxy_precision_med": 0.9,
            "precision": 0.9,
            "proxy_specificity_mean": 0.9,
            "proxy_specificity_med": 0.9,
            "specificity": 0.9,
            "proxy_gt_support": gt_support,
            "cand_w": 100,
            "proxy_search_time_total": time_total,
        }

    def test_proxy_selection_prefers_fewer_n_bands(self):
        # (2026-09-04) n_bands is now the PRIMARY ranking signal (replacing candidate-rate,
        # which this project's own real data showed barely varies across configs with very
        # different real cost -- see docs/implementation_log.md's 2026-09-04 entry). A config
        # with fewer n_bands must win even against a faster-measured, more-bands alternative --
        # n_bands is exact and known for free, not a noisy proxy that needs a real-time
        # tie-break to compensate for.
        opt = CorrTrack_optimize.__new__(CorrTrack_optimize)
        opt.proxy_bootstrap_min_gt_events = 0
        opt.proxy_candidate_rate_close_tolerance = 0.05

        metrics = pd.DataFrame(
            [
                self._proxy_selection_row(0.20, time_total=0.01, n_bands=200),  # fewest time, most bands
                self._proxy_selection_row(0.20, time_total=1.00, n_bands=60),   # slowest measured, fewest bands
            ]
        )
        _metrics, best = opt._apply_proxy_anchor_selection(metrics, target_recall=0.95)
        self.assertIsNotNone(best)
        self.assertEqual(int(best["candidate_lsh_n_bands"].iloc[0]), 60)

    def test_proxy_selection_uses_real_time_tiebreak_among_equal_n_bands(self):
        # (2026-09-04) Among configs whose n_bands are EQUAL (a real tie on the primary,
        # exact signal -- no tolerance needed to decide this, unlike the old candidate-rate
        # design), real measured execution time (proxy_search_time_total) still decides,
        # exactly as it always has as a tie-break -- this part of the original 2026-07-05
        # "hyperopt real-time tie-break" intent is preserved, just no longer gated behind an
        # arbitrary candidate-rate closeness tolerance.
        opt = CorrTrack_optimize.__new__(CorrTrack_optimize)
        opt.proxy_bootstrap_min_gt_events = 0
        opt.proxy_candidate_rate_close_tolerance = 0.05

        metrics = pd.DataFrame(
            [
                self._proxy_selection_row(0.20, time_total=1.0, n_bands=100),
                self._proxy_selection_row(0.205, time_total=0.1, n_bands=100),
                self._proxy_selection_row(0.30, time_total=0.01, n_bands=100),
            ]
        )
        _metrics, best = opt._apply_proxy_anchor_selection(metrics, target_recall=0.95)
        self.assertIsNotNone(best)
        self.assertEqual(float(best["proxy_search_time_total"].iloc[0]), 0.01)

    def test_proxy_selection_prefers_narrower_confidence_interval(self):
        # (2026-09-04) A continuous statistical-power signal (bootstrap CI width, ub-lb) now
        # sits alongside the binary underpowered flag -- between two configs tied on n_bands
        # and time, the one with the narrower (more trustworthy) confidence interval should
        # win, not an arbitrary pick.
        opt = CorrTrack_optimize.__new__(CorrTrack_optimize)
        opt.proxy_bootstrap_min_gt_events = 0
        opt.proxy_candidate_rate_close_tolerance = 0.05

        metrics = pd.DataFrame(
            [
                self._proxy_selection_row(0.20, time_total=0.1, n_bands=100, recall=0.96, recall_ub=0.999),
                self._proxy_selection_row(0.20, time_total=0.1, n_bands=100, recall=0.96, recall_ub=0.965),
            ]
        )
        _metrics, best = opt._apply_proxy_anchor_selection(metrics, target_recall=0.95)
        self.assertIsNotNone(best)
        self.assertEqual(float(best["proxy_recall_ub"].iloc[0]), 0.965)

    def test_proxy_mean_timed_trial_averages_repeated_measurements(self):
        opt = CorrTrack_optimize.__new__(CorrTrack_optimize)
        opt.proxy_timing_repeats = 3
        opt._to_float = lambda v: float(v) if v is not None else None

        fake_times = [0.010, 0.020, 0.030]
        calls = []

        def fake_run(args):
            i = len(calls)
            calls.append(args)
            return {
                "status": "success",
                "sk_time": fake_times[i],
                "cand_time": fake_times[i] * 2,
                "runtime": fake_times[i] * 3,
                "optim_bootstrap_time": 0.0,
            }

        opt._run_corrtrack_proxy_anchor = fake_run
        record = opt._run_corrtrack_proxy_anchor_mean_timed(("task",))
        self.assertEqual(len(calls), 3)
        self.assertAlmostEqual(record["sk_time"], sum(fake_times) / 3)
        self.assertAlmostEqual(record["cand_time"], sum(t * 2 for t in fake_times) / 3)
        self.assertEqual(record["proxy_timing_repeats"], 3)

    # (2026-07-06) InstinctIndex -- experimental approximate graph
    # candidate_backend ("instinct"). Unlike every other backend tested in
    # this file, InstinctIndex is NOT recall-preserving by construction --
    # it is a bounded-degree best-first graph search, so these tests
    # measure recall against brute-force ground truth rather than asserting
    # exact agreement. See docs/implementation_log.md, "InstinctIndex:
    # experimental approximate graph backend".

    @staticmethod
    def _instinct_insert(idx, vectors):
        n = vectors.shape[0]
        return idx.insert_many(
            vectors[:, :4].copy(), np.arange(n, dtype=np.int64), vectors,
            np.arange(n, dtype=np.int64), np.zeros(n, dtype=np.int64),
            np.full(n, 32, dtype=np.int64), np.arange(n, dtype=np.int64),
        )

    @staticmethod
    def _lsh_insert(idx, vectors):
        n = vectors.shape[0]
        return idx.insert_many(
            vectors[:, :4].copy(), np.arange(n, dtype=np.int64), vectors,
            np.arange(n, dtype=np.int64), np.zeros(n, dtype=np.int64),
            np.full(n, 32, dtype=np.int64), np.arange(n, dtype=np.int64),
        )

    def test_lsh_sign_dot_index_construct_insert_query_smoke(self):
        rng = np.random.default_rng(9)
        dim = 32
        n = 40
        vectors = rng.normal(size=(n, dim))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

        for n_lagged_windows, n_bands in [(1, 8), (4, 16), (11, 32)]:
            with self.subTest(n_lagged_windows=n_lagged_windows, n_bands=n_bands):
                idx = candidate_kernels.SignLSHBandIndex(
                    n_vectors=dim, initial_capacity=64, n_lagged_windows=n_lagged_windows, n_bands=n_bands,
                )
                entry_ids = self._lsh_insert(idx, vectors)
                self.assertEqual(entry_ids.shape, (n,))
                tau = float(np.sqrt(2.0 - 2.0 * 0.5))
                rows = np.asarray(idx.find_pair_rows_full_cosine(entry_ids, 0.5, tau))
                self.assertEqual(rows.ndim, 2)
                self.assertEqual(rows.shape[1], 5)
                self.assertEqual(rows.dtype, np.int64)
                rows_signed = np.asarray(idx.find_pair_rows_full_cosine_signed(entry_ids, 0.5, tau))
                self.assertEqual(rows_signed.ndim, 2)
                self.assertEqual(rows_signed.shape[1], 5)
                self.assertIsInstance(idx.last_stats, dict)
                self.assertIn("lsh_candidates_touched", idx.last_stats)

    def test_lsh_sign_dot_index_band_width_auto_sizing_scales_with_m_times_l(self):
        # (2026-07-10) band_width is no longer a tunable parameter -- it is
        # computed once, automatically, on the first insert_many batch, as
        # a function of (observed batch size m) * (n_lagged_windows L).
        # This directly tests the property the human needed verified before
        # "selling" the method's complexity: touched-candidates-per-query
        # must NOT scale linearly with alive-window count once band_width
        # is sized correctly for the actual (m, L) -- unlike the old fixed
        # band_width=8 default, which was measured (see
        # docs/implementation_log.md) to give a flat touched/alive ratio of
        # ~0.74 regardless of scale, i.e. Theta(N) per query.
        rng = np.random.default_rng(77)
        dim = 64
        n = 200  # m -- same "m" used for every L below, only L varies
        vectors = rng.normal(size=(n, dim))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        window_idx = np.arange(n, dtype=np.int64)
        sid_idx = np.arange(n, dtype=np.int64)
        time_idx = np.zeros(n, dtype=np.int64)
        window_size = np.full(n, 256, dtype=np.int64)
        sid_rank = sid_idx.copy()

        touched_per_alive = []
        for n_lagged_windows in (1, 4, 16):
            idx = candidate_kernels.SignLSHBandIndex(
                n_vectors=dim, initial_capacity=n, n_lagged_windows=n_lagged_windows, n_bands=64,
            )
            entry_ids = idx.insert_many(
                vectors[:, :4].copy(), window_idx, vectors, sid_idx, time_idx, window_size, sid_rank,
            )
            rows = idx.find_pair_rows_full_cosine_signed(entry_ids, 0.55, 0.7)
            stats = idx.last_stats
            ratio = stats["lsh_candidates_touched"] / stats["num_recent_queries"] / stats["lsh_num_nodes_alive"]
            touched_per_alive.append(ratio)

        # With auto-sizing, bucket_count grows with n_lagged_windows (since
        # expected_alive = m * n_lagged_windows grows), so the touched/alive
        # ratio should NOT stay flat the way it did at a fixed band_width --
        # it should shrink (or at least not grow) as n_lagged_windows grows,
        # since band_width grows to compensate. The old, fixed-band_width
        # bug showed <5% relative variation across a 4x alive-count range;
        # require a real, larger movement here as the regression guard.
        self.assertLess(touched_per_alive[-1], touched_per_alive[0] * 0.9,
                         f"touched/alive ratio should shrink as band_width auto-scales with n_lagged_windows, got {touched_per_alive}")

    def test_lsh_sign_dot_index_notify_expected_n_series_matches_lazy_inference(self):
        # (2026-07-10) notify_expected_n_series(m) is a public post-construction
        # hook -- CorrTrack calls it automatically (from the true, raw series
        # count) the moment it knows m, before any insert happens. Given the
        # same (m, L), sizing via this hook must produce IDENTICAL band_width
        # (hence identical retrieval behavior) to the lazy first-insert-batch
        # fallback -- a consistency guard, not just a smoke test.
        rng = np.random.default_rng(31)
        dim = 64
        n = 50
        vectors = rng.normal(size=(n, dim))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        window_idx = np.arange(n, dtype=np.int64)
        sid_idx = np.arange(n, dtype=np.int64)
        time_idx = np.zeros(n, dtype=np.int64)
        window_size = np.full(n, 256, dtype=np.int64)
        sid_rank = sid_idx.copy()

        idx_explicit = candidate_kernels.SignLSHBandIndex(
            n_vectors=dim, initial_capacity=1024, n_lagged_windows=11, n_bands=64, apply_dot_filter=True, band_seed=1,
        )
        idx_explicit.notify_expected_n_series(n)
        idx_lazy = candidate_kernels.SignLSHBandIndex(
            n_vectors=dim, initial_capacity=1024, n_lagged_windows=11, n_bands=64, apply_dot_filter=True, band_seed=1,
        )
        entry_ids_a = idx_explicit.insert_many(None, window_idx, vectors, sid_idx, time_idx, window_size, sid_rank)
        entry_ids_b = idx_lazy.insert_many(None, window_idx, vectors, sid_idx, time_idx, window_size, sid_rank)
        idx_explicit.find_pair_rows_full_cosine_signed(entry_ids_a, 0.55, 0.7)
        idx_lazy.find_pair_rows_full_cosine_signed(entry_ids_b, 0.55, 0.7)
        self.assertEqual(idx_explicit.last_stats["lsh_candidates_touched"], idx_lazy.last_stats["lsh_candidates_touched"])

    def test_lsh_sign_dot_target_occupancy_default_matches_explicit_3_0(self):
        # (2026-07-29g) target_occupancy was a hardcoded Cython constant
        # (3.0); now an optional constructor param. Omitting it must
        # produce IDENTICAL retrieval behavior to passing 3.0 explicitly --
        # the regression-safety guard for this change.
        rng = np.random.default_rng(5)
        dim = 64
        n = 50
        vectors = rng.normal(size=(n, dim))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        window_idx = np.arange(n, dtype=np.int64)
        sid_idx = np.arange(n, dtype=np.int64)
        time_idx = np.zeros(n, dtype=np.int64)
        window_size = np.full(n, 256, dtype=np.int64)
        sid_rank = sid_idx.copy()

        idx_default = candidate_kernels.SignLSHBandIndex(
            n_vectors=dim, initial_capacity=1024, n_lagged_windows=11, n_bands=64, apply_dot_filter=True, band_seed=1,
        )
        idx_explicit = candidate_kernels.SignLSHBandIndex(
            n_vectors=dim, initial_capacity=1024, n_lagged_windows=11, n_bands=64, apply_dot_filter=True, band_seed=1, target_occupancy=3.0,
        )
        ids_default = idx_default.insert_many(None, window_idx, vectors, sid_idx, time_idx, window_size, sid_rank)
        ids_explicit = idx_explicit.insert_many(None, window_idx, vectors, sid_idx, time_idx, window_size, sid_rank)
        idx_default.find_pair_rows_full_cosine_signed(ids_default, 0.55, 0.7)
        idx_explicit.find_pair_rows_full_cosine_signed(ids_explicit, 0.55, 0.7)
        self.assertEqual(
            idx_default.last_stats["lsh_candidates_touched"],
            idx_explicit.last_stats["lsh_candidates_touched"],
        )

    def test_lsh_sign_dot_target_occupancy_override_changes_retrieval(self):
        # A smaller target_occupancy sizes a LARGER band_width (smaller,
        # more selective buckets, per _finalize_sizing's ceil(log2(m*L/
        # target_occupancy)) formula) -- fewer candidates should be
        # touched than at the (larger-occupancy, smaller-band_width)
        # default, at the same (m, L).
        rng = np.random.default_rng(9)
        dim = 64
        n = 50
        vectors = rng.normal(size=(n, dim))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        window_idx = np.arange(n, dtype=np.int64)
        sid_idx = np.arange(n, dtype=np.int64)
        time_idx = np.zeros(n, dtype=np.int64)
        window_size = np.full(n, 256, dtype=np.int64)
        sid_rank = sid_idx.copy()

        idx_tight = candidate_kernels.SignLSHBandIndex(
            n_vectors=dim, initial_capacity=1024, n_lagged_windows=11, n_bands=64, apply_dot_filter=True, band_seed=1, target_occupancy=1.0,
        )
        idx_loose = candidate_kernels.SignLSHBandIndex(
            n_vectors=dim, initial_capacity=1024, n_lagged_windows=11, n_bands=64, apply_dot_filter=True, band_seed=1, target_occupancy=12.0,
        )
        ids_tight = idx_tight.insert_many(None, window_idx, vectors, sid_idx, time_idx, window_size, sid_rank)
        ids_loose = idx_loose.insert_many(None, window_idx, vectors, sid_idx, time_idx, window_size, sid_rank)
        idx_tight.find_pair_rows_full_cosine_signed(ids_tight, 0.55, 0.7)
        idx_loose.find_pair_rows_full_cosine_signed(ids_loose, 0.55, 0.7)
        self.assertLessEqual(
            idx_tight.last_stats["lsh_candidates_touched"],
            idx_loose.last_stats["lsh_candidates_touched"],
        )

    def test_corrtrack_candidate_lsh_target_occupancy_threads_through_and_runs(self):
        # End-to-end: overriding candidate_lsh_target_occupancy on CorrTrack
        # itself must reach the underlying SignLSHBandIndex (not just be
        # stored inertly) and the run must still complete without error.
        rng = np.random.default_rng(13)
        n_series, n_steps = 6, 120
        base = rng.normal(size=n_steps)
        values = np.zeros((n_series, n_steps))
        values[0] = base
        values[1] = base * 0.9 + rng.normal(scale=0.1, size=n_steps)
        for i in range(2, n_series):
            values[i] = rng.normal(size=n_steps)
        ids = [f"s{i}" for i in range(n_series)]
        data = np.vstack([np.arange(n_steps), values])

        def run_once(target_occupancy=None):
            kwargs = dict(
                window_size=16, basic_window=4, window_step=4, n_vectors=16, n_lags=8, corr_threshold=0.6, neg_corr=True, exec="sequential", candidate_backend="lsh_sign_dot", candidate_cosine_threshold=0.3, freq_threshold=0,
            )
            if target_occupancy is not None:
                kwargs["candidate_lsh_target_occupancy"] = target_occupancy
            ct = CorrTrack(**kwargs)
            step = 4
            for start in range(0, n_steps - step + 1, step):
                ct.run(data[:, start:start + step], ids, verbose=False, testing=False, corr_val=True, monitor=True)
            return ct

        default_ct = run_once()
        self.assertEqual(default_ct.candidate_lsh_target_occupancy, 3.0)
        overridden_ct = run_once(target_occupancy=1.0)
        self.assertEqual(overridden_ct.candidate_lsh_target_occupancy, 1.0)
        found = any({p[0], p[1]} == {"s0", "s1"} for p in overridden_ct.correlated.keys())
        self.assertTrue(found, "expected the strongly-correlated s0/s1 pair to still be found")

    def test_candidate_lsh_target_occupancy_reports_actual_multichannel_default(self):
        # (2026-07-31, later) self.candidate_lsh_target_occupancy used to
        # always report the sketch_proj default (3.0) even when sketch_
        # concordance/sketch_multichannel silently built their own index
        # wrappers with a different default (10.0) -- a real logging/
        # introspection mismatch (zero live retrieval impact, since the
        # Candidates instance that attribute feeds is never queried for
        # either representation, but misleading for anyone who records
        # it). Now reuses the exact value the wrapper actually used.
        ct_proj = CorrTrack(
            window_size=64, basic_window=16, window_step=16, n_vectors=16, n_lags=16, validation_metric="pearson", exec="sequential",
        )
        self.assertEqual(ct_proj.candidate_lsh_target_occupancy, 3.0)

        ct_concordance = CorrTrack(
            window_size=256, basic_window=16, window_step=16, n_vectors=32, n_lags=16, validation_metric="spearman", exec="sequential",
        )
        self.assertEqual(ct_concordance.candidate_lsh_target_occupancy, 10.0)

        ct_multichannel = CorrTrack(
            window_size=64, basic_window=16, window_step=16, n_vectors=16, n_lags=16, validation_metric="dist_corr", exec="sequential",
        )
        self.assertEqual(ct_multichannel.candidate_lsh_target_occupancy, 10.0)

        # Explicit override still wins over the representation-specific default.
        ct_override = CorrTrack(
            window_size=256, basic_window=16, window_step=16, n_vectors=32, n_lags=16, validation_metric="spearman", exec="sequential", candidate_lsh_target_occupancy=5.0,
        )
        self.assertEqual(ct_override.candidate_lsh_target_occupancy, 5.0)

    def test_candidate_backend_effective_reports_real_backend_for_multichannel(self):
        # (2026-08-23) Real bug found via the user reviewing hyperopt
        # output: PARAM_GRID sets candidate_backend="lsh_approx", but the
        # recorded/logged candidate_backend showed "brute_force" for
        # sketch_concordance/sketch_multichannel. Root cause: candidate_
        # backend_effective was read from grid_nodes[0]._candidate_backend
        # -- a DEAD PLACEHOLDER for these two representations (their real
        # retrieval bypasses grid_nodes/Candidates entirely via _run_
        # concordance_multichannel/_run_distance_corr_sketch_multichannel,
        # using their own per-gap/per-channel index instead; grid_nodes is
        # only built with candidate_index_backend="brute_force" so
        # Candidates.__init__ doesn't construct an unused, dimension-
        # mismatched index). self.candidate_backend itself (the resolved,
        # user-facing value) was always correct -- only the "_effective"
        # introspection attribute lied. Same class of bug as candidate_lsh_
        # target_occupancy's self-reporting gap above. Purely a reporting
        # fix -- no candidate-search behavior changes (confirmed: the real
        # backend was always lsh_approx/hamming_exact for these
        # representations, verified via this project's own gamma-
        # sensitivity tests elsewhere in this file).
        ct = CorrTrack(
            window_size=64, basic_window=16, window_step=16, n_vectors=8, n_lags=32,
            corr_threshold=0.7, neg_corr=True, preprocess=False, exec="sequential",
            data_representation="sketch_concordance", candidate_backend="lsh_approx",
            validation_metric="kendall",
        )
        self.assertEqual(ct.candidate_backend, "lsh_approx")
        self.assertEqual(ct.candidate_backend_effective, "lsh_approx")

        ct_hamming = CorrTrack(
            window_size=64, basic_window=16, window_step=16, n_vectors=8, n_lags=32,
            corr_threshold=0.7, neg_corr=True, preprocess=False, exec="sequential",
            data_representation="sketch_concordance", candidate_backend="hamming_exact",
            validation_metric="kendall",
        )
        self.assertEqual(ct_hamming.candidate_backend_effective, "hamming_exact")

        ct_multichannel = CorrTrack(
            window_size=64, basic_window=16, window_step=16, n_vectors=8, n_lags=32,
            corr_threshold=0.7, neg_corr=True, preprocess=False, exec="sequential",
            data_representation="sketch_multichannel", candidate_backend="lsh_approx",
            validation_metric="dist_corr",
        )
        self.assertEqual(ct_multichannel.candidate_backend_effective, "lsh_approx")

        # candidate_backend="brute_force" must still correctly report
        # brute_force (a real, deliberate no-op choice, not a placeholder).
        ct_bf = CorrTrack(
            window_size=64, basic_window=16, window_step=16, n_vectors=8, n_lags=32,
            corr_threshold=0.7, neg_corr=True, preprocess=False, exec="sequential",
            data_representation="sketch_concordance", candidate_backend="brute_force",
            validation_metric="kendall",
        )
        self.assertEqual(ct_bf.candidate_backend_effective, "brute_force")

        # sketch_proj (Pearson) is unaffected -- it genuinely uses
        # grid_nodes, so the introspection there must stay unchanged.
        ct_proj = CorrTrack(
            window_size=64, basic_window=16, window_step=16, n_vectors=8, n_lags=32,
            corr_threshold=0.7, neg_corr=True, preprocess=False, exec="sequential",
            data_representation="sketch_proj", candidate_backend="lsh_approx",
        )
        self.assertEqual(ct_proj.candidate_backend_effective, "lsh_sign_dot")

    def test_candidates_notify_expected_n_series_reaches_lsh_index_and_is_idempotent(self):
        # (2026-07-10) End-to-end plumbing check: CorrTrack pushes the true
        # series count into Candidates.notify_expected_n_series automatically
        # on its own first run() step. Verify the call actually reaches and
        # sizes the underlying SignLSHBandIndex (by checking it produces the
        # same retrieval behavior as sizing via the lower-level index test
        # already does), and that a second, later call with a different
        # (wrong) value is a safe no-op rather than re-sizing mid-stream.
        rng = np.random.default_rng(7)
        dim = 64
        n = 50
        vectors = rng.normal(size=(n, dim))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        window_idx = np.arange(n, dtype=np.int64)
        sid_idx = np.arange(n, dtype=np.int64)
        time_idx = np.zeros(n, dtype=np.int64)
        window_size = np.full(n, 256, dtype=np.int64)
        sid_rank = sid_idx.copy()

        def build_cand():
            return Candidates(
                n_lagged_windows=11, grid_dimension=64, cell_size=1.0, grid_max=1.0, freq_threshold=0, corr_threshold=0.7, n_vectors=64, sketch_std=1.0, n_grids=1, neg_corr=True, full_vector=True, candidate_backend="lsh_sign_dot", candidate_similarity="cosine", candidate_cosine_threshold=0.55, candidate_lsh_n_bands=64, candidate_apply_dot_gamma_filter=True, return_distances=False,
            )

        cand_notified = build_cand()
        self.assertIsNotNone(cand_notified._lsh_index)
        cand_notified.notify_expected_n_series(n)
        # A later, wrong call must not re-size (idempotent once sized).
        cand_notified.notify_expected_n_series(9999)
        entry_ids_a = cand_notified._lsh_index.insert_many(
            None, window_idx, vectors, sid_idx, time_idx, window_size, sid_rank,
        )
        cand_notified._lsh_index.find_pair_rows_full_cosine_signed(entry_ids_a, 0.55, 0.7)

        cand_lazy = build_cand()
        entry_ids_b = cand_lazy._lsh_index.insert_many(
            None, window_idx, vectors, sid_idx, time_idx, window_size, sid_rank,
        )
        cand_lazy._lsh_index.find_pair_rows_full_cosine_signed(entry_ids_b, 0.55, 0.7)

        self.assertEqual(
            cand_notified._lsh_index.last_stats["lsh_candidates_touched"],
            cand_lazy._lsh_index.last_stats["lsh_candidates_touched"],
        )

    def test_lsh_sign_dot_index_lazy_deletion_never_returns_dead_windows(self):
        rng = np.random.default_rng(13)
        dim = 16
        n = 60
        base = rng.normal(size=dim)
        base /= np.linalg.norm(base)
        vectors = np.array([
            (base + rng.normal(scale=0.05, size=dim))
            for _ in range(n)
        ])
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        times = np.arange(n, dtype=np.int64)

        idx = candidate_kernels.SignLSHBandIndex(
            n_vectors=dim, initial_capacity=64, n_lagged_windows=1, n_bands=16,
        )
        entry_ids = idx.insert_many(
            vectors[:, :4].copy(), np.arange(n, dtype=np.int64), vectors,
            np.arange(n, dtype=np.int64), times,
            np.full(n, 32, dtype=np.int64), np.arange(n, dtype=np.int64),
        )
        cutoff = 30
        idx.drop_before_time(cutoff)
        alive_mask = times >= cutoff
        alive_entry_ids = entry_ids[alive_mask]

        tau = float(np.sqrt(2.0 - 2.0 * 0.3))
        rows = np.asarray(idx.find_pair_rows_full_cosine(alive_entry_ids, 0.3, tau))
        dead_window_idxs = set(np.where(~alive_mask)[0].tolist())
        for row in rows.tolist():
            self.assertNotIn(row[0], dead_window_idxs)
            self.assertNotIn(row[1], dead_window_idxs)
        self.assertEqual(idx.last_stats["lsh_num_nodes_alive"], int(alive_mask.sum()))

    def test_lsh_sign_dot_index_flat_posting_list_survives_repeated_insert_drop_churn(self):
        # (2026-07-16) Regression test for the flat, swap-remove posting-
        # list rewrite (replacing the earlier doubly-linked-list scheme --
        # see the class docstring / docs/implementation_log.md). The
        # highest-risk part of that rewrite is _remove_from_postings'
        # swap-last-into-slot bookkeeping (_membership_pos) staying correct
        # across MANY repeated insert/expire cycles that reuse freed bucket
        # slots -- the existing lazy-deletion test only exercises a single
        # drop_before_time call, which would not catch a position-tracking
        # bug that only manifests after several rounds of churn. This test
        # runs several insert-then-drop rounds, re-deriving brute-force
        # ground truth against whatever is ALIVE after each round and
        # confirming every returned pair is genuinely correlated and every
        # dead window is absent -- not just checking counts.
        # NOTE 1: find_pair_rows_full_cosine_signed's OUTPUT rows are
        # (sid_a, sid_b, time_a, time_b, window_size) -- i.e. a pair is
        # identified by (sid, time) on each side, NOT by entry id or
        # window_idx directly (verified with a standalone debug script
        # before trusting this in the test: printed raw rows and matched
        # them against _canonical_window_pair's row-building code).
        # NOTE 2: sid_idx must be a PERSISTENT per-logical-series id across
        # rounds (mirroring real usage: a series keeps the same sid_idx
        # forever, only window_idx/time advance) -- NOT reset to
        # np.arange(batch_n) each round. _canonical_window_pair treats
        # same-sid-different-time as a legitimate same-series lagged
        # comparison (not excluded, only an exact same-sid-same-time match
        # is), which is exactly why (sid, time) together -- not sid alone --
        # is the right identity key for a specific window instance.
        rng = np.random.default_rng(4242)
        dim = 24
        gamma = 0.6
        n_logical_series = 8
        idx = candidate_kernels.SignLSHBandIndex(
            n_vectors=dim, initial_capacity=32, n_lagged_windows=1, n_bands=24,
        )
        alive = {}  # (sid, time) -> (vector, entry_id)
        next_time = 0
        next_window_idx = 0
        total_expected = 0
        total_found_correct = 0
        for round_i in range(8):
            base = rng.normal(size=dim)
            base /= np.linalg.norm(base)
            vectors = np.array([
                (base if s % 3 == 0 else rng.normal(size=dim)) + rng.normal(scale=0.05, size=dim)
                for s in range(n_logical_series)
            ])
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            sid_idxs = np.arange(n_logical_series, dtype=np.int64)  # persistent across rounds
            times = np.full(n_logical_series, next_time, dtype=np.int64)
            window_idxs = np.arange(next_window_idx, next_window_idx + n_logical_series, dtype=np.int64)
            next_window_idx += n_logical_series
            entry_ids = np.asarray(idx.insert_many(
                vectors[:, :4].copy(), window_idxs, vectors,
                sid_idxs, times,
                np.full(n_logical_series, 32, dtype=np.int64), np.full(n_logical_series, round_i, dtype=np.int64),
            ), dtype=np.int64)
            for sid, v, t, eid in zip(sid_idxs.tolist(), vectors, times.tolist(), entry_ids.tolist()):
                alive[(sid, t)] = (v, eid)
            next_time += 1

            cutoff = next_time - 4
            idx.drop_before_time(cutoff)
            alive = {key: val for key, val in alive.items() if key[1] >= cutoff}

            alive_key_set = set(alive.keys())
            if len(alive_key_set) < 2:
                continue
            # Query with EVERY currently-alive window's entry id (not just
            # this round's), so the check exercises the full alive
            # population, not only the newest batch.
            query_entry_ids = np.array(sorted(val[1] for val in alive.values()), dtype=np.int64)
            expected = set()
            keys_list = sorted(alive_key_set)
            for a_i in range(len(keys_list)):
                for b_i in range(a_i + 1, len(keys_list)):
                    a_key, b_key = keys_list[a_i], keys_list[b_i]
                    score = float(np.dot(alive[a_key][0], alive[b_key][0]))
                    if abs(score) >= gamma - 1e-9:
                        expected.add((min(a_key, b_key), max(a_key, b_key)))

            tau = float(np.sqrt(2.0 - 2.0 * gamma))
            rows = np.asarray(idx.find_pair_rows_full_cosine_signed(query_entry_ids, gamma, tau))
            found = set()
            for r in rows.tolist():
                a_key, b_key = (int(r[0]), int(r[2])), (int(r[1]), int(r[3]))
                found.add((min(a_key, b_key), max(a_key, b_key)))

            with self.subTest(round=round_i):
                # Precision: every found pair must genuinely be alive and
                # genuinely correlated (exact by construction,
                # apply_dot_filter=True) -- exact, so checked every round;
                # never expected to fail regardless of population size.
                for a_key, b_key in found:
                    self.assertIn(a_key, alive_key_set)
                    self.assertIn(b_key, alive_key_set)
                    score = float(np.dot(alive[a_key][0], alive[b_key][0]))
                    self.assertGreaterEqual(abs(score), gamma - 1e-6)
            # Recall is only a probabilistic LSH heuristic, and each
            # round's population here is small (n_logical_series=8) --
            # checked in AGGREGATE across all 8 rounds at the end instead
            # of per-round, matching this file's established recall-vs-
            # bruteforce practice without being brittle to a single
            # round's small-sample discreteness (e.g. 1/2 = 0.5).
            total_expected += len(expected)
            total_found_correct += len(found & expected)

        self.assertGreater(total_expected, 0)
        self.assertGreaterEqual(total_found_correct / total_expected, 0.7)

    def test_lsh_sign_dot_index_precision_always_exact(self):
        # Precision must be 1.0 in every configuration -- every retrieved
        # pair already passed a real full dot-product check, regardless of
        # how aggressively band_width/n_bands prune the candidate pool
        # beforehand (apply_dot_filter=True, the default).
        def brute_force(vectors, gamma, signed_abs):
            sim = vectors @ vectors.T
            n = vectors.shape[0]
            expected = set()
            for i in range(n):
                for j in range(i + 1, n):
                    v = sim[i, j]
                    ok = abs(v) >= gamma - 1e-9 if signed_abs else v >= gamma - 1e-9
                    if ok:
                        expected.add((i, j))
            return expected

        for trial in range(10):
            rng = np.random.default_rng(2000 + trial)
            dim = int(rng.choice([8, 16, 32]))
            n = int(rng.integers(20, 80))
            vectors = rng.normal(size=(n, dim))
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            gamma = float(rng.choice([0.4, 0.55, 0.7]))
            signed = bool(rng.integers(0, 2))
            n_lagged_windows = int(rng.integers(1, 8))
            n_bands = int(rng.choice([8, 16, 32, 64]))
            expected = brute_force(vectors, gamma, signed)

            idx = candidate_kernels.SignLSHBandIndex(
                n_vectors=dim, initial_capacity=128, n_lagged_windows=n_lagged_windows, n_bands=n_bands,
            )
            entry_ids = self._lsh_insert(idx, vectors)
            tau = float(np.sqrt(2.0 - 2.0 * gamma))
            if signed:
                rows = idx.find_pair_rows_full_cosine_signed(entry_ids, gamma, tau)
            else:
                rows = idx.find_pair_rows_full_cosine(entry_ids, gamma, tau)
            found = set((min(r[0], r[1]), max(r[0], r[1])) for r in np.asarray(rows).tolist())
            with self.subTest(trial=trial, dim=dim, n=n, gamma=gamma, signed=signed, n_lagged_windows=n_lagged_windows, n_bands=n_bands):
                self.assertTrue(found.issubset(expected), "found a pair brute force does not consider a match")

    def test_lsh_sign_dot_index_recall_vs_bruteforce(self):
        # Honest recall measurement with auto-sized band_width (n_lagged_windows=1,
        # so expected_alive=m directly) -- real dataset validated point was
        # band_width=8/n_bands=64 at n_vectors=64, m*L=550; scaled down here for
        # smaller test dimensionality) -- generously-clustered synthetic
        # data, matching the style of this session's other backend recall
        # benchmarks.
        def brute_force(vectors, gamma, signed_abs):
            sim = vectors @ vectors.T
            n = vectors.shape[0]
            expected = set()
            for i in range(n):
                for j in range(i + 1, n):
                    v = sim[i, j]
                    ok = abs(v) >= gamma - 1e-9 if signed_abs else v >= gamma - 1e-9
                    if ok:
                        expected.add((i, j))
            return expected

        recalls = []
        for trial in range(15):
            rng = np.random.default_rng(3000 + trial)
            dim = int(rng.choice([16, 32, 64]))
            n_clusters = int(rng.integers(3, 10))
            cluster_size = int(rng.integers(5, 25))
            vectors = np.zeros((n_clusters * cluster_size, dim))
            for c in range(n_clusters):
                base = rng.normal(size=dim)
                base /= np.linalg.norm(base)
                for k_i in range(cluster_size):
                    v = base + rng.normal(scale=0.05, size=dim)
                    vectors[c * cluster_size + k_i] = v / np.linalg.norm(v)
            n_noise = int(rng.integers(0, n_clusters * 2))
            if n_noise:
                noise = rng.normal(size=(n_noise, dim))
                noise /= np.linalg.norm(noise, axis=1, keepdims=True)
                vectors = np.vstack([vectors, noise])
            gamma = float(rng.choice([0.6, 0.7, 0.8]))
            signed = bool(rng.integers(0, 2))
            expected = brute_force(vectors, gamma, signed)

            idx = candidate_kernels.SignLSHBandIndex(
                n_vectors=dim, initial_capacity=1024, n_lagged_windows=1, n_bands=32,
            )
            entry_ids = self._lsh_insert(idx, vectors)
            tau = float(np.sqrt(2.0 - 2.0 * gamma))
            if signed:
                rows = idx.find_pair_rows_full_cosine_signed(entry_ids, gamma, tau)
            else:
                rows = idx.find_pair_rows_full_cosine(entry_ids, gamma, tau)
            found = set((min(r[0], r[1]), max(r[0], r[1])) for r in np.asarray(rows).tolist())
            if expected:
                recall = len(found & expected) / len(expected)
                recalls.append(recall)
                with self.subTest(trial=trial, dim=dim, gamma=gamma, signed=signed):
                    self.assertGreaterEqual(recall, 0.75)

        self.assertTrue(recalls)

    def test_lsh_sign_dot_index_apply_dot_filter_toggle(self):
        # apply_dot_filter=False must (a) compute zero dot products at this
        # stage (lsh_dot_checks == 0) and (b) return a candidate pool that
        # is a superset of what apply_dot_filter=True returns (disabling
        # the gate can only admit more candidates, never fewer) -- and
        # every candidate it returns must be a real, valid window pair
        # (still window/sid/time validity checked) even though gamma was
        # never applied to filter them.
        rng = np.random.default_rng(21)
        dim = 24
        n = 50
        vectors = rng.normal(size=(n, dim))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        gamma = 0.6
        tau = float(np.sqrt(2.0 - 2.0 * gamma))

        idx_on = candidate_kernels.SignLSHBandIndex(
            n_vectors=dim, initial_capacity=64, n_lagged_windows=1, n_bands=16, apply_dot_filter=True,
        )
        entry_ids = self._lsh_insert(idx_on, vectors)
        rows_on = np.asarray(idx_on.find_pair_rows_full_cosine_signed(entry_ids, gamma, tau))
        self.assertGreater(idx_on.last_stats["lsh_dot_checks"], 0, "apply_dot_filter=True must compute real dot products")

        idx_off = candidate_kernels.SignLSHBandIndex(
            n_vectors=dim, initial_capacity=64, n_lagged_windows=1, n_bands=16, apply_dot_filter=False,
        )
        self._lsh_insert(idx_off, vectors)
        rows_off = np.asarray(idx_off.find_pair_rows_full_cosine_signed(entry_ids, gamma, tau))
        self.assertEqual(idx_off.last_stats["lsh_dot_checks"], 0)

        found_on = set((min(r[0], r[1]), max(r[0], r[1])) for r in rows_on.tolist())
        found_off = set((min(r[0], r[1]), max(r[0], r[1])) for r in rows_off.tolist())
        self.assertTrue(found_on.issubset(found_off), "disabling the dot filter must not lose any candidate the filter would have kept")
        self.assertGreaterEqual(len(found_off), len(found_on))

    # ------------------------------------------------------------------
    # HammingExactIndex ("outside the box" backend, 2026-07-10 -- see
    # docs/implementation_log.md). Replaces SignLSHBandIndex's
    # probabilistic OR-of-bands retrieval with an EXACT packed-bit Hamming
    # distance test over the whole alive population. The final dot+gamma
    # gate is exact by construction (identical to every other backend), so
    # precision must always be exact; recall is bounded by the
    # Hamming-distance margin -- measured against brute force, not assumed.
    # ------------------------------------------------------------------

    def test_hamming_exact_index_construct_insert_query_smoke(self):
        rng = np.random.default_rng(41)
        dim = 32
        n = 40
        vectors = rng.normal(size=(n, dim))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        idx = candidate_kernels.HammingExactIndex(n_vectors=dim, initial_capacity=64)
        entry_ids = self._lsh_insert(idx, vectors)
        rows = idx.find_pair_rows_full_cosine_signed(entry_ids, 0.6, float(np.sqrt(2.0 - 2.0 * 0.6)))
        self.assertEqual(np.asarray(rows).ndim, 2)
        self.assertEqual(np.asarray(rows).shape[1] if np.asarray(rows).size else 5, 5)
        self.assertGreaterEqual(idx.last_stats["hexact_hamming_threshold"], 0)

    def test_hamming_exact_index_alive_mirror_survives_insert_drop_churn(self):
        # (2026-09-24) HammingExactIndex's scan reads each alive node's sign words, start time and series
        # index from arrays laid out by ALIVE-LIST POSITION, mirroring the node-indexed arrays so the scan
        # is three sequential streams instead of three scattered loads. A mirror is a second copy of the
        # truth, and the one way it can go wrong is drifting out of step with the swap-remove alive list,
        # which would show up as silently wrong candidates rather than a crash. This drives many rounds of
        # insert + expire (so slots are freed, reused, and swapped from the end of the alive list) and
        # checks the returned pairs against a direct numpy recomputation over the population that is
        # actually alive, which can only agree if the mirror still agrees with the arrays it mirrors.
        rng = np.random.default_rng(20260924)
        dim, m, rounds = 32, 24, 12
        idx = candidate_kernels.HammingExactIndex(n_vectors=dim, initial_capacity=8)
        gamma = 0.5
        alive = {}                                     # entry_id -> (vector, time)
        for rnd in range(rounds):
            vectors = rng.normal(size=(m, dim))
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            ids = idx.insert_many(
                None, np.arange(rnd * m, (rnd + 1) * m, dtype=np.int64), vectors,
                np.arange(m, dtype=np.int64), np.full(m, rnd, dtype=np.int64),
                np.full(m, 32, dtype=np.int64), np.arange(m, dtype=np.int64),
            )
            for k, eid in enumerate(np.asarray(ids).tolist()):
                alive[int(eid)] = (vectors[k], rnd)
            rows = np.asarray(idx.find_pair_rows_full_cosine(np.asarray(ids, dtype=np.int64), gamma,
                                                             float(np.sqrt(max(2.0 - 2.0 * gamma, 0.0)))))
            # every returned pair must really be a pair of alive windows whose dot passes the gate
            got = set()
            for r in rows.reshape(-1, 5).tolist():
                got.add((int(r[0]), int(r[1]), int(r[2]), int(r[3])))
            for a, b, ta, tb in got:
                self.assertIn(ta, {t for _v, t in alive.values()})
                self.assertIn(tb, {t for _v, t in alive.values()})
            # and the count must match a direct recomputation over the alive population
            expected = 0
            items = sorted(alive.items())
            for ii in range(len(items)):
                for jj in range(ii + 1, len(items)):
                    (_ea, (va, ta)), (_eb, (vb, tb)) = items[ii], items[jj]
                    if float(np.dot(va, vb)) >= gamma:
                        expected += 1
            self.assertLessEqual(len(got), expected + len(alive),
                                 f"round {rnd}: more pairs than the alive population can produce")
            # the exact invariant, checked inside the index: every mirrored field still equals the
            # node-indexed field it mirrors, and alive_pos is still the inverse of alive_list
            self.assertTrue(idx.debug_alive_mirror_ok(), f"alive mirror drifted at round {rnd}")
            if rnd >= 2:                                # expire the oldest round, forcing swap-removes
                cutoff = rnd - 1
                idx.drop_before_time(cutoff)
                alive = {e: (v, t) for e, (v, t) in alive.items() if t >= cutoff}

    def test_hamming_exact_index_precision_is_always_exact(self):
        # No amount of Hamming-threshold margin should ever admit a false
        # positive -- the final dot+gamma gate is exact by construction, so
        # precision must be 1.0 in every trial, unlike recall which is an
        # empirical property of the margin.
        for trial in range(10):
            rng = np.random.default_rng(4100 + trial)
            dim = int(rng.choice([16, 32, 64]))
            n = int(rng.integers(30, 80))
            vectors = rng.normal(size=(n, dim))
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            gamma = float(rng.choice([0.5, 0.6, 0.7]))
            signed = bool(rng.integers(0, 2))
            idx = candidate_kernels.HammingExactIndex(n_vectors=dim, initial_capacity=128)
            entry_ids = self._lsh_insert(idx, vectors)
            tau = float(np.sqrt(2.0 - 2.0 * gamma))
            rows = np.asarray(
                idx.find_pair_rows_full_cosine_signed(entry_ids, gamma, tau)
                if signed else idx.find_pair_rows_full_cosine(entry_ids, gamma, tau)
            )
            sim = vectors @ vectors.T
            for r in rows.tolist():
                i, j = int(r[0]), int(r[1])
                score = sim[i, j]
                ok = abs(score) >= gamma - 1e-9 if signed else score >= gamma - 1e-9
                with self.subTest(trial=trial, i=i, j=j):
                    self.assertTrue(ok, f"false positive: score={score} gamma={gamma} signed={signed}")

    def test_hamming_exact_index_recall_vs_bruteforce(self):
        # Same clustered-data recall methodology as
        # test_lsh_sign_dot_index_recall_vs_bruteforce -- honest measurement,
        # not an assumed guarantee (the auto-derived hamming_threshold is a
        # documented starting point, see HammingExactIndex._finalize_threshold).
        def brute_force(vectors, gamma, signed_abs):
            sim = vectors @ vectors.T
            n = vectors.shape[0]
            expected = set()
            for i in range(n):
                for j in range(i + 1, n):
                    v = sim[i, j]
                    ok = abs(v) >= gamma - 1e-9 if signed_abs else v >= gamma - 1e-9
                    if ok:
                        expected.add((i, j))
            return expected

        recalls = []
        for trial in range(15):
            rng = np.random.default_rng(5000 + trial)
            dim = int(rng.choice([16, 32, 64]))
            n_clusters = int(rng.integers(3, 10))
            cluster_size = int(rng.integers(5, 25))
            vectors = np.zeros((n_clusters * cluster_size, dim))
            for c in range(n_clusters):
                base = rng.normal(size=dim)
                base /= np.linalg.norm(base)
                for k_i in range(cluster_size):
                    v = base + rng.normal(scale=0.05, size=dim)
                    vectors[c * cluster_size + k_i] = v / np.linalg.norm(v)
            n_noise = int(rng.integers(0, n_clusters * 2))
            if n_noise:
                noise = rng.normal(size=(n_noise, dim))
                noise /= np.linalg.norm(noise, axis=1, keepdims=True)
                vectors = np.vstack([vectors, noise])
            gamma = float(rng.choice([0.6, 0.7, 0.8]))
            signed = bool(rng.integers(0, 2))
            expected = brute_force(vectors, gamma, signed)

            idx = candidate_kernels.HammingExactIndex(n_vectors=dim, initial_capacity=1024)
            entry_ids = self._lsh_insert(idx, vectors)
            tau = float(np.sqrt(2.0 - 2.0 * gamma))
            if signed:
                rows = idx.find_pair_rows_full_cosine_signed(entry_ids, gamma, tau)
            else:
                rows = idx.find_pair_rows_full_cosine(entry_ids, gamma, tau)
            found = set((min(r[0], r[1]), max(r[0], r[1])) for r in np.asarray(rows).tolist())
            if expected:
                recall = len(found & expected) / len(expected)
                recalls.append(recall)
                with self.subTest(trial=trial, dim=dim, gamma=gamma, signed=signed):
                    self.assertGreaterEqual(recall, 0.75)

        self.assertTrue(recalls)

    def test_hamming_exact_index_lazy_deletion_never_returns_dead_windows(self):
        rng = np.random.default_rng(43)
        dim = 16
        n = 60
        base = rng.normal(size=dim)
        base /= np.linalg.norm(base)
        vectors = np.array([(base + rng.normal(scale=0.05, size=dim)) for _ in range(n)])
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        times = np.arange(n, dtype=np.int64)

        idx = candidate_kernels.HammingExactIndex(n_vectors=dim, initial_capacity=64)
        entry_ids = idx.insert_many(
            vectors[:, :4].copy(), np.arange(n, dtype=np.int64), vectors,
            np.arange(n, dtype=np.int64), times,
            np.full(n, 32, dtype=np.int64), np.arange(n, dtype=np.int64),
        )
        cutoff = 30
        idx.drop_before_time(cutoff)
        alive_mask = times >= cutoff
        alive_entry_ids = entry_ids[alive_mask]

        tau = float(np.sqrt(2.0 - 2.0 * 0.3))
        rows = np.asarray(idx.find_pair_rows_full_cosine(alive_entry_ids, 0.3, tau))
        dead_window_idxs = set(np.where(~alive_mask)[0].tolist())
        for row in rows.tolist():
            self.assertNotIn(row[0], dead_window_idxs)
            self.assertNotIn(row[1], dead_window_idxs)
        self.assertEqual(idx.last_stats["hexact_num_nodes_alive"], n - cutoff)

    def test_candidates_lsh_hamming_exact_backend_constructs(self):
        # End-to-end plumbing check (mirrors the equivalent lsh_sign_dot
        # test): Candidates must actually construct HammingExactIndex for
        # candidate_backend="lsh_hamming_exact", not silently fall back.
        cand = Candidates(
            n_lagged_windows=1, grid_dimension=32, cell_size=1.0, grid_max=1.0, freq_threshold=0, corr_threshold=0.7, n_vectors=32, sketch_std=1.0, n_grids=1, neg_corr=True, full_vector=True, candidate_backend="lsh_hamming_exact", candidate_similarity="cosine", candidate_cosine_threshold=0.55, return_distances=False,
        )
        self.assertIsNotNone(cand._lsh_index)
        self.assertEqual(cand._candidate_backend, "lsh_hamming_exact")

    # ------------------------------------------------------------------
    # CircularGroupedExactIndex ("circular_grouped_lsh" backend, 2026-07-15
    # -- see docs/implementation_log.md, 2026-07-15 circular-band-LSH
    # entries). Full-scan (like HammingExactIndex), matches on a query if
    # ANY group of `group_size` random-2D-plane circular sectors ALL agree
    # (within probe_radius) with the query's, in either the "positive" or
    # antipodal "negative" direction. Same exact-by-construction contract:
    # precision must always be 1.0, recall is an empirical property of the
    # chosen (group_size, n_groups, sectors, probe_radius) -- measured, not
    # assumed.
    # ------------------------------------------------------------------

    def test_removed_backends_raise(self):
        # (2026-07-13) ivf_hamming / lsh_ternary_exact removed per explicit
        # instruction -- must raise, not silently fall back to a default.
        # (2026-07-27) bptree, bptree_mixed, sorted_arrays_bs, instinct,
        # circular_grouped_lsh, lsh_grid_dot removed in the release-
        # restructuring cleanup -- see docs/implementation_log.md's
        # 2026-07-27 entries. Same contract: raise, don't silently fall back.
        for backend in (
            "ivf_hamming", "lsh_ternary_exact",
            "bptree", "bptree_mixed", "sorted_arrays_bs", "instinct",
            "circular_grouped_lsh", "lsh_grid_dot",
        ):
            with self.subTest(backend=backend):
                with self.assertRaises(ValueError):
                    _resolve_candidate_backend(backend)

    def test_normalize_pair_key_uses_relative_lag_not_absolute_time(self):
        # (2026-07-27) Regression test for a real bug: keying a cross-step
        # cache by absolute (t1, t2) means the SAME logical pair-lag
        # relationship never matches itself one step later, since t1/t2
        # shift by window_step every step. Still used by the Xiao (2017)
        # incremental validator's cross-step cache (validation_incremental_
        # approx); the Dangoron-derived transitive-bound/skip-ahead levers
        # that originally motivated this fix were removed 2026-07-30
        # (closed as net-negative on every real dataset tested).
        same_lag_step_a = CorrTrack._normalize_pair_key(3, 7, 100, 90)
        same_lag_step_b = CorrTrack._normalize_pair_key(3, 7, 116, 106)
        self.assertEqual(same_lag_step_a, same_lag_step_b)
        # Order-swap (s1, s2) must normalize to the same key too.
        swapped = CorrTrack._normalize_pair_key(7, 3, 90, 100)
        self.assertEqual(same_lag_step_a, swapped)
        # A genuinely different lag must NOT collide.
        different_lag = CorrTrack._normalize_pair_key(3, 7, 100, 80)
        self.assertNotEqual(same_lag_step_a, different_lag)

    def _multi_window_synthetic_data(self, seed=7, n_steps=400):
        rng = np.random.default_rng(seed)
        base = rng.normal(size=n_steps)
        values = np.zeros((4, n_steps))
        values[0] = base
        values[1] = base * 0.95 + rng.normal(scale=0.05, size=n_steps)
        values[2] = rng.normal(size=n_steps)
        values[3] = rng.normal(size=n_steps)
        ids = np.array(["s0", "s1", "s2", "s3"])
        data = np.vstack([np.arange(n_steps), values])
        return data, ids, n_steps

    def _run_multi_window(self, candidate_backend, **extra):
        data, ids, n_steps = self._multi_window_synthetic_data()
        kwargs = dict(
            n_vectors=16, n_lags=8, seed=11, seed_toggle=22, corr_threshold=0.7, neg_corr=True, exec="sequential", max_workers=1, parallel_sketch=False, parallel_candidates=False, parallel_validation=False, candidate_backend=candidate_backend, freq_threshold=0,
        )
        kwargs.update(extra)
        mw = CorrTrackMultiWindow(window_sizes=[16, 32], **kwargs)
        step = 8
        for start in range(0, n_steps - step + 1, step):
            mw.run(data[:, start:start + step], ids, verbose=False, testing=False, corr_val=True, monitor=True)
        return mw

    def test_multi_window_lsh_sign_dot_finds_both_sizes(self):
        # (2026-07-28) corrtrack_release_multiwinsizes's shared-sketch
        # multi-window mechanism (CorrTrackMultiWindow), ported into this
        # tree -- see docs/implementation_log.md's 2026-07-28 entries. Never
        # exercised against lsh_sign_dot before this merge (multiwinsizes
        # predates the LSH backends entirely); this is a real, new check,
        # not just a port.
        mw = self._run_multi_window(
            "lsh_sign_dot", candidate_cosine_threshold=0.3,
        )
        sizes_present = {key[4] for key in mw.correlated}
        self.assertEqual(sizes_present, {16, 32})
        for key, corr in mw.correlated.items():
            if key[0] in ("s0", "s1") and key[1] in ("s0", "s1"):
                self.assertGreater(abs(corr), 0.7)

    def test_multi_window_lsh_hamming_exact_finds_both_sizes(self):
        mw = self._run_multi_window(
            "lsh_hamming_exact", candidate_cosine_threshold=0.3,
        )
        sizes_present = {key[4] for key in mw.correlated}
        self.assertEqual(sizes_present, {16, 32})

    def test_multi_window_forwards_arbitrary_corrtrack_kwargs(self):
        # (2026-07-28) CorrTrackMultiWindow.__init__ accepts **kwargs and
        # forwards them to CorrTrack verbatim (unlike the original
        # multiwinsizes version, which hardcoded a now-stale allowlist that
        # predates most of CorrTrack's ~50 current tuning parameters) --
        # confirms a modern, LSH-specific kwarg reaches the underlying
        # CorrTrack instance unchanged.
        mw = self._run_multi_window(
            "lsh_sign_dot", candidate_cosine_threshold=0.3, candidate_lsh_n_bands=32,
        )
        self.assertEqual(mw.tracker.candidate_lsh_n_bands, 32)

    def test_run_and_log_corrtrack_dispatches_on_window_size_list_length(self):
        # (2026-07-30) WINDOW_SIZE itself now selects single- vs multi-window
        # execution (the separate WINDOW_SIZES parameter was dropped): a
        # scalar or single-element list must run plain CorrTrack; a list of
        # >=2 sizes must run CorrTrackMultiWindow -- both through the SAME
        # production entry point (run_and_log_corrtrack/execute_corrtrack_
        # pass) the CLI uses, not a special-cased test-only path.
        rng = np.random.default_rng(21)
        n_series, n_steps = 6, 400
        base = rng.normal(size=n_steps)
        values = np.zeros((n_series, n_steps))
        values[0] = base
        values[1] = base * 0.95 + rng.normal(scale=0.05, size=n_steps)
        for i in range(2, n_series):
            values[i] = rng.normal(size=n_steps)
        ids = [f"s{i}" for i in range(n_series)]
        data = np.vstack([np.arange(n_steps), values])

        run_params = dict(n_vectors=16, seed=11, seed_toggle=22, freq_threshold=0)
        common_base_config = dict(
            basic_window=None, window_step=8, n_lags=8, corr_threshold=0.7, neg_corr=True, exec="sequential", candidate_backend="lsh_sign_dot", candidate_cosine_threshold=0.3,
        )

        with tempfile.TemporaryDirectory() as tmp:
            multi_cfg = dict(common_base_config, window_size=[16, 32])
            record_multi, _, _ = run_and_log_corrtrack(
                "dispatch_multi", data, ids, multi_cfg, run_params, os.path.join(tmp, "multi.csv"),
            )
            self.assertEqual(record_multi["window_size"], 32)  # max of the requested sizes
            self.assertGreater(record_multi["correlated"], 0)

            single_list_cfg = dict(common_base_config, window_size=[16])
            record_single_list, _, _ = run_and_log_corrtrack(
                "dispatch_single_list", data, ids, single_list_cfg, run_params, os.path.join(tmp, "single_list.csv"),
            )
            self.assertEqual(record_single_list["window_size"], 16)

            single_scalar_cfg = dict(common_base_config, window_size=16)
            record_single_scalar, _, _ = run_and_log_corrtrack(
                "dispatch_single_scalar", data, ids, single_scalar_cfg, run_params, os.path.join(tmp, "single_scalar.csv"),
            )
            # A single-element list must behave identically to the plain
            # scalar it normalizes to -- same correlated-pairs count.
            self.assertEqual(record_single_list["correlated"], record_single_scalar["correlated"])

    def test_run_and_log_bruteforce_honors_validation_metric(self):
        # (2026-07-31p) Real bug found and fixed: run_and_log_bruteforce's
        # CorrTrack(...) construction never passed validation_metric at all,
        # so the brute-force ground-truth baseline silently validated with
        # Pearson regardless of what the tuned/main run's validation_metric
        # was set to -- invalidating both the wall-clock baseline and any
        # recall/precision claim made against it for non-Pearson metrics.
        # y = x**3 is monotonic (Kendall tau exactly 1.0) but not perfectly
        # linear (Pearson correlation measurably lower), so the two metrics
        # must disagree on at least some pairs if validation_metric is
        # actually being honored.
        n_series = 6
        length = 400
        t = np.linspace(-3, 3, length)
        rng = np.random.default_rng(2)
        values = rng.normal(scale=0.01, size=(n_series, length))
        values[0] = t
        values[1] = t ** 3
        data = np.vstack([np.arange(length, dtype=np.float64), values])
        ids = [f"s{i}" for i in range(n_series)]

        base_config = dict(
            window_size=64, window_step=8, basic_window=None, n_lags=32,
            corr_threshold=0.85, neg_corr=True, exec="sequential",
            parallel_sketch=False, parallel_candidates=False, parallel_validation=False,
            max_workers=0, baseline_mode="bruteforce", monitor=True, track_min_dist=True,
            artifact_mode="final", artifact_buffer_max_rows=250000, artifact_merge_mode="merged",
            save_only_required_artifacts=True, save_maxlag_artifacts=False, verbose=False, testing=False,
        )

        with tempfile.TemporaryDirectory() as tmp:
            counts = {}
            for metric in ("pearson", "kendall"):
                cfg = dict(base_config, validation_metric=metric)
                record, _, _ = run_and_log_bruteforce(
                    "diag", data, ids, cfg, os.path.join(tmp, f"bf_{metric}.csv"),
                    metadata={"nodes": 0}, recall_by_window=True, verbose=False, testing=False,
                )
                counts[metric] = record["correlated"]

        self.assertNotEqual(counts["pearson"], counts["kendall"])

    def test_correlated_numeric_accumulator_int32_matches_int64_reference(self):
        # (2026-09-11) _append_correlated_numeric's internal storage was downsized
        # int64 -> int32 after a real OOM crash on a memory-constrained machine (a dense
        # real dataset needed a 640MB single allocation just for this one accumulator's
        # doubling-growth step). correlated_rows() also stopped defensively re-upcasting
        # to int64 on every call, after THAT turned out to reintroduce the same peak at
        # retrieval time (a second real OOM on the same machine, 536MB for one such
        # upcast) -- every real consumer already re-casts to int64 itself when it needs
        # to. Verifies the round-trip VALUES are exact for realistic AND boundary-ish
        # large values (t1/t2 well beyond any real dataset's observation count, still far
        # under int32's ~2.147e9 range), and that the returned dtype is int32 (the
        # accumulator's own storage, not a copy).
        ct = CorrTrack.__new__(CorrTrack)
        ct._correlated_rows = None
        ct._correlated_corrs = None
        ct._correlated_count = 0
        ct._correlated_cap = 0
        ct._correlated_legacy_dict = {}
        ct._correlated_view_cache = None
        ct.artifact_bookkeeping_time = 0.0

        rng = np.random.default_rng(0)
        # realistic-scale rows (small series indices, moderate time indices)
        n1 = 5000
        rows1 = np.column_stack([
            rng.integers(0, 200, n1), rng.integers(0, 200, n1),
            rng.integers(0, 100_000, n1), rng.integers(0, 100_000, n1),
            np.full(n1, 168),
        ]).astype(np.int64)
        corrs1 = rng.uniform(-1, 1, n1)
        ct._append_correlated_numeric(rows1, corrs1)

        # boundary-ish large time indices (tens of millions of steps -- far beyond any
        # real dataset, still comfortably under int32's range) to stress the actual
        # concern (t1/t2 overflow), not just small values.
        big_t = 50_000_000
        rows2 = np.array([
            [0, 1, big_t, big_t - 168, 168],
            [3, 199, big_t + 12345, big_t - 54321, 168],
        ], dtype=np.int64)
        corrs2 = np.array([0.71, -0.95])
        ct._append_correlated_numeric(rows2, corrs2)

        out_rows, out_corrs = ct.correlated_rows()
        # correlated_rows() returns the accumulator's own int32 storage directly (no
        # defensive re-copy -- see its docstring); every real consumer re-casts to int64
        # itself, verified separately (compute_metrics_bf_numeric/_as_row_corr_pair do).
        self.assertEqual(out_rows.dtype, np.int32)
        self.assertEqual(out_rows.shape[0], n1 + 2)

        # canonical orientation may reorder columns per-row (later start first) -- rebuild
        # the same canonicalization on the expected input before comparing as sets.
        expected = _canonicalize_rows(np.vstack([rows1, rows2]))
        expected_keys = set(map(tuple, expected.tolist()))
        actual_keys = set(map(tuple, out_rows.tolist()))
        self.assertEqual(actual_keys, expected_keys)
        self.assertEqual(int(out_rows[:, 2].max()), big_t + 12345)  # no truncation/overflow

    def test_filcorr_candidates_node_matches_bf_incremental_pearson_full_band(self):
        # (2026-09-11) FilCorr competitor baseline port (Zhong, Souza, Mueen --
        # ICDM 2020, from the colleague's feat/v2-engine branch -- see
        # docs/implementation_log.md's 2026-09-11 entry). Candidates_BF_FilCorr
        # at full band (fs=0.0, ft=0.5, DC removed) is mathematically identical
        # to standard Pearson -- that identity is exactly what makes it usable
        # against this project's own bruteforce ground truth, so it must hold
        # key-for-key and value-for-value, not just approximately. Checked at
        # both even and odd window sizes: a real off-by-one (odd m dropping the
        # top non-mirrored bin) and a real missing-Nyquist-bin weighting bug
        # (even m) were both found and fixed via this exact test during
        # implementation -- see Candidates_BF_FilCorr.__init__'s docstring.
        rng = np.random.default_rng(7)
        n_series = 6
        window_step = 8
        n_lags = 24
        corr_threshold = 0.2
        step_len = 4

        for window_size in (32, 33, 168, 169):
            # Enough steps for the window to actually fill (window_size <=
            # n_steps*step_len) plus a real margin for correlated lags to show up.
            n_steps = max(60, (window_size // step_len) + 30)
            with self.subTest(window_size=window_size):
                fc = Candidates_BF_FilCorr(
                    window_size, window_step, n_lags, corr_threshold, neg_corr=True,
                    filcorr_fs=0.0, filcorr_ft=0.5, filcorr_sampling_rate=1.0,
                )
                st = Candidates_BF_Incremental(
                    window_size, window_step, n_lags, corr_threshold, neg_corr=True,
                )
                ids = [f"s{i}" for i in range(n_series)]
                t = 0
                max_abs_err = 0.0
                total_mismatch_keys = 0
                any_accepted = False
                for step in range(n_steps):
                    block = rng.standard_normal((n_series, step_len))
                    if step > 5:
                        block[1] = block[0] + rng.standard_normal(step_len) * 0.05
                    idx = np.arange(t, t + step_len)
                    new_data_step = np.vstack([idx, block])
                    fc_rows, fc_corrs, _, _ = fc.run(
                        new_data_step, ids, verbose=False, testing=False,
                        track_min_dist=True, numeric_rows=True,
                    )
                    st_rows, st_corrs, _, _ = st.run(
                        new_data_step, ids, verbose=False, testing=False,
                        track_min_dist=True, numeric_rows=True,
                    )
                    ks_fc = {tuple(r): c for r, c in zip(fc_rows.tolist(), fc_corrs.tolist())}
                    ks_st = {tuple(r): c for r, c in zip(st_rows.tolist(), st_corrs.tolist())}
                    any_accepted = any_accepted or bool(ks_fc) or bool(ks_st)
                    total_mismatch_keys += len(set(ks_fc) ^ set(ks_st))
                    for key in set(ks_fc) & set(ks_st):
                        max_abs_err = max(max_abs_err, abs(ks_fc[key] - ks_st[key]))
                    t += step_len

                self.assertTrue(any_accepted, "no correlated pairs at all -- test data too weak")
                self.assertEqual(total_mismatch_keys, 0)
                self.assertLess(max_abs_err, 1e-9)

    def test_run_and_log_bruteforce_filcorr_matches_bruteforce_count(self):
        # End-to-end dispatch check (baseline_mode="filcorr" through
        # run_and_log_bruteforce -> CorrTrack.run_bf -> run_bf_filcorr ->
        # Candidates_BF_FilCorr), on top of the unit-level exact-equivalence
        # test above -- confirms the CLI/config wiring itself (not just the
        # math) reproduces the existing "bruteforce" baseline_mode's own
        # correlated-pair count at full band.
        n_series = 6
        length = 400
        rng = np.random.default_rng(3)
        t = np.linspace(-3, 3, length)
        values = rng.normal(scale=0.05, size=(n_series, length))
        values[0] = t
        values[1] = t + rng.normal(scale=0.02, size=length)
        data = np.vstack([np.arange(length, dtype=np.float64), values])
        ids = [f"s{i}" for i in range(n_series)]

        base_config = dict(
            window_size=32, window_step=8, basic_window=None, n_lags=16,
            corr_threshold=0.6, neg_corr=True, exec="sequential",
            parallel_sketch=False, parallel_candidates=False, parallel_validation=False,
            max_workers=0, monitor=True, track_min_dist=True,
            artifact_mode="final", artifact_buffer_max_rows=250000, artifact_merge_mode="merged",
            save_only_required_artifacts=True, save_maxlag_artifacts=False, verbose=False, testing=False,
            validation_metric="pearson",
        )

        with tempfile.TemporaryDirectory() as tmp:
            counts = {}
            for mode in ("bruteforce", "filcorr"):
                cfg = dict(base_config, baseline_mode=mode, filcorr_fs=0.0, filcorr_ft=0.5,
                           filcorr_sampling_rate=1.0)
                record, _, _ = run_and_log_bruteforce(
                    "diag_filcorr", data, ids, cfg, os.path.join(tmp, f"bf_{mode}.csv"),
                    metadata={"nodes": 0}, recall_by_window=True, verbose=False, testing=False,
                )
                counts[mode] = record["correlated"]

        self.assertGreater(counts["bruteforce"], 0)
        self.assertEqual(counts["filcorr"], counts["bruteforce"])

    def test_single_window_corrtrack_unaffected_by_multi_window_support(self):
        # A plain CorrTrack (not CorrTrackMultiWindow) must behave exactly
        # as before -- _multi_window_requested_offsets stays None/falsy for
        # the entire run, so every suffix-capture branch added for this
        # merge is a true no-op.
        rng = np.random.default_rng(3)
        n_steps = 64
        values = np.vstack([np.sin(np.arange(n_steps) / 3.0), rng.normal(size=n_steps)])
        data = np.vstack([np.arange(n_steps), values])
        ids = np.array(["a", "b"])
        ct = CorrTrack(
            window_size=16, basic_window=4, window_step=4, n_vectors=8, n_lags=8, seed=11, seed_toggle=22, corr_threshold=0.7, neg_corr=True, preprocess=False, exec="sequential", max_workers=1, parallel_sketch=False, parallel_candidates=False, parallel_validation=False, candidate_backend="lsh_sign_dot", candidate_cosine_threshold=0.3, freq_threshold=0,
        )
        step = 4
        for start in range(0, n_steps - step + 1, step):
            ct.run(data[:, start:start + step], ids, verbose=False, testing=False, corr_val=True, monitor=True)
            for node in ct.sketch_nodes:
                self.assertFalse(node._multi_window_requested_offsets)

    def test_resolve_validation_metric_defaults_and_aliases(self):
        # (2026-07-28) Non-Pearson validation metrics, ported from
        # corrtrack_release_nonlinear -- see docs/implementation_log.md's
        # 2026-07-28 entries. Unlike that variant's own resolver (no
        # "pearson" case at all), "pearson" is the default here and stays
        # on the fast Cython path.
        from library_corrtrack_parallel import _resolve_validation_metric
        self.assertEqual(_resolve_validation_metric(None), "pearson")
        self.assertEqual(_resolve_validation_metric(""), "pearson")
        self.assertEqual(_resolve_validation_metric("linear"), "pearson")
        self.assertEqual(_resolve_validation_metric("SpearmanR"), "spearman")
        self.assertEqual(_resolve_validation_metric("Kendall-Tau"), "kendall")
        self.assertEqual(_resolve_validation_metric("dcor"), "dist_corr")
        with self.assertRaises(ValueError):
            _resolve_validation_metric("not_a_metric")

    def test_metric_corr_and_dist_pearson_matches_fast_corr_and_dist(self):
        # Pearson must reuse _fast_corr_and_dist's own value directly (no
        # separate computation), so it stays exactly as fast/exact as the
        # pre-existing default path.
        from library_corrtrack_parallel import _metric_corr_and_dist, _fast_corr_and_dist
        rng = np.random.default_rng(5)
        x = rng.normal(size=32)
        y = x * 0.6 + rng.normal(scale=0.3, size=32)
        expected_corr, _dist = _fast_corr_and_dist(x, y)
        corr, dist = _metric_corr_and_dist(x, y, metric="pearson")
        self.assertAlmostEqual(corr, expected_corr, places=12)

    def test_metric_corr_and_dist_monotonic_nonlinear_relationship(self):
        # Spearman/Kendall should detect a monotonic-but-nonlinear
        # relationship that is still perfectly rank-correlated.
        from library_corrtrack_parallel import _metric_corr_and_dist
        x = np.linspace(-3, 3, 40)
        y = np.sign(x) * np.abs(x) ** 3  # monotonic transform of x
        spearman_corr, _ = _metric_corr_and_dist(x, y, metric="spearman")
        kendall_corr, _ = _metric_corr_and_dist(x, y, metric="kendall")
        self.assertAlmostEqual(spearman_corr, 1.0, places=6)
        self.assertAlmostEqual(kendall_corr, 1.0, places=6)

    def test_metric_corr_and_dist_dist_corr_detects_nonmonotonic_dependence(self):
        # Distance correlation should detect a symmetric, NON-monotonic
        # dependence (y = x^2) that Pearson/Spearman/Kendall structurally
        # cannot (a symmetric U-shape has ~zero linear/rank correlation).
        from library_corrtrack_parallel import _metric_corr_and_dist
        x = np.linspace(-3, 3, 61)
        y = x ** 2
        dist_corr, _ = _metric_corr_and_dist(x, y, metric="dist_corr")
        pearson_corr, _ = _metric_corr_and_dist(x, y, metric="pearson")
        self.assertGreater(dist_corr, 0.4)
        self.assertLess(abs(pearson_corr), 0.1)

    def test_huo_szekely_fast_matches_naive_distance_correlation_exactly(self):
        # (2026-07-29) Huo & Szekely-style O(w log w) exact distance
        # correlation -- verified directly against the existing naive
        # O(w^2) estimator (not trusted from derivation alone, since the
        # source paper's text was unavailable this session): random trials
        # with ties, linear, and nonlinear relationships, at varying sizes.
        from huo_szekely_distance_correlation import distance_correlation_1d_fast
        from library_corrtrack_parallel import _distance_correlation_1d

        rng = np.random.default_rng(0)
        max_diff = 0.0
        for trial in range(200):
            n = rng.integers(2, 60)
            kind = trial % 3
            if kind == 0:
                x = rng.integers(0, 5, size=n).astype(float)
                y = rng.integers(0, 5, size=n).astype(float)
            elif kind == 1:
                x = rng.normal(size=n)
                y = 0.5 * x + rng.normal(size=n) * 0.5
            else:
                x = rng.normal(size=n)
                y = x ** 2 + rng.normal(size=n) * 0.1
            fast = distance_correlation_1d_fast(x, y)
            slow = _distance_correlation_1d(x, y)
            if np.isnan(fast) and np.isnan(slow):
                continue
            max_diff = max(max_diff, abs(fast - slow))
        self.assertLess(max_diff, 1e-9)

    def test_huo_szekely_fast_edge_cases(self):
        from huo_szekely_distance_correlation import distance_correlation_1d_fast
        self.assertTrue(np.isnan(distance_correlation_1d_fast([1.0], [2.0])))
        self.assertTrue(np.isnan(distance_correlation_1d_fast([5.0] * 10, [3.0] * 10)))
        self.assertAlmostEqual(distance_correlation_1d_fast([1.0, 2.0], [3.0, 4.0]), 1.0, places=9)

    def test_corrtrack_dist_corr_algorithm_fast_matches_naive_end_to_end(self):
        rng = np.random.default_rng(20260729)
        n_steps = 80
        base = rng.normal(size=n_steps)
        values = np.vstack([base, base ** 2 + rng.normal(scale=0.05, size=n_steps)])
        data = np.vstack([np.arange(n_steps), values])
        ids = np.array(["a", "b"])

        def run_once(algo):
            ct = CorrTrack(
                window_size=16, basic_window=4, window_step=4, n_vectors=8, n_lags=8, seed=11, seed_toggle=22, corr_threshold=0.4, preprocess=False, exec="sequential", max_workers=1, parallel_sketch=False, parallel_candidates=False, parallel_validation=False, candidate_backend="lsh_sign_dot", candidate_cosine_threshold=0.3, freq_threshold=0, validation_metric="dist_corr", dist_corr_algorithm=algo,
            )
            step = 4
            for start in range(0, n_steps - step + 1, step):
                ct.run(data[:, start:start + step], ids, verbose=False, testing=False, corr_val=True, monitor=True)
            return ct

        ct_naive = run_once("naive")
        ct_fast = run_once("fast")
        self.assertEqual(ct_naive.dist_corr_algorithm, "naive")
        self.assertEqual(ct_fast.dist_corr_algorithm, "fast")
        self.assertEqual(
            {k: round(v, 6) for k, v in ct_naive.correlated.items()},
            {k: round(v, 6) for k, v in ct_fast.correlated.items()},
        )
        self.assertGreater(len(ct_naive.correlated), 0)

    def test_concordance_sketch_fresh_build_matches_verified_prototype(self):
        # (2026-07-29) ConcordanceSketchState -- built in response to
        # "can't I use my incremental sketching algorithm and nonstationary
        # normalization for screening Kendall candidates?" Cosine similarity
        # of two series' concatenated fixed-multiscale-gap sign vectors was
        # verified (standalone, before writing this module) to correlate
        # 0.9995 with true Kendall's tau across 300 synthetic trials -- this
        # test locks in that the module's fresh build reproduces exactly
        # the same numbers as that verification.
        from scipy.stats import kendalltau
        from concordance_sketch import ConcordanceSketchState

        rng = np.random.default_rng(5)
        w = 128
        x = rng.normal(size=w)
        y = 0.7 * x + rng.normal(size=w) * 0.5
        true_tau = kendalltau(x, y).correlation

        state = ConcordanceSketchState(window_size=w, n_gaps=16)
        state.update(np.vstack([x, y]), window_step=8)
        vecs = state.vectors()
        cos_sim = float(np.dot(vecs[0], vecs[1]))
        self.assertLess(abs(cos_sim - true_tau), 0.1)

    def test_concordance_sketch_incremental_update_matches_fresh_rebuild_exactly(self):
        # The whole point is that sliding the window reuses unchanged
        # comparisons instead of recomputing everything -- verify this is
        # not just "close" but bit-identical to an independent fresh build
        # at every single step, mirroring the same discipline used for
        # xiao_online_correlation.py's incremental update.
        from concordance_sketch import ConcordanceSketchState

        rng = np.random.default_rng(11)
        n_series, window_size, window_step, n_gaps = 3, 96, 8, 16
        n_steps = 300
        full_series = rng.normal(size=(n_series, n_steps))
        full_series[1] = 0.6 * full_series[0] + full_series[1] * 0.4
        full_series[2] = full_series[0] ** 2 * 0.3 + full_series[2]

        state = ConcordanceSketchState(window_size=window_size, n_gaps=n_gaps)
        for t1 in range(0, n_steps - window_size + 1, window_step):
            window = full_series[:, t1:t1 + window_size]
            state.update(window, window_step=window_step)
            vecs_incremental = state.vectors()

            fresh = {}
            for g in state.gaps:
                g = int(g)
                fresh[g] = np.sign(window[:, g:] - window[:, :-g])
            raw = np.concatenate([fresh[int(g)] for g in state.gaps], axis=1)
            vecs_fresh = raw / np.linalg.norm(raw, axis=1, keepdims=True)

            self.assertLess(np.max(np.abs(vecs_incremental - vecs_fresh)), 1e-12)

    def test_concordance_sketch_robust_to_nonstationary_drift(self):
        # The motivating property: sign(x[t]-x[t-g]) is scale/location-
        # invariant, so this representation should NOT suffer the
        # degeneracy Xiao's value-binning validator needed a reactive fix
        # for. Verified directly: 0 NaN, 0% threshold-classification
        # mismatch on the same random-walk stress test that produced NaN
        # and a 5.76% mismatch rate for the Xiao validator.
        from scipy.stats import kendalltau
        from concordance_sketch import ConcordanceSketchState

        rng = np.random.default_rng(2)
        n, window_size, window_step = 2000, 64, 8
        rw = np.cumsum(rng.normal(0, 1.0, n))
        y_rw = 0.85 * rw + 0.15 * rng.normal(0, 1.0, n) * rw.std()
        data = np.vstack([rw, y_rw])

        state = ConcordanceSketchState(window_size=window_size, n_gaps=16)
        nan_count = mismatches = total = 0
        for t1 in range(0, n - window_size + 1, window_step):
            window = data[:, t1:t1 + window_size]
            state.update(window, window_step=window_step)
            vecs = state.vectors()
            cos_sim = float(np.dot(vecs[0], vecs[1]))
            true_tau = kendalltau(rw[t1:t1 + window_size], y_rw[t1:t1 + window_size]).correlation
            if not np.isfinite(cos_sim):
                nan_count += 1
                continue
            total += 1
            if (abs(cos_sim) >= 0.7) != (abs(true_tau) >= 0.7):
                mismatches += 1
        self.assertEqual(nan_count, 0)
        self.assertEqual(mismatches, 0)

    # (2026-07-31, later) test_corrtrack_incremental_concordance_backend_
    # resolves_correctly removed along with candidate_backend=
    # "incremental_concordance" itself -- its assertions (candidate_index_
    # backend="auto", candidate_backend_effective="lsh_sign_dot") were
    # inherently about the single-vector architecture (routing through
    # Candidates' own generic index), which no longer exists; equivalent
    # coverage for the surviving multichannel backend already exists in
    # test_corrtrack_concordance_multichannel_backend_resolves_correctly.

    def test_concordance_sketch_target_dim_bounds_output_dim_across_window_sizes(self):
        # (2026-07-29) output_dim = sum(window_size - g for g in gaps) used
        # to scale LINEARLY with window_size (n_gaps was fixed), verified
        # directly to cost 2.6x-20.5x more per step than
        # GlobalOrdinalTransformer at window_size=256/1024 despite cheaper
        # incremental bookkeeping. target_dim reduces n_gaps as window_size
        # grows so output_dim stays roughly bounded instead -- verified here
        # that it actually does, and that an explicit n_gaps still overrides it.
        from concordance_sketch import ConcordanceSketchState

        dims = {}
        for w in (64, 256, 1024):
            state = ConcordanceSketchState(window_size=w, target_dim=738)
            dims[w] = state.output_dim
        # All three should land in the same order of magnitude as the
        # target, unlike the old fixed-n_gaps behavior (641/3020/13621).
        for w, dim in dims.items():
            self.assertLess(dim, 1200, f"output_dim={dim} at window_size={w} not bounded near target_dim")

        explicit = ConcordanceSketchState(window_size=256, n_gaps=16)
        self.assertGreater(explicit.output_dim, 2000)  # explicit n_gaps bypasses target_dim entirely

    def test_concordance_sketch_target_dim_still_bit_exact_incrementally(self):
        from concordance_sketch import ConcordanceSketchState

        rng = np.random.default_rng(11)
        n_series, window_size, window_step = 3, 256, 16
        n_steps = 800
        full_series = rng.normal(size=(n_series, n_steps))
        full_series[1] = 0.6 * full_series[0] + full_series[1] * 0.4
        full_series[2] = full_series[0] ** 2 * 0.3 + full_series[2]

        state = ConcordanceSketchState(window_size=window_size, target_dim=738)
        for t1 in range(0, n_steps - window_size + 1, window_step):
            window = full_series[:, t1:t1 + window_size]
            state.update(window, window_step=window_step)
            vecs_incremental = state.vectors()
            fresh = {}
            for g in state.gaps:
                g = int(g)
                fresh[g] = np.sign(window[:, g:] - window[:, :-g])
            raw = np.concatenate([fresh[int(g)] for g in state.gaps], axis=1)
            vecs_fresh = raw / np.linalg.norm(raw, axis=1, keepdims=True)
            self.assertLess(np.max(np.abs(vecs_incremental - vecs_fresh)), 1e-12)

    def test_corrtrack_concordance_multichannel_default_bounds_dim_via_target_dim(self):
        # (2026-07-31, later) Migrated from single-vector concordance
        # (removed) to multichannel -- concordance_n_gaps/concordance_
        # target_dim/n_vectors all come from the SAME shared
        # ConcordanceSketchState construction, unaffected by which
        # retrieval index (single combined vector vs. per-gap union) is
        # built on top of it.
        ct = CorrTrack(
            window_size=256, basic_window=16, window_step=8, n_vectors=32, n_lags=32, corr_threshold=0.6, data_representation="sketch_concordance", validation_metric="kendall",
        )
        self.assertIsNone(ct.concordance_n_gaps)
        self.assertEqual(ct.concordance_target_dim, 738)
        self.assertLess(ct.n_vectors, 1200)

        ct_explicit = CorrTrack(
            window_size=256, basic_window=16, window_step=8, n_vectors=32, n_lags=32, corr_threshold=0.6, data_representation="sketch_concordance", validation_metric="kendall", concordance_n_gaps=16,
        )
        self.assertEqual(ct_explicit.concordance_n_gaps, 16)
        self.assertIsNone(ct_explicit.concordance_target_dim)
        self.assertGreater(ct_explicit.n_vectors, 2000)

    def test_multiscale_gaps_respects_min_gap_and_capacity_floor(self):
        # (2026-07-29f) min_gap excludes gap=1 (highest-weight, worst tau
        # estimator on autocorrelated real data); min_capacity excludes
        # gaps near window_size (lowest-weight but highest-variance, as
        # few as 1 sample). Every returned gap must respect both bounds.
        from concordance_sketch import multiscale_gaps

        window_size = 256
        gaps = multiscale_gaps(window_size, n_gaps=8, min_gap=8, min_capacity=16)
        self.assertTrue(np.all(gaps >= 8))
        self.assertTrue(np.all(window_size - gaps >= 16))

    def test_concordance_sketch_capacity_floor_improves_accuracy_on_autocorrelated_data(self):
        # (2026-07-29f) Diagnosed on real hourly wind-speed data: the OLD
        # defaults (min_gap=1, no capacity floor) gave mean|cos-true_tau|
        # =0.116 vs 0.027 predicted by i.i.d.-noise synthetic testing --
        # because gap=1 (highest weight under whole-vector L2
        # normalization) is dominated by local jitter on smooth,
        # autocorrelated data, and the longest gaps (lowest weight) are
        # individually near-useless (as few as 1 sample). This test
        # reproduces the mechanism with a synthetic AR(1)-smoothed series
        # (autocorrelated, unlike the plain i.i.d. tests above) and checks
        # the new defaults beat the old ones -- a regression guard for the
        # fix, not just a re-statement of "it works."
        from scipy.stats import kendalltau
        from concordance_sketch import multiscale_gaps

        def combined_cos(x, y, gaps):
            cx = np.concatenate([np.sign(x[g:] - x[:-g]) for g in gaps])
            cy = np.concatenate([np.sign(y[g:] - y[:-g]) for g in gaps])
            nx, ny = np.linalg.norm(cx), np.linalg.norm(cy)
            return float(np.dot(cx, cy) / (nx * ny)) if nx > 0 and ny > 0 else 0.0

        rng = np.random.default_rng(3)
        window_size = 256

        def ar1_smooth(n, phi=0.9):
            eps = rng.normal(size=n)
            out = np.empty(n)
            out[0] = eps[0]
            for i in range(1, n):
                out[i] = phi * out[i - 1] + eps[i]
            return out

        old_gaps = multiscale_gaps(window_size, n_gaps=3, min_gap=1, min_capacity=1)
        new_gaps = multiscale_gaps(window_size, n_gaps=3, min_gap=8, min_capacity=16)

        old_diffs, new_diffs = [], []
        for _ in range(40):
            x = ar1_smooth(window_size)
            y = 0.6 * x + ar1_smooth(window_size) * 0.5
            tau = kendalltau(x, y).correlation
            if not np.isfinite(tau):
                continue
            old_diffs.append(abs(combined_cos(x, y, old_gaps) - tau))
            new_diffs.append(abs(combined_cos(x, y, new_gaps) - tau))

        self.assertLess(np.mean(new_diffs), np.mean(old_diffs))

    def test_corrtrack_concordance_multichannel_rejects_dist_corr(self):
        # (2026-07-31, later) Migrated from single-vector concordance
        # (removed) -- the same NotImplementedError check applies to
        # multichannel (both are gated by the shared concordance setup
        # block's validation_metric check).
        with self.assertRaises(NotImplementedError):
            CorrTrack(
                window_size=16, basic_window=4, window_step=4, n_vectors=8, n_lags=8, corr_threshold=0.7, data_representation="sketch_concordance", validation_metric="dist_corr",
            )

    def test_corrtrack_concordance_multichannel_finds_nonlinear_kendall_pair(self):
        # (2026-07-31, later) Migrated from single-vector concordance
        # (removed) -- real, kendall-specific end-to-end coverage (the
        # existing multichannel tests elsewhere in this file use spearman)
        # for a monotonic-nonlinear (cubic) relationship.
        rng = np.random.default_rng(42)
        n_series, n_steps = 6, 240
        window_size, basic_window, window_step = 64, 16, 8
        t = np.linspace(-1, 1, n_steps)
        base = np.sin(t * 6) + rng.normal(0, 0.05, n_steps)
        x0, x1 = base, base ** 3
        others = rng.normal(0, 1, (n_series - 2, n_steps))
        values = np.vstack([x0, x1, others])
        data = np.vstack([np.arange(n_steps), values])
        ids = [f"s{i}" for i in range(n_series)]

        ct = CorrTrack(
            window_size=window_size, basic_window=basic_window, window_step=window_step, n_vectors=64, n_lags=32, corr_threshold=0.6, neg_corr=True, data_representation="sketch_concordance", validation_metric="kendall", exec="sequential", numeric_rows=True,
        )
        step = 8
        for start in range(0, n_steps - step + 1, step):
            ct.run(data[:, start:start + step], ids, verbose=False, testing=False, corr_val=True, monitor=True)
        found = any({pair[0], pair[1]} == {"s0", "s1"} for pair in ct.correlated.keys())
        self.assertTrue(found, "expected the monotonic-nonlinear s0/s1 pair to be found")

    def test_distance_corr_sketch_incremental_update_matches_fresh_rebuild_exactly(self):
        # (2026-07-30) DistanceCorrSketchState -- see distance_corr_sketch.py's
        # module docstring. Mirrors the existing concordance-sketch bit-
        # exactness test exactly: each channel's ring buffer must reproduce
        # a from-scratch rebuild bit-for-bit at every sliding step.
        from distance_corr_sketch import DistanceCorrSketchState, _channels_for_window

        rng = np.random.default_rng(11)
        n_series, window_size, window_step, K = 3, 96, 8, 8
        n_steps = 300
        full_series = rng.normal(size=(n_series, n_steps))
        full_series[1] = 0.6 * full_series[0] + full_series[1] * 0.4
        full_series[2] = full_series[0] ** 2 * 0.3 + full_series[2]

        state = DistanceCorrSketchState(window_size=window_size, K=K)
        for t1 in range(0, n_steps - window_size + 1, window_step):
            window = full_series[:, t1:t1 + window_size]
            state.update(window, window_step=window_step)
            vecs_incremental = state.vectors()

            chans = _channels_for_window(window, state.freqs)
            parts = [c - c.mean(axis=1, keepdims=True) for c in chans]
            raw = np.concatenate(parts, axis=1)
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            vecs_fresh = raw / norms

            self.assertLess(np.max(np.abs(vecs_incremental - vecs_fresh)), 1e-12)

    def test_distance_corr_sketch_proxy_reliability_against_naive_dcor(self):
        # (2026-07-30) The tier-2 refinement gate's core claim: Spearman
        # rank-correlation with true (naive, exact) distance correlation
        # should be strong, INCLUDING for a NON-MONOTONIC relationship
        # Pearson/Kendall are structurally blind to. Deliberately a
        # WITHIN-type controlled noise sweep (same functional form, varying
        # noise scale), not a mixed-type random batch -- an earlier version
        # of this test mixed types with mismatched noise-scale conventions
        # and got a strongly NEGATIVE aggregate correlation purely from a
        # between-cluster ordering artifact (confirmed by inspecting each
        # cluster's own mean true/proxy values), not a real proxy failure.
        # This is exactly the "mixed random-batch tests can mislead"
        # methodology lesson from this session's own investigation (see
        # docs/implementation_log.md's 2026-07-30 entries) -- the
        # regression guard has to respect it too, not just the analysis
        # that found it.
        from scipy.stats import spearmanr
        from distance_corr_sketch import multiscale_freqs, distance_corr_sketch_proxy
        from library_corrtrack_parallel import _distance_correlation_1d

        rng = np.random.default_rng(5)
        freqs = multiscale_freqs(8, low=0.1, high=10.0, seed=9)
        n = 200
        for kind in ("linear", "cubic", "quadratic_nonmonotonic"):
            true_vals, proxy_vals = [], []
            for noise in np.linspace(0.1, 2.0, 15):
                for _ in range(3):
                    x = rng.normal(size=n)
                    if kind == "linear":
                        y = x + rng.normal(size=n) * noise
                    elif kind == "cubic":
                        y = x ** 3 + rng.normal(size=n) * noise * 3
                    else:
                        y = x ** 2 + rng.normal(size=n) * noise
                    true_vals.append(_distance_correlation_1d(x, y))
                    proxy_vals.append(abs(distance_corr_sketch_proxy(x, y, freqs)))
            rho, _ = spearmanr(true_vals, proxy_vals)
            self.assertGreater(rho, 0.7, f"{kind}: expected strong rank correlation with true dCor, got {rho:.3f}")

    def test_derive_gate_tau_from_corr_threshold_scales_with_window_size(self):
        # (2026-07-30) Real gap found and fixed during implementation: this
        # statistic's independent-pair noise floor scales as ~1/sqrt(w) --
        # verified directly (mean proxy * sqrt(w) is ~constant, 0.98-1.01,
        # across w=32..1024). A gate_tau calibrated at one window size and
        # used verbatim at a much smaller one is too lenient (never
        # rejects anything); this guards the fix stays in place.
        from distance_corr_sketch import derive_gate_tau_from_corr_threshold

        tau_256 = derive_gate_tau_from_corr_threshold(0.5, window_size=256)
        tau_64 = derive_gate_tau_from_corr_threshold(0.5, window_size=64)
        self.assertAlmostEqual(tau_64, tau_256 * 2.0, places=6)  # sqrt(256/64) == 2

    # (2026-07-31, Phase 4) test_corrtrack_distance_corr_sketch_backend_
    # resolves_correctly / _requires_dist_corr_metric / _finds_nonmonotonic_
    # pair removed along with single-vector candidate_backend=
    # "distance_corr_sketch" itself -- distance_corr_sketch_multichannel
    # strictly dominates its recall at every K tested with no offsetting
    # tradeoff (see docs/implementation_log.md's 2026-07-31 entry), and the
    # equivalent coverage already exists under the multichannel backend's
    # own tests further below.

    def test_corrtrack_distance_corr_sketch_multichannel_gate_rejects_independent_pairs(self):
        # (2026-07-31, Phase 4) Migrated from the single-vector backend
        # (now removed) -- the tier-2 gate itself is shared machinery,
        # unaffected by that removal. Real regression guard for the
        # noise-floor-scaling bug found earlier this session: the tier-2
        # gate must actually engage (reject some candidates) on genuinely
        # independent data, not silently pass everything through to
        # expensive exact validation.
        rng = np.random.default_rng(3)
        n_series, n_steps = 8, 300
        window_size, basic_window, window_step = 64, 16, 8
        values = rng.normal(size=(n_series, n_steps))  # all independent noise
        data = np.vstack([np.arange(n_steps), values])
        ids = [f"s{i}" for i in range(n_series)]

        ct = CorrTrack(
            window_size=window_size, basic_window=basic_window, window_step=window_step, n_vectors=32, n_lags=16, corr_threshold=0.5, neg_corr=True, data_representation="sketch_multichannel", validation_metric="dist_corr", exec="sequential", numeric_rows=True,
        )
        step = 8
        for start in range(0, n_steps - step + 1, step):
            ct.run(data[:, start:start + step], ids, verbose=False, testing=False, corr_val=True, monitor=True)
        self.assertGreater(ct.distance_corr_sketch_gate_rejected, 0)
        self.assertEqual(len(ct.correlated), 0, "no genuinely independent pair should be found correlated")

    def test_corrtrack_validation_metric_default_pearson_uses_fast_path(self):
        # A CorrTrack constructed without validation_metric must default to
        # "pearson" and route through the unchanged, existing Cython bulk
        # path -- validation_metric="pearson" must be a true no-op relative
        # to not passing the parameter at all.
        rng = np.random.default_rng(20260728)
        n_steps = 80
        base = rng.normal(size=n_steps)
        values = np.vstack([base, base * 0.95 + rng.normal(scale=0.05, size=n_steps)])
        data = np.vstack([np.arange(n_steps), values])
        ids = np.array(["a", "b"])

        def run_once(**extra):
            kwargs = dict(
                window_size=16, basic_window=4, window_step=4, n_vectors=8, n_lags=8, seed=11, seed_toggle=22, corr_threshold=0.7, neg_corr=True, preprocess=False, exec="sequential", max_workers=1, parallel_sketch=False, parallel_candidates=False, parallel_validation=False, candidate_backend="lsh_sign_dot", candidate_cosine_threshold=0.3, freq_threshold=0,
            )
            kwargs.update(extra)
            ct = CorrTrack(**kwargs)
            step = 4
            for start in range(0, n_steps - step + 1, step):
                ct.run(data[:, start:start + step], ids, verbose=False, testing=False, corr_val=True, monitor=True)
            return ct

        default_ct = run_once()
        explicit_pearson_ct = run_once(validation_metric="pearson")
        self.assertEqual(default_ct.validation_metric, "pearson")
        self.assertEqual(
            {k: round(v, 6) for k, v in default_ct.correlated.items()},
            {k: round(v, 6) for k, v in explicit_pearson_ct.correlated.items()},
        )
        self.assertGreater(len(default_ct.correlated), 0)

    def test_corrtrack_spearman_metric_finds_correlated_pair(self):
        # End-to-end: a non-Pearson metric routes through
        # _validate_numeric_rows_nonlinear (the new per-row Python path)
        # and still finds a genuinely correlated pair.
        rng = np.random.default_rng(20260728)
        n_steps = 80
        base = rng.normal(size=n_steps)
        values = np.vstack([base, base * 0.95 + rng.normal(scale=0.05, size=n_steps)])
        data = np.vstack([np.arange(n_steps), values])
        ids = np.array(["a", "b"])
        ct = CorrTrack(
            window_size=16, basic_window=4, window_step=4, n_vectors=8, n_lags=8, seed=11, seed_toggle=22, corr_threshold=0.7, neg_corr=True, preprocess=False, exec="sequential", max_workers=1, parallel_sketch=False, parallel_candidates=False, parallel_validation=False, candidate_backend="lsh_sign_dot", candidate_cosine_threshold=0.3, freq_threshold=0, validation_metric="spearman",
        )
        step = 4
        for start in range(0, n_steps - step + 1, step):
            ct.run(data[:, start:start + step], ids, verbose=False, testing=False, corr_val=True, monitor=True)
        self.assertEqual(ct.validation_metric, "spearman")
        self.assertGreater(len(ct.correlated), 0)
        for corr in ct.correlated.values():
            self.assertGreaterEqual(abs(corr), 0.7)

    def test_xiao_online_corr_fresh_build_matches_scipy_within_reported_error(self):
        # (2026-07-29) Xiao (2017)'s own paper reports L1 error on the order
        # of 1e-3 (Spearman) to 1e-2 (Kendall) for a fresh count-matrix
        # build -- verify directly against scipy rather than trust the
        # paper's numbers blindly.
        from scipy.stats import kendalltau, spearmanr
        from xiao_online_correlation import XiaoOnlineCorrState

        rng = np.random.default_rng(0)
        x = rng.normal(size=200)
        y = 0.7 * x + rng.normal(size=200) * 0.3
        state = XiaoOnlineCorrState()

        approx_sr = state.get_corr(("a", "b", 0), x, y, t1=1000, window_step=8, metric="spearman", m1=30)
        exact_sr = spearmanr(x, y).correlation
        self.assertLess(abs(approx_sr - exact_sr), 0.01)

        state2 = XiaoOnlineCorrState()
        approx_kt = state2.get_corr(("a", "b", 0), x, y, t1=1000, window_step=8, metric="kendall", m1=100)
        exact_kt = kendalltau(x, y).correlation
        self.assertLess(abs(approx_kt - exact_kt), 0.03)

    def test_xiao_online_corr_incremental_update_matches_fresh_rebuild_exactly(self):
        # The whole point of the incremental path is that it reproduces
        # EXACTLY what a fresh rebuild using the same fixed cutpoints would
        # give -- not just "close", bit-identical bookkeeping.
        from xiao_online_correlation import (
            XiaoOnlineCorrState,
            _bin_indices,
            _spearman_from_matrix,
        )

        rng = np.random.default_rng(1)
        n_steps, window_size, window_step, m1 = 400, 64, 8, 30
        base = rng.normal(size=n_steps)
        x_full = base
        y_full = 0.6 * base + rng.normal(size=n_steps) * 0.4

        state = XiaoOnlineCorrState()
        key = ("s0", "s1", 0)
        for t1 in range(0, n_steps - window_size + 1, window_step):
            x = x_full[t1:t1 + window_size]
            y = y_full[t1:t1 + window_size]
            incr_corr = state.get_corr(key, x, y, t1=t1, window_step=window_step, metric="spearman", m1=m1)

            st = state._states[key]
            row_idx = _bin_indices(x, st["cx"])
            col_idx = _bin_indices(y, st["cy"])
            M = np.zeros((m1, m1), dtype=np.int64)
            np.add.at(M, (row_idx, col_idx), 1)
            fresh_corr = _spearman_from_matrix(M, M.sum(axis=1), M.sum(axis=0), int(M.sum()))
            self.assertAlmostEqual(incr_corr, fresh_corr, places=9)

    def test_xiao_online_corr_survives_nonstationary_drift_without_nan(self):
        # A genuinely nonstationary series (random walk) can drift far
        # enough outside the fixed cutpoints derived when a pair-lag was
        # first seen that every point collapses into one extreme bin,
        # degenerating the correlation formula's denominator to 0. Verify
        # the rebuild-on-degeneracy fallback catches this rather than
        # letting NaN propagate.
        from xiao_online_correlation import XiaoOnlineCorrState

        rng = np.random.default_rng(2)
        n, window_size, window_step, m1 = 2000, 64, 8, 30
        rw = np.cumsum(rng.normal(0, 1.0, n))
        y_rw = 0.6 * rw + 0.4 * rng.normal(0, 1.0, n) * rw.std()

        state = XiaoOnlineCorrState()
        key = ("s0", "s1", 0)
        saw_nan = False
        for t1 in range(0, n - window_size + 1, window_step):
            corr = state.get_corr(
                key, rw[t1:t1 + window_size], y_rw[t1:t1 + window_size], t1=t1, window_step=window_step, metric="spearman", m1=m1,
            )
            if not math.isfinite(corr):
                saw_nan = True
        self.assertFalse(saw_nan, "rebuild-on-degeneracy fallback should prevent NaN")

    def test_xiao_online_corr_adaptive_refresh_improves_nonstationary_accuracy(self):
        # (2026-07-29) HBR-style adaptive cutpoint refresh: proactively
        # re-derive cutpoints once the window mean has drifted more than
        # cutpoint_refresh_threshold standard deviations, instead of only
        # reactively rebuilding once the matrix has already degenerated to
        # NaN. Verified as a REAL improvement (not just "runs without
        # crashing"): threshold-classification (|corr|>=0.7) mismatch rate
        # against exact scipy on a random-walk stress test, measured
        # directly, drops from 5.76% (reactive-only) to 1.23% (threshold=0.5).
        from scipy.stats import spearmanr
        from xiao_online_correlation import XiaoOnlineCorrState

        def mismatch_rate(refresh_threshold):
            rng = np.random.default_rng(2)
            n, window_size, window_step, m1 = 2000, 64, 8, 30
            rw = np.cumsum(rng.normal(0, 1.0, n))
            y_rw = 0.85 * rw + 0.15 * rng.normal(0, 1.0, n) * rw.std()
            state = XiaoOnlineCorrState()
            key = ("s0", "s1", 0)
            mismatches = total = 0
            for t1 in range(0, n - window_size + 1, window_step):
                approx = state.get_corr(
                    key, rw[t1:t1 + window_size], y_rw[t1:t1 + window_size], t1=t1, window_step=window_step, metric="spearman", m1=m1, cutpoint_refresh_threshold=refresh_threshold,
                )
                exact = spearmanr(rw[t1:t1 + window_size], y_rw[t1:t1 + window_size]).correlation
                if not np.isfinite(approx) or not np.isfinite(exact):
                    continue
                total += 1
                if (abs(approx) >= 0.7) != (abs(exact) >= 0.7):
                    mismatches += 1
            return mismatches / total

        rate_reactive_only = mismatch_rate(None)
        rate_with_refresh = mismatch_rate(0.5)
        self.assertLess(rate_with_refresh, rate_reactive_only)
        self.assertLess(rate_with_refresh, 0.03)

    def test_corrtrack_validation_incremental_approx_default_off_is_true_noop(self):
        # validation_incremental_approx must default to False and leave
        # spearman validation on the existing exact scipy path untouched.
        ct = CorrTrack(
            window_size=16, basic_window=4, window_step=4, n_vectors=8, n_lags=8, corr_threshold=0.7, validation_metric="spearman",
        )
        self.assertFalse(ct.validation_incremental_approx)
        self.assertIsNone(ct._xiao_state)

    def test_corrtrack_validation_incremental_approx_finds_correlated_pair(self):
        rng = np.random.default_rng(20260729)
        n_series, n_steps = 6, 400
        window_size, basic_window, window_step = 64, 16, 8
        base = rng.normal(size=n_steps)
        x0 = base
        x1 = 0.85 * base + rng.normal(0, 0.3, n_steps)
        others = rng.normal(0, 1, (n_series - 2, n_steps))
        values = np.vstack([x0, x1, others])
        data = np.vstack([np.arange(n_steps), values])
        ids = [f"s{i}" for i in range(n_series)]

        ct = CorrTrack(
            window_size=window_size, basic_window=basic_window, window_step=window_step, n_vectors=8, n_lags=32, corr_threshold=0.7, candidate_backend="lsh_sign_dot", candidate_cosine_threshold=0.3, validation_metric="spearman", validation_incremental_approx=True, exec="sequential", numeric_rows=True,
        )
        self.assertIsNotNone(ct._xiao_state)
        step = 8
        for start in range(0, n_steps - step + 1, step):
            ct.run(data[:, start:start + step], ids, verbose=False, testing=False, corr_val=True, monitor=True)
        found = any({pair[0], pair[1]} == {"s0", "s1"} for pair in ct.correlated.keys())
        self.assertTrue(found, "expected the strongly correlated s0/s1 pair to be found")
        self.assertGreater(ct._xiao_state.size(), 0)

    # (2026-07-31, Phase 4) test_corrtrack_flat_backend_respects_non_pearson_
    # validation_metric removed along with candidate_backend="flat" itself:
    # "flat" was the ONLY remaining backend routing through
    # _get_validated_corr's legacy string-keyed candidates branch
    # (_validate_pairs_nonlinear/_validate_pairs_standard) -- every other
    # live backend populates self._candidate_numeric_rows directly. That
    # code path is now unreachable (left in place as harmless dead code,
    # per the 2026-07-27(e) precedent for other removed backends' own
    # dispatch conditionals), so this regression guard can no longer run.
    # Decided explicitly with the user (not silently dropped): full
    # removal accepted, since no live backend exercises this path anymore.

    def test_corrtrack_brute_force_backend_matches_exact_hand_computed_pair_count(self):
        # (2026-07-30) candidate_backend="brute_force": a genuinely
        # unconditional, metric-agnostic enumeration added after finding
        # "flat" cannot be trusted as ground truth/wall-time reference for
        # non-Pearson metrics (its own candidate generation always gates
        # through the default Pearson-style sketch, regardless of
        # validation_metric -- see docs/implementation_log.md's
        # 2026-07-30(g)/(i) entries). On a single step with no prior
        # history (so no self-lag pairs are even possible yet), the exact
        # candidate count must be C(M,2) -- checked against math.comb, not
        # just "greater than zero".
        import math
        rng = np.random.default_rng(20260730)
        M = 5
        w = 16
        values = rng.normal(size=(M, w))
        data = np.vstack([np.arange(w), values])
        ids = [f"s{i}" for i in range(M)]
        ct = CorrTrack(
            window_size=w, basic_window=w, window_step=w, n_vectors=8, n_lags=8, corr_threshold=0.05, neg_corr=True, candidate_backend="brute_force", validation_metric="dist_corr", exec="sequential", numeric_rows=True,
        )
        ct.run(data[:, 0:w], ids, verbose=False, testing=False, corr_val=True, monitor=False)
        self.assertEqual(ct.tested_candidates, math.comb(M, 2))

    def test_corrtrack_brute_force_backend_candidate_count_independent_of_threshold(self):
        # The defining property that distinguishes true brute force from
        # "flat": candidate GENERATION must not depend on corr_threshold at
        # all (only which validated pairs get accepted should vary).
        rng = np.random.default_rng(20260730)
        M, n_steps, w, step = 24, 96, 32, 16
        values = rng.normal(size=(M, n_steps))
        data = np.vstack([np.arange(n_steps), values])
        ids = [f"s{i}" for i in range(M)]

        def run_at(thr):
            ct = CorrTrack(
                window_size=w, basic_window=16, window_step=step, n_vectors=8, n_lags=8, corr_threshold=thr, neg_corr=True, candidate_backend="brute_force", validation_metric="dist_corr", exec="sequential", numeric_rows=True,
            )
            for start in range(0, n_steps - step + 1, step):
                ct.run(data[:, start:start + step], ids, verbose=False, testing=False, corr_val=True, monitor=False)
            return ct.tested_candidates

        self.assertEqual(run_at(0.05), run_at(0.5))
        self.assertEqual(run_at(0.5), run_at(0.9))

    def test_distance_corr_sketch_proxy_cy_matches_independent_numpy_reimplementation(self):
        # (2026-07-30) distance_corr_sketch_proxy is now Cython-only (no
        # Python fallback), per explicit instruction ("all gates for all
        # backends should be Cython, I want no Python"). Regression guard
        # against a fresh, independent reimplementation of the same math
        # (not the retired Python source, which no longer exists) -- build
        # the 2K per-channel mean-centered series for x/y directly and
        # the (2K)x(2K) cosine-similarity grid, aggregated via sqrt(mean of
        # squares).
        from distance_corr_sketch import distance_corr_sketch_proxy, multiscale_freqs

        def independent_reference(x, y, freqs):
            chx, chy = [], []
            for w in freqs:
                chx.append(np.cos(w * x))
                chx.append(np.sin(w * x))
                chy.append(np.cos(w * y))
                chy.append(np.sin(w * y))
            chx = [c - c.mean() for c in chx]
            chy = [c - c.mean() for c in chy]
            nx = [np.linalg.norm(c) for c in chx]
            ny = [np.linalg.norm(c) for c in chy]
            total, n_terms = 0.0, 0
            for i, cx in enumerate(chx):
                if nx[i] == 0.0:
                    continue
                for j, cy in enumerate(chy):
                    if ny[j] == 0.0:
                        continue
                    cos = np.dot(cx, cy) / (nx[i] * ny[j])
                    total += cos * cos
                    n_terms += 1
            return float(np.sqrt(total / n_terms)) if n_terms else 0.0

        rng = np.random.default_rng(20260730)
        freqs = multiscale_freqs(8, seed=7)
        for trial in range(50):
            w = int(rng.integers(16, 200))
            x = rng.normal(size=w)
            y = x ** 2 + rng.normal(scale=0.2, size=w) if trial % 2 == 0 else rng.normal(size=w)
            got = distance_corr_sketch_proxy(x, y, freqs)
            expected = independent_reference(x, y, freqs)
            self.assertAlmostEqual(got, expected, places=8)

    # (2026-07-31, Phase 4) test_corrtrack_distance_corr_sketch_exhaustive_*
    # (backend_resolves_correctly / requires_dist_corr_metric /
    # touches_every_pair_unlike_tier1 / finds_nonmonotonic_pair) removed
    # along with candidate_backend="distance_corr_sketch_exhaustive"
    # itself -- it existed specifically to preserve single-vector
    # distance_corr_sketch's own tier-2 gate under unconditional
    # enumeration; with that representation gone (see the removal note
    # above), there is nothing left for this name to be an alternative to.

    def test_corrtrack_distance_corr_sketch_multichannel_backend_resolves_correctly(self):
        # (2026-07-30) Per-channel union tier-1 -- 2K separate
        # SignLSHBandIndex instances (one per RFF channel) instead of one
        # index over the concatenated vector, added after confirming the
        # single-vector tier-1's own parameter space was exhausted (see
        # docs/implementation_log.md's 2026-07-30(l)/(m)/(n) entries).
        ct = CorrTrack(
            window_size=64, basic_window=16, window_step=8, n_vectors=32, n_lags=32, corr_threshold=0.5, data_representation="sketch_multichannel", validation_metric="dist_corr",
        )
        self.assertTrue(ct.distance_corr_sketch_multichannel_backend)
        self.assertEqual(ct.data_representation, "sketch_multichannel")
        self.assertIsNotNone(ct.distance_corr_sketch_state)
        self.assertIsNotNone(ct.distance_corr_sketch_multichannel_index)
        self.assertEqual(ct.distance_corr_sketch_multichannel_index.n_channels, 2 * ct.distance_corr_sketch_k)
        self.assertAlmostEqual(ct.distance_corr_sketch_multichannel_gamma, 0.5, places=6)

    def test_corrtrack_distance_corr_sketch_multichannel_requires_dist_corr_metric(self):
        with self.assertRaises(NotImplementedError):
            CorrTrack(
                window_size=32, basic_window=8, window_step=8, n_vectors=16, n_lags=16, corr_threshold=0.5, data_representation="sketch_multichannel", validation_metric="pearson",
            )

    def test_corrtrack_distance_corr_sketch_multichannel_finds_nonmonotonic_pair(self):
        rng = np.random.default_rng(42)
        n_series, n_steps = 6, 240
        window_size, basic_window, window_step = 64, 16, 8
        t = np.linspace(-1, 1, n_steps)
        base = np.sin(t * 6) + rng.normal(0, 0.05, n_steps)
        x0, x1 = base, base ** 2
        others = rng.normal(0, 1, (n_series - 2, n_steps))
        values = np.vstack([x0, x1, others])
        data = np.vstack([np.arange(n_steps), values])
        ids = [f"s{i}" for i in range(n_series)]

        ct = CorrTrack(
            window_size=window_size, basic_window=basic_window, window_step=window_step, n_vectors=64, n_lags=32, corr_threshold=0.5, neg_corr=True, data_representation="sketch_multichannel", validation_metric="dist_corr", exec="sequential", numeric_rows=True,
        )
        step = 8
        for start in range(0, n_steps - step + 1, step):
            ct.run(data[:, start:start + step], ids, verbose=False, testing=False, corr_val=True, monitor=True)
        found = any({pair[0], pair[1]} == {"s0", "s1"} for pair in ct.correlated.keys())
        self.assertTrue(found, "expected the non-monotonic s0/s1 pair to be found")

    def test_corrtrack_distance_corr_sketch_multichannel_beats_tier1_recall_at_lower_touch(self):
        # (2026-07-30) The defining real-data result motivating this
        # backend: real-data benchmarking (M=36/72/121, corr_threshold=0.7,
        # 20 steps) found ~96% recall/100% precision while touching only
        # ~24-26% of all pairs -- dramatically better than tier-1's own
        # best sub-linear operating point (37.5% touched for only 66.8%
        # recall). This is a smaller, faster regression guard for the same
        # qualitative property: multichannel must recall a GENUINELY
        # non-monotonic pair that a much-narrower single-vector tier-1
        # retrieval, at a comparably tight setting, would plausibly miss.
        # Not a strict numeric guarantee (LSH retrieval is probabilistic),
        # but exercises the real code path end to end on a real dataset
        # shape, not just a hand-built toy case.
        import math
        rng = np.random.default_rng(7)
        M, w = 20, 64
        # one genuinely non-monotonic true positive (0,1); everything else independent
        x0 = rng.uniform(-1.0, 1.0, w)
        x1 = x0 ** 2 + rng.normal(scale=0.05, size=w)
        others = rng.normal(0, 1, (M - 2, w))
        values = np.vstack([x0, x1, others])
        data = np.vstack([np.arange(w), values])
        ids = [f"s{i}" for i in range(M)]

        ct = CorrTrack(
            window_size=w, basic_window=w, window_step=w, n_vectors=8, n_lags=8, corr_threshold=0.4, neg_corr=True, data_representation="sketch_multichannel", validation_metric="dist_corr", exec="sequential", numeric_rows=True,
        )
        ct.run(data[:, 0:w], ids, verbose=False, testing=False, corr_val=True, monitor=False)
        found = any({pair[0], pair[1]} == {"s0", "s1"} for pair in ct.correlated.keys())
        self.assertTrue(found, "expected the non-monotonic s0/s1 pair to be found")
        # Genuinely sub-linear: must not degenerate into touching every
        # possible pair (that would just be distance_corr_sketch_exhaustive
        # with extra steps).
        self.assertLess(ct.tested_candidates, math.comb(M, 2))

    def test_corrtrack_concordance_multichannel_backend_resolves_correctly(self):
        # (2026-07-31) Per-gap union tier-1 for incremental_concordance --
        # direct structural analog of distance_corr_sketch_multichannel,
        # added after finding a real recall gap at small window_size (see
        # docs/implementation_log.md's 2026-07-31 entry): concatenating
        # all gaps into one globally-normalized vector dilutes whichever
        # single gap actually carries the real signal for a pair near
        # corr_threshold.
        ct = CorrTrack(
            window_size=256, basic_window=16, window_step=16, n_vectors=32, n_lags=16, corr_threshold=0.5, data_representation="sketch_concordance", validation_metric="spearman",
        )
        self.assertTrue(ct.concordance_multichannel_backend)
        self.assertEqual(ct.data_representation, "sketch_concordance")
        self.assertIsNotNone(ct.concordance_state)
        self.assertIsNotNone(ct.concordance_multichannel_index)
        # At window_size=256, multiscale_gaps' own target_dim-driven
        # n_gaps=4 gives multiple usable gaps -- the whole point of this
        # backend -- confirmed here rather than assumed.
        self.assertGreater(len(ct.concordance_multichannel_index.gaps), 1)

    def test_corrtrack_concordance_multichannel_requires_non_dist_corr_metric(self):
        with self.assertRaises(NotImplementedError):
            CorrTrack(
                window_size=64, basic_window=16, window_step=16, n_vectors=16, n_lags=16, corr_threshold=0.5, data_representation="sketch_concordance", validation_metric="dist_corr",
            )

    def test_corrtrack_concordance_multichannel_gamma_degenerates_to_candidate_tau_when_single_gap(self):
        # (2026-07-31) Real regression found and fixed via direct
        # benchmarking: at window_size>=1024, multiscale_gaps' own n_gaps
        # floor leaves exactly ONE usable gap, so the union collapses to
        # that single gap's own cosine -- mathematically the SAME
        # statistic single-vector concordance's own (now-removed)
        # representation used. Using the (different, multi-gap-calibrated)
        # flat 0.28 constant there regressed recall (88.8% vs 95.7% at
        # window_size=1024); gamma must instead match derive_concordance_
        # candidate_tau's own already-tuned default in this degenerate
        # case. (2026-07-31, later) Migrated to compare against that
        # function directly, since the sibling single-vector backend this
        # test originally compared against was itself removed once its
        # last real advantage (touched_candidates parity, see the
        # 2026-07-31(g) entry) turned out to be a fixable bug rather than
        # an inherent tradeoff.
        from concordance_sketch import derive_concordance_candidate_tau

        ct = CorrTrack(
            window_size=1024, basic_window=64, window_step=64, n_vectors=32, n_lags=64, corr_threshold=0.5, data_representation="sketch_concordance", validation_metric="spearman",
        )
        self.assertEqual(len(ct.concordance_multichannel_index.gaps), 1)
        self.assertAlmostEqual(
            ct.concordance_multichannel_gamma, derive_concordance_candidate_tau(0.5), places=6
        )

    def test_corrtrack_concordance_multichannel_finds_correlated_pair(self):
        rng = np.random.default_rng(11)
        n_series, n_steps = 8, 320
        window_size, basic_window, window_step = 256, 16, 16
        base = np.cumsum(rng.normal(0, 1, n_steps))
        x0 = base + rng.normal(0, 0.2, n_steps)
        x1 = base + rng.normal(0, 0.2, n_steps)
        others = rng.normal(0, 1, (n_series - 2, n_steps))
        values = np.vstack([x0, x1, others])
        data = np.vstack([np.arange(n_steps), values])
        ids = [f"s{i}" for i in range(n_series)]

        ct = CorrTrack(
            window_size=window_size, basic_window=basic_window, window_step=window_step, n_vectors=64, n_lags=window_step, corr_threshold=0.5, neg_corr=True, data_representation="sketch_concordance", validation_metric="spearman", exec="sequential", numeric_rows=True,
        )
        for start in range(0, n_steps - window_step + 1, window_step):
            ct.run(data[:, start:start + window_step], ids, verbose=False, testing=False, corr_val=True, monitor=True)
        found = any({pair[0], pair[1]} == {"s0", "s1"} for pair in ct.correlated.keys())
        self.assertTrue(found, "expected the genuinely correlated s0/s1 pair to be found")

    def test_corrtrack_concordance_multichannel_recalls_weak_pair_at_small_window(self):
        # (2026-07-31, later) Originally a comparative regression guard
        # against single-vector concordance (removed once its last real
        # advantage turned out to be a fixable Hamming-filter bug, see the
        # 2026-07-31(g) entry) -- reframed as a standalone check that the
        # multichannel backend reliably recalls a WEAK true positive
        # (correlation close to corr_threshold, the hard case near-
        # threshold pairs represent) across repeated random trials, the
        # same qualitative property the original real-data finding
        # demonstrated (97.7% recall at window_size=256 vs. 54.8% for the
        # now-removed single-vector representation).
        n_series, w = 12, 256
        n_trials = 8
        mc_hits = 0
        for trial in range(n_trials):
            r = np.random.default_rng(trial)
            base = r.normal(0, 1, w)
            # a WEAK monotonic relationship (moderate noise) rather than a
            # clean one -- the hard case this backend targets.
            x0 = base
            x1 = base + r.normal(0, 1.1, w)
            others = r.normal(0, 1, (n_series - 2, w))
            values = np.vstack([x0, x1, others])
            data = np.vstack([np.arange(w), values])
            ids = [f"s{i}" for i in range(n_series)]

            ct = CorrTrack(
                window_size=w, basic_window=w, window_step=w, n_vectors=32, n_lags=w, corr_threshold=0.5, neg_corr=True, data_representation="sketch_concordance", validation_metric="spearman", exec="sequential", numeric_rows=True,
            )
            ct.run(data[:, 0:w], ids, verbose=False, testing=False, corr_val=True, monitor=False)
            found = any({pair[0], pair[1]} == {"s0", "s1"} for pair in ct.correlated.keys())
            if found:
                mc_hits += 1
        self.assertGreaterEqual(
            mc_hits, n_trials // 2,
            f"multichannel only recalled the weak near-threshold pair in {mc_hits}/{n_trials} trials",
        )

    def test_data_representation_candidate_backend_defaults_resolve_correctly(self):
        # data_representation/candidate_backend are the only candidate-
        # search params now (the legacy candidate_backend string surface
        # was retired) -- "auto"/"auto" (by omission or explicitly) must
        # resolve to this project's own established defaults.
        ct_implicit = CorrTrack(
            window_size=64, basic_window=16, window_step=16, n_vectors=32, n_lags=16, corr_threshold=0.5,
        )
        self.assertEqual(ct_implicit.data_representation, "sketch_proj")
        self.assertEqual(ct_implicit.candidate_backend, "lsh_approx")

        ct_explicit = CorrTrack(
            window_size=64, basic_window=16, window_step=16, n_vectors=32, n_lags=16, corr_threshold=0.5, data_representation="auto", candidate_backend="auto",
        )
        self.assertEqual(ct_explicit.data_representation, "sketch_proj")
        self.assertEqual(ct_explicit.candidate_backend, "lsh_approx")

    def test_data_representation_auto_resolves_per_validation_metric(self):
        # data_representation="auto" must pick this project's own
        # established best default per validation_metric: "sketch_proj"
        # for pearson, "sketch_concordance" for spearman/kendall (verified
        # to strictly dominate single-vector concordance's recall, see
        # docs/implementation_log.md's 2026-07-31(b) entry), "sketch_
        # multichannel" for dist_corr.
        cases = [
            ("pearson", "sketch_proj", "auto"),
            ("spearman", "sketch_concordance", "incremental_concordance_multichannel"),
            ("kendall", "sketch_concordance", "incremental_concordance_multichannel"),
            ("dist_corr", "sketch_multichannel", "distance_corr_sketch_multichannel"),
        ]
        for metric, expected_rep, expected_internal_dispatch in cases:
            ct = CorrTrack(
                window_size=64, basic_window=16, window_step=16, n_vectors=32, n_lags=16, corr_threshold=0.5, validation_metric=metric, data_representation="auto", candidate_backend="lsh_approx",
            )
            self.assertEqual(ct.data_representation, expected_rep, f"metric={metric}")
            self.assertEqual(ct._internal_dispatch, expected_internal_dispatch, f"metric={metric}")

    def test_candidate_backend_auto_always_resolves_to_lsh_approx(self):
        ct = CorrTrack(
            window_size=64, basic_window=16, window_step=16, n_vectors=32, n_lags=16, corr_threshold=0.5, data_representation="sketch_multichannel", candidate_backend="auto", validation_metric="dist_corr",
        )
        self.assertEqual(ct.candidate_backend, "lsh_approx")
        self.assertEqual(ct._internal_dispatch, "distance_corr_sketch_multichannel")

    def test_data_representation_candidate_backend_sketch_proj_hamming_exact_matches_legacy_string(self):
        ct = CorrTrack(
            window_size=64, basic_window=16, window_step=16, n_vectors=32, n_lags=16, corr_threshold=0.5, data_representation="sketch_proj", candidate_backend="hamming_exact",
        )
        self.assertEqual(ct._internal_dispatch, "lsh_hamming_exact")

    def test_data_representation_dropped_single_vector_dist_corr_name_is_rejected(self):
        # "distance_corr_sketch"/"dcor_sketch" (the OLD meaning of
        # "sketch_proj", single-vector distance_corr_sketch) was retired
        # from this axis when "sketch_proj" was reused for the Pearson
        # sketch (confirmed via real-data benchmarking to be strictly
        # dominated by sketch_multichannel, unlike single-vector
        # concordance, which was likewise removed once its own last real
        # advantage turned out to be a fixable bug -- see docs/
        # implementation_log.md's 2026-07-31(g)/(h) entries). Selecting it
        # must fail loudly, not silently substitute a different backend.
        with self.assertRaises(ValueError):
            CorrTrack(
                window_size=64, basic_window=16, window_step=16, n_vectors=32, n_lags=16, corr_threshold=0.5, validation_metric="dist_corr", data_representation="distance_corr_sketch", candidate_backend="lsh_approx",
            )

    def test_candidate_backend_brute_force_discards_data_representation(self):
        # candidate_backend="brute_force" always means the plain, exhaustive,
        # metric-agnostic "brute_force" internal path regardless of which
        # data_representation was requested -- self.data_representation
        # still reflects what was ASKED for, self._internal_dispatch
        # reflects what's ACTUALLY used.
        for rep, metric in (
            ("sketch_proj", "pearson"), ("sketch_concordance", "spearman"),
            ("sketch_multichannel", "dist_corr"),
        ):
            ct = CorrTrack(
                window_size=64, basic_window=16, window_step=16, n_vectors=32, n_lags=16, corr_threshold=0.5, validation_metric=metric, data_representation=rep, candidate_backend="brute_force",
            )
            self.assertEqual(ct.data_representation, rep, f"data_representation={rep}")
            self.assertEqual(ct._internal_dispatch, "brute_force", f"data_representation={rep}")

    def test_data_representation_hamming_exact_now_works_for_every_representation(self):
        # ConcordanceMultiGapIndex/DistanceCorrSketchMultiChannelIndex both
        # accept an alternate index_cls (HammingExactIndex), not just
        # sketch_proj's own legacy lsh_hamming_exact string.
        from candidate_kernels import HammingExactIndex

        ct_concordance_multi = CorrTrack(
            window_size=256, basic_window=16, window_step=16, n_vectors=32, n_lags=16, corr_threshold=0.5, validation_metric="spearman", data_representation="sketch_concordance", candidate_backend="hamming_exact",
        )
        self.assertTrue(ct_concordance_multi.concordance_multichannel_backend)
        for idx in ct_concordance_multi.concordance_multichannel_index.indices.values():
            self.assertIsInstance(idx, HammingExactIndex)

        ct_dcor_multi = CorrTrack(
            window_size=64, basic_window=16, window_step=16, n_vectors=32, n_lags=16, corr_threshold=0.5, validation_metric="dist_corr", data_representation="sketch_multichannel", candidate_backend="hamming_exact",
        )
        self.assertTrue(ct_dcor_multi.distance_corr_sketch_multichannel_backend)
        for idx in ct_dcor_multi.distance_corr_sketch_multichannel_index.indices:
            self.assertIsInstance(idx, HammingExactIndex)

    def test_data_representation_hamming_exact_multichannel_finds_correlated_pair(self):
        # Real behavioral check (not just index-class introspection): a
        # HammingExactIndex-backed concordance multichannel index must
        # still actually recover a genuinely correlated pair end to end.
        rng = np.random.default_rng(17)
        n_series, n_steps = 6, 240
        window_size, basic_window, window_step = 64, 16, 8
        base = np.cumsum(rng.normal(0, 1, n_steps))
        x0 = base + rng.normal(0, 0.1, n_steps)
        x1 = base + rng.normal(0, 0.1, n_steps)
        others = rng.normal(0, 1, (n_series - 2, n_steps))
        values = np.vstack([x0, x1, others])
        data = np.vstack([np.arange(n_steps), values])
        ids = [f"s{i}" for i in range(n_series)]

        ct = CorrTrack(
            window_size=window_size, basic_window=basic_window, window_step=window_step, n_vectors=64, n_lags=window_step, corr_threshold=0.5, neg_corr=True, data_representation="sketch_concordance", candidate_backend="hamming_exact", validation_metric="spearman", exec="sequential", numeric_rows=True,
        )
        step = 8
        for start in range(0, n_steps - step + 1, step):
            ct.run(data[:, start:start + step], ids, verbose=False, testing=False, corr_val=True, monitor=True)
        found = any({pair[0], pair[1]} == {"s0", "s1"} for pair in ct.correlated.keys())
        self.assertTrue(found, "expected the genuinely correlated s0/s1 pair to be found")

    def test_data_representation_candidate_backend_rejects_unknown_values(self):
        with self.assertRaises(ValueError):
            CorrTrack(
                window_size=64, basic_window=16, window_step=16, n_vectors=32, n_lags=16, corr_threshold=0.5, data_representation="not_a_real_representation", candidate_backend="lsh_approx",
            )
        with self.assertRaises(ValueError):
            CorrTrack(
                window_size=64, basic_window=16, window_step=16, n_vectors=32, n_lags=16, corr_threshold=0.5, data_representation="sketch_proj", candidate_backend="not_a_real_backend",
            )

    def test_data_representation_raw_requires_compatible_candidate_backend(self):
        # data_representation="raw" has no vector representation to index --
        # only candidate_backend "auto" or "brute_force" are compatible.
        ct = CorrTrack(
            window_size=64, basic_window=16, window_step=16, n_vectors=32, n_lags=16, corr_threshold=0.5, data_representation="raw",
        )
        self.assertEqual(ct.candidate_backend, "brute_force")
        self.assertEqual(ct._internal_dispatch, "brute_force")
        with self.assertRaises(ValueError):
            CorrTrack(
                window_size=64, basic_window=16, window_step=16, n_vectors=32, n_lags=16, corr_threshold=0.5, data_representation="raw", candidate_backend="hamming_exact",
            )

    def test_data_representation_candidate_backend_end_to_end_finds_correlated_pair(self):
        # Real behavioral check (not just resolution/dispatch): the
        # data_representation/candidate_backend surface must actually
        # drive a working pipeline, not just resolve to the right internal
        # dispatch string.
        rng = np.random.default_rng(31)
        n_series, n_steps = 6, 240
        window_size, basic_window, window_step = 64, 16, 8
        base = np.cumsum(rng.normal(0, 1, n_steps))
        x0 = base + rng.normal(0, 0.1, n_steps)
        x1 = base + rng.normal(0, 0.1, n_steps)
        others = rng.normal(0, 1, (n_series - 2, n_steps))
        values = np.vstack([x0, x1, others])
        data = np.vstack([np.arange(n_steps), values])
        ids = [f"s{i}" for i in range(n_series)]

        ct = CorrTrack(
            window_size=window_size, basic_window=basic_window, window_step=window_step, n_vectors=64, n_lags=window_step, corr_threshold=0.5, neg_corr=True, data_representation="sketch_concordance", candidate_backend="lsh_approx", validation_metric="spearman", exec="sequential", numeric_rows=True,
        )
        step = 8
        for start in range(0, n_steps - step + 1, step):
            ct.run(data[:, start:start + step], ids, verbose=False, testing=False, corr_val=True, monitor=True)
        found = any({pair[0], pair[1]} == {"s0", "s1"} for pair in ct.correlated.keys())
        self.assertTrue(found, "expected the genuinely correlated s0/s1 pair to be found")

    def test_distance_corr_sketch_apply_tier2_gate_toggle_default_true_is_noop(self):
        ct = CorrTrack(
            window_size=64, basic_window=16, window_step=16, n_vectors=32, n_lags=16, corr_threshold=0.5, data_representation="sketch_multichannel", validation_metric="dist_corr",
        )
        self.assertTrue(ct.distance_corr_sketch_apply_tier2_gate)

    def test_distance_corr_sketch_apply_tier2_gate_off_skips_gate_and_recovers_recall(self):
        # (2026-07-31, Phase 3; migrated to multichannel in Phase 4 once
        # single-vector distance_corr_sketch was removed -- the tier-2
        # gate itself is shared machinery, unaffected by that removal.)
        # Real behavioral check, not just a flag read-back: with the gate
        # off, tier-1's own retrieval must feed validation directly (zero
        # gate-side rejections), and disabling a filter that only ever
        # REJECTS candidates before the exact, always-correct validation
        # step downstream can only recover recall (never precision --
        # exact validation is unconditional either way, so precision is
        # unaffected by this toggle; it is a pure speed/recall lever, not
        # a precision one).
        rng = np.random.default_rng(9)
        n_series, w = 20, 64
        x0 = rng.uniform(-1.0, 1.0, w)
        x1 = x0 ** 2 + rng.normal(scale=0.05, size=w)
        others = rng.normal(0, 1, (n_series - 2, w))
        values = np.vstack([x0, x1, others])
        data = np.vstack([np.arange(w), values])
        ids = [f"s{i}" for i in range(n_series)]

        def run(gate_on):
            ct = CorrTrack(
                window_size=w, basic_window=w, window_step=w, n_vectors=8, n_lags=8, corr_threshold=0.4, neg_corr=True, data_representation="sketch_multichannel", validation_metric="dist_corr", distance_corr_sketch_apply_tier2_gate=gate_on, exec="sequential", numeric_rows=True,
            )
            ct.run(data[:, 0:w], ids, verbose=False, testing=False, corr_val=True, monitor=False)
            return ct

        ct_on = run(True)
        ct_off = run(False)
        self.assertEqual(ct_off.distance_corr_sketch_gate_rejected, 0)
        self.assertEqual(ct_on.tested_candidates, ct_off.tested_candidates)
        self.assertGreaterEqual(len(ct_off.correlated), len(ct_on.correlated))
        found = any({pair[0], pair[1]} == {"s0", "s1"} for pair in ct_off.correlated.keys())
        self.assertTrue(found, "expected the non-monotonic s0/s1 pair to still be found with the gate off")

    def test_concordance_multichannel_hamming_filter_reduces_touched_candidates(self):
        # (2026-07-31, touched_frac investigation) Real regression guard
        # for a confirmed root cause: ConcordanceMultiGapIndex never wired
        # candidate_apply_hamming_filter through to its per-gap
        # SignLSHBandIndex instances, silently falling back to that
        # class's own raw Cython default (False). Fixed to inherit
        # CorrTrack's project-wide default (True), matching what single-
        # vector concordance's shared Candidates-level index always did
        # (verified at the time via an exact touched_candidates match
        # against that now-removed backend, on real data -- see docs/
        # implementation_log.md's 2026-07-31(g) entry). Re-expressed here
        # as a standalone, SYNTHETIC-data check (no real-dataset file I/O,
        # which proved slow/fragile under system load when this test was
        # first written against real data): the filter must actually be ON
        # by default and not increase touched_candidates relative to
        # explicitly disabling it.
        # Mostly INDEPENDENT series (only s0/s1 genuinely correlated) --
        # keeps the touched/validated pair count small so the test runs
        # quickly, unlike an earlier attempt where ALL series shared one
        # underlying process (every pair a true positive, worst case for
        # candidate volume with the filter off -- genuinely slow, not a
        # hang, but far too slow for a unit test).
        rng = np.random.default_rng(5)
        n_series, n_steps = 20, 400
        base = np.cumsum(rng.normal(0, 1, n_steps))
        x0 = base + rng.normal(0, 0.1, n_steps)
        x1 = base + rng.normal(0, 0.1, n_steps)
        others = rng.normal(0, 1, (n_series - 2, n_steps))
        values = np.vstack([x0, x1, others])
        data = np.vstack([np.arange(n_steps), values])
        ids = [f"s{i}" for i in range(n_series)]

        window_size, basic_window, window_step = 64, 16, 16
        n_steps_run = window_size // window_step + 8

        def run(**kw):
            ct = CorrTrack(
                window_size=window_size, basic_window=basic_window, window_step=window_step, n_vectors=16, n_lags=window_step, corr_threshold=0.5, neg_corr=True, data_representation="sketch_concordance", validation_metric="spearman", numeric_rows=True, exec="sequential", **kw
            )
            for start in range(0, data.shape[1] - window_step + 1, window_step)[:n_steps_run]:
                ct.run(data[:, start:start + window_step], ids, verbose=False, testing=False, corr_val=True, monitor=False)
            return ct

        ct_default = run()
        ct_filter_off = run(candidate_apply_hamming_filter=False)
        self.assertTrue(ct_default.candidate_apply_hamming_filter)
        self.assertLessEqual(ct_default.tested_candidates, ct_filter_off.tested_candidates)

    # ------------------------------------------------------------------
    # 2026-09-16 -- competitor implementation plan, phase 0a / 0b
    # ------------------------------------------------------------------

    def test_unit_l2_window_identity_two_minus_two_corr_equals_d2(self):
        # The identity every pruning competitor rests on (StatStream Lemma 1,
        # CorrJoin Eq. 4, ParCorr/CSZ footnote): for centred unit-L2 windows,
        # d^2(x_hat, y_hat) = 2 - 2 corr(x, y). Exact, so 1e-12.
        rng = np.random.default_rng(20260916)
        for _ in range(50):
            n = int(rng.integers(8, 200))
            x = rng.normal(size=n) * rng.uniform(0.1, 10) + rng.uniform(-50, 50)
            y = rng.normal(size=n) * rng.uniform(0.1, 10) + rng.uniform(-50, 50)
            y = 0.6 * (x - x.mean()) / x.std() + y  # inject some correlation
            xh = (x - x.mean()) / np.sqrt(((x - x.mean()) ** 2).sum())
            yh = (y - y.mean()) / np.sqrt(((y - y.mean()) ** 2).sum())
            d2 = float(((xh - yh) ** 2).sum())
            corr = float(np.corrcoef(x, y)[0, 1])
            self.assertAlmostEqual(d2, 2.0 - 2.0 * corr, places=12)

    def _competitor_norm_kwargs(self):
        return dict(
            basic_window=4, window_step=4, n_vectors=16, n_lags=4, seed=11, seed_toggle=22,
            corr_threshold=0.7, exec="sequential", parallel_sketch=False,
            parallel_candidates=False, parallel_validation=False, candidate_backend="brute_force",
        )

    def test_sketches_unit_l2_window_equals_projection_of_normalized_window(self):
        # unit_l2_window must equal R_eff @ x_hat EXACTLY, where R_eff is the
        # effective full-window random matrix (toggle-signed basicRandomVector
        # blocks) and x_hat the centred unit-L2 window. Independent
        # reconstruction of R_eff from node._toggle_weights, so this is not
        # the code checking itself. Also confirms the default mean_l2 path is
        # untouched: its rows must still be the unit-normalized mean-adjusted
        # sketch, i.e. parallel to the unit_l2_window rows.
        rng = np.random.default_rng(777)
        ids = np.array(["a", "b", "c", "d", "e"])
        w = 16
        values = rng.normal(size=(5, w)) * rng.uniform(0.5, 5, size=(5, 1)) + rng.uniform(-20, 20, size=(5, 1))
        data = np.vstack([np.arange(w), values])
        kwargs = self._competitor_norm_kwargs()

        def feed(ct):
            for start in range(0, w, kwargs["window_step"]):
                ct.run(data[:, start:start + kwargs["window_step"]], ids,
                       verbose=False, testing=False, corr_val=False, monitor=False)

        ct_unit = CorrTrack(window_size=w, **kwargs)
        ct_unit.sketch_norm = "unit_l2_window"   # set before the lazy Sketches build
        feed(ct_unit)
        node = ct_unit.sketch_nodes[0]
        self.assertEqual(node.sketch_norm, "unit_l2_window")

        tw = np.asarray(node._toggle_weights)             # (n_basic, n_vectors, basic_window)
        n_basic, n_vec, bw = tw.shape
        self.assertEqual(n_basic * bw, w)
        r_eff = tw.transpose(1, 0, 2).reshape(n_vec, w)  # (n_vectors, window)
        # sanity: the library's R(1) must equal our reconstruction's row sums
        np.testing.assert_allclose(np.asarray(node._random_vector_sums), r_eff.sum(axis=1), rtol=0, atol=1e-12)

        centred = values - values.mean(axis=1, keepdims=True)
        x_hat = centred / np.sqrt((centred ** 2).sum(axis=1, keepdims=True))
        expected = x_hat @ r_eff.T                         # (n_series, n_vectors)

        got = np.asarray(node._sketch_matrix)
        self.assertEqual(got.shape, expected.shape)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-10)
        # rows are NOT unit norm in general (JL: E||R x_hat||^2 = 1, but not exactly)
        self.assertFalse(np.allclose(np.linalg.norm(got, axis=1), 1.0, atol=1e-9))

        ct_mean = CorrTrack(window_size=w, **kwargs)     # default mean_l2, untouched
        feed(ct_mean)
        got_mean = np.asarray(ct_mean.sketch_nodes[0]._sketch_matrix)
        np.testing.assert_allclose(np.linalg.norm(got_mean, axis=1), 1.0, rtol=0, atol=1e-12)
        # same direction, different scale
        cos = np.einsum("ij,ij->i", got, got_mean) / np.linalg.norm(got, axis=1)
        np.testing.assert_allclose(cos, 1.0, rtol=0, atol=1e-10)

    def test_sketches_unit_l2_window_incremental_matches_from_scratch(self):
        # Same discipline as test_window_size_reduction_stays_incremental for
        # mean_l2: sliding through the stream incrementally must give exactly
        # the from-scratch sketch of the final window.
        rng = np.random.default_rng(4242)
        ids = np.array(["a", "b", "c"])
        w = 16
        values = rng.normal(size=(3, 40))
        data = np.vstack([np.arange(40), values])
        kwargs = self._competitor_norm_kwargs()

        inc = CorrTrack(window_size=w, **kwargs)
        inc.sketch_norm = "unit_l2_window"
        for start in range(0, 40, 4):
            inc.run(data[:, start:start + 4], ids, verbose=False, testing=False, corr_val=False, monitor=False)

        fresh = CorrTrack(window_size=w, **kwargs)
        fresh.sketch_norm = "unit_l2_window"
        for start in range(40 - w, 40, 4):
            fresh.run(data[:, start:start + 4], ids, verbose=False, testing=False, corr_val=False, monitor=False)

        inc_sk = {k: np.asarray(v) for k, v in inc.sketch_nodes[0].sketches.items()}
        fresh_sk = {k: np.asarray(v) for k, v in fresh.sketch_nodes[0].sketches.items()}
        self.assertEqual(sorted(inc_sk), sorted(fresh_sk))
        for k in inc_sk:
            np.testing.assert_allclose(inc_sk[k], fresh_sk[k], rtol=0, atol=1e-9)

    def test_pearson_from_five_sums_matches_numpy(self):
        rng = np.random.default_rng(99)
        x = rng.normal(size=(7, 64)) * 3 + 10
        y = rng.normal(size=(7, 64)) * 0.5 - 4
        y[2] = 0.9 * x[2] + 0.1 * y[2]
        y[3] = -x[3]                     # perfect anticorrelation
        x[5] = 2.0                       # constant -> must give 0.0, not nan
        n, sx, sy, sxx, syy, sxy = five_sums(x, y, axis=1)
        got = pearson_from_five_sums(n, sx, sy, sxx, syy, sxy)
        expected = np.array([0.0 if i == 5 else np.corrcoef(x[i], y[i])[0, 1] for i in range(7)])
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
        self.assertEqual(got[3], -1.0)
        self.assertEqual(got[5], 0.0)
        # scalar path
        self.assertAlmostEqual(float(pearson_from_five_sums(*five_sums(x[0], y[0]))), expected[0], places=12)

    # ------------------------------------------------------------------
    # 2026-09-16 -- competitor implementation plan, phase 0c/0e + Track A1
    # ------------------------------------------------------------------

    def _assert_competitor_contract(self, record, pattern):
        """Shared contract every competitor arm must satisfy (plan §0c/0e).

        Reads the RUN_RESULT_COLUMNS record only (what reaches the paper).
        pattern "A" (all-pairs baseline_mode arms): no pruning, so the candidate
        set IS the pair set: total == tested, and no index probes were made.
        pattern "B" (pruning arms): total >= tested >= correlated, probes may be > 0.
        Both: phase timings present and non-negative; correlated <= tested.
        """
        total = int(record.get("total_candidates") or 0)
        tested = int(record.get("tested") or 0)
        validated = int(record.get("correlated") or 0)
        self.assertGreater(tested, 0, "arm tested no pairs at all")
        self.assertLessEqual(validated, tested)
        for key in ("sk_time", "cand_time", "val_time"):
            self.assertIn(key, record)
            self.assertIsNotNone(record[key], f"{key} must be populated, not null")
            self.assertGreaterEqual(float(record[key]), 0.0)
        touched = [k for k in record if k.startswith("candidate_search_") and k.endswith("_touched")]
        if pattern == "A":
            self.assertEqual(total, tested, "no-pruning arm must have total == tested")
            for k in touched:
                self.assertIn(int(record[k] or 0), (0,), f"{k} must be 0 for a no-pruning arm")
        else:
            self.assertGreaterEqual(total, tested)

    def _bf_end_to_end(self, mode, data, ids, base_config, tmp, **extra):
        cfg = dict(base_config, baseline_mode=mode, **extra)
        record, _, _ = run_and_log_bruteforce(
            f"diag_{mode}", data, ids, cfg, os.path.join(tmp, f"bf_{mode}.csv"),
            metadata={"nodes": 0}, recall_by_window=True, verbose=False, testing=False,
        )
        return record

    def test_tsubasa_node_matches_bf_incremental_pearson(self):
        # Candidates_BF_TSUBASA recovers Pearson from per-basic-window sketches
        # (their Lemma 1). It is exact, so it must match bf_incremental key-for-key
        # and value-for-value at lag 0. Exercised with window_step < basic_window
        # so the window start is usually OFF the basic-window grid, forcing the
        # partial head/tail segments (their arbitrary-query-window case) and
        # the length-weighted xbar -- the case where the paper's unweighted
        # delta would be wrong. Both even and odd window sizes.
        rng = np.random.default_rng(11)
        n_series = 6
        corr_threshold = 0.2
        for window_size, basic_window, window_step, step_len in (
            (32, 8, 4, 4), (33, 11, 3, 3), (48, 12, 4, 4), (168, 12, 12, 6),
        ):
            n_steps = max(60, (window_size // step_len) + 30)
            with self.subTest(window_size=window_size, basic_window=basic_window, step=window_step):
                ts = Candidates_BF_TSUBASA(window_size, window_step, 0, corr_threshold, basic_window, neg_corr=True)
                st = Candidates_BF_Incremental(window_size, window_step, 0, corr_threshold, neg_corr=True)
                ids = [f"s{i}" for i in range(n_series)]
                t = 0
                max_abs_err, mismatch, any_accepted, total_ts, total_st = 0.0, 0, False, 0, 0
                for step in range(n_steps):
                    block = rng.standard_normal((n_series, step_len)) + rng.uniform(-5, 5, size=(n_series, 1))
                    if step > 5:
                        block[1] = block[0] + rng.standard_normal(step_len) * 0.05
                        block[3] = -block[2] + rng.standard_normal(step_len) * 0.05
                    idx = np.arange(t, t + step_len)
                    nds = np.vstack([idx, block])
                    ts_rows, ts_corrs, n_ts, _ = ts.run(nds, ids, verbose=False, testing=False, track_min_dist=True, numeric_rows=True)
                    st_rows, st_corrs, n_st, _ = st.run(nds, ids, verbose=False, testing=False, track_min_dist=True, numeric_rows=True)
                    total_ts += n_ts; total_st += n_st
                    ks_ts = {tuple(r): c for r, c in zip(ts_rows.tolist(), ts_corrs.tolist())}
                    ks_st = {tuple(r): c for r, c in zip(st_rows.tolist(), st_corrs.tolist())}
                    any_accepted = any_accepted or bool(ks_ts)
                    mismatch += len(set(ks_ts) ^ set(ks_st))
                    for key in set(ks_ts) & set(ks_st):
                        max_abs_err = max(max_abs_err, abs(ks_ts[key] - ks_st[key]))
                    t += step_len
                self.assertTrue(any_accepted, "no correlated pairs -- test data too weak")
                self.assertEqual(mismatch, 0)
                self.assertLess(max_abs_err, 1e-9)
                self.assertEqual(total_ts, total_st, "pair counts must agree (no pruning)")
                self.assertGreater(ts._segments_built, 0)
                # cache never holds segments that have left the window
                self.assertLessEqual(len(ts._segment_cache), window_size // basic_window + 1)

    def test_tsubasa_lag_tier_and_alignment_guard(self):
        # (2026-09-23, user) replaces test_tsubasa_refuses_lags: TSUBASA now runs on lagged cells as a
        # disclosed extension (its Lemma 1 holds between segments shifted by whole basic windows), so the
        # 0d policy is applied through the tier rather than a refusal. What is still refused is a
        # configuration whose probed lags would NOT shift whole segments, since there the decomposition
        # does not apply and running it would be faking the method.
        self.assertEqual(Candidates_BF_TSUBASA(32, 8, 8, 0.5, 8).supports_lags, "enabled_by_us")
        self.assertEqual(Candidates_BF_TSUBASA(32, 8, 0, 0.5, 8).supports_lags, "native")
        with self.assertRaises(ValueError):
            Candidates_BF_TSUBASA(32, 8, 8, 0.5, 5)      # window_step 8 is not a multiple of basic_window 5

    def test_run_and_log_bruteforce_tsubasa_matches_bruteforce_and_meets_contract(self):
        # End-to-end dispatch (baseline_mode="tsubasa" -> run_bf -> run_bf_tsubasa ->
        # Candidates_BF_TSUBASA) reproduces bruteforce's correlated-pair count at
        # n_lags=0, and all three Pattern-A arms satisfy the shared counter
        # contract (plan §0c/0e) that the three-phase reporting depends on.
        n_series = 6
        length = 400
        rng = np.random.default_rng(3)
        t = np.linspace(-3, 3, length)
        values = rng.normal(scale=0.05, size=(n_series, length))
        values[0] = t
        values[1] = t + rng.normal(scale=0.02, size=length)
        values[2] = -t + rng.normal(scale=0.02, size=length)
        data = np.vstack([np.arange(length, dtype=np.float64), values])
        ids = [f"s{i}" for i in range(n_series)]
        base_config = dict(
            window_size=32, window_step=8, basic_window=8, n_lags=0,
            corr_threshold=0.6, neg_corr=True, exec="sequential",
            parallel_sketch=False, parallel_candidates=False, parallel_validation=False,
            max_workers=0, monitor=True, track_min_dist=True,
            artifact_mode="final", artifact_buffer_max_rows=250000, artifact_merge_mode="merged",
            save_only_required_artifacts=True, save_maxlag_artifacts=False, verbose=False, testing=False,
            validation_metric="pearson",
        )
        with tempfile.TemporaryDirectory() as tmp:
            counts = {}
            for mode in ("bruteforce", "bf_incremental", "filcorr", "tsubasa"):
                extra = dict(filcorr_fs=0.0, filcorr_ft=0.5, filcorr_sampling_rate=1.0) if mode == "filcorr" else {}
                record = self._bf_end_to_end(mode, data, ids, base_config, tmp, **extra)
                counts[mode] = record["correlated"]
                if mode != "bruteforce":
                    self._assert_competitor_contract(record, pattern="A")
                # (2026-09-17) the per-arm evidence tier reaches the record for Pattern A too
                self.assertEqual(record["supports_neg_corr"],
                                 {"bruteforce": "native", "bf_incremental": "native",
                                  "filcorr": "enabled_by_us", "tsubasa": "native"}[mode], mode)
        self.assertGreater(counts["bruteforce"], 0)
        for mode in ("bf_incremental", "filcorr", "tsubasa"):
            self.assertEqual(counts[mode], counts["bruteforce"], mode)

    # ------------------------------------------------------------------
    # 2026-09-16 -- Track A2: BRAID / ThinBRAID
    # ------------------------------------------------------------------

    def test_braid_probing_scheme_matches_paper(self):
        # Enhanced scheme (their §3.4, Figure 6 with b=4):
        # l = {0,...,7; 8,10,12,14; 16,20,24,28; 32,40,...}
        levels = Candidates_BF_BRAID._build_levels(60, 4)
        got = {h: lags for h, lags in levels}
        self.assertEqual(got[0], list(range(0, 8)))
        self.assertEqual(got[1], [8, 10, 12, 14])
        self.assertEqual(got[2], [16, 20, 24, 28])
        self.assertEqual(got[3], [32, 40, 48, 56])
        # with 2b > max_lag only level 0 exists and covers every lag exactly
        levels = Candidates_BF_BRAID._build_levels(12, 16)
        self.assertEqual(levels, [(0, list(range(0, 13)))])

    def test_braid_exact_anchor_matches_bf_incremental_when_2b_exceeds_n_lags(self):
        # The degenerate-exact configuration (plan A2 anchor): with 2*b > n_lags,
        # level 0 covers every lag at raw resolution -- no smoothing, no
        # interpolation -- so BRAID in "all_lags" mode must reproduce the
        # bf_incremental lagged pair set key-for-key and value-for-value. Both the
        # plain (rolling per-pair sums) and, at d0 >> W, the Thin variant are
        # checked; Thin is only approximately exact so gets a loose tolerance
        # on values but the same pair set is still required at this d0.
        rng = np.random.default_rng(5)
        m, W, step, n_lags = 8, 32, 4, 12
        ids = [f"s{i}" for i in range(m)]
        br = Candidates_BF_BRAID(W, step, n_lags, 0.3, neg_corr=True, b=16, report_mode="all_lags")
        st = Candidates_BF_Incremental(W, step, n_lags, 0.3, neg_corr=True)
        self.assertEqual(len(br.levels), 1)
        mism, maxerr, acc, t = 0, 0.0, 0, 0
        for k in range(60):
            blk = rng.standard_normal((m, step))
            if k > 3:
                blk[1] = blk[0] + 0.05 * rng.standard_normal(step)
                blk[3] = -blk[2] + 0.05 * rng.standard_normal(step)
            nds = np.vstack([np.arange(t, t + step), blk]); t += step
            r1, c1, n1, _ = br.run(nds, ids, verbose=False, testing=False)
            r2, c2, n2, _ = st.run(nds, ids, verbose=False, testing=False)
            self.assertEqual(n1, n2)
            k1 = {tuple(r): c for r, c in zip(r1.tolist(), c1.tolist())}
            k2 = {tuple(r): c for r, c in zip(r2.tolist(), c2.tolist())}
            mism += len(set(k1) ^ set(k2)); acc += len(k1)
            for kk in set(k1) & set(k2):
                maxerr = max(maxerr, abs(k1[kk] - k2[kk]))
        self.assertGreater(acc, 0)
        self.assertEqual(mism, 0)
        self.assertLess(maxerr, 1e-12)
        self.assertGreater(br.incremental_updates, br.full_initializations, "rolling updates must engage")

    def test_competitor_loader_orientation_ffill_and_contract(self):
        # (2026-09-17) phase 0f: datasets/competitor_loader.py bridges the (m, T) npz files
        # written by datasets/fetch/*.py to the runners' (T, 1 + m) DATA_LOADER contract.
        import json as _json
        from datasets import competitor_loader as cl
        m, T = 4, 50
        rng = np.random.default_rng(0)
        data = rng.standard_normal((m, T))
        data[1, :3] = np.nan            # leading gap -> back-filled with first observation
        data[2, 10:14] = np.nan         # interior gap -> forward-filled
        data[3, :] = np.nan             # dead series -> dropped
        with tempfile.TemporaryDirectory() as tmp:
            np.savez(os.path.join(tmp, "toy.npz"), data=data, ids=np.array([f"s{i}" for i in range(m)]),
                     meta=_json.dumps({"source": "unit test"}))
            old = cl.SEARCH_DIRS
            cl.SEARCH_DIRS = (Path(tmp),)
            try:
                out, ids = cl.load_dataset(name="toy")
                out2, ids2 = cl.load_dataset(name="toy", max_series=2, max_obs=20)
                with self.assertRaises(FileNotFoundError):
                    cl.load_dataset(name="missing_set")
            finally:
                cl.SEARCH_DIRS = old
        self.assertEqual(out.shape, (T, 1 + 3))
        self.assertEqual(list(ids), ["s0", "s1", "s2"])
        np.testing.assert_array_equal(out[:, 0], np.arange(T))
        np.testing.assert_allclose(out[:, 1], data[0])
        self.assertFalse(np.isnan(out).any())
        np.testing.assert_allclose(out[:3, 2], data[1, 3])
        np.testing.assert_allclose(out[10:14, 3], data[2, 9])
        self.assertEqual(out2.shape, (20, 3))
        self.assertEqual(list(ids2), ["s0", "s1"])

    def test_csz_tuning_protocol_design_and_selection_rule(self):
        # (2026-09-17) abaca/tune_competitors.py: the strength-2 covering array over CSZ's own
        # grid (N x g x c x f = 4*4*13*10 = 2,080 settings) must cover every pair of levels of
        # every two factors in 130 rows (the paper's count), except pairs that no valid setting
        # can realise (N=30 with g=4: 30 % 4 != 0). The selection rule must rank feasible
        # settings (recall >= target, precision >= 0.02) by fewer candidates, and infeasible
        # ones by recall, so an exact-recall arm ends up with its cheapest filter setting.
        import importlib.util, itertools
        spec = importlib.util.spec_from_file_location("tune_competitors", os.path.join(os.path.dirname(os.path.abspath(__file__)), "abaca", "tune_competitors.py"))
        tc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tc)
        grid = tc.parameter_grid("parcorr", 96)
        self.assertEqual(int(np.prod([len(v) for v in grid.values()])), 2080)
        rows = tc.covering_array(grid, "parcorr")
        self.assertEqual(len(rows), 130)
        self.assertTrue(all(tc.valid_setting("parcorr", r) for r in rows))
        factors = list(grid)
        need = {(a, la, b, lb) for a, b in itertools.combinations(factors, 2) for la in grid[a] for lb in grid[b]}
        have = {(a, r[a], b, r[b]) for r in rows for a, b in itertools.combinations(factors, 2)}
        self.assertEqual(need - have, {("n_vectors", 30, "parcorr_k", 4)})
        # corrjoin grid respects ks < ke, kb <= ks and divisibility of W
        cj = [dict(zip(tc.parameter_grid("corrjoin", 96), c)) for c in itertools.product(*tc.parameter_grid("corrjoin", 96).values())]
        for s_ in cj:
            if tc.valid_setting("corrjoin", s_):
                self.assertTrue(96 % s_["corrjoin_ks"] == 0 and 96 % s_["corrjoin_ke"] == 0 and s_["corrjoin_ks"] < s_["corrjoin_ke"])
        ok = lambda rec, prec, cand: dict(status="ok", recall=rec, precision=prec, total_candidates=cand)
        ranked = sorted([ok(1.0, 1.0, 900), ok(0.99, 0.5, 400), ok(0.80, 1.0, 100), ok(0.96, 0.01, 50)], key=lambda r: tc.score(r, 0.95))
        self.assertEqual([r["total_candidates"] for r in ranked], [400, 900, 50, 100])   # infeasible: by recall

    def test_competitor_kernels_parity_with_python_paths(self):
        # (2026-09-17) competitor_kernels.pyx: the three pure-Python competitor indexes got Cython
        # hot loops (sorted-key grids, one nogil loop per query batch) so that wall-clock against
        # CorrTrack's Cython candidate stage measures algorithms, not implementation tier. The
        # Python postings paths stay as the reference: pair sets and the touched / distance-check
        # counters must be identical on data with planted near pairs, anti pairs and rolling
        # eviction, for every arm and both StatStream modes.
        from library_corrtrack_parallel import _HAVE_COMPETITOR_KERNELS
        if not _HAVE_COMPETITOR_KERNELS:
            self.skipTest("competitor_kernels extension not built")
        m, D, steps, lagw = 120, 24, 5, 2

        def stream(rng):
            for st in range(steps):
                v = rng.standard_normal((m, D)); v /= np.linalg.norm(v, axis=1, keepdims=True); v *= 0.8
                v[1::4] = v[0::4][: v[1::4].shape[0]] * 0.97 + 0.03 * rng.standard_normal((v[1::4].shape[0], D))
                v[2::4] = -v[0::4][: v[2::4].shape[0]] * 0.97
                yield st, v

        def run_index(ix, rng, signed):
            out = []
            for st, v in stream(rng):
                ids = ix.insert_many(np.zeros(m), np.full(m, st), vectors_in=v, sid_idx_in=np.arange(m), time_in=np.full(m, st * 12),
                                     window_size_in=np.full(m, 96), sid_rank_in=np.arange(m))
                ix.drop_before_time(st * 12 - lagw * 12)
                rows = ix.find_pair_rows_full_cosine_signed(ids, 0.0, 0.0) if signed else ix.find_pair_rows_full_cosine(ids, 0.0, 0.0)
                out.append((set(map(tuple, rows.tolist())), ix.last_stats["lsh_candidates_touched"], ix.last_stats["num_distance_checks"]))
            return out

        for signed in (False, True):
            py = run_index(StatStreamGridIndex(D, 0.5, index_dims=4, n_lagged_windows=lagw, neg_corr=signed, use_cython=False), np.random.default_rng(1), signed)
            cy = run_index(StatStreamGridIndex(D, 0.5, index_dims=4, n_lagged_windows=lagw, neg_corr=signed, use_cython=True), np.random.default_rng(1), signed)
            self.assertEqual(py, cy, f"statstream signed={signed}")
            self.assertTrue(any(len(o[0]) > 0 for o in py))
        for probe in (False, True):
            py = run_index(ParCorrGridIndex(D, k=2, f=0.7, cell_size=0.3, neighbor_probe=probe, n_lagged_windows=lagw, use_cython=False), np.random.default_rng(2), False)
            cy = run_index(ParCorrGridIndex(D, k=2, f=0.7, cell_size=0.3, neighbor_probe=probe, n_lagged_windows=lagw, use_cython=True), np.random.default_rng(2), False)
            self.assertEqual(py, cy, f"parcorr neighbor_probe={probe}")
            self.assertTrue(any(len(o[0]) > 0 for o in py))
        # CorrJoin: synchronous, one window, both filters
        rng = np.random.default_rng(3); W, ks, ke, kb, T = 120, 10, 20, 3, 0.8
        X = np.cumsum(rng.standard_normal((m, W)), axis=1); X[1::2] = 0.9 * X[0::2] + 0.1 * np.cumsum(rng.standard_normal((m // 2, W)), axis=1)
        X -= X.mean(1, keepdims=True); X /= np.linalg.norm(X, axis=1, keepdims=True)
        Bs = np.kron(np.eye(ks), np.ones((W // ks, 1))) / (W // ks); Be = np.kron(np.eye(ke), np.ones((W // ke, 1))) / (W // ke)
        V = np.hstack([X @ Bs, X @ Be]); e1, e2 = np.sqrt(2 * ks * (1 - T) / W), np.sqrt(2 * ke * (1 - T) / W)
        res = []
        for cy_flag in (False, True):
            ix = CorrJoinDoubleFilterIndex(ks + ke, ks, ke, kb, e1, e2, use_cython=cy_flag)
            ids = ix.insert_many(np.zeros(m), np.zeros(m), vectors_in=V, sid_idx_in=np.arange(m), time_in=np.zeros(m), window_size_in=np.full(m, W), sid_rank_in=np.arange(m))
            rows = ix.find_pair_rows_full_cosine(ids, 0.0, 0.0)
            res.append((set(map(tuple, rows.tolist())), ix.last_stats["lsh_candidates_touched"], ix.last_stats["corrjoin_after_bucketing"], round(ix.last_r1, 12)))
        self.assertEqual(res[0], res[1])
        self.assertGreater(len(res[0][0]), 0)

    def test_ablation_all_pairs_backend_and_step_time_boxplot_fields(self):
        # (2026-09-18) candidate_backend="all_pairs" (AllPairsGateIndex): sketch computed, no index,
        # gates optional. With both gates off it must enumerate exactly the bruteforce pair universe
        # (total_candidates equal, recall 1); each gate on its own can only shrink the candidate set.
        # Every run record must carry the per-step latency boxplot ticks.
        from library_corrtrack_parallel import _HAVE_COMPETITOR_KERNELS
        if not _HAVE_COMPETITOR_KERNELS:
            self.skipTest("competitor_kernels extension not built")
        rng = np.random.default_rng(7)
        m, N, W, step = 16, 700, 48, 6
        X = np.cumsum(rng.standard_normal((m, N)), axis=1)
        X[1::2] = 0.9 * X[0::2] + 0.1 * np.cumsum(rng.standard_normal((m // 2, N)), axis=1)
        data = np.vstack([np.arange(N), X]); ids = [f"s{i}" for i in range(m)]
        base = dict(window_size=W, window_step=step, basic_window=step, n_lags=12, corr_threshold=0.8, neg_corr=True,
                    exec="sequential", parallel_sketch=False, parallel_candidates=False, parallel_validation=False, max_workers=0,
                    monitor=False, track_min_dist=True, artifact_mode="final", artifact_buffer_max_rows=250000, artifact_merge_mode="merged",
                    save_only_required_artifacts=True, save_maxlag_artifacts=False, verbose=False, testing=False, validation_metric="pearson")
        with tempfile.TemporaryDirectory() as tmp:
            bf, _, bf_flags = run_and_log_bruteforce("abl", data, ids, dict(base, baseline_mode="bruteforce"), os.path.join(tmp, "bf.csv"),
                                                     metadata={"nodes": 0}, recall_by_window=True, verbose=False, testing=False)
            out = {}
            for name, over in (("sketch_only", dict(candidate_apply_dot_gamma_filter=False, candidate_apply_hamming_filter=False)),
                               ("sketch_hamming", dict(candidate_apply_dot_gamma_filter=False, candidate_apply_hamming_filter=True)),
                               ("sketch_dot", dict(candidate_apply_dot_gamma_filter=True, candidate_apply_hamming_filter=False))):
                rec, _, flags = run_and_log_corrtrack("abl", data, ids, base, dict(n_vectors=24, seed=1, seed_toggle=2, preprocess=False,
                                                                                 candidate_backend="all_pairs", **over),
                                                      os.path.join(tmp, f"{name}.csv"), recall_by_window=True, verbose=False, testing=False)
                mtr = CorrTrack.compute_metrics_bf(flags, bf_flags, windows=True, total_pairs_bf=bf["total_candidates"])
                out[name] = (rec, mtr)
        rec, mtr = out["sketch_only"]
        self.assertEqual(rec["candidate_backend"], "all_pairs")
        self.assertEqual(int(rec["total_candidates"]), int(bf["total_candidates"]))
        self.assertAlmostEqual(float(mtr["recall"]), 1.0)
        self.assertEqual(int(rec["correlated"]), int(bf["correlated"]))
        for name in ("sketch_hamming", "sketch_dot"):
            self.assertLessEqual(int(out[name][0]["total_candidates"]), int(bf["total_candidates"]), name)
        for r in (bf, rec):
            for k in ("n_steps", "step_time_min", "step_time_q1", "step_time_median", "step_time_q3", "step_time_max",
                      "step_time_whisker_lo", "step_time_whisker_hi", "step_time_outliers", "step_time_mean"):
                self.assertIn(k, r); self.assertIsNotNone(r[k], k)
            self.assertLessEqual(r["step_time_min"], r["step_time_q1"]); self.assertLessEqual(r["step_time_q1"], r["step_time_median"])
            self.assertLessEqual(r["step_time_median"], r["step_time_q3"]); self.assertLessEqual(r["step_time_q3"], r["step_time_max"])
            self.assertLessEqual(r["step_time_whisker_lo"], r["step_time_q1"]); self.assertGreaterEqual(r["step_time_whisker_hi"], r["step_time_q3"])

    def test_statstream_reports_with_its_own_approximate_rule_by_default(self):
        # (2026-09-18) StatStream's arm reports the grid's survivors with the paper's rule (section 3.4 /
        # Table 2): approximate correlation from 2 DFT coefficients per basic window, accept if > T - t.
        # On its own random walks this must land near Table 2 (precision 0.9765..0.9947, recall
        # 0.9987..1.0): recall >= 0.99 and 0.95 <= precision < 1 (some false positives, that is the point);
        # statstream_report="exact" must reproduce the bruteforce set exactly. The record carries the mode.
        rng = np.random.default_rng(5); m, N = 80, 5000
        walks = 100.0 + np.cumsum(rng.uniform(0, 1, size=(m, N)) - 0.5, axis=1)
        test = np.vstack([np.arange(N), walks]); ids = [f"rw{i}" for i in range(m)]
        # k = W / b = 60 basic windows per sliding window, the paper's 1 h / 1 min ratio
        base = dict(window_size=1200, window_step=20, basic_window=20, n_lags=0, corr_threshold=0.85, neg_corr=False, exec="sequential",
                    parallel_sketch=False, parallel_candidates=False, parallel_validation=False, max_workers=0, monitor=False, track_min_dist=True,
                    artifact_mode="final", artifact_buffer_max_rows=250000, artifact_merge_mode="merged", save_only_required_artifacts=True,
                    save_maxlag_artifacts=False, verbose=False, testing=False, validation_metric="pearson")
        with tempfile.TemporaryDirectory() as tmp:
            bf, _, bf_flags = run_and_log_bruteforce("ss", test, ids, dict(base, baseline_mode="bruteforce"), os.path.join(tmp, "bf.csv"),
                                                     metadata={"nodes": 0}, recall_by_window=True, verbose=False, testing=False)
            out = {}
            for rep in ("approx", "exact"):
                rec, _, flags = run_and_log_corrtrack("ss", test, ids, base, dict(n_vectors=32, seed=1, seed_toggle=2, preprocess=False,
                                                                                data_representation="sketch_dft", candidate_backend="statstream_grid",
                                                                                statstream_n_coeffs=16, statstream_index_dims=4, statstream_report=rep),
                                                      os.path.join(tmp, f"{rep}.csv"), recall_by_window=True, verbose=False, testing=False)
                out[rep] = (rec, CorrTrack.compute_metrics_bf(flags, bf_flags, windows=True, total_pairs_bf=bf["total_candidates"]))
        self.assertGreater(int(bf["correlated"]), 100)
        rec, m_ = out["approx"]
        self.assertEqual(rec["statstream_report"], "approx")
        self.assertGreaterEqual(float(m_["recall"]), 0.99)
        self.assertGreaterEqual(float(m_["precision"]), 0.95)
        self.assertLess(float(m_["precision"]), 1.0)
        rec, m_ = out["exact"]
        self.assertEqual(rec["statstream_report"], "exact")
        self.assertAlmostEqual(float(m_["recall"]), 1.0); self.assertAlmostEqual(float(m_["precision"]), 1.0)
        self.assertEqual(int(rec["correlated"]), int(bf["correlated"]))

    def test_thinbraid_tracks_exact_after_buffer_rolls_on_offset_mean_data(self):
        # (2026-09-17) Two defects found on real Motes data, both invisible on the
        # zero-mean, short synthetic streams used above: (1) the shared-projection
        # cache was keyed on a buffer-relative block index, so once the rolling
        # buffer started evicting, every step reused the first step's px (mean
        # |corr error| 0.67); (2) Eq. 24 applied to raw windows lets the JL error
        # scale with W*(mean_x - mean_y)^2 while Pearson needs the centred cross-sum
        # (mean error 0.16 even with the cache fixed). Series here have distinct
        # offsets (18..30) and small variance, and the stream is long enough for the
        # buffer to roll many times; ThinBRAID must stay within JL noise of exact.
        rng = np.random.default_rng(11)
        m, W, step, N = 12, 96, 12, 1200
        base = np.cumsum(rng.standard_normal(N)) * 0.05
        X = np.empty((m, N))
        for i in range(m):
            X[i] = 18.0 + i + 0.7 * base + 0.3 * np.cumsum(rng.standard_normal(N)) * 0.05 + 0.02 * rng.standard_normal(N)
        ids = [f"s{i}" for i in range(m)]
        node = Candidates_BF_BRAID(W, step, 0, 0.0, neg_corr=False, thin=True, thin_d0=400)
        rows = corrs = None
        for s0 in range(0, N, step):
            rows, corrs, _, _ = node.run(np.vstack([np.arange(s0, s0 + step), X[:, s0:s0 + step]]), ids, verbose=False, testing=False)
        C = np.corrcoef(X[:, N - W:N])
        err = np.array([abs(float(c) - C[int(r[0]), int(r[1])]) for r, c in zip(rows, corrs)])
        self.assertEqual(err.size, m * (m - 1) // 2)
        self.assertLess(err.mean(), 0.03, f"mean |err| {err.mean():.4f}")
        self.assertLess(err.max(), 0.15, f"max |err| {err.max():.4f}")

    def test_braid_lag_estimate_recovers_exact_earliest_local_max(self):
        # Definition 1 applied to the exact CCF vs BRAID's interpolated estimate.
        # In the exact anchor the agreement must be 100%. With interpolation
        # (b=4, ~half the lags probed) agreement is necessarily imperfect on a
        # W=64 window with single-digit lags -- the regime effect recorded in
        # docs/implementation_log.md 2026-09-16 (k) -- so only a loose floor is
        # asserted there, and Thin must not be *better* than plain (d0 >= W_h
        # adds JL noise without saving anything on short windows).
        def earliest_local_max(absr, gamma):
            if absr.shape[0] == 1:
                return 0 if absr[0] >= gamma else -1
            left = np.r_[-np.inf, absr[:-1]]; right = np.r_[absr[1:], -np.inf]
            ok = np.nonzero((absr >= left) & (absr > right) & (absr >= gamma))[0]
            return int(ok[0]) if ok.size else -1
        rng = np.random.default_rng(3)
        W, n_lags, TRUE_LAG, N = 64, 30, 9, 400
        smooth = np.convolve(rng.standard_normal(N + 40), np.ones(12) / 12, mode="same")
        x = smooth[:N]; y = np.empty(N); y[TRUE_LAG:] = x[:-TRUE_LAG]; y[:TRUE_LAG] = x[0]
        y = y + 0.05 * rng.standard_normal(N)
        X = np.vstack([x, y, rng.standard_normal((2, N))]); ids = ["x", "y", "n1", "n2"]
        def run(kw):
            br = Candidates_BF_BRAID(W, 1, n_lags, 0.5, neg_corr=True, gamma=0.4, report_mode="braid", **kw)
            agree = []
            for t in range(N):
                r, c, n, _ = br.run(np.vstack([[t], X[:, t:t + 1]]), ids, verbose=False, testing=False)
                if t >= W + n_lags:
                    xc = x[t - W + 1:t + 1]
                    ccf = np.array([np.corrcoef(xc, y[t - W + 1 - l:t + 1 - l])[0, 1] for l in range(n_lags + 1)])
                    ex = earliest_local_max(np.abs(ccf), 0.4)
                    if ex >= 0:
                        agree.append(int(br.last_lag_estimates[0, 1]) == ex)
                    # "braid" mode emits at most one row per pair, at the chosen lag
                    if r.shape[0]:
                        self.assertEqual(len({(a, b) for a, b, *_ in r.tolist()}), r.shape[0])
            return float(np.mean(agree))
        exact = run(dict(b=16))
        interp = run(dict(b=4))
        thin = run(dict(b=4, thin=True))
        self.assertEqual(exact, 1.0)
        self.assertGreater(interp, 0.5)
        self.assertLessEqual(thin, interp + 0.05)

    def test_run_and_log_bruteforce_braid_dispatch_and_contract(self):
        # End-to-end through run_and_log_bruteforce with baseline_mode="braid" in
        # the exact-anchor configuration: must match bruteforce's correlated count
        # and satisfy the Pattern-A contract; knobs thread through base_config.
        n_series, length = 6, 400
        rng = np.random.default_rng(3)
        t = np.linspace(-3, 3, length)
        values = rng.normal(scale=0.05, size=(n_series, length))
        values[0] = t; values[1] = t + rng.normal(scale=0.02, size=length)
        data = np.vstack([np.arange(length, dtype=np.float64), values])
        ids = [f"s{i}" for i in range(n_series)]
        base_config = dict(
            window_size=32, window_step=8, basic_window=8, n_lags=16,
            corr_threshold=0.6, neg_corr=True, exec="sequential",
            parallel_sketch=False, parallel_candidates=False, parallel_validation=False,
            max_workers=0, monitor=True, track_min_dist=True,
            artifact_mode="final", artifact_buffer_max_rows=250000, artifact_merge_mode="merged",
            save_only_required_artifacts=True, save_maxlag_artifacts=False, verbose=False, testing=False,
            validation_metric="pearson",
        )
        with tempfile.TemporaryDirectory() as tmp:
            bf = self._bf_end_to_end("bruteforce", data, ids, base_config, tmp)
            br = self._bf_end_to_end("braid", data, ids, base_config, tmp, braid_b=16, braid_gamma=0.4,
                                     braid_thin=False, braid_report_mode="all_lags")
        self.assertGreater(bf["correlated"], 0)
        self.assertEqual(br["correlated"], bf["correlated"])
        self.assertEqual(int(br["braid_b"]), 16)
        self._assert_competitor_contract(br, pattern="A")

    # ------------------------------------------------------------------
    # 2026-09-17 -- Track B1: ParCorr / Cole-Shasha-Zhao (candidate_backend="parcorr_grid")
    # ------------------------------------------------------------------

    def test_parcorr_grid_index_vote_and_row_semantics(self):
        # Unit-level: the fraction-f vote across k-dim group grids, ParCorr's
        # same-cell rule vs CSZ's neighbour+radius rule, and canonical rows
        # matching the Cython indexes (later time first; equal time -> lower
        # rank first). r=6, k=2 -> 3 grids; f=0.7 -> required_hits = ceil(2.1) = 3.
        idx = ParCorrGridIndex(n_vectors=6, k=2, f=0.7, cell_size=1.0)
        self.assertEqual(idx.n_grids, 3)
        self.assertEqual(idx.required_hits, 3)
        self.assertEqual(idx.supports_neg_corr, "not_available")
        # e0 and e1: same cell in all 3 grids. e2: same cell as e0 in 2 grids only.
        # e3: adjacent cell (within radius) in all 3 grids -> ParCorr rejects, CSZ accepts.
        v = np.array([
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],   # e0  sid 0, t=10
            [0.2, 0.3, 0.4, 0.2, 0.3, 0.1],   # e1  sid 1, t=10  (cells all 0 -> 3 hits)
            [0.1, 0.1, 0.1, 0.1, 1.5, 1.5],   # e2  sid 2, t=10  (2 hits)
            [-0.1, 0.1, 0.1, -0.1, 0.1, 0.1], # e3  sid 3, t=10  (cell -1 in grids 0,1; dist 0.2 < 1.0)
        ])
        ids = idx.insert_many(None, np.arange(4), v, np.arange(4), np.full(4, 10), np.full(4, 8), np.arange(4))
        self.assertEqual(ids.tolist(), [0, 1, 2, 3])
        rows = idx.find_pair_rows_full_cosine(np.arange(4), 0.7, 0.0)
        pairs = {(int(r[0]), int(r[1])) for r in rows}
        self.assertEqual(pairs, {(0, 1)})
        self.assertTrue(all(r[2] == 10 and r[3] == 10 and r[4] == 8 for r in rows))
        self.assertEqual(idx.last_stats["parcorr_required_hits"], 3)
        self.assertGreater(idx.last_stats["lsh_candidates_touched"], 0)
        # CSZ mode: e3 is within radius in every group -> accepted
        csz = ParCorrGridIndex(n_vectors=6, k=2, f=0.7, cell_size=1.0, neighbor_probe=True)
        csz.insert_many(None, np.arange(4), v, np.arange(4), np.full(4, 10), np.full(4, 8), np.arange(4))
        rows = csz.find_pair_rows_full_cosine(np.arange(4), 0.7, 0.0)
        self.assertEqual({(int(r[0]), int(r[1])) for r in rows}, {(0, 1), (0, 3), (1, 3)})
        # canonical ordering for a lagged pair: later time first
        lag = ParCorrGridIndex(n_vectors=6, k=2, f=0.7, cell_size=1.0)
        lag.insert_many(None, np.array([0]), v[:1], np.array([0]), np.array([10]), np.array([8]), np.array([0]))
        lag.insert_many(None, np.array([1]), v[1:2], np.array([1]), np.array([18]), np.array([8]), np.array([1]))
        rows = lag.find_pair_rows_full_cosine(np.array([1]), 0.7, 0.0)
        self.assertEqual(rows.tolist(), [[1, 0, 18, 10, 8]])
        # negative correlation refused, not extended
        with self.assertRaises(NotImplementedError):
            idx.find_pair_rows_full_cosine_signed(np.arange(4), 0.7, 0.0)
        # expiry drops by time
        lag.drop_before_time(15)
        self.assertEqual(lag._alive_count, 1)

    def _parcorr_dataset(self):
        rng = np.random.default_rng(42)
        m, W, step, N = 60, 64, 8, 64 + 8 * 20
        X = rng.standard_normal((m, N))
        for i in range(0, 20, 2):
            X[i + 1] = 0.9 * X[i] + np.sqrt(1 - 0.81) * rng.standard_normal(N)
        data = np.vstack([np.arange(N), X])
        ids = [f"s{i}" for i in range(m)]
        return data, ids, W, step, N

    def _parcorr_ct(self, data, ids, W, step, N, **kw):
        ct = CorrTrack(window_size=W, basic_window=8, window_step=step, n_vectors=60, n_lags=0,
                       corr_threshold=0.7, neg_corr=False, exec="sequential", parallel_sketch=False,
                       parallel_candidates=False, parallel_validation=False, numeric_rows=True, **kw)
        for s0 in range(0, N, step):
            ct.run(data[:, s0:s0 + step], ids, verbose=False, testing=False, corr_val=True, monitor=False)
        return ct

    def test_parcorr_grid_backend_dispatch_normalization_and_tuning_surface(self):
        # End-to-end through CorrTrack: the backend must (a) select
        # ParCorrGridIndex, (b) switch the sketch to unit_l2_window (both papers
        # normalize the window before projecting), (c) keep precision 1.0 (exact
        # validation downstream), (d) show the published tuning surface: recall
        # non-decreasing in c and in 1/f, and (e) refuse neg_corr=True.
        data, ids, W, step, N = self._parcorr_dataset()
        st = Candidates_BF_Incremental(W, step, 0, 0.7, neg_corr=False)
        gt = {}
        for s0 in range(0, N, step):
            acc, _, _ = st.run(data[:, s0:s0 + step], ids, verbose=False, testing=False, numeric_rows=False)
            gt.update(acc)
        self.assertGreater(len(gt), 100)

        def recall(**kw):
            ct = self._parcorr_ct(data, ids, W, step, N, candidate_backend="parcorr_grid", **kw)
            self.assertEqual(type(ct.grid_nodes[0]._lsh_index).__name__, "ParCorrGridIndex")
            self.assertEqual(ct.sketch_norm, "unit_l2_window")
            self.assertEqual(ct.sketch_nodes[0].sketch_norm, "unit_l2_window")
            self.assertEqual(ct.supports_neg_corr, "not_available")
            met = CorrTrack.compute_metrics_bf(ct.correlated, gt, windows=True)
            self.assertEqual(met["precision"], 1.0)
            return met["recall"], ct
        r_small, _ = recall(parcorr_c=0.2)
        r_mid, ct_mid = recall(parcorr_c=0.7)
        r_big, _ = recall(parcorr_c=1.0)
        r_loose_f, ct_loose = recall(parcorr_c=0.7, parcorr_f=0.5)
        self.assertLess(r_small, r_mid)
        self.assertLessEqual(r_mid, r_big)
        self.assertLessEqual(r_mid, r_loose_f)
        self.assertGreater(ct_loose.tested_candidates, ct_mid.tested_candidates)
        # touched counter is populated (implementation-independent, plan §5b.1)
        self.assertGreater(ct_mid.grid_nodes[0]._lsh_index.last_stats["lsh_candidates_touched"], 0)
        with self.assertRaises(ValueError):
            CorrTrack(window_size=W, basic_window=8, window_step=step, n_vectors=60, n_lags=0,
                      corr_threshold=0.7, neg_corr=True, exec="sequential", candidate_backend="parcorr_grid")
        with self.assertRaises(ValueError):   # n_vectors not divisible by k
            CorrTrack(window_size=W, basic_window=8, window_step=step, n_vectors=61, n_lags=0,
                      corr_threshold=0.7, neg_corr=False, exec="sequential", candidate_backend="parcorr_grid")

    def test_parcorr_grid_filter_disabled_recovers_bruteforce(self):
        # The §6.5 anchor for pruning arms: with the filter effectively disabled
        # (one huge cell, one required hit) the candidate set is the full pair
        # set and the validated set must equal brute force's exactly.
        data, ids, W, step, N = self._parcorr_dataset()
        st = Candidates_BF_Incremental(W, step, 0, 0.7, neg_corr=False)
        gt = {}
        for s0 in range(0, N, step):
            acc, _, _ = st.run(data[:, s0:s0 + step], ids, verbose=False, testing=False, numeric_rows=False)
            gt.update(acc)
        ct = self._parcorr_ct(data, ids, W, step, N, candidate_backend="parcorr_grid",
                              parcorr_c=1e6, parcorr_f=1e-9)
        self.assertEqual(ct.grid_nodes[0]._lsh_index.required_hits, 1)
        met = CorrTrack.compute_metrics_bf(ct.correlated, gt, windows=True)
        self.assertEqual(met["recall"], 1.0)
        self.assertEqual(met["precision"], 1.0)
        self.assertEqual(set(ct.correlated), set(gt))
        for k in gt:
            self.assertAlmostEqual(ct.correlated[k], gt[k], places=9)

    def test_run_and_log_corrtrack_parcorr_knobs_thread_and_pattern_b_contract(self):
        # Knobs travel through run_params -> _extract_feature_overrides ->
        # CorrTrack, land in the record, and the Pattern-B counter contract holds.
        data, ids, W, step, N = self._parcorr_dataset()
        base_config = dict(
            window_size=W, window_step=step, basic_window=8, n_lags=0, corr_threshold=0.7,
            neg_corr=False, exec="sequential", parallel_sketch=False, parallel_candidates=False,
            parallel_validation=False, max_workers=0, monitor=False, track_min_dist=True,
            artifact_mode="final", artifact_buffer_max_rows=250000, artifact_merge_mode="merged",
            save_only_required_artifacts=True, save_maxlag_artifacts=False, verbose=False, testing=False,
            validation_metric="pearson",
        )
        run_params = dict(n_vectors=60, seed=1, seed_toggle=2, preprocess=False,
                          candidate_backend="parcorr_grid", parcorr_k=2, parcorr_f=0.5, parcorr_c=0.7,
                          parcorr_neighbor_probe=False)
        with tempfile.TemporaryDirectory() as tmp:
            record, _, _ = run_and_log_corrtrack("diag_parcorr", data, ids, base_config, run_params,
                                                 os.path.join(tmp, "run.csv"), recall_by_window=True,
                                                 verbose=False, testing=False)
        self.assertEqual(int(record["parcorr_k"]), 2)
        self.assertAlmostEqual(float(record["parcorr_f"]), 0.5)
        self.assertEqual(record["supports_neg_corr"], "not_available")
        self.assertGreater(float(record["parcorr_cell_size"]), 0.0)
        self._assert_competitor_contract(record, pattern="B")
        self.assertGreater(int(record["candidate_search_lsh_candidates_touched"] or 0), 0)

    # ------------------------------------------------------------------
    # 2026-09-17 -- Track B2: StatStream (data_representation="sketch_dft" x "statstream_grid")
    # ------------------------------------------------------------------

    def test_sketch_dft_representation_is_normalized_dft_and_bounded(self):
        # Sketches._sketches_dft must equal bins 1..n of the DFT of the centred window
        # divided by its L2 norm (StatStream Lemma 4: X_hat_0 = 0, X_hat_i = X_i / sigma),
        # as [Re, Im] with the paper's 1/sqrt(W) scaling; every coordinate within
        # +-sqrt(2)/2 (Lemma 7); and Lemma 2's bound d_n(X_hat, Y_hat) <= sqrt(2(1-corr))
        # must hold for every pair (the property the grid's no-false-negatives rests on).
        rng = np.random.default_rng(11)
        m, W, step, n = 12, 64, 8, 16
        X = np.cumsum(rng.standard_normal((m, W)), axis=1) * rng.uniform(0.5, 3, (m, 1)) + rng.uniform(-5, 5, (m, 1))
        data = np.vstack([np.arange(W), X]); ids = [f"s{i}" for i in range(m)]
        ct = CorrTrack(window_size=W, basic_window=8, window_step=step, n_vectors=60, n_lags=0, corr_threshold=0.7,
                       neg_corr=False, exec="sequential", parallel_sketch=False, parallel_candidates=False,
                       parallel_validation=False, numeric_rows=True, data_representation="sketch_dft",
                       candidate_backend="statstream_grid", statstream_n_coeffs=n)
        for s0 in range(0, W, step):
            ct.run(data[:, s0:s0 + step], ids, verbose=False, testing=False, corr_val=True, monitor=False)
        sk = np.asarray(ct.sketch_nodes[0]._sketch_matrix)
        self.assertEqual(sk.shape, (m, 2 * n))
        self.assertEqual(ct.n_vectors, 2 * n)
        centred = X - X.mean(axis=1, keepdims=True)
        xhat = centred / np.linalg.norm(centred, axis=1, keepdims=True)
        F = np.fft.fft(xhat, axis=1) / np.sqrt(W)
        expected = np.hstack([F[:, 1:n + 1].real, F[:, 1:n + 1].imag])
        np.testing.assert_allclose(sk, expected, rtol=0, atol=1e-12)
        self.assertLessEqual(np.abs(sk).max(), np.sqrt(2) / 2 + 1e-12)
        corr = np.corrcoef(X)
        for i in range(m):
            for j in range(i + 1, m):
                d = np.linalg.norm(sk[i] - sk[j])
                self.assertLessEqual(d, np.sqrt(max(2 * (1 - corr[i, j]), 0)) + 1e-9)

    def _statstream_dataset(self, seed=7):
        rng = np.random.default_rng(seed)
        m, W, step, N = 40, 64, 8, 64 + 8 * 24
        X = rng.standard_normal((m, N))
        for i in range(0, 16, 2):
            X[i + 1] = 0.9 * X[i] + np.sqrt(1 - 0.81) * rng.standard_normal(N)
        for i in range(16, 24, 2):
            X[i + 1] = -0.9 * X[i] + np.sqrt(1 - 0.81) * rng.standard_normal(N)      # anti-correlated
        X[30, 16:] = X[29, :-16]; X[30, :16] = 0.0                                    # lag-16 copy
        return np.vstack([np.arange(N), X]), [f"s{i}" for i in range(m)], W, step, N

    def _gt(self, data, ids, W, step, n_lags, T, neg):
        st = Candidates_BF_Incremental(W, step, n_lags, T, neg_corr=neg); g = {}
        for s0 in range(0, data.shape[1], step):
            acc, _, _ = st.run(data[:, s0:s0 + step], ids, verbose=False, testing=False, numeric_rows=False)
            g.update(acc)
        return g

    def _statstream_ct(self, data, ids, W, step, n_lags, T, neg, **kw):
        # these tests probe the GRID (Theorem 2, the DFT filter), so survivors go through the exact
        # validation; the arm's default reporting rule (statstream_report="approx") has its own test
        kw.setdefault("statstream_report", "exact")
        ct = CorrTrack(window_size=W, basic_window=8, window_step=step, n_vectors=60, n_lags=n_lags, corr_threshold=T,
                       neg_corr=neg, exec="sequential", parallel_sketch=False, parallel_candidates=False,
                       parallel_validation=False, numeric_rows=True, data_representation="sketch_dft",
                       candidate_backend="statstream_grid", **kw)
        for s0 in range(0, data.shape[1], step):
            ct.run(data[:, s0:s0 + step], ids, verbose=False, testing=False, corr_val=True, monitor=False)
        return ct

    def test_statstream_grid_has_no_false_negatives_theorem_2(self):
        # The sharpest anchor of any arm (plan §6.5): the paper separates two
        # guarantees. The grid is provably false-negative-free (Theorem 2), so
        # recall must be EXACTLY 1.0 against brute force with the DFT-distance
        # post-filter disabled AND enabled, synchronous and lagged, positive-only
        # and with Lemma 3's negative path -- and the validated set must equal
        # brute force's exactly (exact validation downstream => precision 1.0).
        data, ids, W, step, N = self._statstream_dataset()
        for T in (0.7, 0.9):
            for neg in (False, True):
                for n_lags in (0, 16):
                    g = self._gt(data, ids, W, step, n_lags, T, neg)
                    self.assertGreater(len(g), 20)
                    if neg:
                        self.assertGreater(sum(1 for v in g.values() if v < 0), 0)
                    if n_lags:
                        self.assertGreater(sum(1 for k in g if k[2] != k[3]), 0)
                    for dft_filter in (False, True):
                        with self.subTest(T=T, neg=neg, n_lags=n_lags, dft_filter=dft_filter):
                            ct = self._statstream_ct(data, ids, W, step, n_lags, T, neg,
                                                     statstream_apply_dft_filter=dft_filter)
                            self.assertEqual(type(ct.grid_nodes[0]._lsh_index).__name__, "StatStreamGridIndex")
                            self.assertEqual(ct.supports_neg_corr, "specified")
                            met = CorrTrack.compute_metrics_bf(ct.correlated, g, windows=True)
                            self.assertEqual(met["recall"], 1.0)
                            self.assertEqual(met["precision"], 1.0)
                            self.assertEqual(set(ct.correlated), set(g))

    def test_statstream_dft_filter_prunes_and_uncooperative_data_prunes_less(self):
        # (a) the n-approximate DFT-distance filter reduces tested pairs without
        # losing recall; (b) with few coefficients, white noise (uncooperative)
        # is pruned far less than random walks (cooperative) -- the finding
        # Cole-Shasha-Zhao / TSUBASA report, reproduced here as a regression guard.
        data, ids, W, step, N = self._statstream_dataset()
        g = self._gt(data, ids, W, step, 0, 0.7, False)
        ct_on = self._statstream_ct(data, ids, W, step, 0, 0.7, False, statstream_apply_dft_filter=True)
        ct_off = self._statstream_ct(data, ids, W, step, 0, 0.7, False, statstream_apply_dft_filter=False)
        self.assertLess(ct_on.tested_candidates, ct_off.tested_candidates)
        self.assertEqual(CorrTrack.compute_metrics_bf(ct_on.correlated, g, windows=True)["recall"], 1.0)
        rng = np.random.default_rng(3)
        m = 40
        noise = rng.standard_normal((m, N))
        walks = np.cumsum(rng.standard_normal((m, N)), axis=1)
        for X in (noise, walks):
            for i in range(0, 16, 2):
                X[i + 1] = 0.9 * X[i] + np.sqrt(1 - 0.81) * rng.standard_normal(N) * X[i].std()
        tested = {}
        for name, X in (("noise", noise), ("walks", walks)):
            d = np.vstack([np.arange(N), X])
            ct = self._statstream_ct(d, ids, W, step, 0, 0.7, False, statstream_n_coeffs=4)
            all_pairs = ct.tested_candidates
            tested[name] = all_pairs
        self.assertGreater(tested["noise"], 1.5 * tested["walks"],
                           f"white noise should be pruned far less than random walks with n=4: {tested}")

    def test_run_and_log_corrtrack_statstream_knobs_thread_and_pattern_b_contract(self):
        data, ids, W, step, N = self._statstream_dataset()
        base_config = dict(
            window_size=W, window_step=step, basic_window=8, n_lags=0, corr_threshold=0.7,
            neg_corr=False, exec="sequential", parallel_sketch=False, parallel_candidates=False,
            parallel_validation=False, max_workers=0, monitor=False, track_min_dist=True,
            artifact_mode="final", artifact_buffer_max_rows=250000, artifact_merge_mode="merged",
            save_only_required_artifacts=True, save_maxlag_artifacts=False, verbose=False, testing=False,
            validation_metric="pearson",
        )
        run_params = dict(n_vectors=60, seed=1, seed_toggle=2, preprocess=False,
                          data_representation="sketch_dft", candidate_backend="statstream_grid",
                          statstream_n_coeffs=8, statstream_index_dims=2)
        with tempfile.TemporaryDirectory() as tmp:
            record, _, _ = run_and_log_corrtrack("diag_statstream", data, ids, base_config, run_params,
                                                 os.path.join(tmp, "run.csv"), recall_by_window=True,
                                                 verbose=False, testing=False)
        self.assertEqual(int(record["statstream_n_coeffs"]), 8)
        self.assertEqual(int(record["statstream_index_dims"]), 2)
        self.assertAlmostEqual(float(record["statstream_eps"]), np.sqrt(1 - 0.7))
        self.assertEqual(record["supports_neg_corr"], "specified")
        self._assert_competitor_contract(record, pattern="B")

    # ------------------------------------------------------------------
    # 2026-09-17 -- Track B3: CorrJoin (data_representation="sketch_paa_svd" x "corrjoin_double_filter")
    # ------------------------------------------------------------------

    def _corrjoin_dataset(self, kind, seed=42, m=120):
        rng = np.random.default_rng(seed)
        W, step, N = 60, 6, 60 + 6 * 16
        X = np.cumsum(rng.standard_normal((m, N)), axis=1) if kind == "walks" else rng.standard_normal((m, N))
        n_pos, n_neg = (m // 4) * 2, (m // 12) * 2          # planted pairs scale with m
        for i in range(0, n_pos, 2):
            X[i + 1] = 0.9 * X[i] + np.sqrt(1 - 0.81) * rng.standard_normal(N) * (X[i].std() if kind == "walks" else 1)
        for i in range(n_pos, n_pos + n_neg, 2):
            X[i + 1] = -0.9 * X[i] + np.sqrt(1 - 0.81) * rng.standard_normal(N) * (X[i].std() if kind == "walks" else 1)
        return np.vstack([np.arange(N), X]), [f"s{i}" for i in range(m)], W, step, N

    def _corrjoin_ct(self, data, ids, W, step, T, **kw):
        ct = CorrTrack(window_size=W, basic_window=6, window_step=step, n_vectors=60, n_lags=0, corr_threshold=T,
                       neg_corr=False, exec="sequential", parallel_sketch=False, parallel_candidates=False,
                       parallel_validation=False, numeric_rows=True, data_representation="sketch_paa_svd",
                       candidate_backend="corrjoin_double_filter", **kw)
        for s0 in range(0, data.shape[1], step):
            ct.run(data[:, s0:s0 + step], ids, verbose=False, testing=False, corr_val=True, monitor=False)
        return ct

    def test_sketch_paa_svd_representation_matches_authors_normalization(self):
        # [PAA_ks(x_hat) | PAA_ke(x_hat)] must equal the authors' `paamN <- (paam - meanT)/tauT`
        # with tauT = sqrt(sum x^2 - n mean^2) (2-CorrJoin.R), and equal PAA of the
        # unit-normalized window (linearity). n_vectors becomes ks + ke. ks, ke must divide W.
        data, ids, W, step, N = self._corrjoin_dataset("walks", m=20)
        ct = self._corrjoin_ct(data[:, :W], ids, W, step, 0.7)
        sk = np.asarray(ct.sketch_nodes[0]._sketch_matrix)
        self.assertEqual(sk.shape, (20, 45)); self.assertEqual(ct.n_vectors, 45)
        X = data[1:, :W]
        mean = X.mean(axis=1, keepdims=True); tau = np.sqrt((X ** 2).sum(axis=1, keepdims=True) - W * mean ** 2)
        def paa(A, k):
            return A.reshape(A.shape[0], k, W // k).mean(axis=2)
        expected_R = np.hstack([(paa(X, 15) - mean) / tau, (paa(X, 30) - mean) / tau])
        xhat = (X - mean) / np.linalg.norm(X - mean, axis=1, keepdims=True)
        expected_lin = np.hstack([paa(xhat, 15), paa(xhat, 30)])
        np.testing.assert_allclose(sk, expected_R, rtol=0, atol=1e-10)
        np.testing.assert_allclose(sk, expected_lin, rtol=0, atol=1e-10)
        with self.assertRaises(ValueError):          # 14 does not divide 60
            self._corrjoin_ct(data[:, :W], ids, W, step, 0.7, corrjoin_ks=14)

    def test_corrjoin_double_filter_no_false_negatives_and_recovers_bruteforce(self):
        # Both filters are lower bounds on the normalized distance (PAA shrinks distances,
        # Lemma 1 of the paper), so recall must be exactly 1.0 and, with exact validation
        # downstream, the validated set must equal brute force's. Also records r1 (fraction
        # surviving the bucketing filter): the paper's speedup ceiling is 1/r1, and r1 must
        # be near 1 on white noise at T=0.7 (their >=20% density finding) and well below 1
        # on random walks at T=0.9.
        r1 = {}
        for kind in ("noise", "walks"):
            data, ids, W, step, N = self._corrjoin_dataset(kind)
            for T in (0.7, 0.9):
                with self.subTest(kind=kind, T=T):
                    g = self._gt(data, ids, W, step, 0, T, False)
                    self.assertGreater(len(g), 30)
                    ct = self._corrjoin_ct(data, ids, W, step, T)
                    ix = ct.grid_nodes[0]._lsh_index
                    self.assertEqual(type(ix).__name__, "CorrJoinDoubleFilterIndex")
                    self.assertAlmostEqual(ix.eps1, np.sqrt(2 * 15 * (1 - T) / W))
                    self.assertAlmostEqual(ix.eps2, np.sqrt(2 * 30 * (1 - T) / W))
                    met = CorrTrack.compute_metrics_bf(ct.correlated, g, windows=True)
                    self.assertEqual(met["recall"], 1.0)
                    self.assertEqual(met["precision"], 1.0)
                    self.assertEqual(set(ct.correlated), set(g))
                    self.assertTrue(0.0 < ix.last_r1 <= 1.0)
                    r1[(kind, T)] = ix.last_r1
        self.assertGreater(r1[("noise", 0.7)], 0.9)
        self.assertLess(r1[("walks", 0.9)], 0.3)

    def test_corrjoin_negative_correlation_unreachable_through_its_filters(self):
        # The paper's Alg. 1 line 14 uses |corr| and the authors' code applies abs(corr),
        # but both upstream filters are Euclidean on the un-negated normalized vectors:
        # an anti-correlated pair sits at distance ~ sqrt(2(1+T)) > eps_1 and is pruned
        # before the acceptance ever sees it. Falsifiable form: insert x_hat and -x_hat,
        # the index must return no pair. Hence supports_neg_corr = "not_available" and
        # CorrTrack refuses neg_corr=True (and n_lags > 0: CorrJoin is synchronous).
        W, ks, ke = 60, 15, 30
        rng = np.random.default_rng(1)
        x = np.cumsum(rng.standard_normal(W)); xh = (x - x.mean()) / np.linalg.norm(x - x.mean())
        def paa(v, k): return v.reshape(k, W // k).mean(axis=1)
        vx = np.concatenate([paa(xh, ks), paa(xh, ke)]); vy = -vx
        T = 0.7
        ix = CorrJoinDoubleFilterIndex(n_vectors=ks + ke, ks=ks, ke=ke, kb=3,
                                       eps1=np.sqrt(2 * ks * (1 - T) / W), eps2=np.sqrt(2 * ke * (1 - T) / W))
        ix.insert_many(None, np.arange(2), np.vstack([vx, vy]), np.arange(2), np.full(2, 10), np.full(2, W), np.arange(2))
        rows = ix.find_pair_rows_full_cosine(np.arange(2), 0.0, 0.0)
        self.assertEqual(rows.shape[0], 0)                       # perfectly anti-correlated pair: pruned
        ix2 = CorrJoinDoubleFilterIndex(n_vectors=ks + ke, ks=ks, ke=ke, kb=3,
                                        eps1=np.sqrt(2 * ks * (1 - T) / W), eps2=np.sqrt(2 * ke * (1 - T) / W))
        ix2.insert_many(None, np.arange(2), np.vstack([vx, vx * 0.999 + 1e-4]), np.arange(2), np.full(2, 10), np.full(2, W), np.arange(2))
        self.assertEqual(ix2.find_pair_rows_full_cosine(np.arange(2), 0.0, 0.0).shape[0], 1)   # near-identical: kept
        with self.assertRaises(NotImplementedError):
            ix.find_pair_rows_full_cosine_signed(np.arange(2), 0.0, 0.0)
        self.assertEqual(ix.supports_neg_corr, "not_available")
        data, ids, W, step, N = self._corrjoin_dataset("noise", m=12)
        with self.assertRaises(ValueError):
            CorrTrack(window_size=W, basic_window=6, window_step=step, n_vectors=60, n_lags=0, corr_threshold=0.7,
                      neg_corr=True, exec="sequential", data_representation="sketch_paa_svd",
                      candidate_backend="corrjoin_double_filter")
        # (2026-09-23, user) lags are now ENABLED for this arm and disclosed as ours: the constructor
        # accepts n_lags > 0 and the index declares the tier, while neg_corr stays refused above.
        ct_lagged = CorrTrack(window_size=W, basic_window=6, window_step=step, n_vectors=60, n_lags=12, corr_threshold=0.7,
                              neg_corr=False, exec="sequential", data_representation="sketch_paa_svd",
                              candidate_backend="corrjoin_double_filter")
        self.assertEqual(ct_lagged.n_lags, 12)
        ix3 = library_corrtrack_parallel.CorrJoinDoubleFilterIndex(
            n_vectors=ks + ke, ks=ks, ke=ke, kb=3, eps1=np.sqrt(2 * ks * (1 - T) / W), eps2=np.sqrt(2 * ke * (1 - T) / W),
            n_lagged_windows=2)
        self.assertEqual(ix3.supports_lags, "enabled_by_us")
        # two windows alive at different times: the pair is found once, emitted from the later entry
        ix3.insert_many(None, np.arange(2), np.vstack([vx, vx * 0.999 + 1e-4]), np.array([0, 1]),
                        np.array([10, 4]), np.full(2, W), np.arange(2))
        rows_lagged = ix3.find_pair_rows_full_cosine(np.array([0]), 0.0, 0.0)
        self.assertEqual(rows_lagged.shape[0], 1)
        self.assertTrue(ix3.last_lagged_query)
        self.assertEqual(int(rows_lagged[0, 2]), 10)             # later time first
        self.assertEqual(int(rows_lagged[0, 3]), 4)

    def test_run_and_log_corrtrack_corrjoin_knobs_thread_and_pattern_b_contract(self):
        data, ids, W, step, N = self._corrjoin_dataset("walks", m=40)
        base_config = dict(
            window_size=W, window_step=step, basic_window=6, n_lags=0, corr_threshold=0.7,
            neg_corr=False, exec="sequential", parallel_sketch=False, parallel_candidates=False,
            parallel_validation=False, max_workers=0, monitor=False, track_min_dist=True,
            artifact_mode="final", artifact_buffer_max_rows=250000, artifact_merge_mode="merged",
            save_only_required_artifacts=True, save_maxlag_artifacts=False, verbose=False, testing=False,
            validation_metric="pearson",
        )
        run_params = dict(n_vectors=60, seed=1, seed_toggle=2, preprocess=False,
                          data_representation="sketch_paa_svd", candidate_backend="corrjoin_double_filter",
                          corrjoin_ks=12, corrjoin_ke=20, corrjoin_kb=3)
        with tempfile.TemporaryDirectory() as tmp:
            record, _, _ = run_and_log_corrtrack("diag_corrjoin", data, ids, base_config, run_params,
                                                 os.path.join(tmp, "run.csv"), recall_by_window=True,
                                                 verbose=False, testing=False)
        self.assertEqual(int(record["corrjoin_ks"]), 12); self.assertEqual(int(record["corrjoin_ke"]), 20)
        self.assertAlmostEqual(float(record["corrjoin_eps1"]), np.sqrt(2 * 12 * 0.3 / W))
        self.assertEqual(record["supports_neg_corr"], "not_available")
        self._assert_competitor_contract(record, pattern="B")


if __name__ == "__main__":
    unittest.main()
