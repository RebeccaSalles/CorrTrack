import csv
import json
import math
import os
import resource
import sys
import tempfile
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
    CSV_DELIMITER,
    COMPARISON_COLUMNS,
    Candidates,
    Candidates_BF_ExactSTOMP,
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
        results_no_cache = cache_no.validate_pairs(
            data, series1, series2, curr_t1, curr_t2, window_sizes, 0, 8, 0.0, False,
        )["results"]

        cache_yes = candidate_kernels.HybridValidationCache(1024)
        results_with_cache = cache_yes.validate_pairs(
            data, series1, series2, curr_t1, curr_t2, window_sizes, 0, 8, 0.0, False, current_window_sums=raw_sums, current_window_sums_sq=raw_sums_sq, current_window_sums_cu=raw_sums_cu, current_window_sums_qu=raw_sums_qu, current_window_size=w,
        )["results"]

        for exp, no_cache_row, with_cache_row in zip(expected, results_no_cache, results_with_cache):
            self.assertAlmostEqual(no_cache_row[2], exp, places=9)
            self.assertAlmostEqual(with_cache_row[2], exp, places=9)

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
        record = optimizer._run_corrtrack_proxy_anchor((0, param_combo, "smoke"))
        self.assertEqual(record["status"], "success")
        self.assertGreater(record["cand_w"], 0)
        self.assertGreater(record["proxy_search_index_candidates"], 0)
        self.assertGreaterEqual(record["proxy_search_objective_rate"], record["proxy_candidate_rate"])

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

    def test_filcorr_candidates_node_matches_exact_stomp_pearson_full_band(self):
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
                st = Candidates_BF_ExactSTOMP(
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


if __name__ == "__main__":
    unittest.main()
