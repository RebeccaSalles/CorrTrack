import csv
import os
import resource
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import corrtrack_run_corrtrack
import candidate_kernels
from library_corrtrack_parallel import (
    CSV_DELIMITER,
    COMPARISON_COLUMNS,
    Candidates,
    CorrTrack,
    CorrTrack_compare,
    CorrTrack_optimize,
    OPTIM_RESULT_COLUMNS,
    RUN_RESULT_COLUMNS,
    Sketches,
    _resolve_candidate_backend,
    _write_correlated_codebook,
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
                os.path.join(tmp, "pred"),
                total_pairs_bf=1,
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
                os.path.join(tmp, "pred"),
                total_pairs_bf=2,
            )

        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1_score"], 1.0)

    def test_blocked_index_reports_multistage_cosine_pruning(self):
        gamma = 0.95
        tau = float(np.sqrt(2.0 - 2.0 * gamma))
        vectors = np.array(
            [
                [0.1, 0.9949874371],
                [0.1, 0.98],
                [0.1, -0.9949874371],
                [0.1, -0.98],
            ],
            dtype=float,
        )
        index = candidate_kernels.BlockedLazyIndex(2, block_size=4, index_dims=2)
        entry_ids = index.insert_many(
            np.zeros(4),
            np.arange(4, dtype=np.int64),
            vectors,
            np.arange(4, dtype=np.int64),
            np.zeros(4, dtype=np.int64),
            np.full(4, 4, dtype=np.int64),
            np.arange(4, dtype=np.int64),
        )

        rows = index.find_pair_rows_full_cosine(entry_ids, gamma, tau)
        stats = index.last_stats

        self.assertEqual(rows.tolist(), [[0, 1, 0, 0, 4], [2, 3, 0, 0, 4]])
        self.assertLessEqual(stats["num_valid_index_candidates"], stats["num_index_candidates"])
        self.assertLessEqual(stats["num_after_coord_filter"], stats["num_valid_index_candidates"])
        self.assertEqual(stats["num_partial_bound_checks"], stats["num_after_coord_filter"])
        self.assertLessEqual(stats["num_after_partial_bound"], stats["num_partial_bound_checks"])
        self.assertLessEqual(stats["num_dot_checks"], stats["num_after_partial_bound"])
        self.assertEqual(stats["num_dot_checks"], stats["num_unique_pre_dot_pairs"])
        self.assertGreater(stats["num_duplicate_pre_dot_pairs"], 0)

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
                    ["k", "v"],
                    key_fn=lambda row: int(row[0]),
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
            gt,
            windows=True,
            use_ground_truth_sign_for_tp=True,
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
            self.assertIn("candidate_block_size_steps", columns)
            self.assertIn("candidate_block_index_dims", columns)
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

    def test_sorted_arrays_bs_backend_rename_keeps_old_name_as_alias(self):
        # (2026-07-05) "bucketed_bst" was renamed to "sorted_arrays_bs" (it
        # never used a BST -- flat sorted arrays + binary search). Old
        # configs/artifacts referencing the previous name must keep working.
        self.assertEqual(_resolve_candidate_backend("bucketed_bst"), "sorted_arrays_bs")
        self.assertEqual(_resolve_candidate_backend("sorted_arrays_bs"), "sorted_arrays_bs")
        self.assertEqual(_resolve_candidate_backend("bucket"), "sorted_arrays_bs")
        self.assertEqual(_resolve_candidate_backend("bucketed"), "sorted_arrays_bs")

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
            "candidate_backend": "sorted_arrays_bs",
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

    def test_dev_blocked_candidate_backends_support_signed_search(self):
        first_partition = {
            ("a", 0, 4): (np.array([1.0, 0.0]), False, 0.0),
            ("b", 0, 4): (np.array([0.7, 0.0]), False, 0.0),
        }
        second_partition = {
            ("c", 4, 4): (np.array([-1.0, 0.0]), False, 0.0),
        }

        def candidate_keys_for(backend):
            candidates = Candidates(
                n_lagged_windows=3,
                grid_dimension=2,
                cell_size=0.05,
                grid_max=1.0,
                freq_threshold=0,
                corr_threshold=0.7,
                n_vectors=2,
                sketch_std=1.0,
                n_grids=1,
                neg_corr=True,
                full_vector=True,
                candidate_backend=backend,
                candidate_similarity="cosine",
                candidate_cosine_threshold=0.5,
                candidate_block_size_steps=1,
                candidate_block_index_dims=1,
                return_distances=False,
            )
            candidates.set_sid_list(["a", "b", "c"])
            candidates.append_partition(0, first_partition)
            candidates.run(2, verbose=False, testing=False, numeric_rows=False)
            candidates.append_partition(1, second_partition)
            freq, selected, _ = candidates.run(1, verbose=False, testing=False, numeric_rows=False)
            return set(freq), set(selected), candidates._candidate_backend

        tree_freq, tree_selected, tree_backend = candidate_keys_for("bptree")
        bucketed_freq, bucketed_selected, bucketed_backend = candidate_keys_for("sorted_arrays_bs")
        self.assertEqual(tree_backend, "bptree")
        self.assertEqual(bucketed_backend, "sorted_arrays_bs")
        self.assertIn(("c", "a", 4, 0, 4), tree_freq)
        self.assertEqual(bucketed_freq, tree_freq)
        self.assertEqual(bucketed_selected, tree_selected)

    def test_signed_cosine_scan_matches_bruteforce_after_fused_sign_pass(self):
        # (2026-07-04) Correctness check for the single-fused-pass rewrite of
        # BlockedLazyIndex/BucketedMultiIndex._scan_block_bucketed_cosine
        # (see docs/implementation_log.md, "sign-canonicalized candidate
        # scan"): the +q/-q search used to run as two full passes; it now
        # runs as one pass with separate marks_pos/marks_neg counters. This
        # cross-checks the fused output against a from-scratch brute-force
        # cosine computation, across several key_dims and thresholds, with
        # data split across a closed block and the still-open current block.
        rng = np.random.default_rng(20260704)
        n = 40
        n_vectors = 8
        window_size = 4
        vectors = rng.normal(size=(n, n_vectors))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

        sim = vectors @ vectors.T

        for backend_cls_name in ("BlockedLazyIndex", "BucketedMultiIndex"):
            for index_dims in (1, 2, 3):
                for gamma in (0.4, 0.6, 0.8):
                    with self.subTest(backend=backend_cls_name, index_dims=index_dims, gamma=gamma):
                        keys = vectors[:, :index_dims].copy()
                        cls = getattr(candidate_kernels, backend_cls_name)
                        split = n // 2
                        index = cls(n_vectors, block_size=split, index_dims=index_dims)
                        if backend_cls_name == "BlockedLazyIndex":
                            index.configure_bucketed()
                        entry_ids = []
                        for lo, hi in ((0, split), (split, n)):
                            entry_ids.append(
                                index.insert_many(
                                    keys[lo:hi],
                                    np.arange(lo, hi, dtype=np.int64),
                                    vectors[lo:hi],
                                    np.arange(lo, hi, dtype=np.int64),
                                    np.zeros(hi - lo, dtype=np.int64),
                                    np.full(hi - lo, window_size, dtype=np.int64),
                                    np.arange(lo, hi, dtype=np.int64),
                                )
                            )
                        entry_ids = np.concatenate(entry_ids)
                        tau = float(np.sqrt(2.0 - 2.0 * gamma))
                        rows = index.find_pair_rows_full_cosine_signed(entry_ids, gamma, tau)
                        got = set(map(tuple, np.asarray(rows).tolist()))

                        expected = set()
                        for i in range(n):
                            for j in range(i + 1, n):
                                if abs(sim[i, j]) >= gamma - 1e-9:
                                    expected.add((i, j, 0, 0, window_size))

                        self.assertEqual(got, expected)

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
        # freq_threshold (0.7 exercises the dict-based candidate path, 0.0
        # with grid_dimension==n_vectors triggers the numeric-rows path via
        # _get_validated_corr_numeric) and checks the exact same correlated
        # pairs and correlation values are found in every case. See
        # docs/implementation_log.md.
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
                basic_window=4,
                window_step=4,
                n_vectors=8,
                n_lags=4,
                grid_dimension=8 if freq_threshold <= 0 else 1,
                cell_size=1,
                seed=11,
                seed_toggle=22,
                corr_threshold=0.7,
                neg_corr=True,
                preprocess=False,
                exec=exec_mode,
                max_workers=2,
                parallel_sketch=False,
                parallel_candidates=False,
                parallel_validation=False,
                candidate_backend="flat",
                hybrid_validation=hybrid,
                hybrid_validation_min_candidates=1,
                hybrid_validation_min_repeat_rate=0.0,
                validation_current_window_cache=use_cache,
                freq_threshold=freq_threshold,
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
            data, rows, 0, 0.0, False,
            current_window_sums=raw_sums,
            current_window_sums_sq=raw_sums_sq,
            current_window_sums_cu=raw_sums_cu,
            current_window_sums_qu=raw_sums_qu,
            current_window_size=w,
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
            x_batch, y_batch, 0.0, False,
            current_window_sums=raw_sums,
            current_window_sums_sq=raw_sums_sq,
            current_window_sums_cu=raw_sums_cu,
            current_window_sums_qu=raw_sums_qu,
            x_series=x_series,
            y_series=y_series,
            x_is_current=x_is_current,
            y_is_current=y_is_current,
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
            data, series1, series2, curr_t1, curr_t2, window_sizes, 0, 8, 0.0, False,
            current_window_sums=raw_sums,
            current_window_sums_sq=raw_sums_sq,
            current_window_sums_cu=raw_sums_cu,
            current_window_sums_qu=raw_sums_qu,
            current_window_size=w,
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

    def test_sorted_arrays_bs_upper_bound_pruning_matches_bruteforce(self):
        # (2026-07-06) Part 1: Cauchy-Schwarz row-level bound and
        # cone/angular block-level bound for BucketedMultiIndex ("sorted_arrays_bs")
        # and BlockedLazyIndex. Both bounds are proven-safe upper bounds (see
        # the module-level helper functions in candidate_kernels.pyx), so
        # enabling this pruning must never drop a true candidate -- verified
        # here against brute-force ground truth across randomized
        # configurations (block sizes, index dims, bound dims, bound-dim
        # selection mode, gamma, signed/unsigned, with and without tight
        # correlated clusters). Also checks that block-level pruning actually
        # fires (num_blocks_pruned_by_ub > 0) on tightly clustered data,
        # since an axis-aligned per-coordinate box bound was tried first and
        # found to essentially never fire on correlated data (see
        # docs/implementation_log.md, "Part 1" and "Part 1 follow-up:
        # cone-based block bound" for the full story) -- this asserts the
        # cone-based replacement actually delivers pruning opportunities, not
        # just correctness.
        def brute_force(vectors, gamma, signed_abs):
            n = vectors.shape[0]
            sim = vectors @ vectors.T
            expected = set()
            for i in range(n):
                for j in range(i + 1, n):
                    v = sim[i, j]
                    ok = abs(v) >= gamma - 1e-9 if signed_abs else v >= gamma - 1e-9
                    if ok:
                        expected.add((i, j))
            return expected

        def run_query(cls, n, dim, block_size, index_dims, bound_dims, bound_dim_var,
                      enable_block, enable_row, gamma, signed_abs, vectors):
            keys = vectors[:, :index_dims].copy()
            idx = cls(dim, block_size, index_dims, 1024, bound_dims, bound_dim_var, enable_block, enable_row)
            if cls.__name__ == "BlockedLazyIndex":
                idx.configure_bucketed()
            entry_ids = idx.insert_many(
                keys, np.arange(n, dtype=np.int64), vectors,
                np.arange(n, dtype=np.int64), np.zeros(n, dtype=np.int64),
                np.full(n, 32, dtype=np.int64), np.arange(n, dtype=np.int64),
            )
            tau = float(np.sqrt(2.0 - 2.0 * gamma))
            if signed_abs:
                rows = idx.find_pair_rows_full_cosine_signed(entry_ids, gamma, tau)
            else:
                rows = idx.find_pair_rows_full_cosine(entry_ids, gamma, tau)
            found = set()
            for r in np.asarray(rows).tolist():
                a, b = r[0], r[1]
                found.add((min(a, b), max(a, b)))
            return found, idx.last_stats

        rng_master = np.random.default_rng(20260706)
        mismatches = []
        for trial in range(120):
            cls = candidate_kernels.BucketedMultiIndex if trial % 2 == 0 else candidate_kernels.BlockedLazyIndex
            n = int(rng_master.integers(5, 150))
            dim = int(rng_master.choice([4, 6, 8, 12, 16, 32]))
            block_size = int(rng_master.choice([4, 8, 16, 32, 64]))
            index_dims = int(rng_master.integers(1, min(4, dim) + 1))
            bound_dims = int(rng_master.integers(0, dim + 1))
            bound_dim_var = bool(rng_master.integers(0, 2))
            enable_block = bool(rng_master.integers(0, 2))
            enable_row = bool(rng_master.integers(0, 2)) if bound_dims > 0 else False
            gamma = float(rng_master.choice([0.1, 0.2, 0.35, 0.5, 0.55, 0.7, 0.85, 0.95]))
            signed_abs = bool(rng_master.integers(0, 2))
            seed = int(rng_master.integers(0, 1_000_000))

            rng_data = np.random.default_rng(seed)
            vectors = rng_data.normal(size=(n, dim))
            if rng_master.random() < 0.4 and n >= 4:
                base = rng_data.normal(size=dim)
                k = int(rng_master.integers(2, min(8, n) + 1))
                idxs = rng_master.choice(n, size=k, replace=False)
                for i in idxs:
                    vectors[i] = base + rng_data.normal(scale=0.05, size=dim)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors /= norms
            expected = brute_force(vectors, gamma, signed_abs)

            with self.subTest(cls=cls.__name__, n=n, dim=dim, block_size=block_size, bound_dims=bound_dims,
                              enable_block=enable_block, enable_row=enable_row, gamma=gamma, signed_abs=signed_abs, seed=seed):
                found, _ = run_query(cls, n, dim, block_size, index_dims, bound_dims, bound_dim_var,
                                      enable_block, enable_row, gamma, signed_abs, vectors)
                if found != expected:
                    mismatches.append((cls.__name__, n, dim, block_size, bound_dims, enable_block, enable_row, gamma, signed_abs, seed))

        self.assertEqual(mismatches, [])

        # Tight angular clusters: block-level (cone) pruning must actually
        # fire here, not just be correct -- an axis-aligned box bound was
        # tried first and never fired on this exact scenario.
        rng = np.random.default_rng(21)
        dim = 32
        block_size = 64
        n_blocks = 20
        n = block_size * n_blocks
        vectors = np.zeros((n, dim))
        for b in range(n_blocks):
            base = rng.normal(size=dim)
            base /= np.linalg.norm(base)
            for k in range(block_size):
                v = base + rng.normal(scale=0.03, size=dim)
                vectors[b * block_size + k] = v / np.linalg.norm(v)
        index_dims = 6
        keys = vectors[:, :index_dims].copy()
        gamma = 0.55
        expected = brute_force(vectors, gamma, False)
        found, stats = run_query(candidate_kernels.BucketedMultiIndex, n, dim, block_size, index_dims, 0, True, True, False, gamma, False, vectors)
        self.assertEqual(found, expected)
        self.assertGreater(stats["num_blocks_pruned_by_ub"], 0)

    def test_block_similarity_assignment_matches_bruteforce_and_actually_prunes(self):
        # (2026-07-06) Part 1 follow-up: block-level cone pruning was found
        # to never fire on real correlation-search workloads, because
        # blocks were formed purely by arrival order (whatever inserted
        # consecutively), unrelated to angular similarity -- the earlier
        # "95.9% block-prune rate" demo above only worked because it
        # inserted each synthetic cluster's members at *consecutive* array
        # positions, so "block == cluster" held by construction of that
        # test, not because the mechanism generally groups correlated rows
        # together. See docs/implementation_log.md, "block-level cone
        # pruning: similarity-aware block assignment".
        #
        # block_similarity_assignment groups each closing block's rows by
        # online "leader" clustering instead. This test builds genuine
        # Pearson-correlation-preserving unit vectors (windows normalized
        # exactly like CorrTrack does, so dot product == real correlation)
        # from several independent correlated clusters, then SHUFFLES series
        # identity before insertion so cluster membership has no relation
        # to array/insertion position -- deliberately avoiding the same
        # methodological mistake. Asserts: (1) correctness is identical to
        # brute force regardless of the new flags (grouping which rows share
        # a block can never drop a true pair -- every row is still in
        # exactly one sealed block); (2) with the flag off, blocks_pruned
        # stays 0 on this realistic order (reproducing the original bug);
        # (3) with it on, blocks_pruned_by_ub is genuinely > 0.
        def brute(vectors, gamma, signed_abs):
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

        def run(cls, vectors, block_size, index_dims, similarity_assignment, max_open, gamma, signed_abs):
            n, dim = vectors.shape
            keys = vectors[:, :index_dims].copy()
            idx = cls(dim, block_size, index_dims, 1024, 0, True, True, True,
                      similarity_assignment, max_open)
            if cls.__name__ == "BlockedLazyIndex":
                idx.configure_bucketed()
            entry_ids = idx.insert_many(
                keys, np.arange(n, dtype=np.int64), vectors,
                np.arange(n, dtype=np.int64), np.zeros(n, dtype=np.int64),
                np.full(n, 32, dtype=np.int64), np.arange(n, dtype=np.int64),
            )
            tau = float(np.sqrt(2.0 - 2.0 * gamma))
            if signed_abs:
                rows = idx.find_pair_rows_full_cosine_signed(entry_ids, gamma, tau)
            else:
                rows = idx.find_pair_rows_full_cosine(entry_ids, gamma, tau)
            found = set()
            for r in np.asarray(rows).tolist():
                a, b = r[0], r[1]
                found.add((min(a, b), max(a, b)))
            return found, idx.last_stats

        rng = np.random.default_rng(20260706)
        n_steps = 300
        n_series = 40
        values = np.zeros((n_series, n_steps))
        n_groups = 8
        group_size = n_series // n_groups
        for g in range(n_groups):
            base = rng.normal(size=n_steps)
            for k in range(group_size):
                idx = g * group_size + k
                values[idx] = base * (0.95 if k % 2 == 0 else -0.9) + rng.normal(scale=0.1, size=n_steps)
        # Deliberately shuffle series identity so cluster membership has no
        # relation to array/insertion position.
        perm = rng.permutation(n_series)
        values = values[perm]
        window = values[:, 40:72]
        window = window - window.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(window, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = window / norms

        gamma = 0.5
        block_size = 8
        index_dims = 6
        for cls in (candidate_kernels.BucketedMultiIndex, candidate_kernels.BlockedLazyIndex):
            for signed_abs in (False, True):
                expected = brute(vectors, gamma, signed_abs)
                found_off, stats_off = run(cls, vectors, block_size, index_dims, False, 4, gamma, signed_abs)
                found_on, stats_on = run(cls, vectors, block_size, index_dims, True, 8, gamma, signed_abs)
                with self.subTest(cls=cls.__name__, signed_abs=signed_abs):
                    self.assertEqual(found_off, expected)
                    self.assertEqual(found_on, expected)
                    self.assertEqual(stats_off["num_blocks_pruned_by_ub"], 0)
                    self.assertGreater(stats_on["num_blocks_pruned_by_ub"], 0)

    def test_corrtrack_sorted_arrays_bs_upper_bound_pruning_end_to_end(self):
        # (2026-07-06) Part 1, end-to-end: streams synthetic correlated series
        # (grouped into clusters with strong within-group correlation) through
        # the real CorrTrack.run() pipeline on candidate_backend="sorted_arrays_bs"
        # with block+row upper-bound pruning enabled vs. disabled, and checks
        # the exact same correlated pairs/values are found either way -- this
        # is the test that would catch a wiring mistake in the new
        # enable_block_ub_pruning/enable_row_ub_pruning/candidate_bound_dims
        # constructor parameters (CorrTrack -> Candidates -> BucketedMultiIndex),
        # not just correctness of the underlying Cython kernel (covered by
        # test_sorted_arrays_bs_upper_bound_pruning_matches_bruteforce above).
        rng = np.random.default_rng(20260706)
        n_steps = 300
        n_series = 40
        values = np.zeros((n_series, n_steps))
        n_groups = 8
        group_size = n_series // n_groups
        for g in range(n_groups):
            base = rng.normal(size=n_steps)
            for k in range(group_size):
                idx = g * group_size + k
                values[idx] = base * (0.95 if k % 2 == 0 else -0.9) + rng.normal(scale=0.1, size=n_steps)
        ids = np.array([f"s{i}" for i in range(n_series)])
        data = np.vstack([np.arange(n_steps), values])

        def run_once(enable_block, enable_row, bound_dims):
            corrtrack = CorrTrack(
                window_size=32, basic_window=8, window_step=8, n_vectors=16, n_lags=8, grid_dimension=16,
                cell_size=1, seed=11, seed_toggle=22, corr_threshold=0.55, neg_corr=True, preprocess=False,
                exec="sequential", parallel_sketch=False, parallel_candidates=False, parallel_validation=False,
                candidate_backend="sorted_arrays_bs", candidate_similarity="cosine", candidate_cosine_threshold=0.5,
                candidate_block_index_dims=6, candidate_block_size_steps=8,
                enable_block_ub_pruning=enable_block, enable_row_ub_pruning=enable_row, candidate_bound_dims=bound_dims,
                hybrid_validation=False,
            )
            step = 8
            for start in range(0, n_steps - step + 1, step):
                corrtrack.run(data[:, start:start + step], ids, verbose=False, testing=False, corr_val=True, monitor=False)
            return corrtrack

        baseline = run_once(False, False, 0)
        pruned = run_once(True, True, 8)

        self.assertGreater(len(baseline.correlated), 0)
        baseline_corr = {k: round(v, 5) for k, v in baseline.correlated.items()}
        pruned_corr = {k: round(v, 5) for k, v in pruned.correlated.items()}
        self.assertEqual(set(baseline_corr.keys()), set(pruned_corr.keys()))
        for key, corr in baseline_corr.items():
            self.assertAlmostEqual(corr, pruned_corr[key], places=4)

    def test_bptree_batched_query_matches_bruteforce_no_dropped_pairs(self):
        # (2026-07-06) Regression test for a real, severe correctness bug in
        # BalancedIndex ("bptree" backend): _scan_tree_full_meta_cosine used
        # to `return` on a duplicate _pair_seen_insert hit -- but that is a
        # recursive tree-walk, not a loop (unlike BlockedLazyIndex/
        # BucketedMultiIndex's row scans, which correctly `continue`), so the
        # `return` silently aborted the entire right-subtree traversal below
        # that point, not just the one duplicate candidate. This meant a
        # batched query (recent_entry_ids with >1 entry) could silently drop
        # true-positive pairs whenever a node's window had already been
        # recorded (accepted or rejected) by an *earlier* query in the same
        # batch -- with the miss rate growing with batch size. See
        # docs/implementation_log.md, "bptree ChatGPT-proposal triage...".
        # This test cross-checks batched bptree queries, at several batch
        # sizes, against a from-scratch brute-force cosine computation.
        rng = np.random.default_rng(20260706)
        n = 24
        n_vectors = 8
        window_size = 32
        gamma = 0.3
        vectors = rng.normal(size=(n, n_vectors))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        sim = vectors @ vectors.T

        window_idx = np.arange(n, dtype=np.int64)
        sid_idx = np.arange(n, dtype=np.int64)
        time_idx = np.zeros(n, dtype=np.int64)
        win_size = np.full(n, window_size, dtype=np.int64)
        sid_rank = np.arange(n, dtype=np.int64)

        for signed in (False, True):
            for batch_size in (1, 8, 16, 24):
                with self.subTest(signed=signed, batch_size=batch_size):
                    index = candidate_kernels.BalancedIndex(n_vectors, 1024, 20260706)
                    index.insert_many(
                        vectors[:, 0].copy(),
                        window_idx,
                        vectors_in=vectors,
                        sid_idx_in=sid_idx,
                        time_in=time_idx,
                        window_size_in=win_size,
                        sid_rank_in=sid_rank,
                    )

                    entry_ids = np.arange(batch_size, dtype=np.int64)
                    if signed:
                        rows = index.find_pair_rows_full_cosine_signed(entry_ids, gamma, 0.0)
                    else:
                        rows = index.find_pair_rows_full_cosine(entry_ids, gamma, 0.0)
                    got = set()
                    for row in np.asarray(rows).tolist():
                        a, b = row[0], row[1]
                        got.add((min(a, b), max(a, b)))

                    expected = set()
                    for i in range(batch_size):
                        for j in range(n):
                            if i == j:
                                continue
                            value = sim[i, j]
                            ok = abs(value) >= gamma - 1e-9 if signed else value >= gamma - 1e-9
                            if ok:
                                expected.add((min(i, j), max(i, j)))

                    self.assertEqual(got, expected)

    def test_bucketed_candidate_backend_supports_random_sign_projection_keys(self):
        candidates = Candidates(
            n_lagged_windows=3,
            grid_dimension=2,
            cell_size=0.05,
            grid_max=1.0,
            freq_threshold=0,
            corr_threshold=0.7,
            n_vectors=2,
            sketch_std=1.0,
            n_grids=1,
            neg_corr=True,
            full_vector=True,
            candidate_backend="sorted_arrays_bs",
            candidate_similarity="cosine",
            candidate_cosine_threshold=0.5,
            candidate_block_size_steps=1,
            candidate_block_index_dims=2,
            candidate_key_mode="random_sign",
            candidate_key_seed=7,
            return_distances=False,
        )
        candidates.set_sid_list(["a", "b", "c"])
        raw_vectors = np.array([[1.0, 0.0], [0.7, 0.0], [-1.0, 0.0]], dtype=np.float64)
        keys = candidates._candidate_key_values(raw_vectors, 2)
        self.assertEqual(keys.shape, (3, 2))
        self.assertFalse(np.allclose(keys[:, 0], raw_vectors[:, 0]))

        candidates.append_partition(
            0,
            {
                ("a", 0, 4): (raw_vectors[0], False, 0.0),
                ("b", 0, 4): (raw_vectors[1], False, 0.0),
            },
        )
        candidates.run(2, verbose=False, testing=False, numeric_rows=False)
        candidates.append_partition(1, {("c", 4, 4): (raw_vectors[2], False, 0.0)})
        freq, selected, _ = candidates.run(1, verbose=False, testing=False, numeric_rows=False)
        self.assertIn(("c", "a", 4, 0, 4), set(freq))
        stats = candidates.candidate_search_stats()
        self.assertGreaterEqual(stats["unique_index_candidates"], stats["after_similarity"])
        self.assertLessEqual(stats["dot_checks"], stats["unique_index_candidates"])

    def test_candidate_key_mode_sampled_sketch_uses_sampled_projection(self):
        candidates = Candidates(
            n_lagged_windows=3,
            grid_dimension=2,
            cell_size=0.05,
            grid_max=1.0,
            freq_threshold=0,
            corr_threshold=0.7,
            n_vectors=2,
            sketch_std=1.0,
            n_grids=1,
            neg_corr=False,
            full_vector=True,
            candidate_backend="sorted_arrays_bs",
            candidate_similarity="cosine",
            candidate_cosine_threshold=0.5,
            candidate_block_size_steps=1,
            candidate_block_index_dims=2,
            candidate_key_mode="sampled_sketch",
            candidate_key_seed=11,
            return_distances=False,
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
            ids,
            window_size=4,
            window_step=4,
            n_lags=4,
            corr_threshold=0.7,
            recall_by_window=True,
            alg="nD",
            neg_corr=False,
            corr_val=False,
            exec="sequential",
            parallel_sketch=False,
            parallel_candidates=False,
            parallel_validation=False,
            candidate_similarity="cosine",
            candidate_cosine_threshold=-1.0,
            proxy_config={"anchor_count": 1, "max_pair_rows": 100, "bootstrap_enabled": False},
        )
        param_combo = {
            "n_vectors": 4,
            "grid_dimension": 4,
            "cell_size": 1.0,
            "seed": 1,
            "seed_toggle": 2,
            "preprocess": False,
            "candidate_backend": "sorted_arrays_bs",
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

    def test_window_size_reduction_stays_incremental(self):
        rng = np.random.default_rng(123)
        ids = np.array(["a", "b", "c"])
        values = rng.normal(size=(3, 32))
        data = np.vstack([np.arange(32), values])
        kwargs = dict(
            basic_window=4,
            window_step=4,
            n_vectors=8,
            n_lags=8,
            grid_dimension=1,
            cell_size=1,
            seed=11,
            seed_toggle=22,
            corr_threshold=0.7,
            exec="sequential",
            parallel_sketch=False,
            parallel_candidates=False,
            parallel_validation=False,
            candidate_backend="flat",
        )
        corrtrack = CorrTrack(window_size=8, **kwargs)
        for start in range(0, 8, 4):
            corrtrack.run(
                data[:, start : start + 4],
                ids,
                verbose=False,
                testing=False,
                corr_val=False,
                monitor=False,
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

    def test_proxy_selection_uses_real_time_tiebreak_within_tolerance(self):
        # (2026-07-05) hyperopt real-time tie-break: candidate-count
        # minimization must remain the dominant objective, but among configs
        # whose candidate rate is within proxy_candidate_rate_close_tolerance
        # of the best achievable, real measured execution time
        # (proxy_search_time_total) should decide the tie -- see
        # docs/implementation_log.md.
        opt = CorrTrack_optimize.__new__(CorrTrack_optimize)
        opt.proxy_bootstrap_min_gt_events = 0
        opt.proxy_candidate_rate_close_tolerance = 0.05

        def row(cand_rate, time_total, recall=0.98, gt_support=100):
            return {
                "hyperopt_strategy": "proxy_anchor",
                "status": "success",
                "proxy_recall_lb": recall,
                "proxy_recall_mean": recall,
                "proxy_recall_med": recall,
                "recall": recall,
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

        # Best candidate rate (0.20) but slow; a config within 5% tolerance
        # (0.205) but fast should win via the time tie-break; a config with a
        # clearly worse candidate rate (0.30, outside tolerance) should never
        # win even though it is the fastest of all three.
        metrics = pd.DataFrame(
            [
                row(0.20, 1.0),
                row(0.205, 0.1),
                row(0.30, 0.01),
            ]
        )
        _metrics, best = opt._apply_proxy_anchor_selection(metrics, target_recall=0.95)
        self.assertIsNotNone(best)
        self.assertEqual(float(best["proxy_candidate_rate"].iloc[0]), 0.205)
        self.assertEqual(float(best["proxy_search_time_total"].iloc[0]), 0.1)

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

    def test_instinct_index_construct_insert_query_smoke(self):
        # No-crash smoke test: construction, insertion, and a query all
        # return well-formed output for each of the 3 implemented query
        # modes (threshold/topk/hybrid). Deliberately does NOT assert
        # anything about recall here -- that is covered separately below.
        rng = np.random.default_rng(7)
        dim = 12
        n = 40
        vectors = rng.normal(size=(n, dim))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

        for mode in ("threshold", "topk", "hybrid"):
            with self.subTest(mode=mode):
                idx = candidate_kernels.InstinctIndex(
                    n_vectors=dim, initial_capacity=64, max_degree=16,
                    ef_insert=64, ef_search=64, entry_points=8,
                    query_mode=mode, top_k=32, min_candidates=8, seed=0,
                )
                entry_ids = self._instinct_insert(idx, vectors)
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
                self.assertIn("instinct_visited_nodes", idx.last_stats)

    def test_instinct_index_lazy_deletion_never_returns_dead_windows(self):
        # Lazy deletion (drop_before_time) must never surface an expired
        # window in a later query's results. Queries only use still-alive
        # entries as anchors -- querying with an id that was just expired
        # is not how CorrTrack actually uses this backend (see
        # docs/implementation_log.md for the test-artifact this avoids).
        rng = np.random.default_rng(11)
        dim = 8
        n = 60
        base = rng.normal(size=dim)
        base /= np.linalg.norm(base)
        vectors = np.array([
            (base + rng.normal(scale=0.05, size=dim))
            for _ in range(n)
        ])
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        times = np.arange(n, dtype=np.int64)

        idx = candidate_kernels.InstinctIndex(
            n_vectors=dim, initial_capacity=64, max_degree=32,
            ef_insert=128, ef_search=128, entry_points=16,
            query_mode="threshold", top_k=64, min_candidates=8, seed=0,
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
        self.assertGreater(idx.last_stats["instinct_dead_nodes_skipped"] + idx.last_stats["instinct_visited_live_nodes"], 0)
        self.assertEqual(idx.last_stats["instinct_num_nodes_alive"], int(alive_mask.sum()))

    def test_instinct_index_recall_vs_bruteforce(self):
        # Honest recall measurement, not a blind threshold: adequately
        # provisioned (max_degree/ef_insert/ef_search/entry_points well
        # above cluster size -- NOT this backend's dead-param-reuse
        # defaults, which measured lower recall on dense clusters in
        # exploratory testing), InstinctIndex reached recall==1.0 against
        # brute-force ground truth across 15 randomized trials spanning
        # dim/cluster-count/cluster-size/gamma/signed/query-mode. This
        # asserts a slightly relaxed floor (0.97) to absorb minor
        # numerical/ordering nondeterminism while still failing loudly if
        # graph connectivity regresses.
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
            rng = np.random.default_rng(1000 + trial)
            dim = int(rng.choice([8, 16, 32]))
            n_clusters = int(rng.integers(3, 10))
            cluster_size = int(rng.integers(5, 25))
            vectors = np.zeros((n_clusters * cluster_size, dim))
            for c in range(n_clusters):
                base = rng.normal(size=dim)
                base /= np.linalg.norm(base)
                for k in range(cluster_size):
                    v = base + rng.normal(scale=0.05, size=dim)
                    vectors[c * cluster_size + k] = v / np.linalg.norm(v)
            n_noise = int(rng.integers(0, n_clusters * 2))
            if n_noise:
                noise = rng.normal(size=(n_noise, dim))
                noise /= np.linalg.norm(noise, axis=1, keepdims=True)
                vectors = np.vstack([vectors, noise])
            n = vectors.shape[0]
            gamma = float(rng.choice([0.6, 0.7, 0.8]))
            signed = bool(rng.integers(0, 2))
            mode = str(rng.choice(["threshold", "hybrid"]))
            expected = brute_force(vectors, gamma, signed)

            idx = candidate_kernels.InstinctIndex(
                n_vectors=dim, initial_capacity=1024, max_degree=64,
                ef_insert=256, ef_search=512, entry_points=32,
                query_mode=mode, top_k=256, min_candidates=64, seed=trial,
            )
            entry_ids = self._instinct_insert(idx, vectors)
            tau = float(np.sqrt(2.0 - 2.0 * gamma))
            if signed:
                rows = idx.find_pair_rows_full_cosine_signed(entry_ids, gamma, tau)
            else:
                rows = idx.find_pair_rows_full_cosine(entry_ids, gamma, tau)
            found = set()
            for r in np.asarray(rows).tolist():
                a, b = r[0], r[1]
                found.add((min(a, b), max(a, b)))
            if expected:
                recall = len(found & expected) / len(expected)
                recalls.append(recall)
                with self.subTest(trial=trial, dim=dim, n=n, gamma=gamma, signed=signed, mode=mode):
                    self.assertGreaterEqual(recall, 0.97)

        self.assertTrue(recalls)


if __name__ == "__main__":
    unittest.main()
