"""Tests of the independent-band-bits mode's occupancy rule and cost-ratio calibration (corrtrack_release_indepbands)."""
import math
import unittest

import numpy as np

import library_corrtrack_parallel as L


class OccupancyRuleTest(unittest.TestCase):
    def test_rule_values_and_shape(self):
        # rho = log2(1 / (1 - arccos(T) / pi)); occupancy = c rho / (1 - rho)
        self.assertAlmostEqual(L.lsh_auto_occupancy(0.8, 11.9), 5.89, places=1)
        self.assertAlmostEqual(L.lsh_auto_occupancy(0.9, 11.9), 3.43, places=1)
        occ = [L.lsh_auto_occupancy(t, 12.0) for t in (0.6, 0.7, 0.8, 0.9, 0.95)]
        self.assertTrue(all(a > b for a, b in zip(occ, occ[1:])))          # falls as the threshold rises
        self.assertAlmostEqual(L.lsh_auto_occupancy(0.8, 24.0), 2 * L.lsh_auto_occupancy(0.8, 12.0), places=9)
        self.assertEqual(L.lsh_auto_occupancy(0.999, 1.0), 1.0)              # floored at 1
        self.assertEqual(L.lsh_auto_occupancy(0.8), L.lsh_auto_occupancy(0.8, L.LSH_DEFAULT_COST_RATIO))

    def _ct(self, **kw):
        base = dict(window_size=60, basic_window=6, window_step=6, n_vectors=64, n_lags=30, corr_threshold=0.8,
                    exec="sequential", candidate_backend="lsh_sign_dot")
        base.update(kw)
        return L.CorrTrack(**base)

    def test_auto_only_in_independent_mode_and_explicit_wins(self):
        self.assertAlmostEqual(self._ct(candidate_lsh_independent_bits=True).candidate_lsh_target_occupancy,
                               L.lsh_auto_occupancy(0.8))
        self.assertAlmostEqual(self._ct(candidate_lsh_independent_bits=True, candidate_lsh_target_occupancy="auto")
                               .candidate_lsh_target_occupancy, L.lsh_auto_occupancy(0.8))
        self.assertAlmostEqual(self._ct(candidate_lsh_independent_bits=True, candidate_lsh_cost_ratio=24.0)
                               .candidate_lsh_target_occupancy, L.lsh_auto_occupancy(0.8, 24.0))
        self.assertEqual(self._ct(candidate_lsh_independent_bits=True, candidate_lsh_target_occupancy=12.0)
                         .candidate_lsh_target_occupancy, 12.0)
        # (2026-10-29) independent bits are the default, with the automatic occupancy; pooled keeps 3.0
        self.assertTrue(self._ct().candidate_lsh_independent_bits)
        self.assertAlmostEqual(self._ct().candidate_lsh_target_occupancy, L.lsh_auto_occupancy(0.8))
        self.assertEqual(self._ct(candidate_lsh_independent_bits=False).candidate_lsh_target_occupancy, 3.0)

    def test_fit_recovers_known_cost_ratio(self):
        rng = np.random.default_rng(1)
        runs = []
        for cell, beta in ((0.8, 0.01), (0.9, 0.03)):
            for occ, nb in ((2, 90), (3, 70), (6, 55), (12, 45), (24, 35), (48, 28)):
                runs.append({"cell": cell, "occupancy": occ, "n_bands": nb,
                             "time": beta * nb * (10.0 + occ) * (1 + 0.005 * rng.standard_normal())})
        c, _ = L._fit_lsh_cost_ratio(runs)
        self.assertAlmostEqual(c, 10.0, delta=0.5)

    def test_calibration_smoke(self):
        out = L.calibrate_lsh_cost_ratio(n_series=60, n_obs=360, group_size=6, thresholds=(0.8,), occupancies=(3, 12),
                                         repeats=1, verbose=False)
        self.assertEqual(len(out["runs"]), 2)
        self.assertTrue(0.5 <= out["cost_ratio"] < 100.0)
        self.assertIn(0.8, out["occupancy_by_threshold"])


if __name__ == "__main__":
    unittest.main()


class IndependentModeSelectionTest(unittest.TestCase):
    """The independent mode's tuning picks the best recall among the feasible configurations whose validation work
    (candidates passing the gate) is within PROXY_VALIDATION_WORK_TOLERANCE of the fewest; time only breaks ties."""

    @staticmethod
    def _frame(indep):
        import pandas as pd
        rows = [  # name, recall lb, recall mean, candidates passing the gate, proxy time, n_bands
            ("A", 0.951, 0.960, 1000, 1.0, 20),
            ("B", 0.990, 0.995, 1050, 1.2, 20),
            ("C", 0.999, 0.999, 2000, 0.5, 20),
            ("D", 0.900, 0.920, 500, 0.4, 10),
        ]
        return pd.DataFrame([{
            "name": n, "hyperopt_strategy": "proxy_anchor", "status": "success", "proxy_recall_lb": lb,
            "proxy_recall_mean": mean, "proxy_recall_ub": min(1.0, lb + 0.01), "proxy_search_after_dot": work,
            "proxy_search_time_total": t, "candidate_lsh_n_bands": nb, "candidate_lsh_target_occupancy": 5.43,
            "candidate_lsh_independent_bits": indep, "proxy_gt_support": 1000, "cand_w": work,
        } for n, lb, mean, work, t, nb in rows])

    def _select(self, indep):
        from types import SimpleNamespace
        dummy = SimpleNamespace(proxy_bootstrap_min_gt_events=0)
        _, best = L.CorrTrack_optimize._apply_proxy_anchor_selection(dummy, self._frame(indep), 0.95)
        return best["name"].iloc[0]

    def test_independent_mode_prefers_safer_recall_among_equally_cheap(self):
        self.assertEqual(self._select(True), "B")      # A barely feasible, C too much work, D infeasible

    def test_pooled_mode_keeps_the_band_count_ranking(self):
        self.assertEqual(self._select(False), "C")     # same bands among feasible rows, then time: C is fastest
