"""CorrTrack hyper-parameter grid for the competitor campaign, lsh_hamming_exact backend (2026-09-21).

experiment_run_param_grid_campaign.py fixes candidate_backend at lsh_approx (lsh_sign_dot), so the
campaign's corrtrack arm was that backend only. The m=500 sweeps report both backends
(hamming_exact + dot gate, and lsh_sign_dot tuned); this grid is the campaign grid restricted to
lsh_hamming_exact, searched by CorrTrack's own proxy hyperopt on the calibration span like the
other one. The LSH occupancy axis does not apply to this backend (no bands), the dot-gate offset
and n_vectors do; the hamming pre-filter fraction is the lsh_approx-only knob, left at its default.
"""
from experiment_run_param_grid_campaign import PARAM_GRID as _CAMPAIGN
PARAM_GRID = dict(_CAMPAIGN)
PARAM_GRID.update({
    "candidate_backend": ["lsh_hamming_exact"],
    "candidate_lsh_target_occupancy": [3.0],
    "candidate_hamming_filter_max_frac": [None],
})
