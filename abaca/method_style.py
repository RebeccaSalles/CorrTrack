"""One visual identity for every method, shared by the campaign and synthetic figure builders
(2026-09-28, user: the legend must read the same way in every plot).

Order: how the methods rank in the campaign, leaders first, counted exactly as the ranking tables
count them (abaca/build_ranking_tables.py): the cells where a method is the fastest, with the same
10% tie band, over the 179 cells of the final m=500 campaign. Firsts / among the two fastest:
CorrTrack-LSH 116 / 129, CorrTrack-Ham 71 / 125, FilCorr 65 / 79, BF_incr 9 / 52, CorrJoin 7 / 11,
StatStream 2 / 9, then the five methods that never lead a cell, ordered by median speedup
(BRAID 1.60, TSUBASA 1.32, ParCorr 0.84, ThinBRAID 0.63, CSZ 0.32). Restricting the count to the
arms that reach recall 0.95, as the qualified ranking table does, gives the same order
(103 / 80 / 67 / 10 / 8 / 2). Recompute if a campaign replaces this one.
"""
from __future__ import annotations

# (result key in nway.json, label): BF_vect is the brute force each cell is timed against
ARMS = [("bf_incremental", "BF_incr"), ("filcorr", "FilCorr"), ("tsubasa", "TSUBASA"), ("braid", "BRAID"),
        ("thinbraid", "ThinBRAID"), ("corrtrack", "CorrTrack-LSH"), ("corrtrack_hamming", "CorrTrack-Ham"),
        ("parcorr", "ParCorr"), ("csz", "CSZ"), ("statstream", "StatStream"), ("corrjoin", "CorrJoin")]
NAMES = [n for _, n in ARMS]
LEAD_ORDER = ["CorrTrack-LSH", "CorrTrack-Ham", "FilCorr", "BF_incr", "CorrJoin", "StatStream",
              "BRAID", "TSUBASA", "ParCorr", "ThinBRAID", "CSZ"]
BF = "BF_vect"
COLOR = {"BF_incr": "#4c78a8", "FilCorr": "#f58518", "TSUBASA": "#8c6d31", "BRAID": "#7b4173",
         "ThinBRAID": "#d67ab1", "CorrTrack-LSH": "#c03d3e", "CorrTrack-Ham": "#e8927c",
         "ParCorr": "#2f8a57", "CSZ": "#8fbf6b", "StatStream": "#57a3c7", "CorrJoin": "#c8b44a",
         BF: "#222222"}
MARK = {"CorrTrack-LSH": "o", "CorrTrack-Ham": "s"}
# where the capability a cell exercises comes from, as the runs record it
SHAPE = {"native": "o", "enabled_by_us": "^", "specified": "s", "not_available": "X", None: "."}
HATCH = {"native": "", "enabled_by_us": "//", "specified": "..", "not_available": "xx", None: ""}
TIER_LABEL = {"native": "evaluated by the authors",
              "enabled_by_us": "capability added by us",
              "specified": "specified by the authors, never evaluated there"}


def lead(names):
    """The given names in the shared legend order, anything unknown appended."""
    return [n for n in LEAD_ORDER if n in names] + [n for n in names if n not in LEAD_ORDER]
