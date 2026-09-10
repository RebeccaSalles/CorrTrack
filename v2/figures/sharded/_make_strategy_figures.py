"""Three figures dedicated to the sharding PARALLELIZATION STRATEGY.

They complement architecture/split/fused (the overview) by focusing on HOW the
work is split between threads, the load balancing and the barriers:

  • strategy_1_sketch_index.png — Block 1: sketch parallel over a WINDOW bundle
    (1 sketcher/worker) → SERIAL index merge (id space, bulk_load O(N)) →
    BARRIER before select.
  • strategy_2_split.png        — Block 2 `split`: TWO successive splits.
    select over WINDOWS (uneven pairs per window) → global arrays (IA,IB) →
    validate over BALANCED PAIRS (equal slices, chunked) ⇒ no straggler threads.
  • strategy_3_fused.png        — Block 2 `fused`: select+validate fused per
    WINDOW, accumulating only the correlations ⇒ memory bounded to 1 window.

Reference model: v2/core/_static_shard.py (see v2/SHARDING.md).

Run it occasionally:

    python -m v2.figures.sharded._make_strategy_figures
"""

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))

# worker colors (same as _make_figures.py, for a consistent deck)
WCOL = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
C_SK, C_IDX = "#cfe8ff", "#bfe3ea"
C_SEL, C_VAL = "#e3d9fb", "#fbd6e9"
C_NOTE, C_WARN, C_OK = "#f3f4f6", "#ffe0e0", "#dcf3df"


# --------------------------------------------------------------------------- helpers
def _box(ax, x, y, w, h, label, fc="white", ec="#333", lw=1.3, fs=9, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                linewidth=lw, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, weight=weight)


def _arrow(ax, p0, p1, color="#333", lw=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=13,
                                 lw=lw, color=color))


def _ltext(ax, x, y, w, h, title, lines, fc, fs=8.5, tc="#222"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                linewidth=1.3, edgecolor="#333", facecolor=fc))
    ax.text(x + 0.12, y + h - 0.20, title, ha="left", va="top",
            fontsize=fs + 1, weight="bold", color=tc)
    ax.text(x + 0.12, y + h - 0.58, "\n".join(lines), ha="left", va="top",
            fontsize=fs, family="monospace")


def _barrier(ax, x, y0, y1, label="⊥ BARRIER"):
    ax.plot([x, x], [y0, y1], color="#b00", lw=2.2, ls=(0, (4, 2)))
    ax.text(x, y1 + 0.10, label, ha="center", va="bottom", fontsize=8.5,
            weight="bold", color="#b00")


# =========================================================================== FIG 1
def fig_sketch_index():
    fig, ax = plt.subplots(figsize=(13.2, 7.6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8.6); ax.axis("off")
    ax.set_title("Sharding — Block 1: sketch parallel over WINDOWS  →  SERIAL index "
                 "merge (id space)  →  barrier", fontsize=12.5, weight="bold")

    # data = windows statically partitioned into 4 bundles (1 per worker)
    ax.text(0.3, 8.0, "data = N_w windows, STATIC partition into W bundles "
            "(1 bundle/worker, fixed up front)", fontsize=9, style="italic")
    nb, per = 4, 3
    x0, ww, gap = 0.3, 0.74, 0.10
    for w in range(nb):
        for j in range(per):
            xx = x0 + (w * per + j) * (ww + gap)
            _box(ax, xx, 7.05, ww, 0.55, f"W{w*per+j}", WCOL[w], fs=7.5)
    ax.text(x0, 6.78, "windows (each colored by its owning worker)",
            fontsize=7.8, style="italic", color="#555")

    # 4 parallel sketch lanes
    ax.text(0.3, 6.35, "① SKETCH  —  parallel", fontsize=10.5, weight="bold")
    lane_x, lane_w, lh, lg = 0.3, 8.4, 0.5, 0.14
    y = 5.85
    for w in range(nb):
        yy = y - w * (lh + lg)
        _box(ax, lane_x, yy, lane_w, lh, "", WCOL[w])
        ax.text(lane_x + 0.12, yy + lh / 2, f"thread {w}", ha="left",
                va="center", fontsize=8.5, weight="bold")
        ax.text(lane_x + 1.6, yy + lh / 2,
                f"sketch( bundle {w} )  ·  1 sketcher, cache NOT shared",
                ha="left", va="center", fontsize=8.2, family="monospace")
    _ltext(ax, 9.1, 3.85, 4.6, 2.5, "WHY it scales", [
        "each worker sketches ITS own",
        "windows, nothing shared →",
        "no contention.",
        "",
        "≈ linear scaling (array-native",
        "compute, GIL released).",
    ], C_OK, fs=8.4)

    # barrier + serial index merge
    _barrier(ax, 9.0, 3.55, 6.45)
    ax.text(0.3, 3.35, "② INDEX  —  SERIAL merge", fontsize=10.5, weight="bold")
    _box(ax, 0.3, 2.25, 8.5, 0.95,
         "serial merge of the sketches  ·  bulk_load O(N) (no insort)  ·  "
         "store/index keyed by INTEGER ID", C_IDX, fs=9)
    ax.text(0.3, 1.95, "each (series, t) gets its id = its row in the contiguous "
            "SK / RAW matrices", fontsize=8.2, style="italic", color="#555")

    _ltext(ax, 0.3, 0.25, 13.4, 1.45, "ID SPACE  (shared key of both blocks)", [
        "store + index keyed by INTEGER ID (≠ the (series, t) key)  →  select returns "
        "id pairs directly: ZERO key→id lookup over the millions of candidates.",
        "contiguous SK / RAW matrices: vectorized SK[ids] gather. Conversion back to "
        "keys only for the ~survivors (canonical order → byte-identical to serial).",
    ], C_NOTE, fs=8.3)

    fig.savefig(os.path.join(HERE, "strategy_1_sketch_index.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)


# =========================================================================== FIG 2
def fig_split():
    fig, ax = plt.subplots(figsize=(13.4, 8.4))
    ax.set_xlim(0, 14); ax.set_ylim(0, 9.4); ax.axis("off")
    ax.set_title("Sharding — Block 2  \"split\" (default): TWO splits — "
                 "select over WINDOWS, validate over BALANCED PAIRS",
                 fontsize=12.5, weight="bold")

    # --- SELECT: per window, uneven pairs ---
    ax.text(0.3, 8.85, "② SELECT  —  parallel over a WINDOW bundle", fontsize=10.5,
            weight="bold")
    # pairs produced per window = uneven (width ∝ number of pairs)
    widths = [0.9, 3.0, 0.7, 1.4, 4.2, 0.6, 1.1, 2.3]   # number of pairs/window
    owner = [0, 0, 1, 1, 2, 2, 3, 3]
    y = 8.25
    x0 = 2.1
    for i, (wv, ow) in enumerate(zip(widths, owner)):
        yy = y - i * 0.30
        _box(ax, x0, yy, wv, 0.24, "", WCOL[ow], lw=0.8, fs=6)
        ax.text(x0 - 0.1, yy + 0.12, f"win {i}", ha="right", va="center", fontsize=7)
        ax.text(x0 + wv + 0.1, yy + 0.12, f"{int(wv*30)} pairs", ha="left",
                va="center", fontsize=6.8, color="#555")
    ax.text(x0, y - 8 * 0.30 + 0.10, "→ the candidate count is VERY uneven across "
            "windows (a few windows concentrate the pairs)", fontsize=8,
            style="italic", color="#a00")

    # global arrays
    _box(ax, 9.6, 6.7, 4.1, 0.95,
         "global id arrays\n(IA, IB)  —  all windows", C_SEL, fs=9, weight="bold")
    _arrow(ax, (8.6, 7.2), (9.55, 7.2))

    # --- VALIDATE: per balanced pairs ---
    ax.text(0.3, 5.60, "validate  —  parallel over a BALANCED PAIR bundle", fontsize=10.5,
            weight="bold")
    ax.text(0.3, 5.32, "the pair total is RE-SPLIT into W EQUAL slices "
            "(≠ per window) → identical load per thread", fontsize=8.2,
            style="italic", color="#070")

    total_w = 11.0
    eqw = total_w / 4
    yv = 4.55
    for w in range(4):
        xx = 0.3 + w * (eqw + 0.12)
        _box(ax, xx, yv, eqw, 0.6, f"thread {w}\n= {int(total_w*30/4)} pairs",
             WCOL[w], fs=8)
    ax.text(0.3, yv - 0.30, "each slice: cosine (sketch filter) + ARRAY-NATIVE "
            "Pearson, GIL released, in CHUNKS of 250k pairs (memory-safe)",
            fontsize=8, style="italic", color="#555")

    # straggler contrast
    _ltext(ax, 0.3, 2.05, 6.5, 2.05, "✗ if validate were split PER WINDOW", [
        "thread 0: ████████████  (win 4 dense)",
        "thread 1: ██",
        "thread 2: █",
        "thread 3: ███",
        "→ 3 threads wait for the straggler",
    ], C_WARN, fs=8.3, tc="#a00")
    _ltext(ax, 7.1, 2.05, 6.6, 2.05, "✓ split PER PAIRS (split)", [
        "thread 0: ██████",
        "thread 1: ██████",
        "thread 2: ██████",
        "thread 3: ██████",
        "→ equal load, ISOLATED validate ≈48×",
    ], C_OK, fs=8.3, tc="#070")

    _ltext(ax, 0.3, 0.25, 13.4, 1.5, "MEMORY & MEASUREMENT", [
        "memory-safe: select converts each window into an id array right away "
        "(no millions of Python tuples); validate bounds the gather with chunks.",
        "the validate span measures ONLY the parallel compute (tuple assembly + "
        "vectorized owner-window attachment via searchsorted = AFTERWARDS, off the hot path).",
        "trade-off vs fused: GLOBAL IA/IB + survivors in RAM — for an isolated, fast validate.",
    ], C_NOTE, fs=8.2)

    fig.savefig(os.path.join(HERE, "strategy_2_split.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)


# =========================================================================== FIG 3
def fig_fused():
    fig, ax = plt.subplots(figsize=(13.2, 8.0))
    ax.set_xlim(0, 14); ax.set_ylim(0, 9.0); ax.axis("off")
    ax.set_title("Sharding — Block 2  \"fused\" (alternative): select+validate "
                 "FUSED per WINDOW  →  bounded memory", fontsize=12.5, weight="bold")

    ax.text(0.3, 8.45, "each worker iterates over ITS window bundle; for EACH "
            "window: select → validate → only the correlations are accumulated",
            fontsize=9, style="italic")

    # 4 lanes, each = a [select|validate] sequence per window
    lane_x, lh, lg = 0.3, 0.78, 0.20
    y = 8.05
    seq = [("sel", C_SEL), ("val", C_VAL)]
    for w in range(4):
        yy = y - w * (lh + lg)
        ax.text(lane_x, yy + lh / 2, f"thread {w}", ha="left", va="center",
                fontsize=9, weight="bold")
        # 3 windows per worker, each chaining select+validate
        bx = lane_x + 1.3
        for f in range(3):
            for name, col in seq:
                _box(ax, bx, yy + 0.12, 0.95, lh - 0.24,
                     f"{name}\nf{w*3+f}", col, fs=7.5)
                bx += 0.95 + 0.04
            bx += 0.28  # separator between windows
        ax.text(bx + 0.1, yy + lh / 2, "→ corr[]", ha="left", va="center",
                fontsize=8, family="monospace", color="#070")
        # worker band on the left
        ax.add_patch(FancyBboxPatch((lane_x + 1.18, yy + 0.05), 0.10, lh - 0.10,
                                    boxstyle="square,pad=0", facecolor=WCOL[w],
                                    edgecolor="none"))

    _ltext(ax, 0.3, 2.45, 6.6, 1.85, "BOUNDED MEMORY", [
        "at any moment a worker only holds",
        "ONE window (its pairs + its SK/RAW",
        "gather), never the global set.",
        "→ safe on very dense data",
        "  (ASOS, large n_lags: ~10⁸ pairs).",
    ], C_OK, fs=8.5, tc="#070")
    _ltext(ax, 7.1, 2.45, 6.6, 1.85, "split  vs  fused", [
        "split: GLOBAL IA/IB in RAM, validate",
        "       ISOLATED and the fastest (≈48×).",
        "fused: memory bounded to 1 window, validate",
        "       INTERLEAVED (span split synthetically),",
        "       safe when the pairs explode.",
    ], C_NOTE, fs=8.4)

    _ltext(ax, 0.3, 0.25, 13.4, 1.95, "COMMON TO BOTH MODES", [
        "STATIC partition: one bundle fixed up front per worker, NO per-window "
        "re-dispatch, NO nested pool (threads, GIL released on the array-native compute).",
        "the correlated.csv result is BYTE-IDENTICAL to serial streaming (causal filter: "
        "the latest window owns the pair; ids converted back in canonical order).",
        "dense regime: select stays GIL-bound (candidate generation in pure Python) and "
        "validate becomes memory-bandwidth bound → moderate gain; decisive gain with few candidates.",
    ], C_NOTE, fs=8.2)

    fig.savefig(os.path.join(HERE, "strategy_3_fused.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)


# =========================================================================== FIG 4
def fig_validate():
    fig, ax = plt.subplots(figsize=(13.6, 8.6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 9.6); ax.axis("off")
    ax.set_title("Sharding — parallel strategy of the VALIDATION (split mode): "
                 "balanced PAIR bundle, chunked, array-native (GIL released)",
                 fontsize=12.3, weight="bold")

    # input: global arrays of id pairs
    _box(ax, 0.3, 8.45, 13.4, 0.7,
         "INPUT = global id arrays  (IA, IB)  —  ALL the candidate pairs of "
         "every window (id space, coming from select)",
         C_SEL, fs=9, weight="bold")

    # balanced partition by NUMBER of pairs
    ax.text(0.3, 7.95, "①  BALANCED PARTITION  —  W EQUAL slices by NUMBER of "
            "pairs (≠ per window → no straggler thread)", fontsize=9.6,
            weight="bold")
    eqw = 13.4 / 4
    for w in range(4):
        xx = 0.3 + w * eqw
        _box(ax, xx + 0.03, 7.15, eqw - 0.06, 0.55,
             f"thread {w}: IA[{w}::4], IB[{w}::4]", WCOL[w], fs=8.2)

    # ② each thread: loop over chunks
    ax.text(0.3, 6.65, "②  EACH THREAD  —  processes its slice in CHUNKS of "
            "250k pairs (env CORRTRACK_SHARD_VALIDATE_CHUNK, memory-safe)",
            fontsize=9.6, weight="bold")

    # pipeline of one chunk (chained boxes)
    steps = [
        ("gather\nSK[IA], SK[IB]", C_IDX, 2.5),
        ("COSINE\n(sketch filter ≥ thr)", C_VAL, 2.8),
        ("gather RAW\non survivors", C_IDX, 2.4),
        ("PEARSON\narray-native", C_VAL, 2.5),
        ("corr[]\n(a,b,r,lag)", C_OK, 2.0),
    ]
    bx = 0.3
    yb = 5.5
    for i, (lab, col, ww) in enumerate(steps):
        _box(ax, bx, yb, ww, 0.95, lab, col, fs=8.3,
             weight="bold" if "COSINE" in lab or "PEARSON" in lab else "normal")
        if i < len(steps) - 1:
            _arrow(ax, (bx + ww, yb + 0.47), (bx + ww + 0.18, yb + 0.47))
        bx += ww + 0.18
    ax.text(0.3, 5.25, "everything in BATCHED numpy → the GIL is released during "
            "the compute → real parallelism between threads", fontsize=8.4,
            style="italic", color="#070")

    # ③ serial post-processing, off the hot path
    _ltext(ax, 0.3, 2.95, 6.6, 1.95, "③  AFTERWARDS (serial, OFF the hot path)", [
        "the threads return ARRAYS.",
        "result-tuple assembly +",
        "OWNER-window attachment through",
        "searchsorted (vectorized)",
        "→ canonical order",
        "⇒ byte-identical to serial.",
    ], C_NOTE, fs=8.4)
    _ltext(ax, 7.1, 2.95, 6.6, 1.95, "WHY BALANCE PER PAIRS", [
        "a few windows concentrate the candidates.",
        "splitting per WINDOW → 1 straggler",
        "thread, the others wait.",
        "splitting per PAIRS → equal load.",
        "→ the validate span measures ONLY the",
        "  parallel compute: ISOLATED ≈ 48×.",
    ], C_OK, fs=8.4, tc="#070")

    _ltext(ax, 0.3, 0.25, 13.4, 2.45, "MEMORY & REGIME", [
        "memory-safe: without the chunking, a bundle of millions of pairs would "
        "materialize (N×n_vec) [SK] and (N×window) [RAW] matrices → OOM",
        "  (e.g. 25M pairs, n_vectors=32, window=168 → tens of GB). The chunk bounds "
        "the SK[ia]/RAW[ia] gather.",
        "two-stage filter: COSINE on the sketches (cheap, prunes most of them) THEN exact "
        "Pearson on the survivors only.",
        "DENSE regime (many candidates): validate becomes memory-bandwidth bound → the gain "
        "plateaus around 2-4 workers; DECISIVE gain with few candidates (validate dominates and scales).",
        "fused alternative: validate interleaved per window (memory bounded to 1 window) → "
        "see strategy_3_fused.png.",
    ], C_NOTE, fs=8.1)

    fig.savefig(os.path.join(HERE, "strategy_4_validate.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
def main():
    fig_sketch_index()
    fig_split()
    fig_validate()
    fig_fused()
    print("written:", ", ".join([
        "strategy_1_sketch_index.png",
        "strategy_2_split.png",
        "strategy_4_validate.png",
        "strategy_3_fused.png",
    ]), "in", HERE)


if __name__ == "__main__":
    main()
