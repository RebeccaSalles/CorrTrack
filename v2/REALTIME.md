# Real-time metrics — throughput, resources, energy

This document describes the **real-time** metrics emitted by the v2 pipeline:
global and per-step **throughput**, its distribution (min/max/median…), its
**curve over time**, and **resource** usage (CPU, memory, energy proxy). The
goal is to judge whether a configuration can sustain a **real-time stream** and
to show *how the system behaves* under load — not just how long it took.

Everything is automatic: **no option to enable**, valid for **every config**
(corrtrack, bf, filcorr, sharded, all backends/indexes).

---

## 1. What is measured

### Throughput

At every *checkpoint* of the streaming loop (by default every `log_every`
seconds, see `--log-every`), a throughput sample is recorded. Five streams are
tracked:

| stream      | unit        | instantaneous throughput definition       | meaning                                |
|-------------|-------------|-------------------------------------------|----------------------------------------|
| `windows`   | windows/s   | Δwindows / **Δwall time**                 | **system throughput** (real-time metric) |
| `sketch`    | windows/s   | Δwindows / Δsketch_time                   | raw capacity of the sketch step        |
| `candidate` | candidates/s| Δcandidates / Δcandidate_time             | candidate generation capacity          |
| `validate`  | pairs/s     | Δtested pairs / Δvalidate_time            | validation capacity (Pearson)          |
| `monitor`   | corr/s      | Δcorrelations / Δmonitor_time             | tracking/monitoring capacity           |

- **`windows`** is relative to **wall time**: this is the throughput actually
  observable at the system output ("is the system keeping up with the arrival
  rate of the windows?"). It is the reference real-time metric.
- The **per-step streams** are relative to the **time spent inside the step**:
  this is the *raw capacity* of the step, independent of what the others do.
  Useful to spot the bottleneck and its ceiling.

Each stream is summarized by: **min, max, median, mean, std, p05, p95**
(over the instantaneous samples) + **overall** = total units / total wall time
(the effective average throughput).

> `tput_sketch_overall` is deliberately `nan`: "total windows / wall time" would
> duplicate `windows_overall`. The instantaneous sketch summary, on the other
> hand, is populated.

### Resources

A background thread samples (every ~0.25 s) the usage of the process **and of
its child processes** (hence correct for the parallel / sharded backends):

| stream        | unit             | meaning                                                    |
|---------------|------------------|------------------------------------------------------------|
| `cpu_pct`     | %                | aggregated CPU % (>100% = several cores)                    |
| `mem_mb`      | MB               | resident memory (RSS)                                       |
| `power_cores` | cores            | power proxy = ΔCPU time / Δwall time (busy cores)           |

Summaries: min/max/median/mean + **peak memory** (`mem_peak_mb`).

### Energy

Real energy in Joules (RAPL / `powermetrics`) requires privileges (sudo on
macOS) → **not attempted**, so as not to add any cost. Instead:

- **`energy_cpu_seconds`** = CPU core·seconds consumed (≈ energy at constant
  TDP). Exact source: `getrusage` (self + children).
- **`power_cores`** (above) is its *instantaneous power* proxy; integrated over
  the duration it gives back the core·seconds.

> If `psutil` is missing, we fall back to `getrusage` at shutdown: only **peak
> memory** and **energy** (core·s) remain available (no CPU/mem time series). A
> very short run (< ~0.3 s) may capture no resource sample at all → CPU/mem at
> `NaN`, but energy and peak memory are still provided.

---

## 2. **Per-step** metrics — time/computation + input→output throughput

On top of the throughput curve sampled over time (§1), each run produces a
**per-step distribution**, measured **at every call** (≈ one window). This is
the instrument dedicated to **per-step speed-up**: two backends are compared on
the *same* step without contamination from the other phases.

Measured steps (corrtrack): `sketch`, `index`, `select`, `validate_sketches`,
`validate`, `monitor` (+ `read`/`windows`/`evict` for time only). In `bf` mode:
`ingest`, `candidates`, `validate`, `monitor`. Every call records
`(duration, n_in, n_out)`, then the summary is:

| quantity               | definition                                             |
|------------------------|--------------------------------------------------------|
| `time` (per call)      | duration of **one** computation, in s — **min/max/median/mean/std/p05/p95** |
| `time_total`           | Σ of the step durations                                 |
| `in_tput` (per call)   | **input** throughput = n_in/duration, items/s (min/max/median…) |
| `out_tput` (per call)  | **output** throughput = n_out/duration, items/s (min/max/median…) |
| `in_overall`           | Σn_in / Σtime (effective input throughput of the step) |
| `out_overall`          | Σn_out / Σtime (effective output throughput of the step) |
| `n_in_total` / `n_out_total` | cumulated input / output elements                |
| `selectivity`          | n_out_total / n_in_total (pass-through rate of the step) |

Per-step input/output (what "enters" and "leaves"):

| step                | input (n_in)            | output (n_out)          | throughput meaning            |
|---------------------|-------------------------|-------------------------|-------------------------------|
| `sketch`            | series of the window    | sketches kept           | sketched series / s           |
| `index`             | sketches                | inserted sketches       | insertions / s                |
| `select`            | queried keys            | candidate pairs         | candidates emitted / s        |
| `validate_sketches` | candidate pairs         | pairs kept (cosine)     | sketch filtering / s          |
| `validate`          | pairs (post-filter)     | correlated pairs        | Pearson validations / s       |
| `monitor`           | correlations            | —                       | correlations tracked / s      |

> `selectivity` reads off the role of the step immediately: `select` ≫ 1
> (combinatorial explosion of candidates), `validate_sketches`/`validate` < 1
> (filters).
>
> **Per-step speed-up** = `phase_<step>_in_overall(B)` / `..._in_overall(A)`
> (or the ratio of `time_total`, or of the per-call `time_median`) between two
> runs A and B. This is the intended use of `phase_metrics.csv`.
>
> **Static sharded** mode: no per-window call → **a single** fused span per step
> (hence `min=max=median`), but the *effective* input/output throughputs remain
> correct. `validate` aggregates cosine+Pearson there (hence `selectivity` =
> correlated/candidates).

---

## 3. Where it shows up

### In `run.log` (readable block at the end of the run)

```
real-time throughput (instantaneous, min/median/max | overall):
      windows        127.9 /     150.6 /     166.2 win/s  | overall     152.3
      sketch         274.8 /     427.0 /     449.6 win/s  | overall       nan
      candidate   180279.9 /  193095.9 /  220828.4 cand/s | overall   35433.3
      validate     74777.8 /   78368.5 /   80240.5 pair/s | overall   34623.9
      monitor    1706908.1 / 2173516.7 / 2338606.1 corr/s | overall    3065.2
      resources | cpu% min/med/max 99/100/100 | mem MB min/med/peak 116/118/120 | energy ~9.5 core·s
per-step metrics (t/call min/med/max ms | in->out items/s overall | sel):
      sketch             n=20    t/call    1.812/   1.881/   2.297 ms |        4226->4226        it/s | sel 1.000
      index              n=20    t/call    0.038/   0.045/   0.053 ms |      177171->177171      it/s | sel 1.000
      select             n=20    t/call    0.073/   0.117/   0.155 ms |       70485->1060792     it/s | sel 15.050
      validate_sketches  n=20    t/call    0.027/   0.117/   0.335 ms |      983343->677477      it/s | sel 0.689
      validate           n=20    t/call    0.512/   3.092/   5.660 ms |       27714->20798       it/s | sel 0.750
      monitor            n=20    t/call    0.006/   0.022/   2.827 ms |      380885->nan         it/s | sel   -
```

### In the result files (per run, under `--output`)

| file               | content                                                                 |
|--------------------|-------------------------------------------------------------------------|
| `summary.csv`      | flat scalars (see §4)                                                   |
| `result.json`      | nested `throughput` / `resources` / `phase_metrics` blocks              |
| **`throughput.csv`** | **the throughput = f(time) curve** — 1 row/checkpoint                  |
| **`resources.csv`**  | CPU % / memory MB / energy-proxy series over time                      |
| **`phase_metrics.csv`** | **1 row per step**: time/computation (min/med/max…) + input/output throughput (§2) |

`throughput.csv` (columns):

```
t, frac, w_done, n_candidates, n_tested, n_correlated,
windows_per_s,   windows_per_s_cum,
sketch_per_s,    sketch_per_s_cum,
candidate_per_s, candidate_per_s_cum,
validate_per_s,  validate_per_s_cum,
monitor_per_s,   monitor_per_s_cum
```

`resources.csv` (columns): `t, cpu_pct, mem_mb, power_cores`.

`phase_metrics.csv` (columns, 1 row/step):

```
phase, calls, time_total,
time_min, time_median, time_max, time_mean, time_std, time_p05, time_p95,
n_in_total, n_out_total, selectivity,
in_overall,  in_min,  in_median,  in_max,
out_overall, out_min, out_median, out_max
```

### In the aggregated report (`comparison.csv` + PDF)

Columns added per run: `tput_win_min`, `tput_win_median`, `tput_win_max`,
`tput_win_overall`, `cpu_pct_median`, `mem_peak_mb`, `energy_cpu_seconds`.

### In the `viz` dashboard

- **Comparison table**: columns `win/s med`, `win/s min`, `mem MB`, `energy`
  (tooltips: min/max/overall throughput, median/max CPU, busy cores).
- **Chart ⑧ "Real-time throughput over time"**: the **per-run throughput
  curve**, with selectors:
  - *metric*: global / cumulative / per-step throughput (sketch/candidate/
    validate/monitor) / CPU % / memory / energy proxy;
  - *log Y*; *X axis*: time (s) or progress (%).
  - Honours the selection/filters; capped at 12 runs (fastest first).

---

## 4. `summary.csv` key reference

```
tput_n_samples                          number of checkpoints (points of the curve)
tput_<stream>_min|max|median|mean|p05|p95 distribution of the instantaneous throughput
tput_<stream>_overall                   total units / wall time
        <stream> ∈ {windows, sketch, candidate, validate, monitor}

phase_<step>_calls                      nb of calls (windows) of the step
phase_<step>_time_total|min|median|max  time PER COMPUTATION (s) — total + distribution
phase_<step>_n_in|n_out                 cumulated input / output elements
phase_<step>_selectivity                n_out_total / n_in_total
phase_<step>_in_overall|out_overall     effective input / output throughput (items/s)
        <step> ∈ {read, windows, sketch, ingest, index, select,
                  validate_sketches, candidates, validate, monitor, evict}

cpu_pct_min|max|median|mean             CPU usage (%)
mem_mb_min|max|median|mean              resident memory (MB)
mem_peak_mb                             peak memory (MB)
power_cores_min|max|median              busy cores (power proxy)
energy_cpu_seconds                      energy proxy (core·seconds)
resources_available                     True if psutil is available (otherwise getrusage only)
resources_n_samples                     number of resource samples
```

---

## 5. Implementation (pointers)

- `v2/core/profiling.py` — `Throughput` (series + summary) and `ResourceSampler`
  (psutil thread + getrusage fallback) classes, `_stats()` helper;
  **`Timings.track()` records `(duration, n_in, n_out)` per call** (through the
  `_CallIO` collector returned by the `with`) and **`Timings.phase_metrics()`**
  derives the per-step distribution from it (time + input/output throughput).
- `v2/core/pipeline.py` — loop instrumentation (`snapshot` per checkpoint +
  `rec.io(n_in=…, n_out=…)` per step), final snapshot for the static sharded
  mode, `_realtime_flat()` / `_phase_metrics_flat()` (summary keys),
  `_phase_metrics_rows()` (CSV), `_log_realtime()` / `_log_phase_metrics()`
  (run.log blocks).
- `v2/core/_static_shard.py` — `_add_timing(..., n_in, n_out)` pushes one fused
  span per step with its effective I/O.
- `v2/steps/persistence.py` — `save_throughput` / `save_resources` /
  `save_phase_metrics`.
- `v2/pipeline.py` — persistence into `result.json`, `comparison.csv` columns.
- `v2/viz.py` — loading (`_read_throughput` / `_read_resources`), table columns,
  `drawTputTimeline` chart.

---

## 6. Quick reading / interpretation

- **`tput_windows_overall` ≥ window arrival rate** ⇒ the config sustains real
  time. Otherwise it falls behind (the stream piles up).
- **`tput_windows_min`** = worst case observed: real-time safety margin.
- **High `std` / min↔max spread** ⇒ unstable throughput (stalls, GC,
  eviction…); the ⑧ chart over time shows *where* it breaks down.
- **Per-step throughput**: the step with the lowest throughput (relative to what
  is required) is the bottleneck. `validate` is typically the hot path (see
  profiling).
- **Per-step speed-up** (`phase_metrics.csv`, §2): to compare two backends on
  the *same* step, take the ratio of `*_in_overall` (or of `time_total`, or of
  the per-call `time_median`). The `time_min/median/max` distribution shows how
  stable the computation is; `selectivity` situates the step (generator vs
  filter).
- **`mem_peak_mb` / `energy_cpu_seconds`**: memory and energy cost of a config,
  to be weighed against its throughput (throughput ↔ resources trade-off).
