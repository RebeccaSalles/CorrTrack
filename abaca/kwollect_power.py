#!/usr/bin/env python3
"""Fetch the node power series recorded by Grid'5000's kwollect for our OAR jobs (2026-09-19).

Energy on Abaca: the RAPL counters are root-only on the nodes (energy_uj: Permission denied,
perf_event_paranoid = 3, /dev/cpu/*/msr root), so the per-process RAPL reading in
abaca/resource_probe.py is None there. What exists is the node-level power exported by the
node exporter when a job is submitted with `-t "monitor=prom_.*"` (narrower regexes such as
`prom_node_hwmon.*` recorded nothing; probes 3122066/68/69/70 on 2026-09-19):
  prom_node_hwmon_power_average_watt   ACPI power meter (acpi000d), 2 s averaging, one sample per 15 s;
                                       verified against a 20-core burn: 64 W idle -> 245 W within one
                                       sample, back to 65 W within one sample (job 3122070)
  prom_node_ipmi_power_watts           BMC "Pwr Consumption", coarse steps and ~1 min lag; fallback only
The mercantour nodes have no wattmeter. The N-way jobs of the campaign are submitted with the
monitor type; this script pulls the two series per job from the API and writes
{host: [[epoch_s, watt], ...]} for abaca/aggregate_campaign.py --power, which integrates the
power over each arm's [t_start_epoch, t_end_epoch] (trapezoid) and subtracts the node's idle
power for the dynamic energy. Resolution caveat, to be stated with the numbers: 15 s samples
make the per-arm energy meaningful for arms running well over a minute; for shorter arms only
the per-cell (per-job) energy is quotable.

    python abaca/kwollect_power.py --results-root ~/corrtrack_abaca_results --out power.json
    python abaca/kwollect_power.py --job 3122066 --out power.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

API = "https://api.grid5000.fr/stable/sites/{site}/metrics"
PREFERRED = ("prom_node_hwmon_power_average_watt", "prom_node_ipmi_power_watts")


def fetch(url: str) -> list[dict]:
    # curl: the frontend has it and the API needs no credentials from inside the site
    r = subprocess.run(["curl", "-s", "-m", "300", url], capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        return []
    try:
        d = json.loads(r.stdout)
    except json.JSONDecodeError:
        return []
    return d if isinstance(d, list) else []


def series_for_job(job_id: int, site: str) -> dict[str, dict[str, list]]:
    # metrics= keeps the answer to the two power series (a 24 h job with monitor=prom_.* holds ~10M rows otherwise)
    rows = fetch(API.format(site=site) + f"?job_id={job_id}&metrics=" + ",".join(PREFERRED))
    out: dict[str, dict[str, list]] = {}
    for x in rows:
        mid = x.get("metric_id")
        if mid not in PREFERRED:
            continue
        host = str(x.get("device_id", "")).split(".")[0]
        t = datetime.fromisoformat(x["timestamp"]).timestamp()
        out.setdefault(host, {}).setdefault(mid, []).append([t, float(x["value"])])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default=None, help="collect the OAR job ids of the N-way cells under nway/*/nway.json")
    ap.add_argument("--job", type=int, action="append", default=[], help="OAR job id (repeatable)")
    ap.add_argument("--site", default="sophia")
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", default=None, help="force a metric (default: hwmon, falling back to ipmi per host)")
    args = ap.parse_args()
    jobs = set(args.job)
    if args.results_root:
        for f in Path(args.results_root).glob("nway/*/nway.json"):
            jid = (json.load(open(f)).get("node") or {}).get("oar_job_id")
            if jid:
                jobs.add(int(jid))
    out_path = Path(args.out)
    power: dict[str, list] = json.load(open(out_path)) if out_path.exists() else {}
    meta = power.pop("_meta", {"jobs": [], "metric_by_host": {}})
    done = set(meta.get("jobs", []))
    for jid in sorted(jobs - done):
        s = series_for_job(jid, args.site)
        for host, by_metric in s.items():
            mid = args.metric or next((m for m in PREFERRED if m in by_metric), None)
            if not mid or mid not in by_metric:
                continue
            power.setdefault(host, []).extend(by_metric[mid])
            meta["metric_by_host"][host] = mid
        meta["jobs"].append(jid)
        print(f"job {jid}: {sum(len(pts) for v in s.values() for pts in v.values())} samples on {sorted(s)}", flush=True)
    for host in power:
        power[host] = sorted({tuple(p) for p in power[host]})
        power[host] = [list(p) for p in power[host]]
    power["_meta"] = meta
    json.dump(power, open(out_path, "w"))
    print(f"wrote {out_path}: {len(power) - 1} hosts, {len(meta['jobs'])} jobs")


if __name__ == "__main__":
    main()
