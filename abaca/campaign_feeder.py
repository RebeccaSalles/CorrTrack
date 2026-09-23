#!/usr/bin/env python3
"""Feed a campaign submit script to OAR while keeping a bounded number of our jobs queued (2026-09-19).

The script written by `campaign_competitors.py --emit` is a header followed by one block per cell
(`# --- <stem>: ...`, six `submit` calls with -a dependencies inside the block). Submitting the
9,606 jobs at once is bad manners on a shared queue and pointless (the N-way jobs wait on their
hyperopt and tuning jobs anyway). This feeder submits whole blocks, in order, only while the number
of our Waiting jobs is below --max-waiting, and re-checks every --poll seconds. It is resumable:
submitted stems are appended to <script>.state and skipped on restart; --dry-run prints instead.
Cells whose N-way result already exists under RESULTS_ROOT (--skip-done) are skipped, so the same
manifest can be re-fed after a failure. Runs on the frontend, e.g.:

    nohup python3 abaca/campaign_feeder.py abaca/campaign_submit.sh --max-waiting 60 --poll 120 \
          > abaca/logs/feeder.log 2>&1 &

Environment passed to the blocks: SNAPSHOT (required by the script), RESULTS_ROOT, PACK_HOSTS, NWAY_HOSTS.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

BLOCK_RE = re.compile(r"^# --- (\S+):", re.M)


def split_script(text: str) -> tuple[str, list[tuple[str, str]]]:
    starts = [m.start() for m in BLOCK_RE.finditer(text)]
    if not starts:
        return text, []
    header = text[:starts[0]]
    blocks = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(text)
        stem = BLOCK_RE.match(text[s:]).group(1)
        blocks.append((stem, text[s:e]))
    return header, blocks


def our_jobs(user: str | None = None) -> dict[str, int]:
    """counts of our OAR jobs by state letter (W, R, T, L, ...)."""
    cmd = ["oarstat", "-u"] + ([user] if user else [])
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        return {}
    counts: dict[str, int] = {}
    for line in out.splitlines()[2:]:
        parts = line.split()
        if len(parts) >= 6:
            counts[parts[5]] = counts.get(parts[5], 0) + 1
    return counts


def block_tags(block: str) -> list[str]:
    """(2026-09-21) the labelled runs a block submits ("pos", "neg", or both): a script may carry one run
    per cell (abaca/campaign_m500_tables.py keeps neg_corr only), so the expected job count and the
    done-check follow the block instead of assuming six jobs / two runs."""
    tags = re.findall(r'^echo "\S+_(pos|neg) H=', block, flags=re.M)
    return tags or ["pos", "neg"]


def nway_done(stem: str, results_root: str, tags=("pos", "neg")) -> bool:
    root = Path(os.path.expandvars(results_root))
    return all((root / "nway" / f"{stem}_{t}" / "nway.json").exists() for t in tags)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("script")
    ap.add_argument("--max-waiting", type=int, default=60, help="submit while fewer than this many of our jobs are Waiting")
    ap.add_argument("--max-total", type=int, default=400, help="and fewer than this many of our jobs exist in any state")
    ap.add_argument("--blocks-per-round", type=int, default=5, help="cells submitted per top-up")
    ap.add_argument("--poll", type=float, default=120.0, help="seconds between checks")
    ap.add_argument("--only", action="append", default=None, help="restrict to stems containing this text (repeatable)")
    ap.add_argument("--skip-done", action="store_true", help="skip cells with an existing nway.json under RESULTS_ROOT")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    text = Path(args.script).read_text()
    header, blocks = split_script(text)
    state_path = Path(args.script + ".state")
    done = set(state_path.read_text().split()) if state_path.exists() else set()
    results_root = os.environ.get("RESULTS_ROOT", "$HOME/corrtrack_abaca_results")
    todo = [(s, b) for s, b in blocks if s not in done and (not args.only or any(o in s for o in args.only))
            and not (args.skip_done and nway_done(s, results_root, block_tags(b)))]
    print(f"{len(blocks)} cells in script, {len(done)} already submitted, {len(todo)} to go; "
          f"limits: waiting<{args.max_waiting}, total<{args.max_total}, {args.blocks_per_round} cells per round, poll {args.poll:.0f}s", flush=True)
    if not args.dry_run and not os.environ.get("SNAPSHOT"):
        sys.exit("SNAPSHOT is not set (export SNAPSHOT=... from abaca/prepare_snapshot.oar)")

    i = 0
    while i < len(todo):
        counts = our_jobs()
        waiting = counts.get("W", 0) + counts.get("T", 0) + counts.get("L", 0)
        total = sum(counts.values())
        if waiting >= args.max_waiting or total >= args.max_total:
            print(f"{time.strftime('%F %T')} hold: waiting={waiting} total={total} ({i}/{len(todo)} cells fed)", flush=True)
            time.sleep(args.poll)
            continue
        room = min(args.blocks_per_round, len(todo) - i)
        for stem, block in todo[i:i + room]:
            if args.dry_run:
                print(f"would submit {stem}")
            else:
                r = subprocess.run(["bash", "-c", header + block], capture_output=True, text=True)
                ids = re.findall(r"(?:H2|H|T|N)=(\d+)", r.stdout)              # the block echoes "<stem>_<tag> H=.. T=.. [H2=..] N=.."
                err = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""
                print(f"{time.strftime('%F %T')} submitted {stem}: jobs {','.join(ids) or 'NONE'} rc={r.returncode} {err}", flush=True)
                if r.returncode == 0 and len(ids) == len(re.findall(r"\$\(submit ", block)):   # one id per submitted job
                    with state_path.open("a") as fh:
                        fh.write(stem + "\n")
                else:
                    print(f"  submission failed for {stem}; stopping the feeder (fix and rerun; state kept)", flush=True)
                    print(r.stdout[-2000:], r.stderr[-2000:], file=sys.stderr)
                    sys.exit(2)
        i += room
        if not args.dry_run:
            time.sleep(5)
    print(f"{time.strftime('%F %T')} all {len(todo)} cells fed", flush=True)


if __name__ == "__main__":
    main()
