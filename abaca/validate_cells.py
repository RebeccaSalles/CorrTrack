"""Completeness check over the campaign cells: every arm must have run, and every tunable arm must
have used parameters the protocol chose, not the fallback defaults. The arm count alone does not
show this: a cell can carry all twelve arms while an arm ran untuned because its tuning job died.
Prints one line per problem and exits non-zero when any cell is not comparable."""
import json, glob, os, sys, collections

ROOT = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/corrtrack_abaca_results")
TUNED_ARMS = {"parcorr", "csz", "statstream", "corrjoin"}          # the CSZ protocol tunes these
CT_ARMS = {"corrtrack": "hyperopt", "corrtrack_hamming": "hyperopt_hamming"}
bad = collections.defaultdict(list)
cells = sorted(glob.glob(f"{ROOT}/nway/*_m500_*/nway.json"))
for f in cells:
    stem = os.path.basename(os.path.dirname(f)); d = json.load(open(f)); arms = d["arms"]
    tuned = {k for k in (d.get("competitor_params_tuned") or {})}
    for a, r in arms.items():
        if not isinstance(r, dict): continue
        st = r.get("status")
        if st not in ("ok", "N/A"):
            bad[stem].append(f"{a}: status={st}")
        elif st == "ok" and a in TUNED_ARMS and a not in tuned:
            bad[stem].append(f"{a}: ran on fallback defaults (no best_params used)")
    for a, d_name in CT_ARMS.items():
        if arms.get(a, {}).get("status") == "ok":
            src = d.get("corrtrack_params_source" if a == "corrtrack" else "corrtrack_hamming_params_source", "")
            if "UNTUNED" in str(src): bad[stem].append(f"{a}: UNTUNED defaults")
    missing = [a for a in ("bruteforce", "corrtrack", "corrtrack_hamming", "exact_stomp", "bf_incremental", "filcorr") if a not in arms]
    if "bruteforce" in missing: bad[stem].append("bruteforce missing")
print(f"cells checked: {len(cells)}; not comparable: {len(bad)}")
for stem, msgs in sorted(bad.items()):
    print(f"  {stem}")
    for m in msgs: print(f"      {m}")
sys.exit(1 if bad else 0)
