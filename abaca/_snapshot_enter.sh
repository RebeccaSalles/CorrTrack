# Sourced by the job wrappers (2026-09-19). With SNAPSHOT set, run from that frozen, prebuilt tree
# (abaca/prepare_snapshot.oar) instead of rebuilding the kernels in the shared clone: no build race
# between concurrent jobs, no per-job test suite, and the same binary for every cell. The CPU model
# must match the one the snapshot was built on (-march=native). Without SNAPSHOT the old behaviour
# (build in $REPO_DIR) is kept for standalone runs. Sets WORK_DIR and DO_BUILD.
SNAPSHOT="${SNAPSHOT:-}"
if [ -n "$SNAPSHOT" ]; then
  if [ ! -f "$SNAPSHOT/SNAPSHOT_OK" ]; then
    echo "ERROR: $SNAPSHOT has no SNAPSHOT_OK (prepare_snapshot.oar not run or failed)" >&2; exit 4
  fi
  SNAP_CPU=$(sed -n 's/^cpu_model=//p' "$SNAPSHOT/SNAPSHOT_OK")
  NODE_CPU=$(lscpu | sed -n 's/^Model name: *//p' | sed 's/  */ /g')
  if [ "$SNAP_CPU" != "$NODE_CPU" ]; then
    echo "ERROR: snapshot built on '$SNAP_CPU', this node is '$NODE_CPU' (-march=native binary); refusing" >&2; exit 4
  fi
  WORK_DIR="$SNAPSHOT"; DO_BUILD=0
  echo "snapshot: $SNAPSHOT ($(sed -n 's/^sha=//p' "$SNAPSHOT/SNAPSHOT_OK") built $(sed -n 's/^date=//p' "$SNAPSHOT/SNAPSHOT_OK") on $(sed -n 's/^built_on=//p' "$SNAPSHOT/SNAPSHOT_OK"))"
else
  WORK_DIR="$REPO_DIR"; DO_BUILD=1
fi
