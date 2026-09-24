#!/usr/bin/env bash
# Two arms on one task family: hypernetwork with and without its CL regulariser.
# Usage: scripts/run_cl_experiment.sh [DEVICE] [SEED] [NUM_TASKS] [OUTDIR] [ENV]
set -uo pipefail

DEVICE="${1:-cuda:0}"
SEED="${2:-0}"
NUM_TASKS="${3:-3}"
OUTDIR="${4:-runs/cl_s3}"
ENVNAME="${5:-spaceEnv_thruster}"
PY="${PYTHON:-python}"

cd "$(dirname "$0")/.."
mkdir -p "$OUTDIR"

PIDS=()
for ARM in hnet noreg; do
    EXTRA=""
    [ "$ARM" = noreg ] && EXTRA="--no-hnet-reg"
    "$PY" -u main.py run --method hnet --device "$DEVICE" --seed "$SEED" \
        --num-tasks "$NUM_TASKS" --savepath "$OUTDIR" --env "$ENVNAME" \
        --cf-experiment --cl-profile --fixed-scenario $EXTRA \
        --name "cl_${ARM}_s${SEED}" > "$OUTDIR/${ARM}_seed${SEED}.log" 2>&1 &
    PIDS+=($!)
    sleep 5
done

STATUS=0
for pid in "${PIDS[@]}"; do wait "$pid" || STATUS=1; done
echo "[done] exit status $STATUS"
exit "$STATUS"
