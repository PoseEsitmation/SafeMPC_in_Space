#!/usr/bin/env bash
# Fault family + identical-task control, side by side on one GPU.
# Usage: scripts/run_cl_experiment.sh [DEVICE] [SEED] [NUM_TASKS] [OUTDIR]
set -uo pipefail

DEVICE="${1:-cuda:0}"
SEED="${2:-0}"
NUM_TASKS="${3:-3}"
OUTDIR="${4:-runs/cl_s1}"
PY="${PYTHON:-python}"

cd "$(dirname "$0")/.."
mkdir -p "$OUTDIR"

PIDS=()
for ENVNAME in spaceEnv_thruster spaceEnv_null; do
    "$PY" -u main.py run --method hnet --device "$DEVICE" --seed "$SEED" \
        --num-tasks "$NUM_TASKS" --savepath "$OUTDIR" --cf-experiment --cl-profile \
        --env "$ENVNAME" --name "cl_${ENVNAME}" > "$OUTDIR/${ENVNAME}_seed${SEED}.log" 2>&1 &
    PIDS+=($!)
    sleep 5
done

STATUS=0
for pid in "${PIDS[@]}"; do wait "$pid" || STATUS=1; done
echo "[done] exit status $STATUS"
exit "$STATUS"
