#!/usr/bin/env bash
# Launch one arm per process on a task family and wait for all of them.
# Usage: scripts/run_cl_experiment.sh [DEVICE] [SEED] [NUM_TASKS] [OUTDIR] [ENV] [ARMS]
# ARMS: space-separated list of hnet | noreg | hnet_replay | noreg_replay
set -uo pipefail

DEVICE="${1:-cuda:0}"
SEED="${2:-0}"
NUM_TASKS="${3:-3}"
OUTDIR="${4:-runs/cl_s3}"
ENVNAME="${5:-spaceEnv_thruster}"
ARMS="${6:-hnet noreg}"
PY="${PYTHON:-python}"

cd "$(dirname "$0")/.."
mkdir -p "$OUTDIR"

PIDS=()
for ARM in $ARMS; do
    EXTRA=""
    case "$ARM" in
        hnet)          EXTRA="" ;;
        noreg)         EXTRA="--no-hnet-reg" ;;
        hnet_replay)   EXTRA="--replay" ;;
        noreg_replay)  EXTRA="--replay --no-hnet-reg" ;;
        *) echo "unknown arm: $ARM" >&2; exit 2 ;;
    esac
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
