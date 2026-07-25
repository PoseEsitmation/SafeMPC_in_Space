#!/usr/bin/env bash
# Catastrophic-forgetting experiment launcher (spaceEnv, hnet).
#
# Runs the 2 conditions x 3 seeds that back the "the safety filter is safe and
# useful, and DAgger works, even in a multi-task HyperCRL setting" claim:
#
#   condition "dagger" : full pipeline (BC + DAgger + CBF/CLF safety loss)
#   condition "bc"     : BC-only ablation (--no-dagger); policy still distils the
#                        MPC expert each dynamics update, but no DAgger rollouts
#
# Filter ON vs OFF is NOT a separate run — every task-boundary evaluation tests
# the same frozen policy both filtered and unfiltered (see
# hnet_exp._eval_forgetting_matrix), so both live inside each run.
#
# Each run writes forgetting_matrix.csv into its timestamped run dir. Afterwards:
#   python scripts/plot_forgetting.py --runs runs/cf --out runs/cf/analysis
#
# Usage:
#   scripts/run_cf_experiment.sh [DEVICE] [SEEDS...]
#   scripts/run_cf_experiment.sh cuda:0 0 1 2
set -euo pipefail

DEVICE="${1:-cuda:0}"; shift || true
SEEDS=("${@:-0 1 2}")
# allow "0 1 2" as one arg or as separate args
if [[ ${#SEEDS[@]} -eq 1 ]]; then read -ra SEEDS <<< "${SEEDS[0]}"; fi

ROOT="runs/cf"
mkdir -p "$ROOT"
echo "Device=$DEVICE  Seeds=${SEEDS[*]}  -> $ROOT"

for seed in "${SEEDS[@]}"; do
  echo "=== seed $seed : DAgger condition ==="
  python main.py run --method hnet --env spaceEnv --device "$DEVICE" \
      --seed "$seed" --savepath "$ROOT" --name "cf_dagger" \
      --cf-experiment

  echo "=== seed $seed : BC-only condition ==="
  python main.py run --method hnet --env spaceEnv --device "$DEVICE" \
      --seed "$seed" --savepath "$ROOT" --name "cf_bc" \
      --cf-experiment --no-dagger
done

echo "All runs done. Analyse with:"
echo "  python scripts/plot_forgetting.py --runs $ROOT --out $ROOT/analysis"
