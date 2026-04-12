#!/usr/bin/env bash

# Lance toutes les configurations daily (GBM/SABR x DeepDPG/CVaRDPG x 1M/3M/1Y) avec seed=42.
# Usage:
#   bash run_night_daily_seed42.sh
#   DRY_RUN=1 bash run_night_daily_seed42.sh

set -u

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

SEED=42
MODE="full"
REBALANCING="daily"
MATURITIES=("0.0833333333" "0.25" "1.0")
AGENTS=("DeepDPG" "CVaRDPG")
PROCESSES=("GBM" "SABR")

log_dir="outputs/nightly_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$log_dir"

echo "Logs: $log_dir"

for process in "${PROCESSES[@]}"; do
  if [ "$process" = "GBM" ]; then
    benchmark="BsDelta"
  else
    benchmark="BartlettDelta"
  fi

  for agent in "${AGENTS[@]}"; do
    for maturity in "${MATURITIES[@]}"; do
      tag="${process}_${agent}_${benchmark}_M${maturity}_S${SEED}"
      log_file="$log_dir/${tag}.log"

      cmd=(
        python main.py
        --mode "$MODE"
        --process "$process"
        --agent "$agent"
        --benchmark "$benchmark"
        --maturity "$maturity"
        --rebalancing "$REBALANCING"
        --seed "$SEED"
      )

      echo "=== START $tag ==="
      if [ "${DRY_RUN:-0}" = "1" ]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
        echo "=== DRY RUN DONE $tag ==="
        continue
      fi

      "${cmd[@]}" >"$log_file" 2>&1
      status=$?
      if [ $status -ne 0 ]; then
        echo "!!! FAILED $tag (code=$status) -> $log_file"
      else
        echo "=== DONE $tag -> $log_file ==="
      fi
    done
  done
done

echo "All nightly runs finished."

