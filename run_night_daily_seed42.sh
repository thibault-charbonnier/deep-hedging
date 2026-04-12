#!/usr/bin/env bash

# Lance toutes les configurations daily (GBM/SABR x DeepDPG/SkewDDPG x 1M/3M/1Y) avec seed=42.
# Usage:
#   bash run_night_daily_seed42.sh
#   DRY_RUN=1 bash run_night_daily_seed42.sh

set -u

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

SEED=42
REBALANCING="1"
# Maturities in years: 1 month = 0.08333, 3 months = 0.25, 12 months = 1.0
MATURITIES=("0.08333333" "0.25" "1.0")
AGENTS=("DeepDPG" "SkewDDPG")
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

