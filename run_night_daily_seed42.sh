#!/usr/bin/env bash

# Grille nightly via configs temporaires (compatible main.py --config).
# Force 5000/1500 épisodes pour toutes les combinaisons.
#
# Usage:
#   bash run_night_daily_seed42.sh
#   DRY_RUN=1 MAX_RUNS=3 bash run_night_daily_seed42.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

BASE_CONFIG="config.json"
REBALANCING=1
SEEDS=(42)
PROCESSES=("GBM" "SABR" "SVJ")
AGENTS=("DeepDPG" "SkewDDPG")
MATURITIES=("0.0833333333" "0.25" "1.0")

# name|actor_lr|critic_lr|noise_decay|risk_lambda|per_alpha
HYPER_PROFILES=(
  "base|1e-4|1e-3|0.9995|1.5|0.6"
  "stable|8e-5|8e-4|0.9998|1.2|0.5"
  "fast|2e-4|1e-3|0.999|1.0|0.6"
)

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="outputs/nightly_logs_${TS}"
TMP_CFG_DIR="outputs/nightly_tmp_configs_${TS}"
mkdir -p "$LOG_DIR" "$TMP_CFG_DIR"

echo "Logs       : $LOG_DIR"
echo "Tmp configs: $TMP_CFG_DIR"

max_runs="${MAX_RUNS:-0}"
run_count=0

for process in "${PROCESSES[@]}"; do
  if [[ "$process" == "SABR" ]]; then
    benchmark="BartlettDelta"
  else
    benchmark="BsDelta"
  fi

  for agent in "${AGENTS[@]}"; do
    for maturity in "${MATURITIES[@]}"; do
      for seed in "${SEEDS[@]}"; do
        for hp in "${HYPER_PROFILES[@]}"; do
          IFS='|' read -r hp_name actor_lr critic_lr eps_decay risk_lambda per_alpha <<< "$hp"

          tag="${process}_${agent}_${benchmark}_M${maturity}_S${seed}_${hp_name}"
          cfg_path="$TMP_CFG_DIR/${tag}.json"
          log_file="$LOG_DIR/${tag}.log"

          python - <<PY
import json
from pathlib import Path

cfg = json.loads(Path("${BASE_CONFIG}").read_text(encoding="utf-8"))

cfg["run"]["process"] = "${process}"
cfg["run"]["agent"] = "${agent}"
cfg["run"]["benchmark"] = "${benchmark}"
cfg["run"]["seed"] = int(${seed})
cfg["run"]["maturity"] = float("${maturity}")
cfg["run"]["rebalancing"] = int(${REBALANCING})

# Force même budget d'entraînement/évaluation sur toute la grille
cfg["training_schedule"]["train_episodes"] = 5000
cfg["training_schedule"]["eval_episodes"] = 1500

ha = cfg["hedging_agent"]
ha["actor_learning_rate"] = float("${actor_lr}")
ha["critic_learning_rate"] = float("${critic_lr}")
ha["exploration_noise_decay"] = float("${eps_decay}")
ha["risk_lambda"] = float("${risk_lambda}")
ha["per_alpha"] = float("${per_alpha}")

Path("${cfg_path}").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
PY

          cmd=(python main.py --config "$cfg_path")

          echo "=== START $tag ==="
          if [[ "${DRY_RUN:-0}" == "1" ]]; then
            printf '%q ' "${cmd[@]}"
            printf '\n'
            echo "=== DRY RUN DONE $tag ==="
          else
            set +e
            "${cmd[@]}" >"$log_file" 2>&1
            status=$?
            set -e
            if [[ $status -ne 0 ]]; then
              echo "!!! FAILED $tag (code=$status) -> $log_file"
            else
              echo "=== DONE $tag -> $log_file ==="
            fi
          fi

          run_count=$((run_count + 1))
          if [[ "$max_runs" -gt 0 && "$run_count" -ge "$max_runs" ]]; then
            echo "Reached MAX_RUNS=$max_runs, stopping early."
            echo "Total processed: $run_count"
            exit 0
          fi
        done
      done
    done
  done
done

echo "All nightly runs finished. Total processed: $run_count"

