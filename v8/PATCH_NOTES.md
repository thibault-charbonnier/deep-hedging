# Patch notes for the deep hedging project

## What this patch contains

- A working `DQNHedgingAgent`
- A working `DoubleQDNHedgingAgent`
- A paper-aligned `DeepDPGHedgingAgent` with two critics for `E[C]` and `E[C^2]`
- A corrected accounting P&L environment
- A corrected Black-Scholes delta benchmark
- A corrected orchestrator and result logger
- A cleaner example config

## Main conceptual correction

The paper does **not** optimize only the cumulative reward.
It minimizes:

`Y(0) = E[C] + lambda * sqrt(E[C^2] - E[C]^2)`

The provided `DeepDPGHedgingAgent` implements this objective directly with two critics.

## Recommended setup for the report

- Main model: `DeepDPGHedgingAgent`
- Baseline 1: `BSDeltaBenchmark`
- Baseline 2: `DQNHedgingAgent`
- Baseline 3: `DoubleQDNHedgingAgent`

## Important experimental advice

- Increase training episodes significantly above 500
- Keep `gamma = 1.0` for consistency with the paper's undiscounted cost formulation
- Compare GBM and SABR
- Show both mean hedging cost and standard deviation
- Report the paper's objective, not only mean reward

## Simulation package added

The patch now also includes:

- `src/simulation/gbm_process.py`
- `src/simulation/sabr_process.py`
- `src/simulation/svj_process.py`
- `src/utils/helpers.py`
- `main.py`
- `config.json`

`config.json` is intentionally lightweight so the project runs quickly for a smoke test.
Use `config.example.json` as the template for a more serious experiment.
