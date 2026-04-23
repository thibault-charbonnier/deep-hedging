"""Aggregate grid-search runs into a single comparable CSV.

Usage:
    python scripts/grid_aggregate.py --manifest outputs/grid_manifest_1713883284.json
    python scripts/grid_aggregate.py --since 2026-04-23

Outputs:
    outputs/grid_summary_<timestamp>.csv

Columns per row (one row per run):
    run_id, timestamp, process, agent, benchmark, seed,
    <all grid keys as columns>,
    train_episodes, eval_episodes,
    mean_rl, std_rl, skew_rl, cvar95_rl, cvar99_rl,
    mean_bm, std_bm, skew_bm, cvar95_bm, cvar99_bm,
    y_rl, y_bm, improvement_pct,
    ok
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "outputs"


def _cvar(values: list[float], alpha: float = 0.95) -> float:
    """Empirical CVaR at level alpha: mean of observations above the alpha quantile.

    Returns NaN if no finite values are available.
    """
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    threshold = float(np.quantile(finite, alpha))
    tail = finite[finite >= threshold]
    if tail.size == 0:
        return float("nan")
    return float(tail.mean())


def _stat_skewness(values: list[float]) -> float:
    """Population skewness (3rd standardised moment), NaN when undefined.

    Uses ddof=0 and requires at least 3 finite values and non-zero std.
    """
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size < 3:
        return float("nan")
    mean = float(finite.mean())
    std = float(finite.std(ddof=0))
    if std == 0:
        return float("nan")
    return float(np.mean(((finite - mean) / std) ** 3))


def _load_run_row(run_dir: Path, grid_keys: list[str]) -> dict | None:
    """Build a single summary row from a run directory.

    Reads config/meta, pulls the requested grid keys (supporting dotted
    paths into nested dicts), and computes agent/benchmark cost metrics
    (mean, std, skew, CVaR95/99, Y objective). Returns None if the run
    directory is missing config or meta; returns a partial row with
    ok=False if evaluation tables are missing.
    """
    cfg_path = run_dir / "config.json"
    meta_path = run_dir / "meta.json"
    if not cfg_path.exists() or not meta_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    row: dict = {
        "run_id": meta.get("run_id", run_dir.name),
        "timestamp": meta.get("timestamp"),
        "process": cfg.get("run", {}).get("process"),
        "agent": cfg.get("run", {}).get("agent"),
        "benchmark": cfg.get("run", {}).get("benchmark"),
        "seed": cfg.get("run", {}).get("seed"),
        "train_episodes": cfg.get("training_schedule", {}).get("train_episodes"),
        "eval_episodes": cfg.get("training_schedule", {}).get("eval_episodes"),
    }
    for key in grid_keys:
        parts = key.split(".")
        val: object = cfg
        for p in parts:
            if isinstance(val, dict) and p in val:
                val = val[p]
            else:
                val = None
                break
        row[key] = val

    rl_sum_path = run_dir / "tables" / "eval_agent_summary.csv"
    bm_sum_path = run_dir / "tables" / "eval_benchmark_summary.csv"
    rl_ep_path = run_dir / "tables" / "eval_agent_episodes.csv"
    bm_ep_path = run_dir / "tables" / "eval_benchmark_episodes.csv"
    ok = all(p.exists() for p in (rl_sum_path, bm_sum_path, rl_ep_path, bm_ep_path))
    row["ok"] = bool(ok)
    if not ok:
        return row

    rl_sum = pd.read_csv(rl_sum_path).iloc[0]
    bm_sum = pd.read_csv(bm_sum_path).iloc[0]
    rl_costs = pd.read_csv(rl_ep_path)["total_cost"].tolist()
    bm_costs = pd.read_csv(bm_ep_path)["total_cost"].tolist()

    row.update({
        "mean_rl": float(rl_sum["mean_total_cost"]),
        "std_rl": float(rl_sum["std_total_cost"]),
        "skew_rl": _stat_skewness(rl_costs),
        "cvar95_rl": _cvar(rl_costs, 0.95),
        "cvar99_rl": _cvar(rl_costs, 0.99),
        "y_rl": float(rl_sum["y_objective"]),
        "mean_bm": float(bm_sum["mean_total_cost"]),
        "std_bm": float(bm_sum["std_total_cost"]),
        "skew_bm": _stat_skewness(bm_costs),
        "cvar95_bm": _cvar(bm_costs, 0.95),
        "cvar99_bm": _cvar(bm_costs, 0.99),
        "y_bm": float(bm_sum["y_objective"]),
    })
    row["improvement_pct"] = (
        100.0 * (row["y_bm"] - row["y_rl"]) / row["y_bm"]
        if row["y_bm"] not in (0.0,) and np.isfinite(row["y_bm"]) else float("nan")
    )
    return row


def _resolve_runs(args: argparse.Namespace) -> tuple[list[Path], list[str]]:
    """Return list of run dirs + discovered grid keys (from the manifest or inferred)."""
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        grid_keys = list(manifest.get("grid_spec", {}).get("grid", {}).keys())
        # After a grid, the runs we care about are the most recent matching the
        # process/agent/benchmark in the grid configs. Simplest: filter by
        # mtime on the run's meta.json, newer than manifest creation time.
        manifest_time = float(Path(args.manifest).stat().st_mtime)
        candidates = [p for p in OUTPUTS.iterdir() if p.is_dir() and not p.name.startswith("_")]
        runs = [p for p in candidates if (p / "meta.json").exists() and (p / "meta.json").stat().st_mtime >= manifest_time - 5]
        runs.sort(key=lambda p: (p / "meta.json").stat().st_mtime)
        return runs, grid_keys

    since = datetime.fromisoformat(args.since) if args.since else None
    grid_keys = args.grid_keys.split(",") if args.grid_keys else []
    runs = []
    for p in OUTPUTS.iterdir():
        if not p.is_dir() or p.name.startswith("_"):
            continue
        meta = p / "meta.json"
        if not meta.exists():
            continue
        if since is not None:
            m = json.loads(meta.read_text(encoding="utf-8"))
            ts = m.get("timestamp")
            if not ts:
                continue
            try:
                if datetime.fromisoformat(ts) < since:
                    continue
            except ValueError:
                continue
        runs.append(p)
    runs.sort(key=lambda p: (p / "meta.json").stat().st_mtime)
    return runs, grid_keys


def main() -> None:
    """CLI entry point: resolve runs, build one row per run, write a summary CSV."""
    parser = argparse.ArgumentParser(description="Aggregate grid-search runs into one CSV.")
    parser.add_argument("--manifest", default=None, help="Path to grid_manifest JSON (written by grid_search.py).")
    parser.add_argument("--since", default=None, help="ISO date/time: only aggregate runs newer than this.")
    parser.add_argument("--grid-keys", default="", help="Comma-separated grid keys to extract (when not using --manifest).")
    parser.add_argument("--out", default=None, help="Output CSV path (default: outputs/grid_summary_<ts>.csv).")
    args = parser.parse_args()

    runs, grid_keys = _resolve_runs(args)
    if not runs:
        print("[agg] no matching runs found")
        return

    rows: list[dict] = []
    for run_dir in runs:
        row = _load_run_row(run_dir, grid_keys)
        if row is not None:
            rows.append(row)

    if not rows:
        print("[agg] no runs could be loaded")
        return

    df = pd.DataFrame(rows)
    # Order: run_id first, then hyperparams, then metrics, then ok flag.
    meta_cols = ["run_id", "timestamp", "process", "agent", "benchmark", "seed",
                 "train_episodes", "eval_episodes"]
    metric_cols = ["mean_rl", "std_rl", "skew_rl", "cvar95_rl", "cvar99_rl", "y_rl",
                   "mean_bm", "std_bm", "skew_bm", "cvar95_bm", "cvar99_bm", "y_bm",
                   "improvement_pct", "ok"]
    ordered = [c for c in meta_cols if c in df.columns] + \
              [c for c in grid_keys if c in df.columns] + \
              [c for c in metric_cols if c in df.columns]
    extras = [c for c in df.columns if c not in ordered]
    df = df[ordered + extras]

    out_path = Path(args.out) if args.out else OUTPUTS / f"grid_summary_{int(runs[-1].stat().st_mtime)}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[agg] wrote {len(df)} rows → {out_path}")
    print(df.to_string(index=False, max_colwidth=20))


if __name__ == "__main__":
    main()
