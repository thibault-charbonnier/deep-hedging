"""Parallel grid search over config.json fields.

Usage:
    python scripts/grid_search.py --grid scripts/grid_example.json

Grid spec (JSON):
    {
        "base_config": "config.json",
        "scenarios": [
            {"run.process": "GBM",  "run.benchmark": "BsDelta"},
            {"run.process": "SABR", "run.benchmark": "SABRPractitionerDelta"}
        ],
        "grid": {
            "run.rebalancing": [5, 3, 2, 1],
            "run.maturity":    [0.0833333, 0.25]
        },
        "parallel_workers": 4,
        "paths_cache_dir": "outputs/_paths_cache"
    }

Behaviour:
- If `scenarios` is provided, the Cartesian product of `grid` is applied
  to each scenario (fixed overrides), producing len(scenarios) * prod(grid)
  total runs. Without `scenarios`, behaviour is unchanged.
- Each run is `python main.py --config <tmp.json>` in a subprocess so the
  global random state / torch threading is isolated.
- `paths_cache_dir` (if set) is written into each run's config so that
  runs sharing the same (process, sim config, seed, n_paths) reuse the
  same simulated paths — big speedup on grids that vary only agent
  hyperparameters.
- `parallel_workers` controls concurrency; each worker is forced to
  OMP_NUM_THREADS=1 to avoid torch contention across processes.
- Failed runs are logged but don't stop the rest of the grid.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _apply_override(cfg: dict, dotted_key: str, value) -> None:
    """Set ``cfg[a][b][c] = value`` for a dotted key ``"a.b.c"``.

    Missing intermediate dicts are created in place.
    """
    parts = dotted_key.split(".")
    ref = cfg
    for p in parts[:-1]:
        if p not in ref or not isinstance(ref[p], dict):
            ref[p] = {}
        ref = ref[p]
    ref[parts[-1]] = value


def _cartesian(grid: dict) -> list[dict]:
    """Expand a ``{key: [values, ...]}`` grid into a list of override dicts.

    One dict is produced per element of the Cartesian product of the
    value lists.
    """
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _build_run_config(base_cfg: dict, overrides: dict, paths_cache_dir: str | None) -> dict:
    """Deep-copy ``base_cfg`` and apply the override dict plus optional paths cache dir."""
    cfg = deepcopy(base_cfg)
    for key, val in overrides.items():
        _apply_override(cfg, key, val)
    if paths_cache_dir is not None:
        _apply_override(cfg, "run.paths_cache_dir", paths_cache_dir)
    return cfg


def _run_one(job: tuple[int, int, dict]) -> dict:
    """Execute a single grid run in a subprocess and return its status dict.

    Writes the config to a temp file, runs ``python main.py --config <tmp>``
    with OMP/MKL threads pinned to 1, captures the last 400 chars of
    stdout/stderr, deletes the temp file, and returns a result dict with
    ``ok``, ``elapsed_sec``, and the captured tails.
    """
    idx, total, payload = job
    overrides = payload["overrides"]
    cfg = payload["config"]
    tag = payload["tag"]

    tmp_dir = Path(tempfile.gettempdir()) / "deep-hedging-grid"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = tmp_dir / f"grid_{tag}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    env = os.environ.copy()
    # Prevent torch/OpenBLAS from spawning many threads per worker.
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")

    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "main.py", "--config", str(cfg_path)],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        ok = proc.returncode == 0
        stdout_tail = proc.stdout[-400:] if proc.stdout else ""
        stderr_tail = proc.stderr[-400:] if proc.stderr else ""
    except Exception as exc:
        ok, stdout_tail, stderr_tail = False, "", f"exception: {exc}"
    elapsed = time.time() - t0

    try:
        cfg_path.unlink()
    except OSError:
        pass

    return {
        "idx": idx,
        "total": total,
        "overrides": overrides,
        "ok": ok,
        "elapsed_sec": elapsed,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


def main() -> None:
    """CLI entry point: expand the grid, run all combos (optionally in parallel), dump a manifest."""
    parser = argparse.ArgumentParser(description="Parallel grid search over config fields.")
    parser.add_argument("--grid", required=True, help="Path to grid JSON spec.")
    args = parser.parse_args()

    grid_spec = json.loads(Path(args.grid).read_text(encoding="utf-8"))
    base_cfg_path = REPO_ROOT / grid_spec.get("base_config", "config.json")
    base_cfg = json.loads(base_cfg_path.read_text(encoding="utf-8"))

    grid = grid_spec.get("grid", {})
    scenarios = grid_spec.get("scenarios")
    workers = int(grid_spec.get("parallel_workers", 1))
    paths_cache_dir = grid_spec.get("paths_cache_dir")
    if paths_cache_dir is not None:
        paths_cache_dir = str((REPO_ROOT / paths_cache_dir).resolve())

    grid_combos = _cartesian(grid) if grid else [{}]
    if scenarios:
        combos = [{**scenario, **grid_combo} for scenario in scenarios for grid_combo in grid_combos]
    else:
        if not grid:
            raise ValueError("Grid spec must contain a non-empty 'grid' dict (or 'scenarios').")
        combos = grid_combos
    total = len(combos)
    print(f"[grid] {total} combinations, {workers} workers")
    if paths_cache_dir:
        print(f"[grid] paths cache: {paths_cache_dir}")
    for i, ovr in enumerate(combos, 1):
        print(f"  [{i:03d}] {ovr}")

    jobs = []
    for i, overrides in enumerate(combos, 1):
        run_cfg = _build_run_config(base_cfg, overrides, paths_cache_dir)
        tag = f"{i:04d}_{os.getpid()}"
        jobs.append((i, total, {"overrides": overrides, "config": run_cfg, "tag": tag}))

    results: list[dict] = []
    t_start = time.time()
    if workers <= 1:
        for job in jobs:
            res = _run_one(job)
            results.append(res)
            _log_job_done(res)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_run_one, job) for job in jobs]
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                _log_job_done(res)

    elapsed_total = time.time() - t_start
    ok_count = sum(1 for r in results if r["ok"])
    print(f"[grid] done: {ok_count}/{total} ok in {elapsed_total/60:.1f} min")

    # Dump a tiny manifest to help the aggregation step know which grid was run.
    manifest = {
        "grid_spec": grid_spec,
        "t_start": t_start,
        "combos": combos,
        "results": [
            {"idx": r["idx"], "overrides": r["overrides"], "ok": r["ok"],
             "elapsed_sec": round(r["elapsed_sec"], 1)}
            for r in sorted(results, key=lambda r: r["idx"])
        ],
    }
    manifest_path = REPO_ROOT / "outputs" / f"grid_manifest_{int(time.time())}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[grid] manifest → {manifest_path}")


def _log_job_done(res: dict) -> None:
    """Print a one-line status for a finished job, with stdout/stderr tails on failure."""
    status = "OK " if res["ok"] else "FAIL"
    print(f"[grid] {status} [{res['idx']:03d}/{res['total']}] {res['elapsed_sec']:.1f}s  {res['overrides']}")
    if not res["ok"]:
        if res["stderr_tail"]:
            print(f"       stderr: {res['stderr_tail']!r}")
        if res["stdout_tail"]:
            print(f"       stdout: {res['stdout_tail']!r}")


if __name__ == "__main__":
    main()
