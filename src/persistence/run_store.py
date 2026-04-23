from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..hedging_result import HedgingResult
from ..utils.helpers import append_csv_row, build_run_id, dump_json, ensure_dir


@dataclass
class RunContext:
    run_id: str
    script: str
    root_dir: Path
    data_dir: Path
    tables_dir: Path


class RunStore:
    def __init__(self, base_dir: str | Path = "outputs") -> None:
        self.base_dir = ensure_dir(base_dir)
        self.index_path = self.base_dir / "runs_index.csv"

    def start_run(
        self,
        *,
        script: str = "main",
        config: dict[str, Any] | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> RunContext:
        run_id = build_run_id(script)
        root = ensure_dir(self.base_dir / run_id)
        ctx = RunContext(
            run_id=run_id,
            script=script,
            root_dir=root,
            data_dir=ensure_dir(root / "data"),
            tables_dir=ensure_dir(root / "tables"),
        )
        if config is not None:
            dump_json(root / "config.json", config)
        meta_payload = {
            "run_id": run_id,
            "script": script,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "python": sys.version,
            "platform": platform.platform(),
        }
        if extra_meta:
            meta_payload.update(extra_meta)
        dump_json(root / "meta.json", meta_payload)
        return ctx

    def save_result(
        self,
        *,
        ctx: RunContext,
        result: HedgingResult,
        label: str,
        risk_lambda: float = 1.5,
        option_price_t0: float | None = None,
    ) -> None:
        step_df = result.step_frame()
        episode_df = result.episode_table()
        summary_df = result.split_summary(risk_lambda=risk_lambda)

        if option_price_t0 is not None and option_price_t0 > 0:
            scale = 100.0 / float(option_price_t0)
            for col in ("cost", "trade_cost", "liquidation_cost"):
                if col in step_df.columns:
                    step_df[col] = step_df[col] * scale
            for col in (
                "total_cost",
                "mean_step_cost",
                "std_step_cost",
                "total_trade_cost",
                "total_liquidation_cost",
            ):
                if col in episode_df.columns:
                    episode_df[col] = episode_df[col] * scale
            for col in ("mean_total_cost", "std_total_cost", "y_objective"):
                if col in summary_df.columns:
                    summary_df[col] = summary_df[col] * scale

        if not step_df.empty:
            step_df.to_csv(ctx.data_dir / f"{label}_steps.csv", index=False)
        if not episode_df.empty:
            episode_df.to_csv(ctx.tables_dir / f"{label}_episodes.csv", index=False)
        if not summary_df.empty:
            summary_df.to_csv(ctx.tables_dir / f"{label}_summary.csv", index=False)

    def finalize(self, *, ctx: RunContext, ok: bool, note: str = "") -> None:
        append_csv_row(
            self.index_path,
            {
                "run_id": ctx.run_id,
                "script": ctx.script,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "ok": int(ok),
                "note": note,
            },
        )
