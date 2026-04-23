from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from ..hedging_result import HedgingResult
from ..utils.helpers import append_csv_row, build_run_id, dump_json, ensure_dir


@dataclass
class RunContext:
    run_id: str
    script: str
    root_dir: Path
    data_dir: Path
    tables_dir: Path
    figures_dir: Path
    profile_dir: Path


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
            figures_dir=ensure_dir(root / "figures"),
            profile_dir=ensure_dir(root / "profile"),
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
        save_figures: bool = True,
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

        if save_figures:
            self._save_figures(ctx=ctx, step_df=step_df, episode_df=episode_df, label=label)

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

    def save_profile_text(self, *, ctx: RunContext, stats_text: str) -> None:
        (ctx.profile_dir / "cprofile.txt").write_text(stats_text, encoding="utf-8")

    def _save_figures(self, *, ctx: RunContext, step_df: pd.DataFrame, episode_df: pd.DataFrame, label: str) -> None:
        if not episode_df.empty and "total_cost" in episode_df.columns:
            plt.figure(figsize=(7, 4))
            for split_name, g in episode_df.groupby("split", sort=False):
                plt.hist(g["total_cost"], bins=30, alpha=0.45, label=split_name)
            plt.title(f"Total cost distribution - {label}")
            plt.xlabel("Total hedging cost")
            plt.ylabel("Count")
            plt.legend()
            plt.tight_layout()
            plt.savefig(ctx.figures_dir / f"{label}_cost_dist.png", dpi=150)
            plt.close()

        if not step_df.empty and {"episode_idx", "action"}.issubset(step_df.columns):
            df = step_df.copy()
            df["prev_action"] = df.groupby(["split", "episode_idx"], sort=False)["action"].shift(1).fillna(0.0)
            sample = df[df["step_idx"] > 0]
            if not sample.empty:
                plt.figure(figsize=(6, 6))
                plt.scatter(sample["prev_action"], sample["action"], s=4, alpha=0.25)
                lo = min(sample["prev_action"].min(), sample["action"].min())
                hi = max(sample["prev_action"].max(), sample["action"].max())
                plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
                plt.title(f"Action stability map - {label}")
                plt.xlabel("Previous holding")
                plt.ylabel("Current holding")
                plt.tight_layout()
                plt.savefig(ctx.figures_dir / f"{label}_action_scatter.png", dpi=150)
                plt.close()

