from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .backtest import cross_section_ic, run_rank_portfolio
from .transition_overlay import build_causal_transition_overlay, transition_summary
from .veighna_adapter import VeighNaExecutionAdapter


class V77ResearchPipeline:
    """V7.7 overlay study built on frozen V7.6 stock alpha and non-cycle overlays."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.v77 = dict(config.get("v7_7", {}))

    def run(self, v76_signals: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
        output_dir.mkdir(parents=True, exist_ok=True)
        out = v76_signals.copy()
        out["datetime"] = pd.to_datetime(out["datetime"])

        transition = build_causal_transition_overlay(
            out,
            target_col="future_return_5",
            cfg=dict(self.v77.get("transition", {})),
        )
        out = out.merge(transition, on="datetime", how="left")
        out["transition_cycle_multiplier"] = pd.to_numeric(
            out["transition_cycle_multiplier"], errors="coerce"
        ).fillna(float(self.v77.get("transition", {}).get("base_exposure", 0.78)))

        # Preserve every V7.6 stock-selection/risk component. V7.7 changes only the cycle overlay.
        non_cycle = pd.to_numeric(out["overlay_no_cycle"], errors="coerce").fillna(1.0)
        out["v7_7_dynamic_position_multiplier"] = (
            non_cycle * out["transition_cycle_multiplier"]
        ).clip(0.20, 1.00)

        bt_cfg = {
            **dict(self.config.get("backtest", {})),
            **dict(self.config.get("v7", {}).get("backtest", {})),
        }
        bt_cfg["round_trip_turnover_cost"] = float(
            self.config.get("v7", {}).get("backtest", {}).get("round_trip_turnover_cost", 0.0017)
        )

        variants = {
            "v7_7_transition_full": ("v7_alpha", True, "v7_7_dynamic_position_multiplier"),
            "v7_6_full": ("v7_alpha", True, "dynamic_position_multiplier"),
            "v7_no_cycle_overlay": ("v7_alpha", True, "overlay_no_cycle"),
            "v6_score_baseline": ("alpha_v6_baseline", False, ""),
        }
        periods = {
            "full": out,
            "development_2020_2024": out.loc[out["datetime"] < pd.Timestamp("2025-01-01")],
            "holdout_2025": out.loc[out["datetime"] >= pd.Timestamp("2025-01-01")],
        }

        rows: list[dict[str, Any]] = []
        full_backtests: dict[str, pd.DataFrame] = {}
        metrics: dict[str, Any] = {}
        for period_name, period_frame in periods.items():
            for variant, (score_col, use_overlay, overlay_col) in variants.items():
                local_cfg = dict(bt_cfg)
                local_cfg["use_overlay"] = use_overlay
                if overlay_col:
                    local_cfg["overlay_col"] = overlay_col
                daily, perf = run_rank_portfolio(period_frame, score_col, local_cfg)
                ic5 = cross_section_ic(period_frame, score_col, "future_return_5")
                record = {
                    "period": period_name,
                    "variant": variant,
                    **perf,
                    **{f"ic5_{k}": v for k, v in ic5.items()},
                }
                rows.append(record)
                metrics[f"{period_name}:{variant}"] = record
                if period_name == "full":
                    full_backtests[variant] = daily

        ablation = pd.DataFrame(rows)
        trans_summary = transition_summary(out, "future_return_5", "2025-01-01")

        out.to_csv(output_dir / "v7_7_signals.csv", index=False, encoding="utf-8-sig")
        transition.to_csv(output_dir / "transition_overlay.csv", index=False, encoding="utf-8-sig")
        trans_summary.to_csv(output_dir / "transition_summary_development.csv", index=False, encoding="utf-8-sig")
        ablation.to_csv(output_dir / "ablation_v7_7.csv", index=False, encoding="utf-8-sig")
        for name, daily in full_backtests.items():
            daily.to_csv(output_dir / f"backtest_{name}.csv", index=False, encoding="utf-8-sig")

        # Reuse the existing execution adapter with the V7.7 multiplier exposed under the name it expects.
        exec_frame = out.copy()
        exec_frame["dynamic_position_multiplier"] = exec_frame["v7_7_dynamic_position_multiplier"]
        intents = VeighNaExecutionAdapter(self.config.get("v7", {}).get("veighna", {})).build_order_intents(exec_frame)
        intents.to_csv(output_dir / "veighna_order_intents_v7_7.csv", index=False, encoding="utf-8-sig")

        summary = {
            "version": "7.7",
            "rows": int(len(out)),
            "dates": int(out["datetime"].nunique()),
            "symbols": int(out["symbol"].nunique()),
            "transition_config": dict(self.v77.get("transition", {})),
            "metrics": metrics,
        }
        (output_dir / "v7_7_metrics.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._write_report(summary, ablation, trans_summary, output_dir)
        return out, summary

    @staticmethod
    def _write_report(
        summary: dict[str, Any],
        ablation: pd.DataFrame,
        trans_summary: pd.DataFrame,
        output_dir: Path,
    ) -> None:
        def pick(period: str, variant: str) -> pd.Series:
            return ablation.loc[
                (ablation["period"] == period) & (ablation["variant"] == variant)
            ].iloc[0]

        dev77 = pick("development_2020_2024", "v7_7_transition_full")
        dev76 = pick("development_2020_2024", "v7_6_full")
        hold77 = pick("holdout_2025", "v7_7_transition_full")
        hold76 = pick("holdout_2025", "v7_6_full")
        full77 = pick("full", "v7_7_transition_full")
        full76 = pick("full", "v7_6_full")

        stable = trans_summary.loc[trans_summary["dates"] >= 10].copy()
        top_pos = stable.nlargest(6, "mean_forward_5d")
        top_neg = stable.nsmallest(6, "mean_forward_5d")
        lines = [
            "# V7.7 情绪周期状态转换动态仓位研究报告",
            "",
            "## 设计纪律",
            "",
            "- V7.6 的个股 alpha、Sector、Momentum、Event 与非周期风控全部冻结。",
            "- V7.7 只替换 Cycle overlay，避免同时改多个模块导致归因失真。",
            "- Transition edge 每个测试年仅使用更早年份训练，并 purge 年末 7 个交易日。",
            "- 2025 不参与参数选择；V7.7 参数在 2020-2024 开发期冻结后再一次性查看 holdout。",
            "- 连续 cycle delta/slope 已输出为诊断列，但因 V7.6 样本中其单变量关系弱且非单调，本版本不强行加入仓位公式。",
            "",
            "## 核心结果",
            "",
            f"- Full V7.7 annual: {float(full77['annual_return']):.2%}, Sharpe: {float(full77['sharpe']):.3f}, MaxDD: {float(full77['max_drawdown']):.2%}",
            f"- Full V7.6 annual: {float(full76['annual_return']):.2%}, Sharpe: {float(full76['sharpe']):.3f}, MaxDD: {float(full76['max_drawdown']):.2%}",
            f"- Development V7.7 annual: {float(dev77['annual_return']):.2%}, Sharpe: {float(dev77['sharpe']):.3f}, MaxDD: {float(dev77['max_drawdown']):.2%}",
            f"- Development V7.6 annual: {float(dev76['annual_return']):.2%}, Sharpe: {float(dev76['sharpe']):.3f}, MaxDD: {float(dev76['max_drawdown']):.2%}",
            f"- 2025 holdout V7.7 annual: {float(hold77['annual_return']):.2%}, Sharpe: {float(hold77['sharpe']):.3f}, MaxDD: {float(hold77['max_drawdown']):.2%}",
            f"- 2025 holdout V7.6 annual: {float(hold76['annual_return']):.2%}, Sharpe: {float(hold76['sharpe']):.3f}, MaxDD: {float(hold76['max_drawdown']):.2%}",
            "",
            "## Ablation",
            "",
            ablation.to_markdown(index=False),
            "",
            "## 开发期样本数 >=10 的高收益 Transition",
            "",
            top_pos.to_markdown(index=False),
            "",
            "## 开发期样本数 >=10 的低收益 Transition",
            "",
            top_neg.to_markdown(index=False),
        ]
        (output_dir / "V7_7_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
