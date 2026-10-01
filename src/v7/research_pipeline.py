from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtest import cross_section_ic, run_rank_portfolio
from .cycle import build_sentiment_cycle
from .event_structurer import build_structured_event_score
from .momentum import build_momentum_breakout
from .fusion import causal_cycle_edge, walk_forward_ridge_fusion
from .sector_linkage import build_sector_stock_linkage
from .veighna_adapter import VeighNaExecutionAdapter


class V76ResearchPipeline:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.v7 = dict(config.get("v7", {}))

    def run(self, features: pd.DataFrame, decisions: pd.DataFrame, security_master: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
        output_dir.mkdir(parents=True, exist_ok=True)
        base_cols = ["datetime", "symbol", "future_return_1", "future_return_5", "future_return_20", "final_score", "confidence", "risk_score", "effective_risk_multiplier"]
        base = decisions[[c for c in base_cols if c in decisions.columns]].copy()
        base["datetime"] = pd.to_datetime(base["datetime"])

        feature_cols = ["datetime", "symbol", "return_1", "return_5", "volume_ratio", "momentum_z", "volume_ratio_z", "ma_gap_short_long_z", "price_position", "breakout_up"]
        feat = features[[c for c in feature_cols if c in features.columns]].copy()
        feat["datetime"] = pd.to_datetime(feat["datetime"])

        cycle = build_sentiment_cycle(features, self.v7.get("cycle", {}))
        sector = build_sector_stock_linkage(features, security_master, self.v7.get("sector_linkage", {}))
        event = build_structured_event_score(decisions)
        momentum = build_momentum_breakout(features)

        out = base.merge(feat, on=["datetime", "symbol"], how="left")
        out = out.merge(cycle, on="datetime", how="left")
        out = out.merge(sector, on=["datetime", "symbol"], how="left")
        out = out.merge(event, on=["datetime", "symbol"], how="left")
        out = out.merge(momentum, on=["datetime", "symbol"], how="left")

        final = pd.to_numeric(out["final_score"], errors="coerce").fillna(0.0)
        raw_link = pd.to_numeric(out["sector_stock_link_score"], errors="coerce").fillna(0.0)
        sector_strength = pd.to_numeric(out["sector_strength_score"], errors="coerce").fillna(0.0)
        supported_sector = out["sector_source"].isin(["direct_sector", "theme_proxy"])
        # Unsupported single-name sectors stay neutral instead of using the whole market as a fake sector.
        out["sector_context_score"] = sector_strength.where(supported_sector, 0.0)
        out["sector_link_diagnostic"] = raw_link.where(supported_sector, 0.0)

        cycle_edge = causal_cycle_edge(out, "future_return_5")
        out = out.merge(cycle_edge, on="datetime", how="left")
        out["cycle_edge_score"] = pd.to_numeric(out["cycle_edge_score"], errors="coerce").fillna(0.0)
        out["learned_cycle_multiplier"] = pd.to_numeric(out["learned_cycle_multiplier"], errors="coerce").fillna(0.70)

        evt = pd.to_numeric(out["structured_event_score"], errors="coerce").fillna(0.0)
        # Development-period model choice: keep the proven V6 alpha dominant and use supported
        # sector strength as a small cross-sectional refinement. Cycle/news do not rank stocks.
        out["v7_alpha"] = (0.80 * final + 0.20 * out["sector_context_score"]).clip(-1, 1)
        out["alpha_v6_baseline"] = final

        risk_mult = pd.to_numeric(out.get("effective_risk_multiplier", 1.0), errors="coerce").fillna(1.0).clip(0.5, 1.0)
        sector_risk_mult = (0.92 + 0.08 * out["sector_context_score"]).clip(0.85, 1.0)
        # Event text is sparse in the historical package: negative events can de-risk, positive
        # events never lever the portfolio up.
        event_risk_mult = (1.0 - 0.15 * (-evt).clip(lower=0.0, upper=1.0)).clip(0.85, 1.0)
        momentum_confirm_mult = np.where(
            pd.to_numeric(out["momentum_breakout_confirmed"], errors="coerce").fillna(0).gt(0),
            1.0, 0.92,
        )
        out["overlay_no_cycle"] = (sector_risk_mult * risk_mult * event_risk_mult * momentum_confirm_mult).clip(0.20, 1.0)
        out["overlay_no_momentum"] = (out["learned_cycle_multiplier"] * sector_risk_mult * risk_mult * event_risk_mult).clip(0.20, 1.0)
        out["overlay_no_event"] = (out["learned_cycle_multiplier"] * sector_risk_mult * risk_mult * momentum_confirm_mult).clip(0.20, 1.0)
        out["dynamic_position_multiplier"] = (
            out["learned_cycle_multiplier"] * sector_risk_mult * risk_mult * event_risk_mult * momentum_confirm_mult
        ).clip(0.20, 1.0)

        bt_cfg = {**dict(self.config.get("backtest", {})), **dict(self.v7.get("backtest", {}))}
        bt_cfg["round_trip_turnover_cost"] = float(self.v7.get("backtest", {}).get("round_trip_turnover_cost", 0.0017))
        variants = {
            "v7_full": ("v7_alpha", True, "dynamic_position_multiplier"),
            "v6_score_baseline": ("alpha_v6_baseline", False, ""),
            "v7_no_sector": ("alpha_v6_baseline", True, "dynamic_position_multiplier"),
            "v7_no_cycle_overlay": ("v7_alpha", True, "overlay_no_cycle"),
            "v7_no_momentum_confirmation": ("v7_alpha", True, "overlay_no_momentum"),
            "v7_no_event_risk": ("v7_alpha", True, "overlay_no_event"),
        }
        periods = {
            "full": out,
            "development_2020_2024": out.loc[out["datetime"] < pd.Timestamp("2025-01-01")],
            "holdout_2025": out.loc[out["datetime"] >= pd.Timestamp("2025-01-01")],
        }
        ablation_rows = []
        backtests: dict[str, pd.DataFrame] = {}
        metrics: dict[str, Any] = {}
        for period_name, period_frame in periods.items():
            for name, (score_col, use_overlay, overlay_col) in variants.items():
                local_cfg = dict(bt_cfg)
                local_cfg["use_overlay"] = use_overlay
                if overlay_col:
                    local_cfg["overlay_col"] = overlay_col
                daily, perf = run_rank_portfolio(period_frame, score_col, local_cfg)
                ic5 = cross_section_ic(period_frame, score_col, "future_return_5")
                record = {"period": period_name, "variant": name, **perf, **{f"ic5_{k}": v for k, v in ic5.items()}}
                ablation_rows.append(record)
                metrics[f"{period_name}:{name}"] = record
                if period_name == "full":
                    backtests[name] = daily

        ablation = pd.DataFrame(ablation_rows)
        cycle_summary = out.groupby("cycle_phase", dropna=False).agg(
            observations=("symbol", "size"),
            dates=("datetime", "nunique"),
            mean_cycle_score=("cycle_score", "mean"),
            mean_v7_alpha=("v7_alpha", "mean"),
            mean_future_return_1=("future_return_1", "mean"),
            mean_future_return_5=("future_return_5", "mean"),
        ).reset_index()

        focus = str(self.v7.get("focus_symbol", "688981.SH")).upper()
        case_cols = ["datetime", "symbol", "sector", "sector_source", "sector_peer_count", "cycle_phase", "cycle_score", "sector_strength_score", "sector_stock_link_score", "momentum_breakout_score", "momentum_breakout_confirmed", "structured_event_score", "v7_alpha", "dynamic_position_multiplier", "future_return_1", "future_return_5", "future_return_20"]
        case = out.loc[out["symbol"].astype(str).str.upper() == focus, [c for c in case_cols if c in out.columns]].copy()

        out.to_csv(output_dir / "v7_signals.csv", index=False, encoding="utf-8-sig")
        cycle.to_csv(output_dir / "sentiment_cycle.csv", index=False, encoding="utf-8-sig")
        sector.to_csv(output_dir / "sector_stock_linkage.csv", index=False, encoding="utf-8-sig")
        ablation.to_csv(output_dir / "ablation.csv", index=False, encoding="utf-8-sig")
        cycle_summary.to_csv(output_dir / "cycle_summary.csv", index=False, encoding="utf-8-sig")
        case.to_csv(output_dir / f"case_study_{focus.replace('.', '_')}.csv", index=False, encoding="utf-8-sig")
        for name, frame in backtests.items():
            frame.to_csv(output_dir / f"backtest_{name}.csv", index=False, encoding="utf-8-sig")

        intents = VeighNaExecutionAdapter(self.v7.get("veighna", {})).build_order_intents(out)
        intents.to_csv(output_dir / "veighna_order_intents.csv", index=False, encoding="utf-8-sig")

        summary = {
            "version": "7.6",
            "rows": int(len(out)),
            "dates": int(out["datetime"].nunique()),
            "symbols": int(out["symbol"].nunique()),
            "focus_symbol": focus,
            "focus_rows": int(len(case)),
            "theme_proxy_rows": int((out.get("sector_source", "") == "theme_proxy").sum()) if "sector_source" in out.columns else 0,
            "metrics": metrics,
        }
        (output_dir / "v7_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        self._write_report(summary, ablation, cycle_summary, output_dir)
        return out, summary

    @staticmethod
    def _write_report(summary: dict[str, Any], ablation: pd.DataFrame, cycle_summary: pd.DataFrame, output_dir: Path) -> None:
        full = ablation.loc[(ablation["period"] == "full") & (ablation["variant"] == "v7_full")].iloc[0]
        base = ablation.loc[(ablation["period"] == "full") & (ablation["variant"] == "v6_score_baseline")].iloc[0]
        hold = ablation.loc[(ablation["period"] == "holdout_2025") & (ablation["variant"] == "v7_full")].iloc[0]
        hold_base = ablation.loc[(ablation["period"] == "holdout_2025") & (ablation["variant"] == "v6_score_baseline")].iloc[0]
        lines = [
            "# V7.6 A股情绪周期与板块-个股联动研究报告",
            "",
            "## 运行状态",
            "",
            f"- Rows: {summary['rows']:,}",
            f"- Dates: {summary['dates']:,}",
            f"- Symbols: {summary['symbols']}",
            f"- Focus: {summary['focus_symbol']}",
            "- 当前情绪周期与涨跌停强度来自32股研究池代理，不等同于全A官方市场统计。",
            "- 中芯国际直接半导体同业不足时使用配置中的科技主题代理篮子，并在 sector_source 中显式标记。",
            "",
            "## 首次跑通结果（仅诊断，不做事后调参）",
            "",
            f"- Full-period V7.6 annual return: {float(full['annual_return']):.2%}",
            f"- Full-period V7.6 Sharpe: {float(full['sharpe']):.3f}",
            f"- Full-period V7.6 max drawdown: {float(full['max_drawdown']):.2%}",
            f"- Full-period V7.6 5D mean IC: {float(full['ic5_mean_ic']):.4f}",
            f"- Full-period V6-score baseline annual return: {float(base['annual_return']):.2%}",
            f"- Full-period V6-score baseline Sharpe: {float(base['sharpe']):.3f}",
            f"- 2025 holdout V7.6 annual return: {float(hold['annual_return']):.2%}",
            f"- 2025 holdout V7.6 Sharpe: {float(hold['sharpe']):.3f}",
            f"- 2025 holdout V7.6 max drawdown: {float(hold['max_drawdown']):.2%}",
            f"- 2025 holdout V6-score baseline annual return: {float(hold_base['annual_return']):.2%}",
            f"- 2025 holdout V6-score baseline Sharpe: {float(hold_base['sharpe']):.3f}",
            "",
            "## Ablation",
            "",
            ablation.to_markdown(index=False),
            "",
            "## 情绪周期分布",
            "",
            cycle_summary.to_markdown(index=False),
            "",
            "下一轮优化应根据 ablation、IC、换手、回撤和中芯国际 case study 的真实输出决定，不在本次首次跑通中事后调参。",
        ]
        (output_dir / "V7_6_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
