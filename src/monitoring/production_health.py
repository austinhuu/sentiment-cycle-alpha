from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STATE_ORDER = ["healthy", "cautious", "fallback", "defensive"]


def build_production_health_report(
    decisions: pd.DataFrame,
    portfolio: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    decision_frame = decisions.copy()
    decision_frame["datetime"] = pd.to_datetime(decision_frame["datetime"], errors="coerce")
    daily_columns = [
        "datetime",
        "model_health_state",
        "model_health_score",
        "rolling_adaptive_ic",
        "rolling_fallback_ic",
        "rolling_relative_ic",
        "fold_validation_ic",
    ]
    daily = decision_frame[[column for column in daily_columns if column in decision_frame]].drop_duplicates("datetime")
    portfolio_frame = portfolio.copy()
    portfolio_frame["datetime"] = pd.to_datetime(portfolio_frame["datetime"], errors="coerce")
    merge_columns = [
        "datetime",
        "strategy_return",
        "drawdown",
        "dynamic_risk_budget_multiplier",
        "gross_exposure",
        "turnover",
        "transaction_cost",
    ]
    daily = daily.merge(
        portfolio_frame[[column for column in merge_columns if column in portfolio_frame]],
        on="datetime",
        how="outer",
    ).sort_values("datetime").reset_index(drop=True)
    recent_window = max(int(config.get("recent_window_dates", 120)), 1)
    recent = daily.tail(recent_window)
    state_counts = recent.get("model_health_state", pd.Series(dtype=str)).value_counts()
    recent_count = max(len(recent), 1)
    fallback_share = float(state_counts.get("fallback", 0) / recent_count)
    defensive_share = float(state_counts.get("defensive", 0) / recent_count)
    recent_ic = float(pd.to_numeric(recent.get("rolling_adaptive_ic"), errors="coerce").dropna().tail(1).iloc[0]) if "rolling_adaptive_ic" in recent and recent["rolling_adaptive_ic"].notna().any() else np.nan
    recent_drawdown = float(pd.to_numeric(recent.get("drawdown"), errors="coerce").dropna().tail(1).iloc[0]) if "drawdown" in recent and recent["drawdown"].notna().any() else np.nan
    recent_risk_multiplier = float(pd.to_numeric(recent.get("dynamic_risk_budget_multiplier"), errors="coerce").dropna().mean()) if "dynamic_risk_budget_multiplier" in recent and recent["dynamic_risk_budget_multiplier"].notna().any() else np.nan

    alerts: list[dict[str, Any]] = []

    def add(level: str, metric: str, value: float, threshold: float, message: str) -> None:
        alerts.append(
            {
                "level": level,
                "metric": metric,
                "value": float(value) if np.isfinite(value) else None,
                "threshold": float(threshold),
                "message": message,
            }
        )

    if fallback_share >= float(config.get("critical_fallback_share", 0.50)):
        add("critical", "fallback_share", fallback_share, float(config.get("critical_fallback_share", 0.50)), "近期多数交易日已进入模型降级状态。")
    elif fallback_share >= float(config.get("warning_fallback_share", 0.25)):
        add("warning", "fallback_share", fallback_share, float(config.get("warning_fallback_share", 0.25)), "近期模型降级占比偏高。")
    if defensive_share >= float(config.get("critical_defensive_share", 0.25)):
        add("critical", "defensive_share", defensive_share, float(config.get("critical_defensive_share", 0.25)), "近期防御状态占比过高。")
    elif defensive_share >= float(config.get("warning_defensive_share", 0.10)):
        add("warning", "defensive_share", defensive_share, float(config.get("warning_defensive_share", 0.10)), "近期防御状态占比偏高。")
    if np.isfinite(recent_ic):
        if recent_ic <= float(config.get("critical_recent_ic", -0.020)):
            add("critical", "recent_adaptive_ic", recent_ic, float(config.get("critical_recent_ic", -0.020)), "近期自适应信号IC显著为负。")
        elif recent_ic <= float(config.get("warning_recent_ic", -0.005)):
            add("warning", "recent_adaptive_ic", recent_ic, float(config.get("warning_recent_ic", -0.005)), "近期自适应信号IC偏弱。")
    if np.isfinite(recent_drawdown):
        if recent_drawdown <= float(config.get("critical_drawdown", -0.22)):
            add("critical", "current_drawdown", recent_drawdown, float(config.get("critical_drawdown", -0.22)), "当前回撤达到严重风险阈值。")
        elif recent_drawdown <= float(config.get("warning_drawdown", -0.15)):
            add("warning", "current_drawdown", recent_drawdown, float(config.get("warning_drawdown", -0.15)), "当前回撤达到警戒阈值。")
    if np.isfinite(recent_risk_multiplier):
        if recent_risk_multiplier <= float(config.get("critical_risk_multiplier", 0.45)):
            add("critical", "risk_budget_multiplier", recent_risk_multiplier, float(config.get("critical_risk_multiplier", 0.45)), "近期风险预算长期处于极低水平。")
        elif recent_risk_multiplier <= float(config.get("warning_risk_multiplier", 0.65)):
            add("warning", "risk_budget_multiplier", recent_risk_multiplier, float(config.get("warning_risk_multiplier", 0.65)), "近期风险预算偏低。")

    levels = [item["level"] for item in alerts]
    system_status = "critical" if "critical" in levels else "warning" if "warning" in levels else "healthy"
    summary = {
        "system_status": system_status,
        "dates": int(daily["datetime"].nunique()),
        "recent_window_dates": int(len(recent)),
        "recent_fallback_share": fallback_share,
        "recent_defensive_share": defensive_share,
        "recent_adaptive_ic": recent_ic,
        "current_drawdown": recent_drawdown,
        "recent_average_risk_budget_multiplier": recent_risk_multiplier,
        "alert_count": len(alerts),
        "warning_count": sum(item["level"] == "warning" for item in alerts),
        "critical_count": sum(item["level"] == "critical" for item in alerts),
    }
    return daily, summary, alerts


def save_production_health_outputs(
    daily: pd.DataFrame,
    summary: dict[str, Any],
    alerts: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(
        output_dir / "production_health_daily.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "production_health_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "production_health_alerts.json").write_text(
        json.dumps(alerts, ensure_ascii=False, indent=2), encoding="utf-8"
    )