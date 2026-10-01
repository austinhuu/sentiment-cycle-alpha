from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.portfolio_engine_v4 import (
    PortfolioBacktestEngineV4,
    PortfolioBacktestResult,
    save_portfolio_backtest,
)


class PortfolioBacktestEngineV5(PortfolioBacktestEngineV4):
    """V4 execution engine plus V5 production-risk and health diagnostics."""

    def run(
        self,
        decisions: pd.DataFrame,
        allocations: pd.DataFrame,
    ) -> PortfolioBacktestResult:
        result = super().run(decisions, allocations)
        allocation_daily = allocations.copy()
        allocation_daily["datetime"] = pd.to_datetime(
            allocation_daily["datetime"], errors="coerce"
        )
        aggregate: dict[str, Any] = {}
        if "dynamic_risk_budget_multiplier" in allocation_daily:
            aggregate["dynamic_risk_budget_multiplier"] = (
                "dynamic_risk_budget_multiplier",
                "median",
            )
        if "dynamic_risk_budget_raw" in allocation_daily:
            aggregate["dynamic_risk_budget_raw"] = (
                "dynamic_risk_budget_raw",
                "median",
            )
        if "base_target_weight" in allocation_daily:
            aggregate["base_gross_exposure"] = (
                "base_target_weight",
                "sum",
            )
        if "risk_budget_portfolio_turnover" in allocation_daily:
            aggregate["risk_budget_portfolio_turnover"] = (
                "risk_budget_portfolio_turnover",
                "max",
            )
        if aggregate:
            daily_extra = allocation_daily.groupby("datetime", as_index=False).agg(**aggregate)
            result.portfolio = result.portfolio.merge(
                daily_extra,
                on="datetime",
                how="left",
            )
        health_daily = decisions.copy()
        health_daily["datetime"] = pd.to_datetime(health_daily["datetime"], errors="coerce")
        if "model_health_state" in health_daily:
            state = (
                health_daily.groupby("datetime")["model_health_state"]
                .agg(lambda values: values.mode().iloc[0] if not values.mode().empty else "unknown")
                .rename("model_health_state")
            )
            result.portfolio = result.portfolio.merge(
                state,
                left_on="datetime",
                right_index=True,
                how="left",
            )
        if "model_health_score" in health_daily:
            score = health_daily.groupby("datetime")["model_health_score"].median().rename(
                "model_health_score"
            )
            result.portfolio = result.portfolio.merge(
                score,
                left_on="datetime",
                right_index=True,
                how="left",
            )

        returns = pd.to_numeric(result.portfolio["strategy_return"], errors="coerce").fillna(0.0)
        downside = returns.clip(upper=0.0)
        result.metrics.update(
            {
                "annualized_downside_deviation": float(
                    downside.std(ddof=0) * np.sqrt(int(self.config.get("annualization_bars", 252)))
                ),
                "expected_shortfall_95": float(
                    returns.loc[returns <= returns.quantile(0.05)].mean()
                )
                if len(returns)
                else 0.0,
                "positive_day_ratio": float((returns > 0).mean()) if len(returns) else 0.0,
                "worst_day_return": float(returns.min()) if len(returns) else 0.0,
                "best_day_return": float(returns.max()) if len(returns) else 0.0,
            }
        )
        if "dynamic_risk_budget_multiplier" in result.portfolio:
            multiplier = pd.to_numeric(
                result.portfolio["dynamic_risk_budget_multiplier"], errors="coerce"
            ).fillna(1.0)
            result.metrics.update(
                {
                    "average_dynamic_risk_budget_multiplier": float(multiplier.mean()),
                    "minimum_dynamic_risk_budget_multiplier": float(multiplier.min()),
                    "low_dynamic_risk_budget_days": int(multiplier.lt(0.65).sum()),
                }
            )
        if "model_health_state" in result.portfolio:
            counts = result.portfolio["model_health_state"].fillna("unknown").value_counts()
            for state_name in ["healthy", "cautious", "fallback", "defensive"]:
                result.metrics[f"model_health_{state_name}_days"] = int(
                    counts.get(state_name, 0)
                )
                result.metrics[f"model_health_{state_name}_share"] = float(
                    counts.get(state_name, 0) / max(len(result.portfolio), 1)
                )
        gross_profit = float(
            pd.to_numeric(result.portfolio.get("gross_return"), errors="coerce")
            .clip(lower=0.0)
            .sum()
        )
        total_cost = float(
            pd.to_numeric(result.portfolio.get("transaction_cost"), errors="coerce")
            .fillna(0.0)
            .sum()
        )
        result.metrics["transaction_cost_to_positive_gross_return"] = (
            total_cost / gross_profit if gross_profit > 1e-12 else 0.0
        )
        return result


def save_portfolio_backtest_v5(
    result: PortfolioBacktestResult,
    output_dir: Path,
) -> None:
    save_portfolio_backtest(result, output_dir)
    summary = {
        key: value
        for key, value in result.metrics.items()
        if isinstance(value, (int, float, np.integer, np.floating))
    }
    (output_dir / "v5_production_risk_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )