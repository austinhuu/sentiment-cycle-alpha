from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.portfolio_engine_v5 import (
    PortfolioBacktestEngineV5,
    save_portfolio_backtest_v5,
)
from src.backtest.portfolio_engine_v4 import PortfolioBacktestResult


class PortfolioBacktestEngineV6(PortfolioBacktestEngineV5):
    """V5 backtest enriched with optimizer and risk-attribution diagnostics."""

    def run(
        self,
        decisions: pd.DataFrame,
        allocations: pd.DataFrame,
        optimizer_daily: pd.DataFrame | None = None,
    ) -> PortfolioBacktestResult:
        result = super().run(decisions, allocations)
        if optimizer_daily is not None and not optimizer_daily.empty:
            daily = optimizer_daily.copy()
            daily["datetime"] = pd.to_datetime(daily["datetime"], errors="coerce")
            columns = [
                "datetime",
                "status",
                "gross_exposure",
                "expected_daily_return",
                "ex_ante_volatility",
                "effective_positions",
                "turnover",
                "maximum_symbol_weight",
                "maximum_sector_weight",
                "covariance_condition_number",
                "average_correlation",
            ]
            available = [column for column in columns if column in daily]
            result.portfolio = result.portfolio.merge(
                daily[available], on="datetime", how="left"
            )
            numeric = daily.select_dtypes(include=[np.number])
            if "ex_ante_volatility" in numeric:
                result.metrics["average_optimizer_ex_ante_volatility"] = float(
                    numeric["ex_ante_volatility"].mean()
                )
                result.metrics["maximum_optimizer_ex_ante_volatility"] = float(
                    numeric["ex_ante_volatility"].max()
                )
            if "effective_positions" in numeric:
                result.metrics["average_optimizer_effective_positions"] = float(
                    numeric["effective_positions"].mean()
                )
            if "turnover" in numeric:
                result.metrics["average_optimizer_turnover"] = float(
                    numeric["turnover"].mean()
                )
                result.metrics["maximum_optimizer_turnover"] = float(
                    numeric["turnover"].max()
                )
            if "status" in daily:
                status = daily["status"].astype(str)
                optimized = status.isin(["converged", "max_iterations"])
                failed = status.isin(["failed", "no_candidates"])
                result.metrics["optimizer_success_share"] = float(
                    optimized.sum() / max(int(optimized.sum() + failed.sum()), 1)
                )
        return result


def save_portfolio_backtest_v6(
    result: PortfolioBacktestResult,
    output_dir: Path,
) -> None:
    save_portfolio_backtest_v5(result, output_dir)
    payload: dict[str, Any] = {
        key: float(value) if isinstance(value, (np.floating,)) else int(value) if isinstance(value, (np.integer,)) else value
        for key, value in result.metrics.items()
        if isinstance(value, (int, float, np.integer, np.floating))
    }
    (output_dir / "v6_optimizer_backtest_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )