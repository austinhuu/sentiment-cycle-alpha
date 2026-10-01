from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtest.fast_portfolio_engine import FastPortfolioBacktestEngine
from src.portfolio.cross_sectional_allocator import CrossSectionalAllocator
from src.risk.dynamic_risk_budget import DynamicRiskBudgetOverlay


def run_production_scenario(
    decisions: pd.DataFrame,
    portfolio_config: dict[str, Any],
    backtest_config: dict[str, Any],
    risk_budget_config: dict[str, Any],
    project_root: Path,
    return_frames: bool = False,
) -> dict[str, Any]:
    allocator = CrossSectionalAllocator(deepcopy(portfolio_config), project_root)
    base_allocations, allocation_diagnostics = allocator.allocate(decisions)
    risk_overlay = DynamicRiskBudgetOverlay(deepcopy(risk_budget_config))
    adjusted_allocations, risk_diagnostics, _ = risk_overlay.apply(
        decisions,
        base_allocations,
    )
    engine = FastPortfolioBacktestEngine(
        {**deepcopy(backtest_config), **deepcopy(portfolio_config)}
    )
    result = engine.run_fast(decisions, adjusted_allocations)
    output = {
        "metrics": result.metrics,
        "exposure_matched_metrics": result.exposure_matched_metrics,
        "equal_weight_metrics": result.equal_weight_metrics,
        "allocation_diagnostics": allocation_diagnostics.to_dict(),
        "risk_diagnostics": risk_diagnostics.to_dict(),
    }
    if return_frames:
        output["portfolio"] = result.portfolio
    return output