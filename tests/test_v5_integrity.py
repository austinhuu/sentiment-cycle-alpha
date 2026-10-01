from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.model_health_controller import ModelHealthController
from src.risk.dynamic_risk_budget import DynamicRiskBudgetOverlay
from src.robustness.bootstrap import run_paired_bootstrap


def _decision_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=80)
    rows = []
    symbols = ["000001.SZ", "000002.SZ", "600000.SH", "600036.SH"]
    for date_index, date in enumerate(dates):
        for symbol_index, symbol in enumerate(symbols):
            technical = 0.30 - 0.05 * symbol_index
            future = technical * 0.01 + 0.0005 * ((date_index + symbol_index) % 3 - 1)
            rows.append(
                {
                    "datetime": date,
                    "symbol": symbol,
                    "technical_score": technical,
                    "technical_confidence": 0.8,
                    "sentiment_score": 0.05,
                    "sentiment_confidence": 0.6,
                    "adaptive_score": technical,
                    "adaptive_confidence": 0.75,
                    "predicted_excess_return": technical * 0.02,
                    "future_return_5": future,
                    "fusion_validation_ic": 0.03,
                    "fusion_fallback_used": 0,
                    "market_return": 0.001 * np.sin(date_index / 5),
                    "market_stress_percentile": 0.3,
                    "market_regime": "neutral",
                    "risk_score": 0.2,
                    "volatility_z": 0.1,
                    "is_tradable_sample": 1,
                    "amount": 1_000_000 + symbol_index,
                    "open": 10.0,
                    "signal_open": 10.0,
                }
            )
    return pd.DataFrame(rows)


def test_model_health_controller_outputs_production_fields() -> None:
    controller = ModelHealthController(
        {
            "target_horizon": 5,
            "rolling_ic_dates": 20,
            "minimum_ic_observations": 10,
            "state_minimum_days": 5,
        }
    )
    output, diagnostics = controller.apply(_decision_frame())
    assert {
        "production_score",
        "production_confidence",
        "model_health_state",
        "rolling_adaptive_ic",
    }.issubset(output.columns)
    assert diagnostics.dates == 80
    assert output["production_confidence"].between(0, 1).all()


def test_dynamic_risk_budget_enforces_turnover() -> None:
    decisions = _decision_frame()
    decisions["model_health_state"] = "healthy"
    allocations = decisions[["datetime", "symbol"]].copy()
    allocations["target_weight"] = 0.20
    overlay = DynamicRiskBudgetOverlay(
        {
            "target_annual_volatility": 0.12,
            "volatility_window": 10,
            "minimum_volatility_observations": 5,
            "maximum_turnover": 0.50,
            "minimum_adaptive_score": 0.10,
        }
    )
    adjusted, diagnostics, daily = overlay.apply(decisions, allocations)
    assert diagnostics.maximum_turnover <= 0.500000001
    assert diagnostics.turnover_violation_dates == 0
    assert adjusted["target_weight"].ge(0).all()
    assert daily["dynamic_risk_budget_multiplier"].between(0.25, 1.0).all()


def test_bootstrap_is_reproducible() -> None:
    portfolio = pd.DataFrame(
        {
            "strategy_return": np.linspace(-0.01, 0.012, 100),
            "exposure_matched_return": np.linspace(-0.008, 0.009, 100),
        }
    )
    config = {
        "simulations": 20,
        "block_length": 10,
        "random_seed": 7,
        "method": "stationary",
    }
    first, first_summary = run_paired_bootstrap(portfolio, config)
    second, second_summary = run_paired_bootstrap(portfolio, config)
    pd.testing.assert_frame_equal(first, second)
    assert first_summary["probability_positive_total_return"] == second_summary[
        "probability_positive_total_return"
    ]