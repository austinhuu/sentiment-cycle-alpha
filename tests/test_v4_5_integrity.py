from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.portfolio_engine_v4 import PortfolioBacktestEngineV4
from src.data.akshare_fundamental_downloader import _resolve_availability_collisions
from src.data.alternative_data import _merge_fundamental
from src.models.adaptive_fusion import _purge_training_dates
from src.portfolio.cross_sectional_allocator import CrossSectionalAllocator


def test_same_day_annual_and_q1_are_preserved_as_package() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ"],
            "report_period": ["2020-12-31", "2021-03-31"],
            "available_date": ["2021-05-01", "2021-05-01"],
            "fundamental_score": [0.4, 0.2],
            "fundamental_confidence": [0.8, 0.7],
        }
    )
    result = _resolve_availability_collisions(frame)
    assert len(result) == 2
    assert set(result["report_type"]) == {"annual", "q1"}
    assert result["available_date"].nunique() == 1
    assert not result.duplicated(["symbol", "available_date", "report_type"]).any()
    assert int(result["availability_date_adjusted"].sum()) == 0


def test_annual_and_quarterly_channels_merge_independently(tmp_path) -> None:
    fundamentals = pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ"],
            "report_period": ["2020-12-31", "2021-03-31"],
            "available_date": ["2021-05-01", "2021-05-01"],
            "report_type": ["annual", "q1"],
            "fundamental_score": [0.5, 0.25],
            "fundamental_confidence": [0.9, 0.8],
            "fundamental_growth_acceleration": [0.1, 0.2],
            "fundamental_quality_change": [0.3, 0.1],
            "fundamental_cash_quality": [0.4, 0.2],
        }
    )
    path = tmp_path / "fundamental_scores.csv"
    fundamentals.to_csv(path, index=False)
    market = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2021-05-06"]),
            "symbol": ["000001.SZ"],
        }
    )
    config = {
        "fundamental_enabled": True,
        "fundamental_csv": str(path),
        "fundamental_annual_max_age_days": 550,
        "fundamental_quarterly_max_age_days": 220,
        "fundamental_annual_weight": 0.45,
        "fundamental_quarterly_weight": 0.55,
        "fundamental_bundle_min_confidence": 0.2,
    }
    result = _merge_fundamental(market, tmp_path, config)
    assert float(result.loc[0, "annual_fundamental_active"]) == 1.0
    assert float(result.loc[0, "quarterly_fundamental_active"]) == 1.0
    assert str(result.loc[0, "annual_report_period"]).startswith("2020-12-31")
    assert str(result.loc[0, "quarterly_report_period"]).startswith("2021-03-31")
    assert float(result.loc[0, "fundamental_active"]) == 1.0


def test_walk_forward_purges_last_six_dates() -> None:
    dates = pd.bdate_range("2022-12-01", periods=20)
    frame = pd.DataFrame(
        {
            "datetime": dates.repeat(2),
            "symbol": ["A", "B"] * len(dates),
        }
    )
    result, removed = _purge_training_dates(frame, purge_days=5, embargo_days=1)
    assert len(removed) == 6
    assert result["datetime"].nunique() == 14
    assert pd.Timestamp(result["datetime"].max()) < pd.Timestamp(removed[0])


def test_drawdown_multiplier_is_monotonic() -> None:
    config = {
        "enabled": True,
        "warning_drawdown": -0.08,
        "warning_multiplier": 0.85,
        "defensive_drawdown": -0.12,
        "defensive_multiplier": 0.65,
        "severe_drawdown": -0.18,
        "severe_multiplier": 0.40,
    }
    assert PortfolioBacktestEngineV4._desired_drawdown_multiplier(-0.02, config) == 1.0
    assert PortfolioBacktestEngineV4._desired_drawdown_multiplier(-0.09, config) == 0.85
    assert PortfolioBacktestEngineV4._desired_drawdown_multiplier(-0.13, config) == 0.65
    assert PortfolioBacktestEngineV4._desired_drawdown_multiplier(-0.20, config) == 0.40


def test_final_turnover_projection_respects_budget(tmp_path) -> None:
    allocator = CrossSectionalAllocator(
        {
            "security_master_csv": "missing.csv",
            "max_active_positions": 3,
            "max_turnover_per_rebalance": 0.5,
            "turnover_tolerance": 1e-10,
        },
        tmp_path,
    )
    index = pd.Index(["A", "B", "C", "D"])
    previous = pd.Series([0.20, 0.20, 0.20, 0.0], index=index)
    desired = pd.Series([0.0, 0.0, 0.0, 0.60], index=index)
    proposed = desired.copy()
    enriched = pd.DataFrame(
        {
            "symbol": index,
            "adaptive_score": [0.1, 0.2, 0.3, 1.0],
        }
    )
    result = allocator._hard_turnover_projection(
        previous,
        desired,
        proposed,
        enriched,
        maximum_turnover=0.5,
    )
    assert float((result - previous).abs().sum()) <= 0.5 + 1e-9
    assert int(result.gt(0).sum()) <= 3
    assert np.isfinite(result.to_numpy()).all()