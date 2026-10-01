from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.data.validator import validate_market_data
from src.features.engineer import add_features

ROOT = Path(__file__).resolve().parents[1]


def run(script: str) -> None:
    """Run an optional project script and fail with its complete terminal output."""
    result = subprocess.run(
        [sys.executable, str(ROOT / script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr


def test_pct_chg_protects_split_discontinuity() -> None:
    """Adjusted economic returns must not inherit raw-price split discontinuities."""
    dates = pd.bdate_range("2024-01-01", periods=180)
    economic = pd.Series(
        100.0 * (1.001 ** pd.RangeIndex(len(dates))),
        dtype=float,
    )

    raw_factor = pd.Series(1.0, index=range(len(dates)))
    raw_factor.iloc[90:] = 0.5

    close = economic * raw_factor.values
    returns = economic.pct_change().fillna(0.0)

    frame = pd.DataFrame(
        {
            "datetime": dates,
            "symbol": "000001.SZ",
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "pre_close": close / (1.0 + returns),
            "change": close - close / (1.0 + returns),
            "pct_chg": returns * 100.0,
            "volume": 1_000_000.0,
            "amount": close * 1_000_000.0,
        }
    )

    cleaned, _ = validate_market_data(frame)

    cfg = {
        "ma_short": 5,
        "ma_mid": 10,
        "ma_long": 20,
        "rsi_window": 14,
        "atr_window": 14,
        "volatility_window": 20,
        "volume_window": 20,
        "breakout_window": 20,
        "momentum_window": 10,
        "drawdown_window": 60,
        "zscore_window": 60,
        "zscore_min_periods": 20,
        "future_horizons": [1, 5],
    }

    features = add_features(
        cleaned,
        cfg,
        sentiment_pool=["000001.SZ"],
    )

    split_row = features.iloc[90]

    assert abs(float(split_row["raw_close_return_1"])) > 0.30
    assert abs(float(split_row["return_1"])) < 0.01
    assert int(split_row["possible_raw_price_gap"]) == 1


def test_optional_legacy_demo_research_pipeline() -> None:
    """Run the old demo workflow only when its complete legacy script set exists.

    V6 no longer requires run_backtest.py, run_evaluation.py or run_ablation.py.
    Therefore a cleaned V6 project should skip this optional compatibility test
    rather than fail because obsolete V1-V4 runner files were archived.
    """
    legacy_scripts = [
        "generate_demo_data.py",
        "run_pipeline.py",
        "run_backtest.py",
        "run_evaluation.py",
        "run_ablation.py",
    ]

    missing_scripts = [
        script
        for script in legacy_scripts
        if not (ROOT / script).is_file()
    ]

    if missing_scripts:
        pytest.skip(
            "Optional legacy demo workflow is not included in the V6 project. "
            f"Missing: {', '.join(missing_scripts)}"
        )

    for script in legacy_scripts:
        run(script)

    expected_outputs = [
        ROOT / "outputs" / "agent_decisions.csv",
        ROOT / "outputs" / "backtest" / "performance_metrics.json",
        ROOT / "outputs" / "evaluation" / "score_ic.csv",
        ROOT / "outputs" / "ablation" / "ablation_results.csv",
    ]

    for output_path in expected_outputs:
        assert output_path.exists(), f"Missing demo output: {output_path}"