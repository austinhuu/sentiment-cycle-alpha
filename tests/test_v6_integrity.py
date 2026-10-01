from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.execution.order_planner import OrderPlanner
from src.optimization.robust_optimizer import RobustPortfolioOptimizer
from src.production.artifact_registry import ArtifactRegistry


def _synthetic_decisions() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2022-01-03", periods=90)
    symbols = [f"00000{i}.SZ" for i in range(1, 7)]
    rows = []
    for symbol_index, symbol in enumerate(symbols):
        price = 10.0 + symbol_index
        for date_index, date in enumerate(dates):
            price *= 1.0 + rng.normal(0.0004 + symbol_index * 0.00005, 0.012)
            score = 0.15 + 0.03 * symbol_index + rng.normal(0, 0.02)
            rows.append(
                {
                    "datetime": date,
                    "symbol": symbol,
                    "signal_open": price,
                    "production_score": score,
                    "production_confidence": 0.75,
                    "production_predicted_excess_return": score * 0.03,
                    "model_health_state": "healthy",
                    "is_tradable_sample": 1,
                }
            )
    return pd.DataFrame(rows)


def _prior(decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, group in decisions.groupby("datetime"):
        selected = group.nlargest(4, "production_score")["symbol"].tolist()
        for symbol in group["symbol"]:
            rows.append(
                {
                    "datetime": date,
                    "symbol": symbol,
                    "target_weight": 0.10 if symbol in selected else 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_optimizer_respects_caps_and_turnover(tmp_path: Path) -> None:
    security = pd.DataFrame(
        {
            "symbol": [f"00000{i}.SZ" for i in range(1, 7)],
            "sector": ["A", "A", "B", "B", "C", "C"],
        }
    )
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    security.to_csv(raw / "security_master.csv", index=False)
    config = {
        "security_master_csv": "data/raw/security_master.csv",
        "candidate_top_k": 6,
        "max_active_positions": 5,
        "max_weight_per_symbol": 0.10,
        "max_weight_per_sector": 0.20,
        "max_gross_exposure": 0.50,
        "default_gross_exposure": 0.40,
        "covariance_lookback_dates": 40,
        "target_annual_volatility": 0.20,
        "rebalance_every_n_dates": 5,
        "max_turnover_per_rebalance": 0.30,
        "expected_return_daily_clip": 0.02,
        "risk_aversion": 6.0,
        "prior_anchor_penalty": 0.2,
        "concentration_penalty": 0.05,
        "learning_rate": 0.05,
        "max_iterations": 80,
    }
    decisions = _synthetic_decisions()
    optimizer = RobustPortfolioOptimizer(config, tmp_path)
    allocations, daily, _, summary = optimizer.optimize(decisions, _prior(decisions))
    assert allocations.groupby("datetime")["target_weight"].max().max() <= 0.1000001
    assert daily["turnover"].max() <= 0.3000001
    assert allocations.groupby("datetime")["target_weight"].apply(lambda x: (x > 0).sum()).max() <= 5
    assert summary.success_share >= 0.95


def test_future_prices_do_not_change_earlier_weights(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    pd.DataFrame(
        {
            "symbol": [f"00000{i}.SZ" for i in range(1, 7)],
            "sector": ["A", "A", "B", "B", "C", "C"],
        }
    ).to_csv(raw / "security_master.csv", index=False)
    config = {
        "security_master_csv": "data/raw/security_master.csv",
        "candidate_top_k": 6,
        "max_active_positions": 6,
        "max_weight_per_symbol": 0.15,
        "max_weight_per_sector": 0.30,
        "max_gross_exposure": 0.60,
        "covariance_lookback_dates": 40,
        "rebalance_every_n_dates": 5,
        "max_turnover_per_rebalance": 0.50,
        "max_iterations": 40,
    }
    decisions = _synthetic_decisions()
    cutoff = pd.Timestamp("2022-03-01")
    altered = decisions.copy()
    altered.loc[altered["datetime"].gt(cutoff), "signal_open"] *= 5.0
    optimizer = RobustPortfolioOptimizer(config, tmp_path)
    first, _, _, _ = optimizer.optimize(decisions, _prior(decisions))
    second, _, _, _ = optimizer.optimize(altered, _prior(altered))
    columns = ["datetime", "symbol", "target_weight"]
    first_early = first.loc[first["datetime"].le(cutoff), columns].reset_index(drop=True)
    second_early = second.loc[second["datetime"].le(cutoff), columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(first_early, second_early, atol=1e-10, rtol=1e-10)


def test_order_planner_rounds_lots_and_does_not_send_orders(tmp_path: Path) -> None:
    dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    decisions = pd.DataFrame(
        {
            "datetime": [dates[0], dates[0], dates[1], dates[1]],
            "symbol": ["000001.SZ", "000002.SZ"] * 2,
            "signal_open": [10.0, 20.0, 10.5, 20.5],
            "is_tradable_sample": [1, 1, 1, 1],
        }
    )
    allocations = pd.DataFrame(
        {
            "datetime": [dates[0], dates[0], dates[1], dates[1]],
            "symbol": ["000001.SZ", "000002.SZ"] * 2,
            "target_weight": [0.10, 0.10, 0.20, 0.0],
        }
    )
    planner = OrderPlanner(
        {
            "capital": 1_000_000,
            "lot_size": 100,
            "minimum_trade_value": 1_000,
            "derive_holdings_from_previous_target": True,
        },
        tmp_path,
    )
    orders, summary = planner.build(decisions, allocations)
    ready = orders.loc[orders["order_status"].eq("READY")]
    assert (ready["order_shares"] % 100 == 0).all()
    assert summary.buy_orders + summary.sell_orders >= 1
    assert set(orders["order_status"]).issubset({"READY", "BLOCKED", "NO_ORDER"})


def test_artifact_registry_hashes_files(tmp_path: Path) -> None:
    path = tmp_path / "example.txt"
    path.write_text("v6", encoding="utf-8")
    registry = ArtifactRegistry(tmp_path)
    manifest = registry.build_manifest(["example.txt"], metadata={"version": 6})
    assert manifest["all_required_exist"]
    assert len(manifest["artifacts"][0]["sha256"]) == 64
    output = tmp_path / "manifest.json"
    registry.save(manifest, output)
    assert json.loads(output.read_text(encoding="utf-8"))["metadata"]["version"] == 6