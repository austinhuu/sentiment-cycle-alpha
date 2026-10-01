from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.robustness.common import (
    moving_block_indices,
    performance_metrics,
    percentile_interval,
    stationary_bootstrap_indices,
)


def run_paired_bootstrap(
    portfolio: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = portfolio.copy()
    strategy = pd.to_numeric(frame["strategy_return"], errors="coerce").fillna(0.0).to_numpy()
    benchmark_column = (
        "exposure_matched_return"
        if "exposure_matched_return" in frame
        else "equal_weight_return"
    )
    benchmark = pd.to_numeric(frame[benchmark_column], errors="coerce").fillna(0.0).to_numpy()
    length = min(len(strategy), len(benchmark))
    strategy = strategy[:length]
    benchmark = benchmark[:length]
    simulations = max(int(config.get("simulations", 2000)), 1)
    block_length = max(int(config.get("block_length", 20)), 1)
    annualization = int(config.get("annualization_bars", 252))
    method = str(config.get("method", "stationary")).lower()
    confidence = float(config.get("confidence_level", 0.95))
    drawdown_threshold = float(config.get("drawdown_threshold", -0.25))
    rng = np.random.default_rng(int(config.get("random_seed", 42)))

    records: list[dict[str, Any]] = []
    for simulation in range(simulations):
        if method == "moving_block":
            indices = moving_block_indices(length, block_length, rng)
        else:
            indices = stationary_bootstrap_indices(length, block_length, rng)
        strategy_metrics = performance_metrics(strategy[indices], annualization)
        benchmark_metrics = performance_metrics(benchmark[indices], annualization)
        records.append(
            {
                "simulation": simulation + 1,
                **{f"strategy_{key}": value for key, value in strategy_metrics.items()},
                **{f"benchmark_{key}": value for key, value in benchmark_metrics.items()},
                "excess_total_return": strategy_metrics["total_return"]
                - benchmark_metrics["total_return"],
                "excess_sharpe": strategy_metrics["sharpe_ratio"]
                - benchmark_metrics["sharpe_ratio"],
            }
        )
    samples = pd.DataFrame(records)
    intervals: dict[str, Any] = {}
    for column in [
        "strategy_total_return",
        "strategy_annualized_return",
        "strategy_sharpe_ratio",
        "strategy_max_drawdown",
        "strategy_calmar_ratio",
        "excess_total_return",
        "excess_sharpe",
    ]:
        intervals[column] = percentile_interval(samples[column], confidence)
    base_strategy = performance_metrics(strategy, annualization)
    base_benchmark = performance_metrics(benchmark, annualization)
    summary = {
        "method": method,
        "simulations": simulations,
        "block_length": block_length,
        "confidence_level": confidence,
        "benchmark_column": benchmark_column,
        "base_strategy": base_strategy,
        "base_benchmark": base_benchmark,
        "intervals": intervals,
        "probability_positive_total_return": float(
            samples["strategy_total_return"].gt(0).mean()
        ),
        "probability_outperform_benchmark": float(
            samples["excess_total_return"].gt(0).mean()
        ),
        "probability_sharpe_above_benchmark": float(
            samples["excess_sharpe"].gt(0).mean()
        ),
        "probability_drawdown_worse_than_threshold": float(
            samples["strategy_max_drawdown"].le(drawdown_threshold).mean()
        ),
    }
    return samples, summary


def save_bootstrap_outputs(
    samples: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples.to_csv(
        output_dir / "bootstrap_samples.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "bootstrap_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )