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
)


def run_monte_carlo_stress(
    portfolio: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = portfolio.copy()
    gross = pd.to_numeric(frame.get("gross_return"), errors="coerce").fillna(0.0).to_numpy()
    cost = pd.to_numeric(frame.get("transaction_cost"), errors="coerce").fillna(0.0).to_numpy()
    benchmark = pd.to_numeric(
        frame.get("exposure_matched_return", frame.get("equal_weight_return")),
        errors="coerce",
    ).fillna(0.0).to_numpy()
    length = min(len(gross), len(cost), len(benchmark))
    gross, cost, benchmark = gross[:length], cost[:length], benchmark[:length]
    simulations = max(int(config.get("simulations", 2000)), 1)
    block_length = max(int(config.get("block_length", 20)), 1)
    annualization = int(config.get("annualization_bars", 252))
    rng = np.random.default_rng(int(config.get("random_seed", 43)))
    years = length / max(annualization, 1)
    drawdown_threshold = float(config.get("drawdown_threshold", -0.30))

    records: list[dict[str, Any]] = []
    for simulation in range(simulations):
        indices = moving_block_indices(length, block_length, rng)
        sampled_gross = gross[indices].copy()
        sampled_cost = cost[indices].copy()
        sampled_benchmark = benchmark[indices].copy()
        cost_multiplier = float(
            rng.triangular(
                float(config.get("cost_multiplier_low", 1.0)),
                float(config.get("cost_multiplier_mode", 1.5)),
                float(config.get("cost_multiplier_high", 3.0)),
            )
        )
        stressed = sampled_gross - sampled_cost * cost_multiplier

        shock_count = int(
            rng.poisson(float(config.get("shock_events_per_year", 1.0)) * years)
        )
        if shock_count > 0 and length > 0:
            shock_dates = rng.choice(length, size=min(shock_count, length), replace=False)
            shock_magnitudes = rng.uniform(
                float(config.get("shock_magnitude_min", 0.02)),
                float(config.get("shock_magnitude_max", 0.06)),
                size=len(shock_dates),
            )
            stressed[shock_dates] -= shock_magnitudes

        failure_probability = 1.0 - np.exp(
            -float(config.get("model_failure_probability_per_year", 0.35)) * years
        )
        failure_triggered = bool(rng.random() < failure_probability)
        failure_start = -1
        failure_duration = 0
        if failure_triggered and length > 0:
            failure_duration = min(
                max(int(config.get("model_failure_duration_days", 20)), 1), length
            )
            failure_start = int(rng.integers(0, max(length - failure_duration + 1, 1)))
            stressed[
                failure_start : failure_start + failure_duration
            ] -= float(config.get("model_failure_daily_penalty", 0.0008))

        strategy_metrics = performance_metrics(stressed, annualization)
        benchmark_metrics = performance_metrics(sampled_benchmark, annualization)
        records.append(
            {
                "simulation": simulation + 1,
                "cost_multiplier": cost_multiplier,
                "shock_count": shock_count,
                "model_failure_triggered": int(failure_triggered),
                "model_failure_start": failure_start,
                "model_failure_duration": failure_duration,
                **{f"strategy_{key}": value for key, value in strategy_metrics.items()},
                **{f"benchmark_{key}": value for key, value in benchmark_metrics.items()},
                "excess_total_return": strategy_metrics["total_return"]
                - benchmark_metrics["total_return"],
            }
        )
    samples = pd.DataFrame(records)
    confidence = 0.95
    summary = {
        "simulations": simulations,
        "block_length": block_length,
        "intervals": {
            column: percentile_interval(samples[column], confidence)
            for column in [
                "strategy_total_return",
                "strategy_annualized_return",
                "strategy_sharpe_ratio",
                "strategy_max_drawdown",
                "strategy_expected_shortfall_95",
                "excess_total_return",
            ]
        },
        "probability_positive_total_return": float(
            samples["strategy_total_return"].gt(0).mean()
        ),
        "probability_outperform_benchmark": float(
            samples["excess_total_return"].gt(0).mean()
        ),
        "probability_drawdown_worse_than_threshold": float(
            samples["strategy_max_drawdown"].le(drawdown_threshold).mean()
        ),
        "probability_model_failure_scenario": float(
            samples["model_failure_triggered"].gt(0).mean()
        ),
        "median_cost_multiplier": float(samples["cost_multiplier"].median()),
    }
    return samples, summary


def save_monte_carlo_outputs(
    samples: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples.to_csv(
        output_dir / "monte_carlo_samples.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "monte_carlo_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )