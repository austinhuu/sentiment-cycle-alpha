from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def performance_metrics(
    returns: pd.Series | np.ndarray,
    annualization: int = 252,
) -> dict[str, float]:
    series = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if series.empty:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "expected_shortfall_95": 0.0,
            "observations": 0.0,
        }
    equity = (1.0 + series).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    observations = len(series)
    annualized_return = (
        float(equity.iloc[-1] ** (annualization / observations) - 1.0)
        if observations > 0 and equity.iloc[-1] > 0
        else -1.0
    )
    volatility = float(series.std(ddof=0) * np.sqrt(annualization))
    sharpe = float(series.mean() / series.std(ddof=0) * np.sqrt(annualization)) if series.std(ddof=0) > 1e-12 else 0.0
    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    calmar = annualized_return / abs(max_drawdown) if max_drawdown < -1e-12 else 0.0
    cutoff = float(series.quantile(0.05))
    tail = series.loc[series <= cutoff]
    expected_shortfall = float(tail.mean()) if not tail.empty else 0.0
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "calmar_ratio": float(calmar),
        "expected_shortfall_95": expected_shortfall,
        "observations": float(observations),
    }


def stationary_bootstrap_indices(
    length: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=int)
    probability = 1.0 / max(int(block_length), 1)
    indices = np.empty(length, dtype=int)
    current = int(rng.integers(0, length))
    for position in range(length):
        if position == 0 or rng.random() < probability:
            current = int(rng.integers(0, length))
        else:
            current = (current + 1) % length
        indices[position] = current
    return indices


def moving_block_indices(
    length: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=int)
    block = max(int(block_length), 1)
    values: list[int] = []
    while len(values) < length:
        start = int(rng.integers(0, length))
        values.extend((start + offset) % length for offset in range(block))
    return np.asarray(values[:length], dtype=int)


def percentile_interval(values: pd.Series | np.ndarray, confidence: float) -> dict[str, float]:
    series = pd.Series(values, dtype=float).dropna()
    if series.empty:
        return {"lower": np.nan, "median": np.nan, "upper": np.nan}
    alpha = (1.0 - float(np.clip(confidence, 0.0, 1.0))) / 2.0
    return {
        "lower": float(series.quantile(alpha)),
        "median": float(series.quantile(0.5)),
        "upper": float(series.quantile(1.0 - alpha)),
    }


def numeric_dict(values: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
            result[str(key)] = float(value)
    return result