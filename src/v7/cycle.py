from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def _rolling_z(series: pd.Series, window: int = 120, min_periods: int = 40) -> pd.Series:
    hist_mean = series.rolling(window, min_periods=min_periods).mean().shift(1)
    hist_std = series.rolling(window, min_periods=min_periods).std().shift(1)
    return ((series - hist_mean) / hist_std.replace(0.0, np.nan)).clip(-3.0, 3.0)


def _squash(series: pd.Series) -> pd.Series:
    return np.tanh(pd.to_numeric(series, errors="coerce").fillna(0.0) / 1.5)


@dataclass(frozen=True)
class CycleConfig:
    z_window: int = 120
    z_min_periods: int = 40
    momentum_lag: int = 3
    strong_up_threshold: float = 0.07
    strong_down_threshold: float = -0.07


def build_sentiment_cycle(features: pd.DataFrame, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Build a point-in-time market sentiment cycle from the current research pool.

    The current project only has a 32-stock reference pool, so strong-up/down breadth is
    deliberately named as a *proxy*, not as official whole-market limit-up statistics.
    """
    cfg = cfg or {}
    c = CycleConfig(
        z_window=int(cfg.get("z_window", 120)),
        z_min_periods=int(cfg.get("z_min_periods", 40)),
        momentum_lag=int(cfg.get("momentum_lag", 3)),
        strong_up_threshold=float(cfg.get("strong_up_threshold", 0.07)),
        strong_down_threshold=float(cfg.get("strong_down_threshold", -0.07)),
    )
    work = features.copy()
    work["datetime"] = pd.to_datetime(work["datetime"])
    ret = pd.to_numeric(work["return_1"], errors="coerce")
    work["_up"] = (ret > 0).astype(float)
    work["_strong_up"] = (ret >= c.strong_up_threshold).astype(float)
    work["_strong_down"] = (ret <= c.strong_down_threshold).astype(float)

    market = work.groupby("datetime", as_index=False).agg(
        cycle_breadth=("_up", "mean"),
        cycle_market_return=("return_1", "mean"),
        cycle_volume_ratio=("volume_ratio", "mean"),
        cycle_cross_vol=("return_1", "std"),
        cycle_strong_up_rate=("_strong_up", "mean"),
        cycle_strong_down_rate=("_strong_down", "mean"),
        cycle_symbol_count=("symbol", "nunique"),
    ).sort_values("datetime")

    for col in [
        "cycle_breadth",
        "cycle_market_return",
        "cycle_volume_ratio",
        "cycle_cross_vol",
        "cycle_strong_up_rate",
        "cycle_strong_down_rate",
    ]:
        market[f"{col}_z"] = _rolling_z(market[col], c.z_window, c.z_min_periods)

    balance_z = market["cycle_strong_up_rate_z"].fillna(0.0) - market["cycle_strong_down_rate_z"].fillna(0.0)
    market["cycle_score"] = (
        0.27 * _squash(market["cycle_breadth_z"])
        + 0.22 * _squash(market["cycle_market_return_z"])
        + 0.12 * _squash(market["cycle_volume_ratio_z"])
        + 0.25 * np.tanh(balance_z / 2.0)
        - 0.14 * _squash(market["cycle_cross_vol_z"].clip(lower=0.0))
    ).clip(-1.0, 1.0)
    market["cycle_momentum"] = market["cycle_score"] - market["cycle_score"].shift(c.momentum_lag)

    def phase(row: pd.Series) -> str:
        s = float(row.get("cycle_score", 0.0) or 0.0)
        m = float(row.get("cycle_momentum", 0.0) or 0.0)
        if s <= -0.48:
            return "REPAIR" if m > 0.04 else "ICE_POINT"
        if s < -0.08:
            return "REPAIR" if m > 0.03 else "EBB"
        if s < 0.38:
            if m > 0.04:
                return "FERMENTATION"
            if m < -0.05:
                return "DIVERGENCE"
            return "REPAIR"
        return "DIVERGENCE" if m < -0.04 else "CLIMAX"

    market["cycle_phase"] = market.apply(phase, axis=1)
    default_mult = {
        "ICE_POINT": 0.25,
        "REPAIR": 0.65,
        "FERMENTATION": 1.00,
        "CLIMAX": 0.85,
        "DIVERGENCE": 0.55,
        "EBB": 0.30,
    }
    multipliers = {**default_mult, **dict(cfg.get("position_multipliers", {}))}
    market["cycle_position_multiplier"] = market["cycle_phase"].map(multipliers).fillna(0.6)
    market["cycle_confidence"] = (
        0.45
        + 0.25 * market["cycle_score"].abs()
        + 0.20 * market["cycle_momentum"].fillna(0.0).abs().clip(0, 0.5) / 0.5
        + 0.10 * (market["cycle_symbol_count"] / max(float(market["cycle_symbol_count"].max()), 1.0))
    ).clip(0.0, 1.0)
    return market
