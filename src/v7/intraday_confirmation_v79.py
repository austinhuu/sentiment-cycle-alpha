from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_WEIGHTS = {
    "vwap": 0.25,
    "momentum_5": 0.15,
    "momentum_15": 0.20,
    "rvol": 0.15,
    "breakout": 0.15,
    "volatility": 0.10,
}

FEATURE_COLUMN = {
    "vwap": "vwap_strength_score",
    "momentum_5": "momentum_5_score",
    "momentum_15": "momentum_15_score",
    "rvol": "rvol_score",
    "breakout": "breakout_score",
    "volatility": "volatility_control_score",
}


def build_confirmation_score(
    features: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
    disabled: set[str] | None = None,
) -> pd.DataFrame:
    """Combine pre-declared intraday features into a bounded confirmation score.

    Weights are fixed before observing real V7.9 backtest output. Ablation disables feature
    families but does not re-fit the remaining weights; active weights are simply renormalized.
    """

    cfg = cfg or {}
    disabled = disabled or set()
    d = features.copy()
    weights = {**DEFAULT_WEIGHTS, **dict(cfg.get("weights", {}))}
    active = {k: float(v) for k, v in weights.items() if k not in disabled and float(v) > 0}
    denom = sum(active.values())
    if denom <= 0:
        raise ValueError("V7.9 confirmation 至少需要一个有效特征权重。")

    score = pd.Series(0.0, index=d.index, dtype=float)
    for name, weight in active.items():
        col = FEATURE_COLUMN[name]
        values = pd.to_numeric(d.get(col, 0.0), errors="coerce").fillna(0.0).clip(-1.0, 1.0)
        score = score + values * (weight / denom)
    d["confirmation_score"] = score.clip(-1.0, 1.0)
    d["confirmation_variant"] = "full" if not disabled else "no_" + "_".join(sorted(disabled))
    return d
