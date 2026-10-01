from __future__ import annotations

import numpy as np
import pandas as pd


def build_momentum_breakout(features: pd.DataFrame) -> pd.DataFrame:
    f = features.copy()
    momentum = pd.to_numeric(f.get("momentum_z", 0.0), errors="coerce").fillna(0.0)
    volume = pd.to_numeric(f.get("volume_ratio_z", 0.0), errors="coerce").fillna(0.0)
    ma_gap = pd.to_numeric(f.get("ma_gap_short_long_z", 0.0), errors="coerce").fillna(0.0)
    pos = pd.to_numeric(f.get("price_position", 0.5), errors="coerce").fillna(0.5).clip(0, 1)
    breakout = pd.to_numeric(f.get("breakout_up", 0.0), errors="coerce").fillna(0.0).clip(0, 1)
    score = (
        0.31 * np.tanh(momentum / 1.5)
        + 0.19 * np.tanh(volume / 1.5)
        + 0.18 * np.tanh(ma_gap / 1.5)
        + 0.16 * (2.0 * pos - 1.0)
        + 0.16 * breakout
    ).clip(-1, 1)
    confirm = ((momentum > 0) & (ma_gap > 0) & ((volume > 0) | (breakout > 0))).astype(int)
    return pd.DataFrame({
        "datetime": pd.to_datetime(f["datetime"]),
        "symbol": f["symbol"].astype(str),
        "momentum_breakout_score": score,
        "momentum_breakout_confirmed": confirm,
    })
