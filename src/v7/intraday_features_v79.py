from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _rolling_z(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    mean = series.rolling(window, min_periods=min_periods).mean().shift(1)
    std = series.rolling(window, min_periods=min_periods).std(ddof=1).shift(1)
    return ((series - mean) / std.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _session_mask(dt: pd.Series) -> pd.Series:
    minutes = dt.dt.hour * 60 + dt.dt.minute
    morning = minutes.between(9 * 60 + 30, 11 * 60 + 30)
    afternoon = minutes.between(13 * 60, 15 * 60)
    return morning | afternoon


def build_intraday_features(bars: pd.DataFrame, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Build causal 5-minute confirmation features.

    Every feature at bar ``t`` uses information available by the close of bar ``t``. The V7.9
    simulator executes any resulting action no earlier than the next bar open.
    """

    cfg = cfg or {}
    if bars is None or bars.empty:
        return pd.DataFrame()

    d = bars.copy()
    d["datetime"] = pd.to_datetime(d["datetime"], errors="coerce")
    d["symbol"] = d["symbol"].astype(str).str.upper()
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["datetime", "symbol", "open", "high", "low", "close"])
    d = d.loc[_session_mask(d["datetime"])].copy()
    d = d.sort_values(["symbol", "datetime"]).drop_duplicates(["symbol", "datetime"], keep="last")
    if d.empty:
        return d

    d["trade_date"] = d["datetime"].dt.normalize()
    d["bar_index"] = d.groupby(["symbol", "trade_date"]).cumcount()
    d["bars_in_day"] = d.groupby(["symbol", "trade_date"])["datetime"].transform("size")

    prev_close = d.groupby("symbol")["close"].shift(1)
    tr = pd.concat(
        [
            (d["high"] - d["low"]).abs(),
            (d["high"] - prev_close).abs(),
            (d["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    d["bar_return"] = d.groupby("symbol")["close"].pct_change().fillna(0.0)
    d["tr_frac"] = tr / prev_close.replace(0.0, np.nan)

    # Intraday VWAP is truly causal: cumulative amount / cumulative volume inside the day.
    cum_amount = d.groupby(["symbol", "trade_date"])["amount"].cumsum()
    cum_volume = d.groupby(["symbol", "trade_date"])["volume"].cumsum()
    d["vwap"] = cum_amount / cum_volume.replace(0.0, np.nan)
    d["vwap"] = d["vwap"].where(d["vwap"].between(d["low"] * 0.5, d["high"] * 1.5))
    d["vwap"] = d["vwap"].fillna(
        d.groupby(["symbol", "trade_date"])["close"].expanding().mean().reset_index(level=[0, 1], drop=True)
    )

    atr_window = int(cfg.get("atr_window_bars", 20))
    atr_min = max(5, min(atr_window, int(cfg.get("atr_min_periods", 8))))
    atr_frac = d.groupby("symbol")["tr_frac"].transform(
        lambda s: s.rolling(atr_window, min_periods=atr_min).mean()
    ).clip(lower=1e-4)
    d["vwap_strength_raw"] = (d["close"] / d["vwap"] - 1.0) / atr_frac
    d["vwap_strength_score"] = np.tanh(d["vwap_strength_raw"].fillna(0.0) / 1.5)

    # 5- and 15-minute momentum, normalized by trailing intraday volatility.
    vol_window = int(cfg.get("volatility_window_bars", 24))
    vol_min = max(6, min(vol_window, int(cfg.get("volatility_min_periods", 8))))
    d["intraday_volatility"] = d.groupby("symbol")["bar_return"].transform(
        lambda s: s.rolling(vol_window, min_periods=vol_min).std(ddof=1)
    ).clip(lower=1e-5)

    d["momentum_5"] = d.groupby("symbol")["close"].pct_change(1)
    d["momentum_15"] = d.groupby("symbol")["close"].pct_change(3)
    d["momentum_5_score"] = np.tanh(
        (d["momentum_5"] / d["intraday_volatility"]).replace([np.inf, -np.inf], np.nan).fillna(0.0) / 1.5
    )
    d["momentum_15_score"] = np.tanh(
        (d["momentum_15"] / (d["intraday_volatility"] * np.sqrt(3.0)))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        / 1.5
    )

    # Causal RVOL: compare each 5-minute slot with the same slot in prior sessions only.
    lookback_days = int(cfg.get("rvol_lookback_days", 20))
    min_days = int(cfg.get("rvol_min_days", 5))
    slot_group = d.groupby(["symbol", "bar_index"], sort=False)["volume"]
    d["expected_slot_volume"] = slot_group.transform(
        lambda s: s.shift(1).rolling(lookback_days, min_periods=min_days).mean()
    )
    d["rvol"] = d["volume"] / d["expected_slot_volume"].replace(0.0, np.nan)
    d["rvol_score"] = np.tanh(np.log(d["rvol"].clip(lower=0.05, upper=20.0)).fillna(0.0) / 1.25)

    # Opening-range breakout. The first N bars define the range; no breakout signal is emitted
    # until the range is complete, avoiding accidental future use of the first 30 minutes.
    opening_bars = int(cfg.get("opening_range_bars", 6))
    g = d.groupby(["symbol", "trade_date"], sort=False)
    opening_high = g["high"].transform(
        lambda s: pd.Series(np.where(np.arange(len(s)) < opening_bars, s, np.nan), index=s.index).max()
    )
    opening_low = g["low"].transform(
        lambda s: pd.Series(np.where(np.arange(len(s)) < opening_bars, s, np.nan), index=s.index).min()
    )
    d["opening_range_high"] = opening_high
    d["opening_range_low"] = opening_low
    ready = d["bar_index"] >= opening_bars
    breakout_raw = (d["close"] / d["opening_range_high"] - 1.0) / d["intraday_volatility"].clip(lower=1e-5)
    d["breakout_score"] = np.where(ready, np.tanh(breakout_raw.fillna(0.0) / 1.5), 0.0)

    # High intraday volatility is a risk penalty, not an alpha source. The z-score baseline uses
    # only prior bars because _rolling_z shifts the trailing moments by one observation.
    z_window = int(cfg.get("volatility_z_window_bars", 120))
    z_min = int(cfg.get("volatility_z_min_periods", 30))
    d["intraday_volatility_z"] = d.groupby("symbol")["intraday_volatility"].transform(
        lambda s: _rolling_z(s, z_window, z_min)
    )
    d["volatility_control_score"] = -np.tanh(
        d["intraday_volatility_z"].clip(lower=0.0).fillna(0.0) / 2.0
    )

    return d.reset_index(drop=True)
