from __future__ import annotations

import numpy as np
import pandas as pd


def _z(s: pd.Series, window: int = 120, min_periods: int = 40) -> pd.Series:
    """Causal rolling z-score: today's value is standardized by information through t-1."""
    x = pd.to_numeric(s, errors="coerce")
    mean = x.rolling(window, min_periods=min_periods).mean().shift(1)
    std = x.rolling(window, min_periods=min_periods).std().shift(1).replace(0.0, np.nan)
    return ((x - mean) / std).clip(-3.0, 3.0)


def _as_decimal_return(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().any() and x.abs().quantile(0.99) > 2.0:
        x = x / 100.0
    return x


def _first_text(frame: pd.DataFrame, column: str, default: str) -> str:
    if column not in frame.columns:
        return default
    values = frame[column].dropna().astype(str)
    values = values.loc[values.str.strip().ne("")]
    return str(values.iloc[-1]) if not values.empty else default


def build_historical_sector_context(
    sector_daily: pd.DataFrame,
    focus_daily: pd.DataFrame,
    *,
    source: str,
    quality: str,
    corr_window: int = 20,
    z_window: int = 120,
    z_min_periods: int = 40,
) -> pd.DataFrame:
    """Build one source-agnostic historical sector context.

    The input must be a historical sector series (published board/index or a
    clearly-labelled traded ETF proxy), never a panel reconstructed from today's
    constituents. All rolling normalisation is causal.
    """
    if sector_daily is None or sector_daily.empty or focus_daily is None or focus_daily.empty:
        return pd.DataFrame()

    s = sector_daily.copy()
    s["datetime"] = pd.to_datetime(s["datetime"], errors="coerce")
    s["sector_ret"] = _as_decimal_return(s.get("pct_chg", pd.Series(index=s.index, dtype=float)))
    sector_amount = pd.to_numeric(s.get("amount"), errors="coerce")
    sector_volume = pd.to_numeric(s.get("volume"), errors="coerce")
    amount_share = float(sector_amount.notna().mean()) if len(s) else 0.0
    if amount_share >= 0.80:
        s["sector_liquidity"] = sector_amount
        liquidity_basis = "amount"
    else:
        s["sector_liquidity"] = sector_volume
        liquidity_basis = "volume"

    s = s.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates("datetime", keep="last")

    s["sector_return_5"] = s["sector_ret"].rolling(5, min_periods=3).sum()
    prior_liquidity = s["sector_liquidity"].rolling(20, min_periods=10).mean().shift(1)
    s["sector_amount_ratio"] = s["sector_liquidity"] / prior_liquidity.replace(0.0, np.nan)

    f = focus_daily.copy()
    f["datetime"] = pd.to_datetime(f["datetime"], errors="coerce")
    if "return_1" in f.columns:
        f["focus_ret"] = pd.to_numeric(f["return_1"], errors="coerce")
    elif "pct_chg" in f.columns:
        f["focus_ret"] = _as_decimal_return(f["pct_chg"])
    else:
        f["focus_ret"] = np.nan

    if "return_5" in f.columns:
        f["focus_return_5"] = pd.to_numeric(f["return_5"], errors="coerce")
    else:
        f["focus_return_5"] = f["focus_ret"].rolling(5, min_periods=3).sum()

    f = (
        f[["datetime", "focus_ret", "focus_return_5"]]
        .dropna(subset=["datetime"])
        .drop_duplicates("datetime", keep="last")
        .sort_values("datetime")
    )

    out = s.merge(f, on="datetime", how="left")
    out["sector_relative_strength_5"] = out["focus_return_5"] - out["sector_return_5"]
    out["sector_sync_corr"] = out["focus_ret"].rolling(
        corr_window, min_periods=max(8, corr_window // 2)
    ).corr(out["sector_ret"])

    for col in ["sector_return_5", "sector_amount_ratio", "sector_relative_strength_5"]:
        out[f"{col}_z"] = _z(out[col], z_window, z_min_periods)

    out["sector_strength_score"] = (
        0.70 * np.tanh(out["sector_return_5_z"].fillna(0.0) / 1.5)
        + 0.30 * np.tanh(out["sector_amount_ratio_z"].fillna(0.0) / 1.5)
    ).clip(-1.0, 1.0)

    out["sector_stock_link_score"] = (
        0.55 * out["sector_strength_score"]
        + 0.25 * np.tanh(out["sector_relative_strength_5_z"].fillna(0.0) / 1.5)
        + 0.20 * out["sector_sync_corr"].fillna(0.0).clip(-1.0, 1.0)
    ).clip(-1.0, 1.0)

    out["sector_source"] = str(source)
    out["sector_data_quality"] = str(quality)
    out["sector_survivorship_flag"] = False
    out["sector_liquidity_basis"] = liquidity_basis
    return out.drop(columns=["focus_ret", "focus_return_5"], errors="ignore")


def build_historical_board_linkage(
    board_daily: pd.DataFrame,
    focus_daily: pd.DataFrame,
    corr_window: int = 20,
    z_window: int = 120,
    z_min_periods: int = 40,
) -> pd.DataFrame:
    """Backward-compatible historical-sector wrapper.

    New behaviour: if the input file carries ``sector_source`` and
    ``sector_quality`` metadata, preserve those labels. This allows V7.8 to use
    a verified ETF proxy without pretending it is the Eastmoney board index.
    Old Eastmoney files remain fully backward compatible.
    """
    source = _first_text(
        board_daily,
        "sector_source",
        "Eastmoney historical semiconductor board",
    )
    quality = _first_text(
        board_daily,
        "sector_quality",
        "REAL_HISTORICAL_BOARD_INDEX",
    )

    generic = build_historical_sector_context(
        board_daily,
        focus_daily,
        source=source,
        quality=quality,
        corr_window=corr_window,
        z_window=z_window,
        z_min_periods=z_min_periods,
    )
    if generic.empty:
        return generic

    rename = {
        "sector_return_5": "historical_board_return_5",
        "sector_amount_ratio": "historical_board_amount_ratio",
        "sector_relative_strength_5": "historical_board_relative_strength_5",
        "sector_sync_corr": "historical_board_sync_corr",
        "sector_return_5_z": "historical_board_return_5_z",
        "sector_amount_ratio_z": "historical_board_amount_ratio_z",
        "sector_relative_strength_5_z": "historical_board_relative_strength_5_z",
        "sector_strength_score": "historical_board_strength_score",
        "sector_stock_link_score": "historical_board_stock_link_score",
        "sector_data_quality": "historical_board_data_quality",
        "sector_source": "historical_board_source",
        "sector_survivorship_flag": "historical_board_survivorship_flag",
        "sector_liquidity_basis": "historical_board_liquidity_basis",
    }
    return generic.rename(columns=rename)


def build_real_semiconductor_linkage(
    semiconductor_daily: pd.DataFrame,
    focus_symbol: str = "688981.SH",
    corr_window: int = 20,
    z_window: int = 120,
    z_min_periods: int = 40,
) -> pd.DataFrame:
    """Diagnostic current-membership peer context.

    Historical prices are real, but the constituent list is today's snapshot.
    This output is never eligible for the primary historical backtest unless
    the caller explicitly overrides the survivorship gate.
    """
    if semiconductor_daily is None or semiconductor_daily.empty:
        return pd.DataFrame()

    d = semiconductor_daily.copy()
    d["datetime"] = pd.to_datetime(d["datetime"], errors="coerce")
    d["symbol"] = d["symbol"].astype(str).str.upper()
    d["ret"] = _as_decimal_return(d.get("pct_chg", pd.Series(index=d.index, dtype=float)))
    d["amount_num"] = pd.to_numeric(d.get("amount"), errors="coerce")
    focus = focus_symbol.upper()

    peer = d.loc[d["symbol"] != focus].copy()
    if peer.empty:
        return pd.DataFrame()

    daily = (
        peer.groupby("datetime", as_index=False)
        .agg(
            real_sector_return=("ret", "mean"),
            real_sector_breadth=("ret", lambda x: float((pd.to_numeric(x, errors="coerce") > 0).mean())),
            real_sector_peer_count=("symbol", "nunique"),
            real_sector_amount=("amount_num", "sum"),
        )
        .sort_values("datetime")
    )
    daily["real_sector_return_5"] = daily["real_sector_return"].rolling(5, min_periods=3).sum()
    daily["real_sector_amount_ratio"] = daily["real_sector_amount"] / daily["real_sector_amount"].rolling(
        20, min_periods=10
    ).mean().shift(1).replace(0.0, np.nan)

    focus_d = d.loc[d["symbol"] == focus, ["datetime", "ret"]].sort_values("datetime")
    focus_d["focus_return_5"] = focus_d["ret"].rolling(5, min_periods=3).sum()
    out = daily.merge(focus_d, on="datetime", how="left")
    out["real_relative_strength_5"] = out["focus_return_5"] - out["real_sector_return_5"]
    out["real_sector_sync_corr"] = out["ret"].rolling(
        corr_window, min_periods=max(8, corr_window // 2)
    ).corr(out["real_sector_return"])

    for col in ["real_sector_return_5", "real_sector_breadth", "real_sector_amount_ratio", "real_relative_strength_5"]:
        out[f"{col}_z"] = _z(out[col], z_window, z_min_periods)

    out["real_sector_strength_score"] = (
        0.45 * np.tanh(out["real_sector_return_5_z"].fillna(0.0) / 1.5)
        + 0.35 * np.tanh(out["real_sector_breadth_z"].fillna(0.0) / 1.5)
        + 0.20 * np.tanh(out["real_sector_amount_ratio_z"].fillna(0.0) / 1.5)
    ).clip(-1.0, 1.0)

    out["real_sector_stock_link_score"] = (
        0.45 * out["real_sector_strength_score"]
        + 0.30 * np.tanh(out["real_relative_strength_5_z"].fillna(0.0) / 1.5)
        + 0.25 * out["real_sector_sync_corr"].fillna(0.0).clip(-1.0, 1.0)
    ).clip(-1.0, 1.0)

    out["real_sector_data_quality"] = np.where(
        out["real_sector_peer_count"] >= 20,
        "REAL_BROAD_CURRENT_MEMBERSHIP",
        np.where(out["real_sector_peer_count"] >= 5, "REAL_PARTIAL_CURRENT_MEMBERSHIP", "INSUFFICIENT"),
    )
    out["real_sector_source"] = "CURRENT_SEMICONDUCTOR_MEMBERSHIP_DIAGNOSTIC"
    out["real_sector_survivorship_flag"] = True
    return out.drop(columns=["ret", "focus_return_5"], errors="ignore")
