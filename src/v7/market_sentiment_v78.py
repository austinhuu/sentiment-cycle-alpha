from __future__ import annotations

import numpy as np
import pandas as pd


def _causal_z(s: pd.Series, window: int = 120, min_periods: int = 40) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    mean = x.rolling(window, min_periods=min_periods).mean().shift(1)
    std = x.rolling(window, min_periods=min_periods).std().shift(1).replace(0.0, np.nan)
    return ((x - mean) / std).clip(-3.0, 3.0)


def _limit_threshold(symbols: pd.Series) -> pd.Series:
    sym = symbols.astype(str).str.upper()
    code = sym.str.split(".", regex=False).str[0]
    threshold = pd.Series(0.095, index=sym.index, dtype=float)
    threshold.loc[code.str.startswith(("300", "301", "688"), na=False)] = 0.195
    threshold.loc[sym.str.endswith(".BJ", na=False)] = 0.295
    return threshold


def aggregate_market_cross_section(all_a_daily: pd.DataFrame) -> pd.DataFrame:
    """Collapse a point-in-time A-share panel into one compact market-state row per date."""
    if all_a_daily is None or all_a_daily.empty:
        return pd.DataFrame()
    d = all_a_daily.copy()
    d["datetime"] = pd.to_datetime(d["datetime"], errors="coerce")
    d["symbol"] = d["symbol"].astype(str).str.upper()
    ret = pd.to_numeric(d.get("pct_chg"), errors="coerce")
    if ret.notna().any() and ret.abs().quantile(0.99) > 2.0:
        ret = ret / 100.0
    d["ret"] = ret
    d["amount_num"] = pd.to_numeric(d.get("amount"), errors="coerce")
    threshold = _limit_threshold(d["symbol"])
    # ST/special-treatment status is not available in the basic daily panel. These counts are
    # therefore approximate and are overridden by exact Eastmoney limit-pool observations on
    # overlapping recent dates.
    d["approx_limit_up"] = d["ret"] >= threshold
    d["approx_limit_down"] = d["ret"] <= -threshold

    daily = d.groupby("datetime", as_index=False).agg(
        market_count=("symbol", "nunique"),
        up_count=("ret", lambda x: int((x > 0).sum())),
        down_count=("ret", lambda x: int((x < 0).sum())),
        flat_count=("ret", lambda x: int((x == 0).sum())),
        strong_up_count=("ret", lambda x: int((x >= 0.05).sum())),
        strong_down_count=("ret", lambda x: int((x <= -0.05).sum())),
        approx_limit_up_count=("approx_limit_up", "sum"),
        approx_limit_down_count=("approx_limit_down", "sum"),
        market_equal_return=("ret", "mean"),
        market_median_return=("ret", "median"),
        cross_section_vol=("ret", "std"),
        total_amount=("amount_num", "sum"),
    ).sort_values("datetime")
    daily["breadth"] = daily["up_count"] / daily["market_count"].replace(0, np.nan)
    daily["strong_up_rate"] = daily["strong_up_count"] / daily["market_count"].replace(0, np.nan)
    daily["strong_down_rate"] = daily["strong_down_count"] / daily["market_count"].replace(0, np.nan)
    daily["limit_up_count"] = pd.to_numeric(daily["approx_limit_up_count"], errors="coerce")
    daily["limit_down_count"] = pd.to_numeric(daily["approx_limit_down_count"], errors="coerce")
    daily["limit_count_source"] = "APPROX_FROM_DAILY_RETURN"
    return daily


def _finalize_market_sentiment(
    daily: pd.DataFrame,
    exact_limit_pool: pd.DataFrame | None,
    z_window: int,
    min_periods: int,
) -> pd.DataFrame:
    daily = daily.copy()
    daily["datetime"] = pd.to_datetime(daily["datetime"], errors="coerce")
    daily = daily.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates("datetime", keep="last")

    # Backward compatibility for compact summaries produced by the downloader.
    if "breadth" not in daily.columns:
        daily["breadth"] = pd.to_numeric(daily.get("up_count"), errors="coerce") / pd.to_numeric(daily.get("market_count"), errors="coerce").replace(0, np.nan)
    if "strong_up_rate" not in daily.columns:
        daily["strong_up_rate"] = pd.to_numeric(daily.get("strong_up_count"), errors="coerce") / pd.to_numeric(daily.get("market_count"), errors="coerce").replace(0, np.nan)
    if "strong_down_rate" not in daily.columns:
        daily["strong_down_rate"] = pd.to_numeric(daily.get("strong_down_count"), errors="coerce") / pd.to_numeric(daily.get("market_count"), errors="coerce").replace(0, np.nan)
    if "limit_up_count" not in daily.columns:
        daily["limit_up_count"] = pd.to_numeric(daily.get("approx_limit_up_count"), errors="coerce")
    if "limit_down_count" not in daily.columns:
        daily["limit_down_count"] = pd.to_numeric(daily.get("approx_limit_down_count"), errors="coerce")
    if "limit_count_source" not in daily.columns:
        daily["limit_count_source"] = "APPROX_FROM_DAILY_RETURN"

    if exact_limit_pool is not None and not exact_limit_pool.empty:
        ex = exact_limit_pool.copy()
        ex["datetime"] = pd.to_datetime(ex["datetime"], errors="coerce")
        cols = [
            c for c in [
                "datetime", "limit_up_count", "limit_down_count", "broken_limit_count",
                "broken_board_rate", "max_streak", "first_board_count", "multi_board_count",
            ] if c in ex.columns
        ]
        ex = ex[cols].drop_duplicates("datetime", keep="last")
        merged = daily.merge(ex, on="datetime", how="left", suffixes=("", "_exact"))
        for col in ["limit_up_count", "limit_down_count"]:
            exact = pd.to_numeric(merged.get(f"{col}_exact"), errors="coerce")
            merged[col] = exact.where(exact.notna(), pd.to_numeric(merged[col], errors="coerce"))
        exact_lu = pd.to_numeric(merged.get("limit_up_count_exact"), errors="coerce")
        merged["limit_count_source"] = np.where(
            exact_lu.notna(), "REAL_EASTMONEY_POOL", merged["limit_count_source"]
        )
        for col in ["broken_limit_count", "broken_board_rate", "max_streak", "first_board_count", "multi_board_count"]:
            exact_col = f"{col}_exact"
            if exact_col in merged.columns:
                base = pd.to_numeric(merged[col], errors="coerce") if col in merged.columns else pd.Series(np.nan, index=merged.index)
                merged[col] = pd.to_numeric(merged[exact_col], errors="coerce").where(
                    pd.to_numeric(merged[exact_col], errors="coerce").notna(), base
                )
        daily = merged.drop(columns=[c for c in merged.columns if c.endswith("_exact")], errors="ignore")

    daily["amount_ratio_20"] = pd.to_numeric(daily["total_amount"], errors="coerce") / pd.to_numeric(daily["total_amount"], errors="coerce").rolling(20, min_periods=10).mean().shift(1)
    daily["limit_balance"] = (
        pd.to_numeric(daily["limit_up_count"], errors="coerce") - pd.to_numeric(daily["limit_down_count"], errors="coerce")
    ) / np.sqrt(pd.to_numeric(daily["market_count"], errors="coerce").clip(lower=1))

    z_cols = ["breadth", "market_equal_return", "cross_section_vol", "amount_ratio_20", "strong_up_rate", "strong_down_rate", "limit_balance"]
    for col in z_cols:
        daily[f"{col}_z"] = _causal_z(daily[col], z_window, min_periods)

    daily["real_sentiment_score"] = (
        0.24 * np.tanh(daily["breadth_z"].fillna(0) / 1.5)
        + 0.18 * np.tanh(daily["market_equal_return_z"].fillna(0) / 1.5)
        + 0.17 * np.tanh(daily["amount_ratio_20_z"].fillna(0) / 1.5)
        + 0.16 * np.tanh(daily["strong_up_rate_z"].fillna(0) / 1.5)
        - 0.12 * np.tanh(daily["strong_down_rate_z"].fillna(0) / 1.5)
        + 0.13 * np.tanh(daily["limit_balance_z"].fillna(0) / 1.5)
    ).clip(-1.0, 1.0)
    daily["real_sentiment_momentum"] = daily["real_sentiment_score"].diff(3) / 3.0

    score = daily["real_sentiment_score"].fillna(0.0)
    momentum = daily["real_sentiment_momentum"].fillna(0.0)
    daily["real_cycle_phase"] = np.select(
        [
            (score <= -0.35) & (momentum <= 0),
            (score <= -0.15) & (momentum > 0),
            (score > -0.15) & (score < 0.45) & (momentum > 0),
            (score >= 0.45) & (momentum >= -0.01),
            (score >= 0.15) & (momentum < -0.01),
        ],
        ["ICE_POINT", "REPAIR", "FERMENTATION", "CLIMAX", "DIVERGENCE"],
        default="EBB",
    )
    daily["real_market_data_quality"] = np.where(
        pd.to_numeric(daily["market_count"], errors="coerce") >= 2500,
        "BROAD_POINT_IN_TIME_OR_CURRENT",
        np.where(pd.to_numeric(daily["market_count"], errors="coerce") >= 500, "PARTIAL_UNIVERSE", "INSUFFICIENT_UNIVERSE"),
    )
    return daily


def build_real_market_sentiment(
    all_a_daily: pd.DataFrame,
    exact_limit_pool: pd.DataFrame | None = None,
    z_window: int = 120,
    min_periods: int = 40,
) -> pd.DataFrame:
    """Create a daily full-market sentiment state from either a raw panel or compact summary."""
    if all_a_daily is None or all_a_daily.empty:
        return pd.DataFrame()
    if {"market_count", "up_count", "total_amount"}.issubset(all_a_daily.columns):
        daily = all_a_daily.copy()
    else:
        daily = aggregate_market_cross_section(all_a_daily)
    return _finalize_market_sentiment(daily, exact_limit_pool, z_window, min_periods)
