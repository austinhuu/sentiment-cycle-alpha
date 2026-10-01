from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _rolling_z(series: pd.Series, window: int = 120, min_periods: int = 40) -> pd.Series:
    mean = series.rolling(window, min_periods=min_periods).mean().shift(1)
    std = series.rolling(window, min_periods=min_periods).std().shift(1)
    return ((series - mean) / std.replace(0.0, np.nan)).clip(-3, 3)


def build_sector_stock_linkage(
    features: pd.DataFrame,
    security_master: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Vectorized sector/theme linkage features using only same-day and historical data."""
    cfg = cfg or {}
    min_direct_peers = int(cfg.get("min_direct_peers", 2))
    corr_window = int(cfg.get("corr_window", 20))
    z_window = int(cfg.get("z_window", 120))
    z_min = int(cfg.get("z_min_periods", 40))
    theme_baskets: dict[str, list[str]] = {
        str(k).upper(): [str(x).upper() for x in v]
        for k, v in dict(cfg.get("theme_baskets", {})).items()
    }

    f = features.copy()
    f["datetime"] = pd.to_datetime(f["datetime"])
    f["symbol"] = f["symbol"].astype(str).str.strip().str.upper()
    master = security_master.copy()
    master["symbol"] = master["symbol"].astype(str).str.strip().str.upper()
    master["sector"] = master["sector"].astype(str)
    sector_map = dict(zip(master["symbol"], master["sector"]))
    sector_members = master.groupby("sector")["symbol"].apply(list).to_dict()
    all_symbols = set(f["symbol"].unique())

    parts: list[pd.DataFrame] = []
    for sym, stock in f.groupby("symbol", sort=False):
        sector = sector_map.get(sym, "UNKNOWN")
        direct = [s for s in sector_members.get(sector, []) if s != sym and s in all_symbols]
        if len(direct) >= min_direct_peers:
            peers, source = direct, "direct_sector"
        else:
            theme = [s for s in theme_baskets.get(sym, []) if s != sym and s in all_symbols]
            if theme:
                peers, source = theme, "theme_proxy"
            else:
                peers, source = [s for s in all_symbols if s != sym], "market_fallback"

        peer = f.loc[f["symbol"].isin(peers)].groupby("datetime", as_index=False).agg(
            sector_return=("return_1", "mean"),
            sector_breadth=("return_1", lambda x: float((pd.to_numeric(x, errors="coerce") > 0).mean())),
            sector_return_5=("return_5", "mean"),
            sector_volume_ratio=("volume_ratio", "mean"),
            sector_peer_count=("symbol", "nunique"),
        )
        s = stock[["datetime", "symbol", "return_1", "return_5"]].sort_values("datetime").copy()
        s["sector"] = sector
        s["sector_source"] = source
        s = s.merge(peer, on="datetime", how="left")
        s["relative_strength_5"] = pd.to_numeric(s["return_5"], errors="coerce") - s["sector_return_5"]
        s["sector_sync_corr"] = pd.to_numeric(s["return_1"], errors="coerce").rolling(
            corr_window, min_periods=max(8, corr_window // 2)
        ).corr(pd.to_numeric(s["sector_return"], errors="coerce"))
        for col in ["sector_return_5", "sector_breadth", "sector_volume_ratio", "relative_strength_5"]:
            s[f"{col}_z"] = _rolling_z(pd.to_numeric(s[col], errors="coerce"), z_window, z_min)
        parts.append(s)

    out = pd.concat(parts, ignore_index=True).sort_values(["symbol", "datetime"])
    out["sector_strength_score"] = (
        0.42 * np.tanh(out["sector_return_5_z"].fillna(0) / 1.5)
        + 0.35 * np.tanh(out["sector_breadth_z"].fillna(0) / 1.5)
        + 0.23 * np.tanh(out["sector_volume_ratio_z"].fillna(0) / 1.5)
    ).clip(-1, 1)
    out["sector_stock_link_score"] = (
        0.45 * out["sector_strength_score"]
        + 0.30 * np.tanh(out["relative_strength_5_z"].fillna(0) / 1.5)
        + 0.25 * out["sector_sync_corr"].fillna(0).clip(-1, 1)
    ).clip(-1, 1)
    out["sector_position_multiplier"] = (0.75 + 0.35 * out["sector_strength_score"]).clip(0.55, 1.10)
    return out.drop(columns=["return_1", "return_5"], errors="ignore")
