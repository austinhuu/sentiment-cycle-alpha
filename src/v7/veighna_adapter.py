from __future__ import annotations

from typing import Any

import pandas as pd


class VeighNaExecutionAdapter:
    """Dependency-light adapter that emits vn.py-friendly order intents.

    It does not import vnpy, so research can run on machines without the gateway installed.
    Live integration can translate these intents to CTA/PortfolioStrategy orders.
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or {}

    def build_order_intents(self, signals: pd.DataFrame) -> pd.DataFrame:
        if signals.empty:
            return pd.DataFrame()
        last_dt = pd.to_datetime(signals["datetime"]).max()
        g = signals.loc[pd.to_datetime(signals["datetime"]) == last_dt].copy()
        threshold = float(self.cfg.get("entry_score", 0.10))
        top_k = int(self.cfg.get("top_k", 8))
        g = g.loc[pd.to_numeric(g["v7_alpha"], errors="coerce") > threshold].nlargest(top_k, "v7_alpha")
        if g.empty:
            return pd.DataFrame(columns=["datetime", "symbol", "action", "target_weight", "score", "cycle_phase", "reason"])
        raw = pd.to_numeric(g["v7_alpha"], errors="coerce").clip(lower=0)
        base = raw / raw.sum()
        overlay = pd.to_numeric(g["dynamic_position_multiplier"], errors="coerce").fillna(1.0).clip(0.15, 1.0)
        weights = base * float(overlay.median())
        return pd.DataFrame({
            "datetime": last_dt,
            "symbol": g["symbol"].astype(str).values,
            "action": "TARGET_LONG",
            "target_weight": weights.values,
            "score": g["v7_alpha"].values,
            "cycle_phase": g["cycle_phase"].values,
            "reason": "V7.6 sentiment-cycle + sector-linkage + momentum confirmation",
        })
