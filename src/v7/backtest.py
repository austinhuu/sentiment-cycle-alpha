from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _metrics(daily: pd.DataFrame, annualization: int = 252) -> dict[str, float]:
    r = pd.to_numeric(daily["net_return"], errors="coerce").fillna(0.0)
    equity = (1.0 + r).cumprod()
    total = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0
    years = max(len(r) / annualization, 1 / annualization)
    ann = float((1.0 + total) ** (1 / years) - 1.0) if 1.0 + total > 0 else -1.0
    std = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    vol = std * np.sqrt(annualization)
    sharpe = float(r.mean() / std * np.sqrt(annualization)) if std > 0 else 0.0
    dd = equity / equity.cummax() - 1.0 if len(equity) else pd.Series(dtype=float)
    max_dd = float(dd.min()) if len(dd) else 0.0
    calmar = float(ann / abs(max_dd)) if max_dd < 0 else 0.0
    return {
        "total_return": total,
        "annual_return": ann,
        "annual_volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "mean_turnover": float(pd.to_numeric(daily["turnover"], errors="coerce").fillna(0).mean()),
        "total_cost": float(pd.to_numeric(daily["cost"], errors="coerce").fillna(0).sum()),
        "trading_days": int(len(daily)),
    }


def run_rank_portfolio(
    signals: pd.DataFrame,
    score_col: str,
    cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Long-only next-open research portfolio with configurable holding/rebalance interval."""
    cfg = cfg or {}
    top_k = int(cfg.get("top_k", 8))
    min_score = float(cfg.get("min_score", 0.05))
    max_gross = float(cfg.get("max_gross", 1.0))
    use_overlay = bool(cfg.get("use_overlay", True))
    ann = int(cfg.get("annualization_bars", 252))
    cost_rate = float(cfg.get("round_trip_turnover_cost", 0.0017))
    rebalance_every = max(1, int(cfg.get("rebalance_every", 5)))

    d = signals.copy()
    d["datetime"] = pd.to_datetime(d["datetime"])
    d[score_col] = pd.to_numeric(d[score_col], errors="coerce")
    d["future_return_1"] = pd.to_numeric(d["future_return_1"], errors="coerce")
    d = d.dropna(subset=[score_col, "future_return_1"])

    previous: dict[str, float] = {}
    current: dict[str, float] = {}
    rows: list[dict[str, float | pd.Timestamp]] = []
    for i, (dt, group) in enumerate(d.groupby("datetime", sort=True)):
        is_rebalance = (i % rebalance_every == 0)
        turnover = 0.0
        cost = 0.0
        if is_rebalance:
            g = group.loc[group[score_col] > min_score].nlargest(top_k, score_col).copy()
            weights: dict[str, float] = {}
            if not g.empty:
                raw = g[score_col].clip(lower=0.0)
                base = raw / raw.sum() if raw.sum() > 0 else pd.Series(1 / len(g), index=g.index)
                gross = max_gross
                overlay_col = str(cfg.get("overlay_col", "dynamic_position_multiplier"))
                if use_overlay and overlay_col in g.columns:
                    gross *= float(pd.to_numeric(g[overlay_col], errors="coerce").fillna(1).median())
                    gross = float(np.clip(gross, 0.20, max_gross))
                base = base * gross
                weights = dict(zip(g["symbol"].astype(str), base.astype(float)))
            universe = set(previous) | set(weights)
            turnover = 0.5 * sum(abs(weights.get(s, 0.0) - previous.get(s, 0.0)) for s in universe)
            cost = turnover * cost_rate
            current = weights
            previous = weights.copy()

        ret_map = dict(zip(group["symbol"].astype(str), group["future_return_1"].astype(float)))
        gross_ret = sum(w * ret_map.get(s, 0.0) for s, w in current.items())
        rows.append({
            "datetime": dt,
            "gross_return": gross_ret,
            "turnover": turnover,
            "cost": cost,
            "net_return": gross_ret - cost,
            "gross_exposure": sum(current.values()),
            "holdings": len(current),
            "is_rebalance": int(is_rebalance),
        })

    daily = pd.DataFrame(rows).sort_values("datetime")
    if not daily.empty:
        daily["equity"] = (1 + daily["net_return"]).cumprod()
        daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1
    return daily, _metrics(daily, ann)


def cross_section_ic(signals: pd.DataFrame, score_col: str, horizon: str = "future_return_5") -> dict[str, float]:
    d = signals[["datetime", score_col, horizon]].copy()
    d[score_col] = pd.to_numeric(d[score_col], errors="coerce")
    d[horizon] = pd.to_numeric(d[horizon], errors="coerce")
    vals = []
    for _, g in d.groupby("datetime"):
        g = g.dropna()
        if len(g) >= 8 and g[score_col].nunique() > 1 and g[horizon].nunique() > 1:
            vals.append(g[score_col].corr(g[horizon], method="spearman"))
    s = pd.Series(vals, dtype=float).dropna()
    return {
        "mean_ic": float(s.mean()) if len(s) else 0.0,
        "ic_std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "ic_ir": float(s.mean() / s.std(ddof=1)) if len(s) > 1 and s.std(ddof=1) > 0 else 0.0,
        "positive_ic_rate": float((s > 0).mean()) if len(s) else 0.0,
        "ic_days": int(len(s)),
    }
