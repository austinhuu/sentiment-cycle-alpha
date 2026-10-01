from __future__ import annotations

import numpy as np
import pandas as pd


def causal_cycle_edge(frame: pd.DataFrame, horizon: str = "future_return_5", purge_dates: int = 7) -> pd.DataFrame:
    """Learn each phase's expected edge from prior calendar years only."""
    d = frame[["datetime", "cycle_phase", horizon]].copy()
    d["datetime"] = pd.to_datetime(d["datetime"])
    d["year"] = d["datetime"].dt.year
    out_parts = []
    for year, test in d.groupby("year", sort=True):
        train = d.loc[d["year"] < year].copy()
        edge = {}
        if not train.empty:
            # Previous calendar years are fully observable at the test-year boundary.
            edge = train.groupby("cycle_phase")[horizon].mean().to_dict()
        t = test.copy()
        t["cycle_edge_return"] = t["cycle_phase"].map(edge).fillna(0.0)
        vals = pd.Series(list(edge.values()), dtype=float)
        scale = float(vals.std(ddof=0)) if len(vals) > 1 else 0.0
        center = float(vals.mean()) if len(vals) else 0.0
        if scale > 1e-12:
            t["cycle_edge_score"] = ((t["cycle_edge_return"] - center) / scale).clip(-2, 2) / 2.0
        else:
            t["cycle_edge_score"] = 0.0
        t["learned_cycle_multiplier"] = (0.70 + 0.30 * t["cycle_edge_score"]).clip(0.45, 1.0)
        out_parts.append(t[["datetime", "cycle_edge_return", "cycle_edge_score", "learned_cycle_multiplier"]])
    return pd.concat(out_parts, ignore_index=True).drop_duplicates("datetime")


def walk_forward_ridge_fusion(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "future_return_5",
    ridge_lambda: float = 20.0,
    purge_dates: int = 7,
) -> tuple[pd.Series, pd.DataFrame]:
    """Annual walk-forward ridge. Test-year labels are never used to fit that year's model."""
    d = frame.copy()
    d["datetime"] = pd.to_datetime(d["datetime"])
    d["year"] = d["datetime"].dt.year
    pred = pd.Series(np.nan, index=d.index, dtype=float)
    coef_rows = []
    years = sorted(d["year"].dropna().unique())

    for year in years:
        test_idx = d.index[d["year"] == year]
        train = d.loc[d["year"] < year].copy()
        if train.empty or len(train) < 1000:
            fallback = (
                0.55 * pd.to_numeric(d.loc[test_idx, "final_score"], errors="coerce").fillna(0)
                + 0.30 * pd.to_numeric(d.loc[test_idx, "momentum_breakout_score"], errors="coerce").fillna(0)
                + 0.10 * pd.to_numeric(d.loc[test_idx, "sector_context_score"], errors="coerce").fillna(0)
                + 0.05 * pd.to_numeric(d.loc[test_idx, "cycle_edge_score"], errors="coerce").fillna(0)
            )
            pred.loc[test_idx] = fallback
            coef_rows.append({"test_year": int(year), "mode": "fallback", **{c: np.nan for c in feature_cols}})
            continue

        train_dates = sorted(train["datetime"].dropna().unique())
        if len(train_dates) > purge_dates:
            cutoff = train_dates[-purge_dates]
            train = train.loc[train["datetime"] < cutoff]

        X = train[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        y = pd.to_numeric(train[target_col], errors="coerce")
        valid = y.notna()
        X, y = X.loc[valid], y.loc[valid].clip(-0.20, 0.20)
        if len(X) < 500:
            pred.loc[test_idx] = pd.to_numeric(d.loc[test_idx, "final_score"], errors="coerce").fillna(0)
            coef_rows.append({"test_year": int(year), "mode": "insufficient", **{c: np.nan for c in feature_cols}})
            continue

        mean = X.mean()
        std = X.std(ddof=0).replace(0.0, 1.0)
        Z = ((X - mean) / std).to_numpy(float)
        Z = np.column_stack([np.ones(len(Z)), Z])
        yy = y.to_numpy(float)
        penalty = np.eye(Z.shape[1]) * float(ridge_lambda)
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(Z.T @ Z + penalty, Z.T @ yy)

        Xt = d.loc[test_idx, feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        Zt = ((Xt - mean) / std).to_numpy(float)
        Zt = np.column_stack([np.ones(len(Zt)), Zt])
        raw_pred = Zt @ beta
        pred.loc[test_idx] = raw_pred
        row = {"test_year": int(year), "mode": "ridge", "intercept": float(beta[0])}
        row.update({c: float(beta[i + 1] / std[c]) for i, c in enumerate(feature_cols)})
        coef_rows.append(row)

    # Robust cross-sectional scaling preserves only ranking/magnitude within each decision date.
    tmp = pd.DataFrame({"datetime": d["datetime"], "raw": pred})
    rank = tmp.groupby("datetime")["raw"].rank(method="average", pct=True)
    score = (2.0 * rank - 1.0).fillna(0.0).clip(-1, 1)
    return score, pd.DataFrame(coef_rows)
