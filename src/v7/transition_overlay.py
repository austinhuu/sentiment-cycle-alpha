from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TransitionOverlayConfig:
    base_exposure: float = 0.78
    sensitivity: float = 0.12
    min_exposure: float = 0.55
    max_exposure: float = 1.00
    phase_prior_strength: float = 30.0
    transition_prior_strength: float = 20.0
    purge_dates: int = 7
    min_train_dates: int = 120


def _robust_scale(series: pd.Series) -> tuple[float, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0.0, 1.0
    center = float(s.median())
    mad = float((s - center).abs().median())
    std = float(s.std(ddof=0)) if len(s) > 1 else 0.0
    scale = max(1.4826 * mad, 0.5 * std, 1e-4)
    return center, scale


def _collapse_to_daily(frame: pd.DataFrame, target_col: str) -> pd.DataFrame:
    required = {"datetime", "cycle_phase", "cycle_score", target_col}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"transition overlay missing columns: {sorted(missing)}")

    agg: dict[str, tuple[str, str]] = {
        "cycle_phase": ("cycle_phase", "first"),
        "cycle_score": ("cycle_score", "first"),
        "target": (target_col, "mean"),
    }
    optional_first = [
        "cycle_momentum",
        "cycle_breadth",
        "cycle_volume_ratio",
        "cycle_cross_vol",
        "cycle_breadth_z",
        "cycle_market_return_z",
        "cycle_volume_ratio_z",
        "cycle_cross_vol_z",
        "cycle_confidence",
    ]
    for col in optional_first:
        if col in frame.columns:
            agg[col] = (col, "first")

    d = frame.copy()
    d["datetime"] = pd.to_datetime(d["datetime"])
    daily = d.groupby("datetime", as_index=False).agg(**agg).sort_values("datetime")
    daily["prev_phase"] = daily["cycle_phase"].shift(1)
    daily["transition"] = daily["prev_phase"].fillna("NA").astype(str) + "→" + daily["cycle_phase"].astype(str)
    daily["delta_cycle_score"] = daily["cycle_score"].diff()
    daily["cycle_score_slope3"] = daily["cycle_score"].diff(3) / 3.0
    if "cycle_breadth" in daily.columns:
        daily["delta_cycle_breadth"] = daily["cycle_breadth"].diff()
        daily["cycle_breadth_slope3"] = daily["cycle_breadth"].diff(3) / 3.0
    if "cycle_volume_ratio" in daily.columns:
        daily["delta_cycle_volume_ratio"] = daily["cycle_volume_ratio"].diff()
    if "cycle_cross_vol" in daily.columns:
        daily["delta_cycle_cross_vol"] = daily["cycle_cross_vol"].diff()
    return daily


def build_causal_transition_overlay(
    frame: pd.DataFrame,
    target_col: str = "future_return_5",
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build a causal, annual walk-forward sentiment-transition exposure overlay.

    Design principles:
    1. Collapse stock rows to one market-state row per date so dates with more symbols do not
       receive larger statistical weight.
    2. For each test year, fit only on prior calendar years and purge the final training dates,
       so overlapping forward-return labels near the year boundary cannot leak into the test year.
    3. Estimate transition edge with hierarchical shrinkage: transition -> current phase -> global.
       Sparse state pairs therefore stay close to the better-supported phase estimate.
    4. Convert the learned edge to a continuous exposure multiplier with a robust tanh mapping.

    Continuous cycle deltas/slopes are exported as diagnostics. In V7.7 they are deliberately not
    forced into the position formula because the V7.6 diagnostic sample showed weak/non-monotonic
    standalone predictive relationships. They remain available for the next evidence-driven step.
    """
    cfg = cfg or {}
    c = TransitionOverlayConfig(
        base_exposure=float(cfg.get("base_exposure", 0.78)),
        sensitivity=float(cfg.get("sensitivity", 0.12)),
        min_exposure=float(cfg.get("min_exposure", 0.55)),
        max_exposure=float(cfg.get("max_exposure", 1.00)),
        phase_prior_strength=float(cfg.get("phase_prior_strength", 30.0)),
        transition_prior_strength=float(cfg.get("transition_prior_strength", 20.0)),
        purge_dates=int(cfg.get("purge_dates", 7)),
        min_train_dates=int(cfg.get("min_train_dates", 120)),
    )
    daily = _collapse_to_daily(frame, target_col)
    daily["year"] = daily["datetime"].dt.year
    parts: list[pd.DataFrame] = []

    for year in sorted(daily["year"].dropna().unique()):
        test = daily.loc[daily["year"] == year].copy()
        train = daily.loc[daily["year"] < year].copy()

        if len(train) > c.purge_dates:
            train_dates = sorted(train["datetime"].dropna().unique())
            cutoff = train_dates[-c.purge_dates]
            train = train.loc[train["datetime"] < cutoff].copy()

        if len(train) < c.min_train_dates:
            test["transition_edge_return"] = 0.0
            test["transition_edge_score"] = 0.0
            test["transition_observations"] = 0.0
            test["transition_confidence"] = 0.0
            test["transition_cycle_multiplier"] = c.base_exposure
            test["transition_train_dates"] = int(len(train))
            parts.append(test)
            continue

        train["target"] = pd.to_numeric(train["target"], errors="coerce")
        train = train.dropna(subset=["target"])
        global_mean = float(train["target"].mean()) if len(train) else 0.0

        phase_stats = train.groupby("cycle_phase")["target"].agg(["mean", "count"])
        phase_stats["edge"] = (
            phase_stats["count"] * phase_stats["mean"] + c.phase_prior_strength * global_mean
        ) / (phase_stats["count"] + c.phase_prior_strength)
        phase_edge = phase_stats["edge"].to_dict()

        trans_stats = train.groupby("transition")["target"].agg(["mean", "count"])
        transition_edge: dict[str, float] = {}
        for transition, row in trans_stats.iterrows():
            current_phase = str(transition).split("→")[-1]
            prior = float(phase_edge.get(current_phase, global_mean))
            transition_edge[str(transition)] = float(
                (float(row["count"]) * float(row["mean"]) + c.transition_prior_strength * prior)
                / (float(row["count"]) + c.transition_prior_strength)
            )

        train_pred = train["transition"].map(transition_edge)
        train_pred = train_pred.fillna(train["cycle_phase"].map(phase_edge)).fillna(global_mean)
        center, scale = _robust_scale(train_pred)

        test["transition_edge_return"] = test["transition"].map(transition_edge)
        test["transition_edge_return"] = (
            test["transition_edge_return"]
            .fillna(test["cycle_phase"].map(phase_edge))
            .fillna(global_mean)
        )
        count_map = trans_stats["count"].to_dict()
        test["transition_observations"] = test["transition"].map(count_map).fillna(0.0).astype(float)
        test["transition_confidence"] = (
            test["transition_observations"]
            / (test["transition_observations"] + c.transition_prior_strength)
        ).clip(0.0, 1.0)
        z = ((test["transition_edge_return"] - center) / scale).clip(-3.0, 3.0)
        test["transition_edge_score"] = np.tanh(z)
        test["transition_cycle_multiplier"] = (
            c.base_exposure + c.sensitivity * test["transition_edge_score"]
        ).clip(c.min_exposure, c.max_exposure)
        test["transition_train_dates"] = int(len(train))
        parts.append(test)

    out = pd.concat(parts, ignore_index=True).sort_values("datetime")
    keep = [
        "datetime",
        "prev_phase",
        "transition",
        "delta_cycle_score",
        "cycle_score_slope3",
        "delta_cycle_breadth",
        "cycle_breadth_slope3",
        "delta_cycle_volume_ratio",
        "delta_cycle_cross_vol",
        "transition_edge_return",
        "transition_edge_score",
        "transition_observations",
        "transition_confidence",
        "transition_cycle_multiplier",
        "transition_train_dates",
    ]
    return out[[c for c in keep if c in out.columns]].copy()


def transition_summary(
    frame: pd.DataFrame,
    target_col: str = "future_return_5",
    end_before: str | pd.Timestamp = "2025-01-01",
) -> pd.DataFrame:
    """Descriptive transition table for the development sample only."""
    daily = _collapse_to_daily(frame, target_col)
    cutoff = pd.Timestamp(end_before)
    d = daily.loc[daily["datetime"] < cutoff].copy()
    return (
        d.groupby("transition", dropna=False)
        .agg(
            dates=("datetime", "count"),
            mean_forward_5d=("target", "mean"),
            median_forward_5d=("target", "median"),
            positive_5d_rate=("target", lambda x: float((pd.to_numeric(x, errors="coerce") > 0).mean())),
            mean_delta_cycle_score=("delta_cycle_score", "mean"),
            mean_cycle_score_slope3=("cycle_score_slope3", "mean"),
        )
        .reset_index()
        .sort_values(["dates", "mean_forward_5d"], ascending=[False, False])
    )
