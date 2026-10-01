from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _spearman(group: pd.DataFrame, score: str, target: str) -> float:
    if score not in group.columns or target not in group.columns:
        return np.nan
    x = pd.to_numeric(group[score], errors="coerce")
    y = pd.to_numeric(group[target], errors="coerce")
    valid = x.notna() & y.notna()
    if valid.sum() < 3 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
        return np.nan
    value = x[valid].corr(y[valid], method="spearman")
    return float(value) if pd.notna(value) else np.nan


def daily_rank_ic(
    decisions: pd.DataFrame,
    score_columns: list[str],
    target_column: str,
) -> pd.DataFrame:
    frame = decisions.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    records: list[dict[str, Any]] = []
    for date, group in frame.groupby("datetime"):
        for score in score_columns:
            records.append(
                {
                    "datetime": date,
                    "score_column": score,
                    "rank_ic": _spearman(group, score, target_column),
                    "symbol_count": int(len(group)),
                }
            )
    return pd.DataFrame(records)


def summarize_rank_ic(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for score, group in daily.groupby("score_column"):
        values = pd.to_numeric(group["rank_ic"], errors="coerce").dropna()
        mean = float(values.mean()) if len(values) else np.nan
        std = float(values.std(ddof=0)) if len(values) else np.nan
        rows.append(
            {
                "score_column": score,
                "dates": int(len(values)),
                "mean_rank_ic": mean,
                "median_rank_ic": float(values.median()) if len(values) else np.nan,
                "rank_ic_std": std,
                "rank_ic_ir": mean / std if len(values) and std > 1e-12 else np.nan,
                "positive_ic_ratio": float((values > 0).mean()) if len(values) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_rank_ic", ascending=False)


def event_forward_performance(decisions: pd.DataFrame, target_column: str) -> pd.DataFrame:
    frame = decisions.copy()
    if "dominant_event_type" not in frame.columns:
        return pd.DataFrame()
    frame[target_column] = pd.to_numeric(frame[target_column], errors="coerce")
    frame["news_score"] = pd.to_numeric(frame.get("news_score"), errors="coerce")
    frame["news_active"] = pd.to_numeric(frame.get("news_active"), errors="coerce").fillna(0)
    active = frame.loc[frame["news_active"].gt(0)].copy()
    if active.empty:
        return pd.DataFrame()
    return active.groupby("dominant_event_type", as_index=False).agg(
        samples=(target_column, "count"),
        mean_forward_return=(target_column, "mean"),
        median_forward_return=(target_column, "median"),
        positive_ratio=(target_column, lambda s: float((s > 0).mean())),
        mean_news_score=("news_score", "mean"),
    ).sort_values("samples", ascending=False)


def model_mode_summary(decisions: pd.DataFrame, target_column: str) -> pd.DataFrame:
    if "fusion_model_mode" not in decisions.columns:
        return pd.DataFrame()
    frame = decisions.copy()
    frame[target_column] = pd.to_numeric(frame.get(target_column), errors="coerce")
    frame["adaptive_score"] = pd.to_numeric(frame.get("adaptive_score"), errors="coerce")
    frame["adaptive_confidence"] = pd.to_numeric(
        frame.get("adaptive_confidence"), errors="coerce"
    )
    rows: list[dict[str, Any]] = []
    for mode, group in frame.groupby("fusion_model_mode", dropna=False):
        rows.append(
            {
                "fusion_model_mode": str(mode),
                "rows": int(len(group)),
                "dates": int(pd.to_datetime(group["datetime"], errors="coerce").nunique()),
                "mean_forward_return": float(group[target_column].mean()),
                "mean_adaptive_score": float(group["adaptive_score"].mean()),
                "mean_adaptive_confidence": float(group["adaptive_confidence"].mean()),
                "fallback_share": float(
                    pd.to_numeric(group.get("fusion_fallback_used"), errors="coerce")
                    .fillna(0)
                    .gt(0)
                    .mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def data_integrity_summary(decisions: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        "rows": int(len(decisions)),
        "symbols": int(decisions["symbol"].nunique()) if "symbol" in decisions else 0,
    }
    for column, output in [
        ("fundamental_active", "fundamental_active_coverage"),
        ("annual_fundamental_active", "annual_fundamental_active_coverage"),
        ("quarterly_fundamental_active", "quarterly_fundamental_active_coverage"),
        ("news_active", "news_active_coverage"),
    ]:
        if column in decisions:
            active = pd.to_numeric(decisions[column], errors="coerce").fillna(0)
            result[output] = float(active.gt(0).mean())

    if "annual_report_period" in decisions:
        annual_period = pd.to_datetime(decisions["annual_report_period"], errors="coerce")
        result["annual_report_period_coverage"] = float(annual_period.notna().mean())
        result["annual_report_rows"] = int(annual_period.dt.month.eq(12).sum())
    if "quarterly_report_period" in decisions:
        quarterly_period = pd.to_datetime(decisions["quarterly_report_period"], errors="coerce")
        result["quarterly_report_period_coverage"] = float(quarterly_period.notna().mean())
    if {"news_score", "news_confidence"}.issubset(decisions.columns):
        score = pd.to_numeric(decisions["news_score"], errors="coerce").fillna(0.0)
        confidence = pd.to_numeric(decisions["news_confidence"], errors="coerce").fillna(0.0)
        result["neutral_news_with_nonzero_confidence"] = int(
            (score.abs().lt(0.03) & confidence.gt(0)).sum()
        )
    if "availability_lag_days" in decisions:
        lag = pd.to_numeric(decisions["availability_lag_days"], errors="coerce").dropna()
        result["fundamental_availability_lag_median"] = float(lag.median()) if len(lag) else np.nan
        result["fundamental_availability_lag_max"] = float(lag.max()) if len(lag) else np.nan
    if "fusion_fallback_used" in decisions:
        fallback = pd.to_numeric(decisions["fusion_fallback_used"], errors="coerce").fillna(0)
        result["fusion_fallback_row_share"] = float(fallback.gt(0).mean())
        if "test_year" in decisions:
            result["fusion_fallback_years"] = sorted(
                int(value)
                for value in pd.to_numeric(
                    decisions.loc[fallback.gt(0), "test_year"], errors="coerce"
                ).dropna().unique()
            )
    return result


def build_v4_evaluation(
    decisions: pd.DataFrame,
    coefficients: pd.DataFrame,
    portfolio_metrics: dict[str, Any],
    target_horizon: int,
) -> dict[str, Any]:
    target = f"future_return_{target_horizon}"
    score_columns = [
        "technical_score",
        "sentiment_score",
        "fundamental_score",
        "annual_fundamental_score",
        "quarterly_fundamental_score",
        "news_score",
        "adaptive_score",
        "predicted_excess_return",
    ]
    score_columns = [column for column in score_columns if column in decisions]
    daily = daily_rank_ic(decisions, score_columns, target)
    summary = summarize_rank_ic(daily)
    event = event_forward_performance(decisions, target)
    modes = model_mode_summary(decisions, target)

    coefficient_summary = pd.DataFrame()
    if not coefficients.empty:
        coefficient_summary = coefficients.groupby("feature", as_index=False).agg(
            mean_coefficient=("standardized_coefficient", "mean"),
            coefficient_std=("standardized_coefficient", "std"),
            mean_absolute_coefficient=("absolute_coefficient", "mean"),
            folds=("test_year", "nunique"),
        ).sort_values("mean_absolute_coefficient", ascending=False)

    integrity = data_integrity_summary(decisions)
    adaptive_row = summary.loc[summary["score_column"].eq("predicted_excess_return")]
    adaptive_ic = (
        float(adaptive_row["mean_rank_ic"].iloc[0]) if not adaptive_row.empty else np.nan
    )
    overall = {
        "portfolio_metrics": portfolio_metrics,
        "data_integrity": integrity,
        "adaptive_oos_mean_rank_ic": adaptive_ic,
        "coefficient_feature_count": int(coefficients["feature"].nunique()) if not coefficients.empty else 0,
        "evaluation_status": (
            "promising" if pd.notna(adaptive_ic) and adaptive_ic > 0.02 else "weak_or_unproven"
        ),
    }
    return {
        "daily_rank_ic": daily,
        "rank_ic_summary": summary,
        "event_performance": event,
        "model_mode_summary": modes,
        "coefficient_summary": coefficient_summary,
        "overall": overall,
    }


def save_v4_evaluation(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for key in [
        "daily_rank_ic",
        "rank_ic_summary",
        "event_performance",
        "model_mode_summary",
        "coefficient_summary",
    ]:
        result[key].to_csv(
            output_dir / f"{key}.csv", index=False, encoding="utf-8-sig"
        )
    (output_dir / "v4_6_evaluation_summary.json").write_text(
        json.dumps(result["overall"], ensure_ascii=False, indent=2), encoding="utf-8"
    )