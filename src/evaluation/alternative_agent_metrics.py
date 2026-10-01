from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _corr(first: pd.Series, second: pd.Series, method: str) -> float:
    valid = pd.concat(
        [
            pd.to_numeric(first, errors="coerce"),
            pd.to_numeric(second, errors="coerce"),
        ],
        axis=1,
    ).dropna()
    if len(valid) < 30 or valid.iloc[:, 0].nunique() < 2:
        return float("nan")
    return float(valid.iloc[:, 0].corr(valid.iloc[:, 1], method=method))


def alternative_agent_summary(
    decisions: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    specifications = [
        (
            "fundamental",
            "fundamental_score",
            "fundamental_confidence",
            "fundamental_age_days",
        ),
        (
            "news",
            "news_score",
            "news_confidence",
            "news_age_days",
        ),
    ]
    for agent, score_column, confidence_column, age_column in specifications:
        if score_column not in decisions.columns:
            continue
        score = pd.to_numeric(decisions[score_column], errors="coerce")
        confidence = pd.to_numeric(
            decisions.get(confidence_column, 0.0), errors="coerce"
        ).fillna(0.0)
        active = confidence > 0
        coverage = float(active.mean())
        age = pd.to_numeric(
            decisions.get(age_column, np.nan), errors="coerce"
        )
        for horizon in horizons:
            target = f"future_return_{horizon}"
            if target not in decisions.columns:
                continue
            records.append(
                {
                    "agent": agent,
                    "horizon": horizon,
                    "coverage_ratio": coverage,
                    "active_rows": int(active.sum()),
                    "mean_confidence": float(confidence[active].mean())
                    if active.any()
                    else np.nan,
                    "median_age_days": float(age[active].median())
                    if active.any()
                    else np.nan,
                    "pearson_ic": _corr(
                        score[active], decisions.loc[active, target], "pearson"
                    ),
                    "spearman_ic": _corr(
                        score[active], decisions.loc[active, target], "spearman"
                    ),
                }
            )
    return pd.DataFrame(records)


def score_bucket_summary(
    decisions: pd.DataFrame,
    score_column: str,
    confidence_column: str,
    horizon: int = 5,
) -> pd.DataFrame:
    target = f"future_return_{horizon}"
    required = [score_column, confidence_column, target]
    if any(column not in decisions.columns for column in required):
        return pd.DataFrame()

    frame = decisions[required].copy()
    frame[score_column] = pd.to_numeric(frame[score_column], errors="coerce")
    frame[confidence_column] = pd.to_numeric(
        frame[confidence_column], errors="coerce"
    )
    frame[target] = pd.to_numeric(frame[target], errors="coerce")
    frame = frame.dropna()
    frame = frame.loc[frame[confidence_column] > 0]
    if frame.empty:
        return pd.DataFrame()

    frame["score_bucket"] = pd.cut(
        frame[score_column],
        bins=[-1.01, -0.20, -0.05, 0.05, 0.20, 1.01],
        labels=["strong_negative", "negative", "neutral", "positive", "strong_positive"],
        include_lowest=True,
    )
    return (
        frame.groupby("score_bucket", observed=False)
        .agg(
            count=(target, "count"),
            mean_score=(score_column, "mean"),
            mean_confidence=(confidence_column, "mean"),
            mean_future_return=(target, "mean"),
            median_future_return=(target, "median"),
            positive_ratio=(target, lambda values: float((values > 0).mean())),
        )
        .reset_index()
    )