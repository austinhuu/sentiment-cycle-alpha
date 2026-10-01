from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def daily_cross_sectional_rank_ic(
    data: pd.DataFrame,
    score_column: str,
    target_column: str,
    minimum_symbols: int = 5,
) -> pd.DataFrame:
    """逐日计算横截面Spearman Rank IC。

    先在每个交易日内计算秩，再用向量化Pearson相关公式，
    避免逐日调用DataFrame.corr造成长时间运行。
    """

    required = [
        "datetime",
        "symbol",
        score_column,
        target_column,
    ]
    missing = [
        column
        for column in required
        if column not in data.columns
    ]
    if missing:
        raise ValueError(f"Rank IC计算缺少字段：{missing}")

    frame = data[required].copy()
    frame["datetime"] = pd.to_datetime(
        frame["datetime"],
        errors="coerce",
    )
    frame[score_column] = pd.to_numeric(
        frame[score_column],
        errors="coerce",
    )
    frame[target_column] = pd.to_numeric(
        frame[target_column],
        errors="coerce",
    )
    frame = frame.dropna()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "datetime",
                "score",
                "target",
                "rank_ic",
                "symbol_count",
            ]
        )

    grouped = frame.groupby("datetime", sort=True)
    frame["symbol_count"] = grouped["symbol"].transform("count")
    frame["score_unique"] = grouped[score_column].transform("nunique")
    frame["target_unique"] = grouped[target_column].transform("nunique")
    frame = frame.loc[
        frame["symbol_count"].ge(minimum_symbols)
        & frame["score_unique"].ge(3)
        & frame["target_unique"].ge(3)
    ].copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "datetime",
                "score",
                "target",
                "rank_ic",
                "symbol_count",
            ]
        )

    frame["score_rank"] = frame.groupby("datetime")[score_column].rank(
        method="average"
    )
    frame["target_rank"] = frame.groupby("datetime")[target_column].rank(
        method="average"
    )
    frame["score_centered"] = (
        frame["score_rank"]
        - frame.groupby("datetime")["score_rank"].transform("mean")
    )
    frame["target_centered"] = (
        frame["target_rank"]
        - frame.groupby("datetime")["target_rank"].transform("mean")
    )
    frame["cross"] = frame["score_centered"] * frame["target_centered"]
    frame["score_square"] = frame["score_centered"] ** 2
    frame["target_square"] = frame["target_centered"] ** 2

    aggregated = frame.groupby("datetime", as_index=False).agg(
        numerator=("cross", "sum"),
        score_sum_square=("score_square", "sum"),
        target_sum_square=("target_square", "sum"),
        symbol_count=("symbol", "count"),
    )
    denominator = np.sqrt(
        aggregated["score_sum_square"]
        * aggregated["target_sum_square"]
    )
    aggregated["rank_ic"] = aggregated["numerator"] / denominator.replace(
        0.0, np.nan
    )
    aggregated = aggregated.dropna(subset=["rank_ic"])
    aggregated["score"] = score_column
    aggregated["target"] = target_column
    return aggregated[
        ["datetime", "score", "target", "rank_ic", "symbol_count"]
    ].reset_index(drop=True)


def summarize_rank_ic(
    daily_ic: pd.DataFrame,
    annualization: int = 252,
) -> dict[str, Any]:
    if daily_ic.empty:
        return {
            "observation_count": 0,
            "mean_ic": np.nan,
            "std_ic": np.nan,
            "icir": np.nan,
            "positive_ratio": np.nan,
            "median_ic": np.nan,
        }

    values = pd.to_numeric(
        daily_ic["rank_ic"],
        errors="coerce",
    ).dropna()

    if values.empty:
        return {
            "observation_count": 0,
            "mean_ic": np.nan,
            "std_ic": np.nan,
            "icir": np.nan,
            "positive_ratio": np.nan,
            "median_ic": np.nan,
        }

    mean_ic = float(values.mean())
    std_ic = float(values.std(ddof=1))
    icir = (
        mean_ic / std_ic * math.sqrt(annualization)
        if std_ic > 0
        else np.nan
    )

    return {
        "observation_count": int(len(values)),
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "icir": float(icir),
        "positive_ratio": float(
            (values > 0).mean()
        ),
        "median_ic": float(values.median()),
    }


def build_rank_ic_outputs(
    data: pd.DataFrame,
    score_columns: list[str],
    target_columns: list[str],
    minimum_symbols: int = 5,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """一次生成每日、总体、年度和个股Rank IC结果。"""

    daily_frames: list[pd.DataFrame] = []
    summary_records: list[dict[str, Any]] = []
    yearly_records: list[dict[str, Any]] = []
    symbol_records: list[dict[str, Any]] = []

    for score_column in score_columns:
        if score_column not in data.columns:
            continue

        for target_column in target_columns:
            if target_column not in data.columns:
                continue

            daily = daily_cross_sectional_rank_ic(
                data,
                score_column,
                target_column,
                minimum_symbols=minimum_symbols,
            )

            if not daily.empty:
                daily_frames.append(daily)

            summary = summarize_rank_ic(daily)
            summary_records.append(
                {
                    "score": score_column,
                    "target": target_column,
                    **summary,
                }
            )

            if not daily.empty:
                yearly = daily.copy()
                yearly["year"] = pd.to_datetime(
                    yearly["datetime"]
                ).dt.year

                for year, group in yearly.groupby(
                    "year"
                ):
                    year_summary = summarize_rank_ic(
                        group
                    )
                    yearly_records.append(
                        {
                            "score": score_column,
                            "target": target_column,
                            "year": int(year),
                            **year_summary,
                        }
                    )

            symbol_frame = data[
                [
                    "symbol",
                    score_column,
                    target_column,
                ]
            ].copy()
            symbol_frame[score_column] = pd.to_numeric(
                symbol_frame[score_column],
                errors="coerce",
            )
            symbol_frame[target_column] = pd.to_numeric(
                symbol_frame[target_column],
                errors="coerce",
            )

            for symbol, group in symbol_frame.groupby(
                "symbol"
            ):
                valid = group[
                    [score_column, target_column]
                ].dropna()

                if len(valid) < 30:
                    continue
                if valid[score_column].nunique() < 3:
                    continue
                if valid[target_column].nunique() < 3:
                    continue

                time_series_ic = valid.corr(
                    method="spearman"
                ).iloc[0, 1]

                if not math.isfinite(
                    float(time_series_ic)
                ):
                    continue

                symbol_records.append(
                    {
                        "symbol": symbol,
                        "score": score_column,
                        "target": target_column,
                        "time_series_rank_ic": float(
                            time_series_ic
                        ),
                        "observation_count": int(
                            len(valid)
                        ),
                    }
                )

    daily_output = (
        pd.concat(
            daily_frames,
            ignore_index=True,
        )
        if daily_frames
        else pd.DataFrame(
            columns=[
                "datetime",
                "score",
                "target",
                "rank_ic",
                "symbol_count",
            ]
        )
    )

    return (
        daily_output,
        pd.DataFrame(summary_records),
        pd.DataFrame(yearly_records),
        pd.DataFrame(symbol_records),
    )