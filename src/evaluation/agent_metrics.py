from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _correlation(
    first: pd.Series,
    second: pd.Series,
    method: str = "pearson",
    minimum_observations: int = 20,
) -> float:
    valid = pd.concat(
        [
            pd.to_numeric(first, errors="coerce"),
            pd.to_numeric(second, errors="coerce"),
        ],
        axis=1,
    ).dropna()

    if len(valid) < minimum_observations:
        return float("nan")
    if valid.iloc[:, 0].nunique() < 2:
        return float("nan")
    if valid.iloc[:, 1].nunique() < 2:
        return float("nan")

    return float(
        valid.iloc[:, 0].corr(
            valid.iloc[:, 1],
            method=method,
        )
    )


def _boolean_series(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(
            False,
            index=frame.index,
            dtype=bool,
        )

    values = frame[column]

    if values.dtype == bool:
        return values.fillna(False)

    return (
        values.astype(str)
        .str.strip()
        .str.lower()
        .isin(
            {
                "1",
                "true",
                "yes",
                "y",
                "t",
                "on",
            }
        )
    )


def score_ic_table(
    decisions: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    """总体混合IC，仅作为辅助诊断。

    正式横截面评价仍以rank_ic系列输出为准。
    """

    score_columns = [
        "technical_score",
        "sentiment_score",
        "debate_revision_delta",
        "base_score",
        "final_score",
    ]
    records: list[dict[str, Any]] = []

    for horizon in horizons:
        target = f"future_return_{horizon}"

        if target not in decisions.columns:
            continue

        for score in score_columns:
            if score not in decisions.columns:
                continue

            records.append(
                {
                    "score": score,
                    "horizon": horizon,
                    "pearson_ic": _correlation(
                        decisions[score],
                        decisions[target],
                        method="pearson",
                    ),
                    "spearman_ic": _correlation(
                        decisions[score],
                        decisions[target],
                        method="spearman",
                    ),
                    "observations": int(
                        decisions[
                            [score, target]
                        ].dropna().shape[0]
                    ),
                }
            )

    return pd.DataFrame(records)


def signal_forward_summary(
    decisions: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for signal, group in decisions.groupby(
        "signal"
    ):
        for horizon in horizons:
            target = f"future_return_{horizon}"

            if target not in group.columns:
                continue

            values = pd.to_numeric(
                group[target],
                errors="coerce",
            ).dropna()

            records.append(
                {
                    "signal": signal,
                    "horizon": horizon,
                    "count": int(len(values)),
                    "mean_future_return": (
                        float(values.mean())
                        if len(values)
                        else np.nan
                    ),
                    "median_future_return": (
                        float(values.median())
                        if len(values)
                        else np.nan
                    ),
                    "positive_ratio": (
                        float((values > 0).mean())
                        if len(values)
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(records)


def confidence_calibration(
    decisions: pd.DataFrame,
    horizon: int = 5,
) -> pd.DataFrame:
    target = f"future_return_{horizon}"

    if target not in decisions.columns:
        return pd.DataFrame()

    frame = decisions[
        ["confidence", "signal", target]
    ].copy().dropna()
    frame = frame.loc[
        frame["signal"] == "BUY"
    ].copy()

    if frame.empty:
        return pd.DataFrame(
            columns=[
                "confidence_bin",
                "count",
                "mean_confidence",
                "positive_ratio",
                "mean_future_return",
                "median_future_return",
            ]
        )

    bins = [
        0.0,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        1.01,
    ]

    frame["confidence_bin"] = pd.cut(
        frame["confidence"],
        bins=bins,
        right=False,
        include_lowest=True,
    )

    result = frame.groupby(
        "confidence_bin",
        observed=False,
    ).agg(
        count=(target, "count"),
        mean_confidence=(
            "confidence",
            "mean",
        ),
        positive_ratio=(
            target,
            lambda values: float(
                (values > 0).mean()
            )
            if len(values)
            else np.nan,
        ),
        mean_future_return=(target, "mean"),
        median_future_return=(
            target,
            "median",
        ),
    ).reset_index()

    result["confidence_bin"] = (
        result["confidence_bin"].astype(str)
    )

    return result


def risk_action_summary(
    decisions: pd.DataFrame,
    horizons: list[int],
    symbol_equity: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """评价原始和真正生效的风险动作。

    raw_veto/raw_force_exit表示风险字段被触发；
    effective_veto/effective_force_exit表示该动作确实改变了仓位。
    """

    frame = (
        symbol_equity.copy()
        if symbol_equity is not None
        and not symbol_equity.empty
        else decisions.copy()
    )

    if frame.empty:
        return pd.DataFrame()

    frame["datetime"] = pd.to_datetime(
        frame["datetime"],
        errors="coerce",
    )
    frame = frame.sort_values(
        ["symbol", "datetime"]
    ).reset_index(drop=True)

    raw_veto = _boolean_series(
        frame,
        "risk_veto_new_entry",
    )
    raw_force = _boolean_series(
        frame,
        "risk_force_exit",
    )
    pre_risk_buy = _boolean_series(
        frame,
        "pre_risk_buy_condition",
    )

    executed = pd.to_numeric(
        frame.get(
            "executed_position",
            pd.Series(0.0, index=frame.index),
        ),
        errors="coerce",
    ).fillna(0.0)
    decision_target = pd.to_numeric(
        frame.get(
            "decision_target_position",
            frame.get(
                "target_position",
                pd.Series(0.0, index=frame.index),
            ),
        ),
        errors="coerce",
    ).fillna(0.0)

    frame["raw_veto_new_entry"] = raw_veto
    frame["raw_force_exit"] = raw_force
    frame["effective_veto_new_entry"] = (
        raw_veto
        & pre_risk_buy
        & executed.le(1e-8)
        & decision_target.le(1e-8)
    )
    frame["effective_force_exit"] = (
        raw_force
        & executed.gt(1e-8)
        & decision_target.le(1e-8)
    )

    action_masks = {
        "none": ~(
            raw_veto | raw_force
        ),
        "raw_veto_new_entry": raw_veto,
        "raw_force_exit": raw_force,
        "effective_veto_new_entry": frame[
            "effective_veto_new_entry"
        ],
        "effective_force_exit": frame[
            "effective_force_exit"
        ],
    }

    records: list[dict[str, Any]] = []

    for action, mask in action_masks.items():
        group = frame.loc[mask].copy()

        for horizon in horizons:
            target = f"future_return_{horizon}"

            if target not in group.columns:
                continue

            values = pd.to_numeric(
                group[target],
                errors="coerce",
            ).dropna()

            records.append(
                {
                    "risk_action": action,
                    "horizon": horizon,
                    "count": int(len(values)),
                    "mean_future_return": (
                        float(values.mean())
                        if len(values)
                        else np.nan
                    ),
                    "median_future_return": (
                        float(values.median())
                        if len(values)
                        else np.nan
                    ),
                    "negative_ratio": (
                        float((values < 0).mean())
                        if len(values)
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(records)


def risk_veto_summary(
    decisions: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    """兼容旧调用。"""

    return risk_action_summary(
        decisions,
        horizons,
        symbol_equity=None,
    )


def sentiment_time_series_summary(
    decisions: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    """评价市场情绪的时间序列预测能力。

    情绪得分在同一天对所有股票相同，因此不能做横截面Rank IC。
    这里把每日情绪与股票池未来等权收益进行时间序列相关分析。
    """

    required = {
        "datetime",
        "sentiment_score",
    }

    if not required.issubset(decisions.columns):
        return pd.DataFrame()

    frame = decisions.copy()
    frame["datetime"] = pd.to_datetime(
        frame["datetime"],
        errors="coerce",
    )

    aggregation: dict[str, str] = {
        "sentiment_score": "mean",
    }

    for horizon in horizons:
        target = f"future_return_{horizon}"
        if target in frame.columns:
            aggregation[target] = "mean"

    daily = frame.groupby(
        "datetime",
        as_index=False,
    ).agg(aggregation)

    records: list[dict[str, Any]] = []

    for horizon in horizons:
        target = f"future_return_{horizon}"

        if target not in daily.columns:
            continue

        valid = daily[
            ["sentiment_score", target]
        ].dropna()

        records.append(
            {
                "horizon": horizon,
                "observations": int(len(valid)),
                "pearson_correlation": _correlation(
                    valid["sentiment_score"],
                    valid[target],
                    method="pearson",
                ),
                "spearman_correlation": _correlation(
                    valid["sentiment_score"],
                    valid[target],
                    method="spearman",
                ),
                "positive_sentiment_mean_return": (
                    float(
                        valid.loc[
                            valid["sentiment_score"] > 0,
                            target,
                        ].mean()
                    )
                    if not valid.empty
                    else np.nan
                ),
                "negative_sentiment_mean_return": (
                    float(
                        valid.loc[
                            valid["sentiment_score"] < 0,
                            target,
                        ].mean()
                    )
                    if not valid.empty
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(records)


def debate_diagnostics(
    decisions: pd.DataFrame,
) -> dict[str, float | int]:
    result: dict[str, float | int] = {}

    pairs = [
        ("debate_net_score", "final_score"),
        ("debate_revision_delta", "final_score"),
        (
            "debate_initial_net_score",
            "debate_net_score",
        ),
        ("debate_disagreement", "confidence"),
        ("sentiment_score", "sentiment_multiplier"),
        ("risk_score", "position_multiplier"),
    ]

    for left, right in pairs:
        if (
            left in decisions.columns
            and right in decisions.columns
        ):
            result[
                f"corr_{left}_vs_{right}"
            ] = _correlation(
                decisions[left],
                decisions[right],
            )

    revision = pd.to_numeric(
        decisions.get(
            "debate_revision_delta",
            pd.Series(dtype=float),
        ),
        errors="coerce",
    )
    result["mean_abs_revision_delta"] = float(
        revision.abs().mean()
    )

    result["raw_risk_veto_new_entry_count"] = int(
        _boolean_series(
            decisions,
            "risk_veto_new_entry",
        ).sum()
    )
    result["raw_risk_force_exit_count"] = int(
        _boolean_series(
            decisions,
            "risk_force_exit",
        ).sum()
    )

    if "risk_action" in decisions.columns:
        actions = decisions["risk_action"].astype(str)
        result["risk_action_veto_count"] = int(
            (actions == "veto_new_entry").sum()
        )
        result["risk_action_force_exit_count"] = int(
            (actions == "force_exit").sum()
        )
        result["risk_action_none_count"] = int(
            (actions == "none").sum()
        )

    if "risk_level" in decisions.columns:
        for level, count in (
            decisions["risk_level"]
            .astype(str)
            .value_counts()
            .items()
        ):
            result[f"risk_level_{level}_count"] = int(
                count
            )

    return result


def regime_distribution(decisions: pd.DataFrame) -> pd.DataFrame:
    """统计市场状态的日期数量、样本数量和占比。"""

    if "market_regime" not in decisions.columns:
        return pd.DataFrame(
            columns=[
                "market_regime",
                "decision_rows",
                "trading_days",
                "row_ratio",
                "day_ratio",
                "mean_regime_confidence",
            ]
        )

    frame = decisions.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame["regime_confidence"] = pd.to_numeric(
        frame.get("regime_confidence", np.nan),
        errors="coerce",
    )

    total_rows = max(len(frame), 1)
    total_days = max(frame["datetime"].nunique(), 1)

    result = frame.groupby("market_regime", as_index=False).agg(
        decision_rows=("symbol", "size"),
        trading_days=("datetime", "nunique"),
        mean_regime_confidence=("regime_confidence", "mean"),
    )
    result["row_ratio"] = result["decision_rows"] / total_rows
    result["day_ratio"] = result["trading_days"] / total_days
    return result.sort_values("trading_days", ascending=False).reset_index(drop=True)


def regime_signal_analysis(
    decisions: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    """按市场状态和交易信号统计未来收益。"""

    if "market_regime" not in decisions.columns:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for (regime, signal), group in decisions.groupby(
        ["market_regime", "signal"]
    ):
        for horizon in horizons:
            target = f"future_return_{horizon}"
            if target not in group.columns:
                continue
            values = pd.to_numeric(group[target], errors="coerce").dropna()
            records.append(
                {
                    "market_regime": regime,
                    "signal": signal,
                    "horizon": horizon,
                    "count": int(len(values)),
                    "mean_future_return": (
                        float(values.mean()) if len(values) else np.nan
                    ),
                    "median_future_return": (
                        float(values.median()) if len(values) else np.nan
                    ),
                    "positive_ratio": (
                        float((values > 0).mean()) if len(values) else np.nan
                    ),
                }
            )

    return pd.DataFrame(records)


def regime_transition_matrix(decisions: pd.DataFrame) -> pd.DataFrame:
    """计算日度市场状态转移概率矩阵。"""

    if "market_regime" not in decisions.columns:
        return pd.DataFrame()

    daily = (
        decisions[["datetime", "market_regime"]]
        .dropna()
        .assign(datetime=lambda x: pd.to_datetime(x["datetime"], errors="coerce"))
        .drop_duplicates("datetime")
        .sort_values("datetime")
    )
    daily["next_regime"] = daily["market_regime"].shift(-1)
    valid = daily.dropna(subset=["next_regime"])
    if valid.empty:
        return pd.DataFrame()

    counts = pd.crosstab(valid["market_regime"], valid["next_regime"])
    probabilities = counts.div(counts.sum(axis=1), axis=0).fillna(0.0)

    records: list[dict[str, Any]] = []
    for current in probabilities.index:
        for following in probabilities.columns:
            records.append(
                {
                    "from_regime": current,
                    "to_regime": following,
                    "transition_count": int(counts.loc[current, following]),
                    "transition_probability": float(
                        probabilities.loc[current, following]
                    ),
                }
            )

    return pd.DataFrame(records)


def regime_performance_summary(
    symbol_equity: pd.DataFrame | None,
    annualization: int = 252,
) -> pd.DataFrame:
    """按市场状态统计真实回测日收益和仓位表现。"""

    if symbol_equity is None or symbol_equity.empty:
        return pd.DataFrame()
    if "market_regime" not in symbol_equity.columns:
        return pd.DataFrame()

    frame = symbol_equity.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    return_column = (
        "strategy_return"
        if "strategy_return" in frame.columns
        else "net_strategy_return"
    )

    for column in [
        return_column,
        "executed_position",
        "turnover",
        "cost",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    daily = frame.groupby("datetime", as_index=False).agg(
        market_regime=("market_regime", "first"),
        portfolio_return=(return_column, "sum"),
        gross_exposure=("executed_position", "sum"),
        turnover=("turnover", "sum"),
        transaction_cost=("cost", "sum"),
    )

    records: list[dict[str, Any]] = []
    for regime, group in daily.groupby("market_regime"):
        returns = group["portfolio_return"].dropna()
        if returns.empty:
            continue
        mean_return = float(returns.mean())
        volatility = float(returns.std())
        annualized_return = float((1.0 + mean_return) ** annualization - 1.0)
        annualized_volatility = volatility * np.sqrt(annualization)
        sharpe = (
            mean_return / volatility * np.sqrt(annualization)
            if volatility > 0
            else np.nan
        )
        equity = (1.0 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        records.append(
            {
                "market_regime": regime,
                "trading_days": int(len(group)),
                "conditional_total_return": float(equity.iloc[-1] - 1.0),
                "annualized_return": annualized_return,
                "annualized_volatility": annualized_volatility,
                "sharpe_ratio": float(sharpe),
                "max_drawdown": float(drawdown.min()),
                "positive_day_ratio": float((returns > 0).mean()),
                "average_gross_exposure": float(group["gross_exposure"].mean()),
                "total_turnover": float(group["turnover"].sum()),
                "total_transaction_cost": float(group["transaction_cost"].sum()),
            }
        )

    return pd.DataFrame(records).sort_values(
        "trading_days",
        ascending=False,
    ).reset_index(drop=True)