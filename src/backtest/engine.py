from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0

    return float(
        (
            equity / equity.cummax() - 1.0
        ).min()
    )


def calculate_performance(
    returns: pd.Series,
    equity: pd.Series,
    annualization: int,
    initial_capital: float,
) -> dict[str, float | int]:
    returns = pd.to_numeric(
        returns,
        errors="coerce",
    ).fillna(0.0)

    if returns.empty or equity.empty:
        return {
            "initial_capital": initial_capital,
            "ending_capital": initial_capital,
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "positive_bar_ratio": 0.0,
            "bars": 0,
        }

    total_return = float(
        equity.iloc[-1]
        / initial_capital
        - 1.0
    )

    years = max(
        len(returns) / annualization,
        1.0 / annualization,
    )

    annualized_return = (
        float(
            (1.0 + total_return)
            ** (1.0 / years)
            - 1.0
        )
        if total_return > -1.0
        else -1.0
    )

    mean_return = float(returns.mean())
    standard_deviation = (
        float(returns.std(ddof=1))
        if len(returns) > 1
        else 0.0
    )

    annualized_volatility = (
        standard_deviation
        * math.sqrt(annualization)
    )
    sharpe_ratio = (
        mean_return
        / standard_deviation
        * math.sqrt(annualization)
        if standard_deviation > 0
        else 0.0
    )

    downside = returns.where(
        returns < 0.0,
        0.0,
    )
    downside_deviation = float(
        np.sqrt(downside.pow(2).mean())
    )
    sortino_ratio = (
        mean_return
        / downside_deviation
        * math.sqrt(annualization)
        if downside_deviation > 0
        else 0.0
    )

    max_drawdown = _max_drawdown(equity)
    calmar_ratio = (
        annualized_return / abs(max_drawdown)
        if max_drawdown < 0
        else 0.0
    )

    return {
        "initial_capital": float(initial_capital),
        "ending_capital": float(equity.iloc[-1]),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": float(
            annualized_volatility
        ),
        "sharpe_ratio": float(sharpe_ratio),
        "sortino_ratio": float(sortino_ratio),
        "max_drawdown": float(max_drawdown),
        "calmar_ratio": float(calmar_ratio),
        "positive_bar_ratio": float(
            (returns > 0).mean()
        ),
        "bars": int(len(returns)),
    }


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False

    text = str(value).strip().lower()
    return text in {
        "1",
        "true",
        "yes",
        "y",
        "t",
    }


def _build_stable_targets(
    group: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """根据BUY、SELL、HOLD构造稳定目标仓位。

    规则：
    - HOLD继承已有目标仓位；
    - 普通SELL在满足最短持仓期后退出；
    - 极端负向技术分数可提前退出；
    - 风险强制退出不受最短持仓期限制；
    - 退出后设置冷却期；
    - 小于再平衡带的仓位变化不执行；
    - 较大的仓位调整使用平滑系数。
    """

    g = group.sort_values("datetime").copy()

    minimum_holding_days = max(
        int(config.get(
            "minimum_holding_days",
            3,
        )),
        0,
    )
    cooldown_days = max(
        int(config.get(
            "cooldown_days",
            2,
        )),
        0,
    )
    use_regime_holding_rules = bool(
        config.get("use_regime_holding_rules", False)
    )
    rebalance_band = max(
        float(config.get(
            "rebalance_band",
            0.02,
        )),
        0.0,
    )
    position_smoothing = min(
        max(
            float(config.get(
                "position_smoothing",
                0.50,
            )),
            0.0,
        ),
        1.0,
    )
    hard_exit_score = float(
        config.get(
            "hard_exit_score",
            -0.45,
        )
    )

    current_target = 0.0
    holding_days = 0
    cooldown_remaining = 0

    targets: list[float] = []
    holding_day_values: list[int] = []
    cooldown_values: list[int] = []
    execution_reasons: list[str] = []
    applied_minimum_holding: list[int] = []
    applied_cooldown: list[int] = []

    for row in g.itertuples(index=False):
        signal = str(
            getattr(row, "signal", "HOLD")
        ).strip().upper()
        proposed_target = max(
            _safe_float(
                getattr(
                    row,
                    "target_position",
                    0.0,
                ),
                0.0,
            ),
            0.0,
        )
        final_score = _safe_float(
            getattr(row, "final_score", 0.0),
            0.0,
        )
        row_minimum_holding_days = minimum_holding_days
        row_cooldown_days = cooldown_days

        if use_regime_holding_rules:
            row_minimum_holding_days = max(
                int(_safe_float(
                    getattr(
                        row,
                        "regime_minimum_holding_days",
                        minimum_holding_days,
                    ),
                    float(minimum_holding_days),
                )),
                0,
            )
            row_cooldown_days = max(
                int(_safe_float(
                    getattr(
                        row,
                        "regime_cooldown_days",
                        cooldown_days,
                    ),
                    float(cooldown_days),
                )),
                0,
            )

        force_exit = _safe_bool(
            getattr(
                row,
                "risk_force_exit",
                False,
            )
        ) or (
            str(
                getattr(
                    row,
                    "risk_action",
                    "none",
                )
            ).strip().lower()
            == "force_exit"
        )

        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        if current_target > 0.0:
            holding_days += 1
        else:
            holding_days = 0

        new_target = current_target
        reason = "hold_previous_target"

        if force_exit:
            new_target = 0.0
            holding_days = 0
            cooldown_remaining = row_cooldown_days
            reason = "risk_force_exit"

        elif current_target <= 0.0:
            if (
                signal == "BUY"
                and cooldown_remaining == 0
                and proposed_target > 0.0
            ):
                new_target = proposed_target
                holding_days = 1
                reason = "new_entry"
            else:
                new_target = 0.0
                reason = (
                    "cooldown"
                    if cooldown_remaining > 0
                    else "flat"
                )

        elif signal == "SELL":
            can_exit = (
                holding_days
                >= row_minimum_holding_days
                or final_score
                <= hard_exit_score
            )

            if can_exit:
                new_target = 0.0
                holding_days = 0
                cooldown_remaining = row_cooldown_days
                reason = "normal_exit"
            else:
                new_target = current_target
                reason = "minimum_holding_lock"

        elif signal == "BUY":
            difference = (
                proposed_target - current_target
            )

            if abs(difference) < rebalance_band:
                new_target = current_target
                reason = "inside_rebalance_band"
            else:
                new_target = (
                    current_target
                    + position_smoothing
                    * difference
                )
                reason = "smoothed_rebalance"

        else:
            # HOLD明确表示不改变当前仓位。
            new_target = current_target
            reason = "hold_previous_target"

        current_target = min(
            max(new_target, 0.0),
            1.0,
        )

        targets.append(current_target)
        holding_day_values.append(holding_days)
        cooldown_values.append(cooldown_remaining)
        execution_reasons.append(reason)
        applied_minimum_holding.append(row_minimum_holding_days)
        applied_cooldown.append(row_cooldown_days)

    g["decision_target_position"] = targets
    g["holding_days"] = holding_day_values
    g["cooldown_remaining"] = cooldown_values
    g["target_update_reason"] = execution_reasons
    g["applied_minimum_holding_days"] = applied_minimum_holding
    g["applied_cooldown_days"] = applied_cooldown

    return g


def _build_symbol_intervals(
    decisions: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for _, group in decisions.groupby(
        "symbol",
        sort=True,
    ):
        g = _build_stable_targets(
            group,
            config,
        )

        # t日收盘形成信号，t+1日开盘执行；
        # t日这一行的资产收益是t日开盘到t+1日开盘。
        # shift(1)后，t行仓位来自t-1日收盘决策。
        g["raw_executed_position"] = (
            g["decision_target_position"]
            .shift(1)
            .fillna(0.0)
        )
        g["asset_return"] = (
            g["signal_open"].shift(-1)
            / g["signal_open"]
            - 1.0
        )

        g = g.loc[
            g["asset_return"].notna()
        ].copy()
        frames.append(g)

    if not frames:
        raise RuntimeError(
            "没有可用于回测的决策记录。"
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def run_long_only_backtest(
    decisions: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    required = [
        "datetime",
        "symbol",
        "signal",
        "target_position",
        "signal_open",
    ]
    missing = [
        column
        for column in required
        if column not in decisions.columns
    ]

    if missing:
        raise ValueError(
            f"回测输入缺少字段：{missing}"
        )

    initial_capital = float(
        config.get(
            "initial_capital",
            1_000_000,
        )
    )
    annualization = int(
        config.get(
            "annualization_bars",
            252,
        )
    )
    max_gross_exposure = float(
        config.get(
            "max_gross_exposure",
            1.0,
        )
    )
    buy_commission = float(
        config.get(
            "buy_commission_rate",
            config.get(
                "commission_rate",
                0.0001,
            ),
        )
    )
    sell_commission = float(
        config.get(
            "sell_commission_rate",
            config.get(
                "commission_rate",
                0.0001,
            ),
        )
    )
    stamp_duty = float(
        config.get(
            "stamp_duty_rate",
            0.0005,
        )
    )
    slippage = float(
        config.get(
            "slippage_rate",
            0.0005,
        )
    )

    frame = decisions.copy()
    frame["datetime"] = pd.to_datetime(
        frame["datetime"],
        errors="coerce",
    )

    for column in [
        "target_position",
        "signal_open",
        "final_score",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

    frame = frame.dropna(
        subset=[
            "datetime",
            "symbol",
            "signal_open",
        ]
    )

    symbol_equity = _build_symbol_intervals(
        frame,
        config,
    )

    gross_by_date = symbol_equity.groupby(
        "datetime"
    )["raw_executed_position"].transform(
        "sum"
    )

    safe_gross = gross_by_date.replace(
        0.0,
        np.nan,
    )
    exposure_scale = np.where(
        gross_by_date > max_gross_exposure,
        max_gross_exposure / safe_gross,
        1.0,
    )

    symbol_equity["exposure_scale"] = (
        pd.Series(
            exposure_scale,
            index=symbol_equity.index,
        ).fillna(1.0)
    )
    symbol_equity["executed_position"] = (
        symbol_equity[
            "raw_executed_position"
        ]
        * symbol_equity["exposure_scale"]
    )

    symbol_equity = symbol_equity.sort_values(
        ["symbol", "datetime"]
    ).reset_index(drop=True)

    previous_position = symbol_equity.groupby(
        "symbol"
    )["executed_position"].shift(1).fillna(0.0)

    symbol_equity["position_change"] = (
        symbol_equity["executed_position"]
        - previous_position
    )
    symbol_equity["buy_turnover"] = (
        symbol_equity["position_change"]
        .clip(lower=0.0)
    )
    symbol_equity["sell_turnover"] = (
        -symbol_equity["position_change"]
    ).clip(lower=0.0)
    symbol_equity["turnover"] = (
        symbol_equity["buy_turnover"]
        + symbol_equity["sell_turnover"]
    )

    symbol_equity["cost"] = (
        symbol_equity["buy_turnover"]
        * (buy_commission + slippage)
        + symbol_equity["sell_turnover"]
        * (
            sell_commission
            + stamp_duty
            + slippage
        )
    )

    symbol_equity["gross_strategy_return"] = (
        symbol_equity["executed_position"]
        * symbol_equity["asset_return"]
    )
    symbol_equity["strategy_return"] = (
        symbol_equity["gross_strategy_return"]
        - symbol_equity["cost"]
    )

    symbol_equity["symbol_equity"] = (
        1.0
        + symbol_equity["strategy_return"]
    ).groupby(
        symbol_equity["symbol"]
    ).cumprod()

    portfolio = symbol_equity.groupby(
        "datetime",
        as_index=False,
    ).agg(
        portfolio_return=(
            "strategy_return",
            "sum",
        ),
        gross_portfolio_return=(
            "gross_strategy_return",
            "sum",
        ),
        gross_exposure=(
            "executed_position",
            "sum",
        ),
        active_symbols=(
            "executed_position",
            lambda values: int(
                (values > 0).sum()
            ),
        ),
        buy_turnover=(
            "buy_turnover",
            "sum",
        ),
        sell_turnover=(
            "sell_turnover",
            "sum",
        ),
        turnover=("turnover", "sum"),
        transaction_cost=("cost", "sum"),
        equal_weight_asset_return=(
            "asset_return",
            "mean",
        ),
    ).sort_values(
        "datetime"
    ).reset_index(drop=True)

    portfolio["equal_weight_return"] = (
        portfolio["equal_weight_asset_return"]
        * max_gross_exposure
    )
    portfolio["exposure_matched_return"] = (
        portfolio["equal_weight_asset_return"]
        * portfolio["gross_exposure"]
    )

    portfolio["equity"] = (
        initial_capital
        * (
            1.0
            + portfolio["portfolio_return"]
        ).cumprod()
    )
    portfolio["equal_weight_equity"] = (
        initial_capital
        * (
            1.0
            + portfolio["equal_weight_return"]
        ).cumprod()
    )
    portfolio["exposure_matched_equity"] = (
        initial_capital
        * (
            1.0
            + portfolio[
                "exposure_matched_return"
            ]
        ).cumprod()
    )

    strategy_metrics = calculate_performance(
        portfolio["portfolio_return"],
        portfolio["equity"],
        annualization,
        initial_capital,
    )
    equal_weight_metrics = calculate_performance(
        portfolio["equal_weight_return"],
        portfolio["equal_weight_equity"],
        annualization,
        initial_capital,
    )
    exposure_matched_metrics = calculate_performance(
        portfolio[
            "exposure_matched_return"
        ],
        portfolio[
            "exposure_matched_equity"
        ],
        annualization,
        initial_capital,
    )

    target_changes = symbol_equity[
        "turnover"
    ] > 1e-12
    entries = (
        symbol_equity["position_change"]
        > 1e-12
    ) & (
        previous_position <= 1e-12
    )
    exits = (
        symbol_equity["position_change"]
        < -1e-12
    ) & (
        symbol_equity["executed_position"]
        <= 1e-12
    )

    strategy_metrics.update(
        {
            "average_gross_exposure": float(
                portfolio["gross_exposure"].mean()
            ),
            "maximum_gross_exposure": float(
                portfolio["gross_exposure"].max()
            ),
            "total_turnover": float(
                portfolio["turnover"].sum()
            ),
            "buy_turnover": float(
                portfolio["buy_turnover"].sum()
            ),
            "sell_turnover": float(
                portfolio["sell_turnover"].sum()
            ),
            "total_transaction_cost": float(
                portfolio[
                    "transaction_cost"
                ].sum()
            ),
            "position_change_count": int(
                target_changes.sum()
            ),
            "entry_count": int(entries.sum()),
            "exit_count": int(exits.sum()),
            "execution_mode": (
                "next_open_to_next_open"
            ),
            "minimum_holding_days": int(
                config.get(
                    "minimum_holding_days",
                    3,
                )
            ),
        }
    )

    metrics: dict[str, Any] = {
        "strategy": strategy_metrics,
        "equal_weight_benchmark": (
            equal_weight_metrics
        ),
        "exposure_matched_benchmark": (
            exposure_matched_metrics
        ),
    }

    return portfolio, symbol_equity, metrics


def annual_performance(
    portfolio: pd.DataFrame,
    annualization: int,
    initial_capital: float,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    frame = portfolio.copy()
    frame["year"] = pd.to_datetime(
        frame["datetime"]
    ).dt.year

    for year, group in frame.groupby("year"):
        returns = group[
            "portfolio_return"
        ].reset_index(drop=True)
        equity = (
            initial_capital
            * (1.0 + returns).cumprod()
        )
        metrics = calculate_performance(
            returns,
            equity,
            annualization,
            initial_capital,
        )
        records.append(
            {
                "year": int(year),
                **metrics,
            }
        )

    return pd.DataFrame(records)


def symbol_performance(
    symbol_equity: pd.DataFrame,
    annualization: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for symbol, group in symbol_equity.groupby(
        "symbol"
    ):
        ordered = group.sort_values("datetime")
        returns = ordered[
            "strategy_return"
        ].reset_index(drop=True)
        equity = (1.0 + returns).cumprod()
        metrics = calculate_performance(
            returns,
            equity,
            annualization,
            1.0,
        )

        records.append(
            {
                "symbol": symbol,
                **metrics,
                "average_position": float(
                    ordered[
                        "executed_position"
                    ].mean()
                ),
                "turnover": float(
                    ordered["turnover"].sum()
                ),
            }
        )

    return pd.DataFrame(records).sort_values(
        "total_return",
        ascending=False,
    )