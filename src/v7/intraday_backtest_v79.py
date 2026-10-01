from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SimulationResult:
    bars: pd.DataFrame
    trades: pd.DataFrame
    daily: pd.DataFrame
    metrics: dict[str, float]


def build_focus_daily_targets(
    v78_signals: pd.DataFrame,
    intraday_dates: pd.Index,
    focus_symbol: str,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Convert frozen V7.8 daily ranking into next-session focus-symbol targets.

    A signal observed on date D may first affect trading on the next available
    intraday trading date. This keeps the V7.8 daily model causal.
    """

    d = v78_signals.copy()
    d["datetime"] = pd.to_datetime(d["datetime"], errors="coerce").dt.normalize()
    d["symbol"] = d["symbol"].astype(str).str.upper()

    score_col = str(cfg.get("score_col", "v7_8_final_score"))
    if score_col not in d.columns:
        score_col = "v7_alpha"

    overlay_col = str(cfg.get("overlay_col", "v7_8_dynamic_position_multiplier"))
    if overlay_col not in d.columns:
        overlay_col = "v7_7_dynamic_position_multiplier"

    top_k = int(cfg.get("top_k", 8))
    min_score = float(cfg.get("min_score", 0.05))
    max_gross = float(cfg.get("max_gross", 1.0))

    rows: list[dict[str, Any]] = []

    for signal_date, group in d.groupby("datetime", sort=True):
        g = group.copy()
        g[score_col] = pd.to_numeric(g[score_col], errors="coerce")
        g = g.loc[g[score_col] > min_score].nlargest(top_k, score_col)

        weight = 0.0

        if not g.empty:
            raw = g[score_col].clip(lower=0.0)

            if raw.sum() > 0:
                base = raw / raw.sum()
            else:
                base = pd.Series(1.0 / len(g), index=g.index)

            gross = max_gross

            if overlay_col in g.columns:
                gross *= float(
                    pd.to_numeric(g[overlay_col], errors="coerce")
                    .fillna(1.0)
                    .median()
                )
                gross = float(np.clip(gross, 0.0, max_gross))

            weights = pd.Series(
                base.values * gross,
                index=g["symbol"].astype(str).values,
            )

            weight = float(weights.get(focus_symbol.upper(), 0.0))

        rows.append(
            {
                "signal_date": signal_date,
                "raw_target_weight": weight,
            }
        )

    targets = pd.DataFrame(rows).sort_values("signal_date")

    dates = pd.DatetimeIndex(
        pd.to_datetime(pd.Index(intraday_dates)).normalize().unique()
    ).sort_values()

    mapped: list[dict[str, Any]] = []

    for row in targets.itertuples(index=False):
        pos = dates.searchsorted(pd.Timestamp(row.signal_date), side="right")

        if pos < len(dates):
            mapped.append(
                {
                    "trade_date": dates[pos],
                    "signal_date": pd.Timestamp(row.signal_date),
                    "target_weight": float(row.raw_target_weight),
                }
            )

    if not mapped:
        return pd.DataFrame(
            columns=["trade_date", "signal_date", "target_weight"]
        )

    out = pd.DataFrame(mapped).drop_duplicates("trade_date", keep="last")

    # Every intraday date gets a target. Before a mapped daily signal exists,
    # exposure is zero rather than forward-filled from the future.
    all_dates = pd.DataFrame({"trade_date": dates})
    out = all_dates.merge(out, on="trade_date", how="left")
    out["target_weight"] = pd.to_numeric(
        out["target_weight"], errors="coerce"
    ).fillna(0.0)

    return out


def _daily_metrics(
    daily: pd.DataFrame,
    initial_capital: float,
    trade_count: int,
    total_cost: float,
) -> dict[str, float]:
    if daily.empty:
        return {
            "total_return": 0.0,
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "trading_days": 0,
            "trade_count": int(trade_count),
            "total_cost": float(total_cost),
        }

    equity = pd.to_numeric(daily["equity"], errors="coerce").dropna()

    if len(equity):
        returns = equity.pct_change()
        returns.iloc[0] = equity.iloc[0] / float(initial_capital) - 1.0
        returns = returns.fillna(0.0)
        total = float(equity.iloc[-1] / float(initial_capital) - 1.0)
    else:
        returns = pd.Series(dtype=float)
        total = 0.0

    years = max(len(equity) / 252.0, 1.0 / 252.0)

    annual = (
        float((1.0 + total) ** (1.0 / years) - 1.0)
        if 1.0 + total > 0
        else -1.0
    )

    std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    vol = std * np.sqrt(252.0)
    sharpe = (
        float(returns.mean() / std * np.sqrt(252.0))
        if std > 0
        else 0.0
    )

    drawdown = equity / equity.cummax() - 1.0
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0
    calmar = float(annual / abs(max_dd)) if max_dd < 0 else 0.0

    return {
        "total_return": total,
        "annual_return": annual,
        "annual_volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "trading_days": int(len(equity)),
        "trade_count": int(trade_count),
        "total_cost": float(total_cost),
    }


def _execute_to_weight(
    cash: float,
    shares: float,
    sellable_shares: float,
    price: float,
    target_weight: float,
    buy_cost: float,
    sell_cost: float,
    allow_buy: bool = True,
) -> tuple[float, float, float, dict[str, float | str] | None]:
    """Move toward a target weight subject to cash and A-share T+1 sellability."""

    equity = cash + shares * price

    if equity <= 0 or price <= 0:
        return cash, shares, sellable_shares, None

    desired_shares = max(target_weight, 0.0) * equity / price
    delta = desired_shares - shares

    if abs(delta) < 1e-12:
        return cash, shares, sellable_shares, None

    if delta > 0:
        if not allow_buy:
            return cash, shares, sellable_shares, None

        max_buy = max(cash, 0.0) / (price * (1.0 + buy_cost))
        qty = min(delta, max_buy)

        if qty <= 0:
            return cash, shares, sellable_shares, None

        value = qty * price
        cost = value * buy_cost

        cash -= value + cost
        shares += qty

        trade = {
            "side": "BUY",
            "shares": qty,
            "price": price,
            "value": value,
            "cost": cost,
        }

        return cash, shares, sellable_shares, trade

    qty = min(-delta, sellable_shares)

    if qty <= 0:
        return cash, shares, sellable_shares, None

    value = qty * price
    cost = value * sell_cost

    cash += value - cost
    shares -= qty
    sellable_shares -= qty

    trade = {
        "side": "SELL",
        "shares": qty,
        "price": price,
        "value": value,
        "cost": cost,
    }

    return cash, shares, sellable_shares, trade


def simulate_focus_strategy(
    feature_bars: pd.DataFrame,
    targets: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
    mode: str = "confirmation",
) -> SimulationResult:
    """Simulate a single A-share position with next-bar execution and T+1.

    Supported modes
    ---------------
    open_baseline
        The frozen V7.8 daily target is aligned at the first 5-minute bar open.
        Both increases and reductions are executed immediately.

    confirmation
        Legacy V7.9 behaviour. 5-minute confirmation can time ENTER, REDUCE and
        EXIT. This mode is retained as the historical V7.9 baseline.

    entry_confirmation_only
        New structural experiment. V7.8 owns position reductions/exits: if the
        daily target is below the carried position, the reduction is executed at
        the first bar open. 5-minute confirmation only controls increases in
        exposure (ENTER / add-to-position). Intraday negative confirmation cannot
        liquidate a position while the frozen daily target still wants it.

    In every confirmation mode, a signal computed from bar t may execute no
    earlier than bar t+1 open. Same-day purchases are not sellable until the next
    trading day.
    """

    cfg = cfg or {}

    valid_modes = {
        "open_baseline",
        "confirmation",
        "entry_confirmation_only",
    }

    if mode not in valid_modes:
        raise ValueError(
            f"Unsupported V7.9 simulation mode: {mode}. "
            f"Expected one of {sorted(valid_modes)}"
        )

    d = feature_bars.copy()
    d["datetime"] = pd.to_datetime(d["datetime"], errors="coerce")
    d["trade_date"] = d["datetime"].dt.normalize()
    d = d.sort_values("datetime").reset_index(drop=True)

    target_map = dict(
        zip(
            pd.to_datetime(targets["trade_date"]).dt.normalize(),
            pd.to_numeric(targets["target_weight"], errors="coerce")
            .fillna(0.0)
            .astype(float),
        )
    )

    initial_capital = float(cfg.get("initial_capital", 1_000_000.0))
    buy_cost = float(cfg.get("buy_cost_rate", 0.0006))
    sell_cost = float(cfg.get("sell_cost_rate", 0.0011))

    enter_threshold = float(cfg.get("enter_threshold", 0.30))
    reduce_threshold = float(cfg.get("reduce_threshold", -0.05))
    exit_threshold = float(cfg.get("exit_threshold", -0.30))

    rebalance_band = float(cfg.get("rebalance_band", 0.005))
    earliest_entry_bar = int(cfg.get("earliest_entry_bar", 1))
    latest_entry_bar = int(cfg.get("latest_entry_bar", 42))

    cash = initial_capital
    shares = 0.0
    sellable_shares = 0.0

    pending: dict[str, Any] | None = None

    bar_rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []

    total_cost = 0.0

    for day, day_bars in d.groupby("trade_date", sort=True):
        day = pd.Timestamp(day)
        day_bars = day_bars.sort_values("datetime").reset_index(drop=True)

        if day_bars.empty:
            continue

        target_weight = float(target_map.get(day, 0.0))

        # Everything carried from a prior session is sellable at today's open.
        # Any shares purchased later today are deliberately excluded from this
        # pool and therefore cannot be sold until the next trading day.
        sellable_shares = shares
        pending = None

        first_open = float(day_bars.iloc[0]["open"])
        start_equity = cash + shares * first_open

        for i, row in day_bars.iterrows():
            open_price = float(row["open"])
            close_price = float(row["close"])

            score = float(
                pd.to_numeric(
                    pd.Series([row.get("confirmation_score", 0.0)]),
                    errors="coerce",
                )
                .fillna(0.0)
                .iloc[0]
            )

            executed_action = ""
            decision = "WAIT"

            # ------------------------------------------------------------
            # 1) Orders already decided by prior information execute here.
            # ------------------------------------------------------------

            if mode == "open_baseline" and i == 0:
                cash, shares, sellable_shares, trade = _execute_to_weight(
                    cash,
                    shares,
                    sellable_shares,
                    open_price,
                    target_weight,
                    buy_cost,
                    sell_cost,
                    allow_buy=True,
                )

                if trade:
                    trade.update(
                        {
                            "datetime": row["datetime"],
                            "signal_datetime": pd.NaT,
                            "action": "OPEN_ALIGN",
                        }
                    )
                    trades.append(trade)
                    total_cost += float(trade["cost"])
                    executed_action = str(trade["side"])
                    decision = "OPEN_ALIGN"

            elif mode == "entry_confirmation_only" and i == 0:
                # Structural change: V7.8 daily target owns reductions/exits.
                # Only reduce here. Never increase at the open in this mode.
                equity_open = cash + shares * open_price
                current_open_weight = (
                    shares * open_price / equity_open
                    if equity_open > 0
                    else 0.0
                )

                if target_weight < current_open_weight - rebalance_band:
                    daily_action = (
                        "DAILY_EXIT"
                        if target_weight <= rebalance_band
                        else "DAILY_REDUCE"
                    )

                    cash, shares, sellable_shares, trade = _execute_to_weight(
                        cash,
                        shares,
                        sellable_shares,
                        open_price,
                        target_weight,
                        buy_cost,
                        sell_cost,
                        allow_buy=False,
                    )

                    if trade:
                        trade.update(
                            {
                                "datetime": row["datetime"],
                                "signal_datetime": pd.NaT,
                                "action": daily_action,
                            }
                        )
                        trades.append(trade)
                        total_cost += float(trade["cost"])
                        executed_action = str(trade["side"])
                        decision = daily_action

            elif mode in {"confirmation", "entry_confirmation_only"} and pending is not None:
                # Confirmation signal from the previous bar is executed now.
                # In entry-only mode pending orders can only be ENTER orders.
                desired = float(pending["target_weight"])
                allow_buy = str(pending["action"]) == "ENTER"

                cash, shares, sellable_shares, trade = _execute_to_weight(
                    cash,
                    shares,
                    sellable_shares,
                    open_price,
                    desired,
                    buy_cost,
                    sell_cost,
                    allow_buy=allow_buy,
                )

                if trade:
                    trade.update(
                        {
                            "datetime": row["datetime"],
                            "signal_datetime": pending["signal_datetime"],
                            "action": pending["action"],
                        }
                    )
                    trades.append(trade)
                    total_cost += float(trade["cost"])
                    executed_action = str(trade["side"])

                pending = None

            # ------------------------------------------------------------
            # 2) Mark current state at this bar close.
            # ------------------------------------------------------------

            equity_close = cash + shares * close_price

            current_weight = (
                shares * close_price / equity_close
                if equity_close > 0
                else 0.0
            )

            sellable_weight = (
                sellable_shares * close_price / equity_close
                if equity_close > 0
                else 0.0
            )

            # ------------------------------------------------------------
            # 3) Generate a decision from current-bar information.
            #    It can only execute at the next bar open.
            # ------------------------------------------------------------

            if mode == "confirmation" and i < len(day_bars) - 1:
                penultimate_signal = i >= len(day_bars) - 2

                if target_weight > current_weight + rebalance_band:
                    if (
                        earliest_entry_bar <= i <= latest_entry_bar
                        and score >= enter_threshold
                    ):
                        decision = "ENTER"
                        pending = {
                            "action": "ENTER",
                            "target_weight": target_weight,
                            "signal_datetime": row["datetime"],
                        }

                elif (
                    target_weight < current_weight - rebalance_band
                    and sellable_weight > 0
                ):
                    desired = max(
                        target_weight,
                        current_weight - sellable_weight,
                    )

                    if (
                        target_weight <= rebalance_band
                        and (
                            score <= exit_threshold
                            or penultimate_signal
                        )
                    ):
                        decision = "EXIT"
                        pending = {
                            "action": "EXIT",
                            "target_weight": desired,
                            "signal_datetime": row["datetime"],
                        }

                    elif score <= reduce_threshold or penultimate_signal:
                        decision = "REDUCE"
                        pending = {
                            "action": "REDUCE",
                            "target_weight": desired,
                            "signal_datetime": row["datetime"],
                        }

            elif mode == "entry_confirmation_only" and i < len(day_bars) - 1:
                # V7.9 is deliberately asymmetric here:
                #   * Daily V7.8 controls reductions/exits.
                #   * 5-minute V7.9 controls only increases in exposure.
                # Negative intraday confirmation cannot liquidate a daily-alpha
                # position that V7.8 still wants to hold.
                if target_weight > current_weight + rebalance_band:
                    if (
                        earliest_entry_bar <= i <= latest_entry_bar
                        and score >= enter_threshold
                    ):
                        decision = "ENTER"
                        pending = {
                            "action": "ENTER",
                            "target_weight": target_weight,
                            "signal_datetime": row["datetime"],
                        }

            bar_rows.append(
                {
                    **row.to_dict(),
                    "simulation_mode": mode,
                    "target_weight": target_weight,
                    "position_weight": current_weight,
                    "sellable_weight": sellable_weight,
                    "decision": decision,
                    "executed_side": executed_action,
                    "cash": cash,
                    "shares": shares,
                    "equity": equity_close,
                }
            )

        final_close = float(day_bars.iloc[-1]["close"])
        end_equity = cash + shares * final_close

        daily_rows.append(
            {
                "trade_date": day,
                "start_equity": start_equity,
                "equity": end_equity,
                "target_weight": target_weight,
                "end_position_weight": (
                    shares * final_close / end_equity
                    if end_equity > 0
                    else 0.0
                ),
            }
        )

    bar_df = pd.DataFrame(bar_rows)
    trade_df = pd.DataFrame(trades)
    daily_df = pd.DataFrame(daily_rows)

    metrics = _daily_metrics(
        daily_df,
        initial_capital,
        len(trade_df),
        total_cost,
    )

    if not trade_df.empty:
        buys = trade_df.loc[trade_df["side"].eq("BUY")]
        sells = trade_df.loc[trade_df["side"].eq("SELL")]

        metrics["entry_count"] = int(len(buys))
        metrics["sell_count"] = int(len(sells))

        if "datetime" in buys.columns and not buys.empty:
            buy_dt = pd.to_datetime(buys["datetime"])
            metrics["mean_entry_minute"] = float(
                (buy_dt.dt.hour * 60 + buy_dt.dt.minute).mean()
            )
        else:
            metrics["mean_entry_minute"] = 0.0

        if "action" in trade_df.columns:
            metrics["daily_reduce_count"] = int(
                trade_df["action"].eq("DAILY_REDUCE").sum()
            )
            metrics["daily_exit_count"] = int(
                trade_df["action"].eq("DAILY_EXIT").sum()
            )
            metrics["intraday_reduce_count"] = int(
                trade_df["action"].eq("REDUCE").sum()
            )
            metrics["intraday_exit_count"] = int(
                trade_df["action"].eq("EXIT").sum()
            )
    else:
        metrics["entry_count"] = 0
        metrics["sell_count"] = 0
        metrics["mean_entry_minute"] = 0.0
        metrics["daily_reduce_count"] = 0
        metrics["daily_exit_count"] = 0
        metrics["intraday_reduce_count"] = 0
        metrics["intraday_exit_count"] = 0

    return SimulationResult(
        bars=bar_df,
        trades=trade_df,
        daily=daily_df,
        metrics=metrics,
    )
