from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.backtest.portfolio_engine_v4 import (
    PortfolioBacktestEngineV4,
    _performance_metrics,
)


@dataclass
class FastBacktestResult:
    portfolio: pd.DataFrame
    metrics: dict[str, float]
    exposure_matched_metrics: dict[str, float]
    equal_weight_metrics: dict[str, float]


class FastPortfolioBacktestEngine(PortfolioBacktestEngineV4):
    """Memory-light metric engine for repeated V5 robustness scenarios."""

    def run_fast(
        self,
        decisions: pd.DataFrame,
        allocations: pd.DataFrame,
    ) -> FastBacktestResult:
        market = self._prepare_market(decisions)
        weights = allocations.copy()
        weights["datetime"] = pd.to_datetime(weights["datetime"], errors="coerce")
        weights["symbol"] = weights["symbol"].astype(str).str.upper()
        merged = weights.merge(
            market[
                [
                    "datetime",
                    "symbol",
                    "execution_date",
                    "next_execution_date",
                    "realized_open_to_open_return",
                ]
            ],
            on=["datetime", "symbol"],
            how="left",
        )
        merged["target_weight_raw"] = pd.to_numeric(
            merged.get("target_weight"), errors="coerce"
        ).fillna(0.0)
        merged["realized_open_to_open_return"] = pd.to_numeric(
            merged["realized_open_to_open_return"], errors="coerce"
        ).fillna(0.0)

        buy_cost = float(self.config.get("buy_commission_rate", 0.0001)) + float(
            self.config.get("slippage_rate", 0.0005)
        )
        sell_cost = (
            float(self.config.get("sell_commission_rate", 0.0001))
            + float(self.config.get("stamp_duty_rate", 0.0005))
            + float(self.config.get("slippage_rate", 0.0005))
        )
        drawdown_config = self.config.get("drawdown_control", {})
        symbols = sorted(merged["symbol"].unique())
        previous = pd.Series(0.0, index=symbols, dtype=float)
        risk_equity = 1.0
        risk_peak = 1.0
        risk_drawdown = 0.0
        current_multiplier = 1.0
        pending: list[tuple[pd.Timestamp, float]] = []
        daily_rows: list[dict[str, Any]] = []

        for date, group in merged.sort_values(["datetime", "symbol"]).groupby(
            "datetime", sort=True
        ):
            signal_date = pd.Timestamp(date)
            still_pending: list[tuple[pd.Timestamp, float]] = []
            for realization_date, realized_net in pending:
                if pd.notna(realization_date) and realization_date <= signal_date:
                    risk_equity *= 1.0 + realized_net
                    risk_peak = max(risk_peak, risk_equity)
                    risk_drawdown = risk_equity / risk_peak - 1.0
                else:
                    still_pending.append((realization_date, realized_net))
            pending = still_pending

            current_multiplier = self._hysteresis_multiplier(
                risk_drawdown, current_multiplier, drawdown_config
            )
            local = group.set_index("symbol").reindex(symbols)
            raw_target = pd.to_numeric(
                local["target_weight_raw"], errors="coerce"
            ).fillna(0.0)
            adjusted_target = raw_target * current_multiplier
            weight_change = adjusted_target - previous
            realized_return = pd.to_numeric(
                local["realized_open_to_open_return"], errors="coerce"
            ).fillna(0.0)
            buy_turnover = weight_change.clip(lower=0.0)
            sell_turnover = -weight_change.clip(upper=0.0)
            transaction_cost = buy_turnover * buy_cost + sell_turnover * sell_cost
            gross_return = float((adjusted_target * realized_return).sum())
            total_cost = float(transaction_cost.sum())
            net_return = gross_return - total_cost
            realization_date = pd.to_datetime(
                local["next_execution_date"], errors="coerce"
            ).max()
            if pd.notna(realization_date):
                pending.append((pd.Timestamp(realization_date), net_return))
            daily_rows.append(
                {
                    "datetime": signal_date,
                    "gross_return": gross_return,
                    "transaction_cost": total_cost,
                    "strategy_return": net_return,
                    "gross_exposure": float(adjusted_target.sum()),
                    "raw_gross_exposure": float(raw_target.sum()),
                    "turnover": float(weight_change.abs().sum()),
                    "selected_count": int(adjusted_target.gt(1e-12).sum()),
                    "drawdown_control_multiplier": current_multiplier,
                }
            )
            previous = adjusted_target

        daily = pd.DataFrame(daily_rows).sort_values("datetime").reset_index(drop=True)
        benchmark = market.groupby("datetime")["realized_open_to_open_return"].mean()
        daily = daily.merge(
            benchmark.rename("equal_weight_return"),
            left_on="datetime",
            right_index=True,
            how="left",
        )
        daily["equal_weight_return"] = daily["equal_weight_return"].fillna(0.0)
        daily["exposure_matched_return"] = (
            daily["equal_weight_return"] * daily["gross_exposure"]
        )
        annualization = int(self.config.get("annualization_bars", 252))
        metrics = _performance_metrics(daily["strategy_return"], annualization)
        metrics.update(
            {
                "average_gross_exposure": float(daily["gross_exposure"].mean()),
                "average_turnover": float(daily["turnover"].mean()),
                "maximum_actual_turnover": float(daily["turnover"].max()),
                "total_turnover": float(daily["turnover"].sum()),
                "total_transaction_cost": float(daily["transaction_cost"].sum()),
                "average_selected_count": float(daily["selected_count"].mean()),
                "observations": int(len(daily)),
            }
        )
        return FastBacktestResult(
            portfolio=daily,
            metrics=metrics,
            exposure_matched_metrics=_performance_metrics(
                daily["exposure_matched_return"], annualization
            ),
            equal_weight_metrics=_performance_metrics(
                daily["equal_weight_return"], annualization
            ),
        )