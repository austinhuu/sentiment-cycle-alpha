from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PortfolioBacktestResult:
    portfolio: pd.DataFrame
    contributions: pd.DataFrame
    metrics: dict[str, float]
    exposure_matched_metrics: dict[str, float]
    equal_weight_metrics: dict[str, float]


def _performance_metrics(returns: pd.Series, annualization: int = 252) -> dict[str, float]:
    values = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    if values.empty:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "positive_day_ratio": 0.0,
        }
    equity = (1.0 + values).cumprod()
    total = float(equity.iloc[-1] - 1.0)
    periods = max(len(values), 1)
    annualized_return = float((1.0 + total) ** (annualization / periods) - 1.0) if total > -1 else -1.0
    volatility = float(values.std(ddof=0) * np.sqrt(annualization))
    mean_annual = float(values.mean() * annualization)
    sharpe = mean_annual / volatility if volatility > 1e-12 else 0.0
    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    calmar = annualized_return / abs(max_drawdown) if max_drawdown < -1e-12 else 0.0
    return {
        "total_return": total,
        "annualized_return": annualized_return,
        "annualized_volatility": volatility,
        "sharpe_ratio": float(sharpe),
        "max_drawdown": max_drawdown,
        "calmar_ratio": float(calmar),
        "positive_day_ratio": float((values > 0).mean()),
    }


class PortfolioBacktestEngineV4:
    """Next-open portfolio backtest with lagged, no-look-ahead drawdown control."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @staticmethod
    def _prepare_market(decisions: pd.DataFrame) -> pd.DataFrame:
        market = decisions.copy()
        market["datetime"] = pd.to_datetime(market["datetime"], errors="coerce")
        market["symbol"] = market["symbol"].astype(str).str.upper()
        market["signal_open"] = pd.to_numeric(
            market.get("signal_open", market.get("open")), errors="coerce"
        )
        market = market.dropna(subset=["datetime", "symbol", "signal_open"])
        market = market.sort_values(["symbol", "datetime"])
        grouped = market.groupby("symbol", sort=False)
        market["execution_date"] = grouped["datetime"].shift(-1)
        market["next_execution_date"] = grouped["datetime"].shift(-2)
        next_open = grouped["signal_open"].shift(-1)
        following_open = grouped["signal_open"].shift(-2)
        market["realized_open_to_open_return"] = following_open / next_open - 1.0
        return market

    @staticmethod
    def _desired_drawdown_multiplier(drawdown: float, config: dict[str, Any]) -> float:
        if not bool(config.get("enabled", True)):
            return 1.0
        warning = float(config.get("warning_drawdown", -0.08))
        defensive = float(config.get("defensive_drawdown", -0.12))
        severe = float(config.get("severe_drawdown", -0.18))
        if drawdown <= severe:
            return float(config.get("severe_multiplier", 0.40))
        if drawdown <= defensive:
            return float(config.get("defensive_multiplier", 0.65))
        if drawdown <= warning:
            return float(config.get("warning_multiplier", 0.85))
        return 1.0

    @staticmethod
    def _hysteresis_multiplier(
        drawdown: float,
        current: float,
        config: dict[str, Any],
    ) -> float:
        desired = PortfolioBacktestEngineV4._desired_drawdown_multiplier(drawdown, config)
        if desired <= current:
            return desired
        buffer = max(float(config.get("recovery_buffer", 0.02)), 0.0)
        severe = float(config.get("severe_drawdown", -0.18))
        defensive = float(config.get("defensive_drawdown", -0.12))
        warning = float(config.get("warning_drawdown", -0.08))
        severe_mult = float(config.get("severe_multiplier", 0.40))
        defensive_mult = float(config.get("defensive_multiplier", 0.65))
        warning_mult = float(config.get("warning_multiplier", 0.85))
        if current <= severe_mult + 1e-12 and drawdown <= severe + buffer:
            return current
        if current <= defensive_mult + 1e-12 and drawdown <= defensive + buffer:
            return current
        if current <= warning_mult + 1e-12 and drawdown <= warning + buffer:
            return current
        return desired

    def run(
        self,
        decisions: pd.DataFrame,
        allocations: pd.DataFrame,
    ) -> PortfolioBacktestResult:
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
        contribution_parts: list[pd.DataFrame] = []
        daily_rows: list[dict[str, Any]] = []

        for date, group in merged.sort_values(["datetime", "symbol"]).groupby("datetime", sort=True):
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
            local = group.copy().set_index("symbol").reindex(symbols)
            raw_target = pd.to_numeric(local["target_weight_raw"], errors="coerce").fillna(0.0)
            adjusted_target = raw_target * current_multiplier
            weight_change = adjusted_target - previous
            realized_return = pd.to_numeric(
                local["realized_open_to_open_return"], errors="coerce"
            ).fillna(0.0)
            buy_turnover = weight_change.clip(lower=0.0)
            sell_turnover = -weight_change.clip(upper=0.0)
            transaction_cost = buy_turnover * buy_cost + sell_turnover * sell_cost
            gross_contribution = adjusted_target * realized_return
            net_contribution = gross_contribution - transaction_cost
            net_return = float(net_contribution.sum())
            realization_date = pd.to_datetime(local["next_execution_date"], errors="coerce").max()
            if pd.notna(realization_date):
                pending.append((pd.Timestamp(realization_date), net_return))

            local["target_weight"] = adjusted_target
            local["previous_weight"] = previous
            local["weight_change"] = weight_change
            local["buy_turnover"] = buy_turnover
            local["sell_turnover"] = sell_turnover
            local["transaction_cost"] = transaction_cost
            local["gross_contribution"] = gross_contribution
            local["net_contribution"] = net_contribution
            local["drawdown_control_multiplier"] = current_multiplier
            local["strategy_drawdown_before_signal"] = risk_drawdown
            local["risk_overlay_active"] = int(current_multiplier < 1.0 - 1e-12)
            local["datetime"] = signal_date
            local["symbol"] = local.index
            contribution_parts.append(local.reset_index(drop=True))

            daily_rows.append(
                {
                    "datetime": signal_date,
                    "execution_date": pd.to_datetime(local["execution_date"], errors="coerce").max(),
                    "next_execution_date": realization_date,
                    "gross_return": float(gross_contribution.sum()),
                    "transaction_cost": float(transaction_cost.sum()),
                    "strategy_return": net_return,
                    "gross_exposure": float(adjusted_target.sum()),
                    "raw_gross_exposure": float(raw_target.sum()),
                    "turnover": float(weight_change.abs().sum()),
                    "selected_count": int(adjusted_target.gt(1e-12).sum()),
                    "drawdown_control_multiplier": current_multiplier,
                    "strategy_drawdown_before_signal": risk_drawdown,
                    "risk_overlay_active": int(current_multiplier < 1.0 - 1e-12),
                }
            )
            previous = adjusted_target

        contributions = pd.concat(contribution_parts, ignore_index=True)
        daily = pd.DataFrame(daily_rows).sort_values("datetime").reset_index(drop=True)
        initial_capital = float(self.config.get("initial_capital", 1_000_000))
        daily["equity"] = initial_capital * (1.0 + daily["strategy_return"]).cumprod()
        daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1.0

        benchmark = market.groupby("datetime")["realized_open_to_open_return"].mean()
        daily = daily.merge(
            benchmark.rename("equal_weight_return"),
            left_on="datetime",
            right_index=True,
            how="left",
        )
        daily["equal_weight_return"] = daily["equal_weight_return"].fillna(0.0)
        daily["exposure_matched_return"] = daily["equal_weight_return"] * daily["gross_exposure"]
        daily["equal_weight_equity"] = initial_capital * (1.0 + daily["equal_weight_return"]).cumprod()
        daily["exposure_matched_equity"] = initial_capital * (
            1.0 + daily["exposure_matched_return"]
        ).cumprod()

        annualization = int(self.config.get("annualization_bars", 252))
        metrics = _performance_metrics(daily["strategy_return"], annualization)
        metrics.update(
            {
                "average_gross_exposure": float(daily["gross_exposure"].mean()),
                "average_raw_gross_exposure": float(daily["raw_gross_exposure"].mean()),
                "average_turnover": float(daily["turnover"].mean()),
                "maximum_actual_turnover": float(daily["turnover"].max()),
                "total_turnover": float(daily["turnover"].sum()),
                "total_transaction_cost": float(daily["transaction_cost"].sum()),
                "average_selected_count": float(daily["selected_count"].mean()),
                "average_drawdown_control_multiplier": float(
                    daily["drawdown_control_multiplier"].mean()
                ),
                "risk_overlay_active_days": int(daily["risk_overlay_active"].sum()),
                "observations": int(len(daily)),
            }
        )
        exposure_metrics = _performance_metrics(daily["exposure_matched_return"], annualization)
        equal_metrics = _performance_metrics(daily["equal_weight_return"], annualization)
        return PortfolioBacktestResult(
            portfolio=daily,
            contributions=contributions,
            metrics=metrics,
            exposure_matched_metrics=exposure_metrics,
            equal_weight_metrics=equal_metrics,
        )


def save_portfolio_backtest(result: PortfolioBacktestResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.portfolio.to_csv(
        output_dir / "portfolio_equity.csv", index=False, encoding="utf-8-sig"
    )
    result.contributions.to_csv(
        output_dir / "symbol_contributions.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "performance_metrics.json").write_text(
        json.dumps(result.metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "exposure_matched_metrics.json").write_text(
        json.dumps(result.exposure_matched_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "equal_weight_metrics.json").write_text(
        json.dumps(result.equal_weight_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )