from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DynamicRiskDiagnostics:
    dates: int
    average_multiplier: float
    minimum_multiplier: float
    maximum_multiplier: float
    average_volatility_multiplier: float
    average_stress_multiplier: float
    average_health_multiplier: float
    average_regime_multiplier: float
    low_budget_dates: int
    maximum_turnover: float
    turnover_violation_dates: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "dates": int(self.dates),
            "average_multiplier": float(self.average_multiplier),
            "minimum_multiplier": float(self.minimum_multiplier),
            "maximum_multiplier": float(self.maximum_multiplier),
            "average_volatility_multiplier": float(self.average_volatility_multiplier),
            "average_stress_multiplier": float(self.average_stress_multiplier),
            "average_health_multiplier": float(self.average_health_multiplier),
            "average_regime_multiplier": float(self.average_regime_multiplier),
            "low_budget_dates": int(self.low_budget_dates),
            "maximum_turnover": float(self.maximum_turnover),
            "turnover_violation_dates": int(self.turnover_violation_dates),
        }


def _safe_numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


class DynamicRiskBudgetOverlay:
    """Ex-ante volatility, stress, regime and model-health risk budget.

    Every multiplier is computed from information available on the signal date.
    The final allocation is a convex interpolation from the previous feasible
    portfolio to the desired scaled portfolio, so symbol/sector/gross caps remain
    valid while the turnover limit is enforced as the final operation.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def _daily_inputs(self, decisions: pd.DataFrame) -> pd.DataFrame:
        frame = decisions.copy()
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
        frame = frame.dropna(subset=["datetime"])
        rows: list[dict[str, Any]] = []
        for date, group in frame.groupby("datetime", sort=True):
            regime = (
                str(group["market_regime"].mode().iloc[0])
                if "market_regime" in group and not group["market_regime"].dropna().empty
                else "neutral"
            )
            health = (
                str(group["model_health_state"].mode().iloc[0])
                if "model_health_state" in group and not group["model_health_state"].dropna().empty
                else "cautious"
            )
            market_return = float(_safe_numeric(group, "market_return").median())
            stress = float(_safe_numeric(group, "market_stress_percentile", 0.5).median())
            breadth = int(
                (
                    _safe_numeric(group, "adaptive_score")
                    >= float(self.config.get("minimum_adaptive_score", 0.10))
                ).sum()
            )
            rows.append(
                {
                    "datetime": pd.Timestamp(date),
                    "market_return": market_return,
                    "market_stress_percentile": float(np.clip(stress, 0.0, 1.0)),
                    "market_regime": regime,
                    "model_health_state": health,
                    "eligible_signal_count": breadth,
                }
            )
        daily = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
        window = max(int(self.config.get("volatility_window", 20)), 2)
        minimum = max(int(self.config.get("minimum_volatility_observations", 10)), 2)
        annualization = int(self.config.get("annualization_bars", 252))
        daily["trailing_market_volatility"] = (
            daily["market_return"]
            .rolling(window, min_periods=minimum)
            .std(ddof=0)
            * np.sqrt(annualization)
        )
        target = float(self.config.get("target_annual_volatility", 0.12))
        vol_min = float(self.config.get("volatility_multiplier_min", 0.45))
        vol_max = float(self.config.get("volatility_multiplier_max", 1.05))
        daily["volatility_multiplier"] = (
            target / daily["trailing_market_volatility"].replace(0.0, np.nan)
        ).clip(vol_min, vol_max).fillna(1.0)
        stress_strength = float(self.config.get("stress_strength", 0.35))
        stress_min = float(self.config.get("stress_multiplier_min", 0.55))
        daily["stress_multiplier"] = (
            1.0 - stress_strength * daily["market_stress_percentile"]
        ).clip(stress_min, 1.0)
        daily["health_multiplier"] = daily["model_health_state"].map(
            self.config.get("health_multipliers", {})
        ).fillna(0.85).astype(float)
        daily["regime_multiplier"] = daily["market_regime"].map(
            self.config.get("regime_multipliers", {})
        ).fillna(0.90).astype(float)
        breadth_min = max(int(self.config.get("breadth_minimum", 4)), 1)
        breadth_penalty = float(self.config.get("breadth_penalty", 0.80))
        daily["breadth_multiplier"] = np.where(
            daily["eligible_signal_count"] < breadth_min,
            breadth_penalty,
            1.0,
        )
        weights = self.config.get("component_weights", {})
        volatility_weight = max(float(weights.get("volatility", 0.35)), 0.0)
        stress_weight = max(float(weights.get("stress", 0.20)), 0.0)
        health_weight = max(float(weights.get("health", 0.30)), 0.0)
        regime_weight = max(float(weights.get("regime", 0.15)), 0.0)
        weight_total = max(
            volatility_weight + stress_weight + health_weight + regime_weight,
            1e-12,
        )
        components = [
            (daily["volatility_multiplier"].clip(lower=1e-6), volatility_weight),
            (daily["stress_multiplier"].clip(lower=1e-6), stress_weight),
            (daily["health_multiplier"].clip(lower=1e-6), health_weight),
            (daily["regime_multiplier"].clip(lower=1e-6), regime_weight),
        ]
        log_budget = sum(weight * np.log(series) for series, weight in components) / weight_total
        raw = np.exp(log_budget) * daily["breadth_multiplier"]
        minimum_multiplier = float(self.config.get("minimum_multiplier", 0.25))
        maximum_multiplier = float(self.config.get("maximum_multiplier", 1.0))
        raw = raw.clip(minimum_multiplier, maximum_multiplier)
        recovery = float(np.clip(self.config.get("recovery_smoothing", 0.20), 0.0, 1.0))
        smoothed: list[float] = []
        current = 1.0
        for value in raw:
            value = float(value)
            current = value if value <= current else current + recovery * (value - current)
            current = float(np.clip(current, minimum_multiplier, maximum_multiplier))
            smoothed.append(current)
        daily["dynamic_risk_budget_multiplier"] = smoothed
        daily["dynamic_risk_budget_raw"] = raw
        return daily

    def apply(
        self,
        decisions: pd.DataFrame,
        allocations: pd.DataFrame,
    ) -> tuple[pd.DataFrame, DynamicRiskDiagnostics, pd.DataFrame]:
        daily = self._daily_inputs(decisions)
        frame = allocations.copy()
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
        frame = frame.merge(daily, on="datetime", how="left")
        frame["base_target_weight"] = _safe_numeric(frame, "target_weight")
        frame["dynamic_risk_budget_multiplier"] = _safe_numeric(
            frame, "dynamic_risk_budget_multiplier", 1.0
        )
        all_symbols = sorted(frame["symbol"].unique())
        previous = pd.Series(0.0, index=all_symbols, dtype=float)
        maximum_turnover = float(self.config.get("maximum_turnover", 0.50))
        tolerance = float(self.config.get("turnover_tolerance", 1e-10))
        records: list[pd.DataFrame] = []
        turnover_values: list[float] = []
        for date, group in frame.sort_values(["datetime", "symbol"]).groupby("datetime", sort=True):
            local = group.copy().set_index("symbol").reindex(all_symbols)
            multiplier = float(
                pd.to_numeric(
                    local["dynamic_risk_budget_multiplier"], errors="coerce"
                ).dropna().median()
            ) if local["dynamic_risk_budget_multiplier"].notna().any() else 1.0
            desired = _safe_numeric(local, "base_target_weight") * multiplier
            delta = desired - previous
            turnover = float(delta.abs().sum())
            if maximum_turnover > 0 and turnover > maximum_turnover + tolerance:
                desired = previous + delta * (maximum_turnover / max(turnover, 1e-12))
                delta = desired - previous
                turnover = float(delta.abs().sum())
            local["target_weight"] = desired.clip(lower=0.0)
            local["risk_budget_previous_weight"] = previous
            local["risk_budget_weight_change"] = delta
            local["risk_budget_portfolio_turnover"] = turnover
            local["datetime"] = pd.Timestamp(date)
            local["symbol"] = local.index
            records.append(local.reset_index(drop=True))
            turnover_values.append(turnover)
            previous = desired.clip(lower=0.0)
        adjusted = pd.concat(records, ignore_index=True).sort_values(
            ["datetime", "symbol"]
        ).reset_index(drop=True)
        diagnostics = DynamicRiskDiagnostics(
            dates=int(daily["datetime"].nunique()),
            average_multiplier=float(daily["dynamic_risk_budget_multiplier"].mean()),
            minimum_multiplier=float(daily["dynamic_risk_budget_multiplier"].min()),
            maximum_multiplier=float(daily["dynamic_risk_budget_multiplier"].max()),
            average_volatility_multiplier=float(daily["volatility_multiplier"].mean()),
            average_stress_multiplier=float(daily["stress_multiplier"].mean()),
            average_health_multiplier=float(daily["health_multiplier"].mean()),
            average_regime_multiplier=float(daily["regime_multiplier"].mean()),
            low_budget_dates=int(daily["dynamic_risk_budget_multiplier"].lt(0.65).sum()),
            maximum_turnover=float(max(turnover_values) if turnover_values else 0.0),
            turnover_violation_dates=int(
                sum(value > maximum_turnover + max(tolerance, 1e-9) for value in turnover_values)
            ),
        )
        return adjusted, diagnostics, daily


def save_dynamic_risk_outputs(
    allocations: pd.DataFrame,
    daily: pd.DataFrame,
    diagnostics: DynamicRiskDiagnostics,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    allocations.to_csv(
        output_dir / "risk_adjusted_target_weights.csv",
        index=False,
        encoding="utf-8-sig",
    )
    daily.to_csv(
        output_dir / "dynamic_risk_budget_daily.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_dir / "dynamic_risk_budget_diagnostics.json").write_text(
        json.dumps(diagnostics.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )