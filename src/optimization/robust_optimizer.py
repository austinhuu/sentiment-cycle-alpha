from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.optimization.covariance import open_to_open_returns, shrink_covariance
from src.optimization.risk_attribution import portfolio_risk_attribution


@dataclass
class OptimizationDiagnostics:
    dates: int
    optimized_dates: int
    carried_dates: int
    failed_dates: int
    success_share: float
    average_ex_ante_volatility: float
    maximum_ex_ante_volatility: float
    average_effective_positions: float
    average_turnover: float
    maximum_turnover: float
    maximum_symbol_weight: float
    maximum_sector_weight: float
    average_covariance_condition_number: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


class RobustPortfolioOptimizer:
    """Long-only robust mean-variance optimizer anchored to the accepted V5 portfolio.

    The optimizer only uses returns observed on or before each signal date. V5
    risk-adjusted weights define the gross risk budget and act as a stable prior.
    """

    def __init__(self, config: dict[str, Any], project_root: Path) -> None:
        self.config = config
        self.project_root = Path(project_root)
        self.security_master = self._load_security_master()

    def _load_security_master(self) -> pd.DataFrame:
        path = self.project_root / self.config.get(
            "security_master_csv", "data/raw/security_master.csv"
        )
        if not path.exists():
            return pd.DataFrame(columns=["symbol", "sector"])
        frame = pd.read_csv(path)
        if "symbol" not in frame.columns:
            return pd.DataFrame(columns=["symbol", "sector"])
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
        sector_column = next(
            (column for column in ["sector", "industry", "industry_name"] if column in frame),
            None,
        )
        frame["sector"] = (
            frame[sector_column].fillna("UNKNOWN").astype(str)
            if sector_column
            else "UNKNOWN"
        )
        return frame[["symbol", "sector"]].drop_duplicates("symbol")

    def _attach_sector(self, frame: pd.DataFrame) -> pd.DataFrame:
        local = frame.copy()
        local["symbol"] = local["symbol"].astype(str).str.upper()
        for column in ["sector_x", "sector_y", "sector_master"]:
            if column in local:
                local = local.drop(columns=column)
        if "sector" not in local:
            local["sector"] = pd.Series("UNKNOWN", index=local.index, dtype=object)
        else:
            local["sector"] = local["sector"].astype(object)
        mapping = self.security_master.rename(columns={"sector": "sector_master"})
        local = local.merge(mapping, on="symbol", how="left")
        current = local["sector"].astype(str)
        invalid = local["sector"].isna() | current.str.upper().isin(
            {"", "NAN", "NONE", "UNKNOWN"}
        )
        local.loc[invalid, "sector"] = local.loc[invalid, "sector_master"]
        local["sector"] = local["sector"].fillna("UNKNOWN").astype(str)
        return local.drop(columns=["sector_master"], errors="ignore")

    def _expected_returns(self, group: pd.DataFrame) -> pd.Series:
        prediction_column = str(
            self.config.get("expected_return_column", "production_predicted_excess_return")
        )
        confidence_column = str(
            self.config.get("confidence_column", "production_confidence")
        )
        horizon = max(int(self.config.get("return_horizon", 5)), 1)
        prediction = _numeric(group, prediction_column)
        confidence = _numeric(group, confidence_column, 0.5).clip(0.0, 1.0)
        health = group.get(
            "model_health_state", pd.Series("healthy", index=group.index)
        ).astype(str)
        health_multiplier_map = self.config.get(
            "health_expected_return_multipliers",
            {"healthy": 1.0, "cautious": 0.80, "fallback": 0.60, "defensive": 0.35},
        )
        health_multiplier = health.map(health_multiplier_map).fillna(0.70).astype(float)
        daily = prediction / horizon
        robust = daily * confidence.pow(float(self.config.get("confidence_power", 1.0)))
        robust *= health_multiplier
        clip = float(self.config.get("expected_return_daily_clip", 0.015))
        return robust.clip(-clip, clip)

    @staticmethod
    def _effective_positions(weights: pd.Series) -> float:
        gross = float(weights.sum())
        if gross <= 1.0e-12:
            return 0.0
        normalized = weights / gross
        return float(1.0 / max(float((normalized**2).sum()), 1.0e-12))

    def _project_caps(
        self,
        weights: pd.Series,
        gross: float,
        sector: pd.Series,
    ) -> pd.Series:
        result = weights.clip(lower=0.0).copy()
        if result.sum() <= 1.0e-12 or gross <= 0:
            return result * 0.0
        result *= gross / result.sum()
        max_symbol = float(self.config.get("max_weight_per_symbol", 0.10))
        max_sector = float(self.config.get("max_weight_per_sector", 0.28))
        for _ in range(80):
            before = result.copy()
            result = result.clip(upper=max_symbol)
            sector_sum = result.groupby(sector).sum()
            for name, value in sector_sum.items():
                if value > max_sector + 1.0e-12:
                    members = sector.index[sector.eq(name)]
                    result.loc[members] *= max_sector / value
            remaining = gross - float(result.sum())
            if remaining > 1.0e-12:
                symbol_room = (max_symbol - result).clip(lower=0.0)
                sector_room = max_sector - result.groupby(sector).transform("sum")
                room = pd.concat([symbol_room, sector_room.clip(lower=0.0)], axis=1).min(axis=1)
                eligible = room[room.gt(1.0e-12)]
                if not eligible.empty:
                    addition = remaining * eligible / eligible.sum()
                    result.loc[eligible.index] += np.minimum(addition, eligible)
            if float((result - before).abs().sum()) <= 1.0e-12:
                break
        if result.sum() > gross + 1.0e-12:
            result *= gross / result.sum()
        return result.clip(lower=0.0)

    def _turnover_projection(
        self,
        previous: pd.Series,
        desired: pd.Series,
        maximum: float,
    ) -> pd.Series:
        previous = previous.reindex(desired.index, fill_value=0.0)
        desired = desired.reindex(previous.index, fill_value=0.0)
        delta = desired - previous
        turnover = float(delta.abs().sum())
        if turnover <= maximum + 1.0e-12:
            return desired
        scale = max(float(maximum), 0.0) / max(turnover, 1.0e-12)
        return (previous + scale * delta).clip(lower=0.0)

    def _optimize_one(
        self,
        date: pd.Timestamp,
        group: pd.DataFrame,
        prior: pd.Series,
        previous: pd.Series,
        history: pd.DataFrame,
    ) -> tuple[pd.Series, dict[str, Any], pd.DataFrame]:
        local = self._attach_sector(group).drop_duplicates("symbol").set_index("symbol")
        mu = self._expected_returns(local)
        score_column = str(self.config.get("score_column", "production_score"))
        confidence_column = str(self.config.get("confidence_column", "production_confidence"))
        score = _numeric(local, score_column)
        confidence = _numeric(local, confidence_column, 0.5)
        tradable = _numeric(local, "is_tradable_sample", 1.0).gt(0)
        minimum_score = float(self.config.get("minimum_score", 0.0))
        minimum_confidence = float(self.config.get("minimum_confidence", 0.15))
        eligible = tradable & score.ge(minimum_score) & confidence.ge(minimum_confidence)
        candidate_top_k = max(int(self.config.get("candidate_top_k", 12)), 1)
        ranked = mu.where(eligible, -np.inf).nlargest(candidate_top_k).index.tolist()
        incumbent = prior[prior.gt(0)].index.tolist() + previous[previous.gt(0)].index.tolist()
        symbols = list(dict.fromkeys(ranked + incumbent))
        max_active = max(int(self.config.get("max_active_positions", 12)), 1)
        if len(symbols) > max_active:
            priority = mu.reindex(symbols, fill_value=-np.inf) + 0.01 * prior.reindex(
                symbols, fill_value=0.0
            )
            symbols = priority.nlargest(max_active).index.tolist()
        if not symbols:
            empty = pd.Series(0.0, index=local.index)
            diagnostics = {
                "datetime": date,
                "status": "no_candidates",
                "ex_ante_volatility": 0.0,
                "effective_positions": 0.0,
                "turnover": float((empty - previous.reindex(empty.index, fill_value=0.0)).abs().sum()),
            }
            return empty, diagnostics, pd.DataFrame()

        gross = min(
            float(prior.sum()),
            float(self.config.get("max_gross_exposure", 0.90)),
        )
        if gross <= 1.0e-12:
            gross = min(
                float(self.config.get("default_gross_exposure", 0.50)),
                float(self.config.get("max_gross_exposure", 0.90)),
            )
        selected_history = history.loc[history.index <= date, symbols].tail(
            int(self.config.get("covariance_lookback_dates", 120))
        )
        estimate = shrink_covariance(
            selected_history,
            shrinkage=float(self.config.get("covariance_shrinkage", 0.45)),
            variance_floor=float(self.config.get("covariance_variance_floor", 1.0e-6)),
        )
        covariance = estimate.matrix
        selected_mu = mu.reindex(symbols, fill_value=0.0).to_numpy(dtype=float)
        selected_prior = prior.reindex(symbols, fill_value=0.0)
        if selected_prior.sum() <= 1.0e-12:
            selected_prior = pd.Series(gross / len(symbols), index=symbols)
        else:
            selected_prior *= gross / selected_prior.sum()
        sector = local["sector"].reindex(symbols).fillna("UNKNOWN")
        weights = self._project_caps(selected_prior, gross, sector)

        risk_aversion = float(self.config.get("risk_aversion", 8.0))
        prior_penalty = float(self.config.get("prior_anchor_penalty", 0.20))
        concentration_penalty = float(self.config.get("concentration_penalty", 0.05))
        learning_rate = float(self.config.get("learning_rate", 0.08))
        maximum_iterations = max(int(self.config.get("max_iterations", 300)), 1)
        tolerance = float(self.config.get("tolerance", 1.0e-8))
        prior_array = selected_prior.to_numpy(dtype=float)
        status = "max_iterations"
        for _ in range(maximum_iterations):
            array = weights.to_numpy(dtype=float)
            gradient = (
                selected_mu
                - risk_aversion * (covariance @ array)
                - 2.0 * prior_penalty * (array - prior_array)
                - 2.0 * concentration_penalty * array
            )
            candidate = pd.Series(array + learning_rate * gradient, index=symbols)
            candidate = self._project_caps(candidate, gross, sector)
            if float((candidate - weights).abs().sum()) <= tolerance:
                weights = candidate
                status = "converged"
                break
            weights = candidate

        variance = float(weights.to_numpy() @ covariance @ weights.to_numpy())
        annualization = int(self.config.get("annualization_bars", 252))
        ex_ante_vol = float(np.sqrt(max(variance, 0.0)) * np.sqrt(annualization))
        target_vol = float(self.config.get("target_annual_volatility", 0.12))
        if ex_ante_vol > target_vol > 0:
            weights *= target_vol / ex_ante_vol
            ex_ante_vol = target_vol
        full = pd.Series(0.0, index=local.index, dtype=float)
        full.loc[weights.index] = weights
        union_index = full.index.union(previous.index)
        full = full.reindex(union_index, fill_value=0.0)
        previous_aligned = previous.reindex(union_index, fill_value=0.0)
        full = self._turnover_projection(
            previous_aligned,
            full,
            maximum=float(self.config.get("max_turnover_per_rebalance", 0.50)),
        )
        full = full.reindex(local.index, fill_value=0.0)
        turnover = float((full.reindex(previous.index.union(full.index), fill_value=0.0) - previous.reindex(previous.index.union(full.index), fill_value=0.0)).abs().sum())
        risk_table = portfolio_risk_attribution(
            symbols,
            full.reindex(symbols, fill_value=0.0).to_numpy(dtype=float),
            covariance,
            annualization,
        )
        diagnostics = {
            "datetime": date,
            "status": status,
            "candidate_count": len(symbols),
            "gross_target": gross,
            "gross_exposure": float(full.sum()),
            "expected_daily_return": float(
                selected_mu @ full.reindex(symbols, fill_value=0.0).to_numpy(dtype=float)
            ),
            "ex_ante_volatility": ex_ante_vol,
            "effective_positions": self._effective_positions(full),
            "turnover": turnover,
            "maximum_symbol_weight": float(full.max()) if len(full) else 0.0,
            "maximum_sector_weight": float(full.groupby(local["sector"]).sum().max()) if len(full) else 0.0,
            "covariance_observations": estimate.observations,
            "covariance_condition_number": estimate.condition_number,
            "average_correlation": estimate.average_correlation,
        }
        risk_table["datetime"] = date
        return full, diagnostics, risk_table

    def optimize(
        self,
        decisions: pd.DataFrame,
        prior_allocations: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, OptimizationDiagnostics]:
        frame = decisions.copy()
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
        frame = frame.dropna(subset=["datetime", "symbol"]).sort_values(
            ["datetime", "symbol"]
        )
        prior_frame = prior_allocations.copy()
        prior_frame["datetime"] = pd.to_datetime(prior_frame["datetime"], errors="coerce")
        prior_frame["symbol"] = prior_frame["symbol"].astype(str).str.upper()
        prior_weight_column = str(self.config.get("prior_weight_column", "target_weight"))
        if prior_weight_column not in prior_frame:
            raise KeyError(f"Prior allocation column not found: {prior_weight_column}")
        prior_frame[prior_weight_column] = pd.to_numeric(
            prior_frame[prior_weight_column], errors="coerce"
        ).fillna(0.0)
        returns = open_to_open_returns(frame)
        dates = [pd.Timestamp(value) for value in sorted(frame["datetime"].unique())]
        symbols = sorted(frame["symbol"].unique())
        previous = pd.Series(0.0, index=symbols, dtype=float)
        rebalance_every = max(int(self.config.get("rebalance_every_n_dates", 5)), 1)
        allocation_parts: list[pd.DataFrame] = []
        diagnostic_rows: list[dict[str, Any]] = []
        risk_parts: list[pd.DataFrame] = []

        for date_index, date in enumerate(dates):
            group = frame.loc[frame["datetime"].eq(date)].copy()
            prior_group = prior_frame.loc[prior_frame["datetime"].eq(date)]
            prior = prior_group.set_index("symbol")[prior_weight_column].reindex(
                symbols, fill_value=0.0
            )
            should_optimize = date_index % rebalance_every == 0 or date_index == 0
            if should_optimize:
                target, diagnostics, risk_table = self._optimize_one(
                    date,
                    group,
                    prior,
                    previous,
                    returns,
                )
                if not risk_table.empty:
                    risk_parts.append(risk_table)
            else:
                target = previous.copy()
                diagnostics = {
                    "datetime": date,
                    "status": "carried",
                    "candidate_count": int(target.gt(0).sum()),
                    "gross_target": float(prior.sum()),
                    "gross_exposure": float(target.sum()),
                    "expected_daily_return": np.nan,
                    "ex_ante_volatility": np.nan,
                    "effective_positions": self._effective_positions(target),
                    "turnover": 0.0,
                    "maximum_symbol_weight": float(target.max()) if len(target) else 0.0,
                    "maximum_sector_weight": np.nan,
                    "covariance_observations": 0,
                    "covariance_condition_number": np.nan,
                    "average_correlation": np.nan,
                }
            local = pd.DataFrame(
                {
                    "datetime": date,
                    "symbol": symbols,
                    "target_weight": target.reindex(symbols, fill_value=0.0).to_numpy(),
                    "v5_prior_weight": prior.reindex(symbols, fill_value=0.0).to_numpy(),
                    "optimizer_status": diagnostics["status"],
                    "optimizer_ex_ante_volatility": diagnostics.get("ex_ante_volatility", np.nan),
                    "optimizer_effective_positions": diagnostics.get("effective_positions", 0.0),
                    "optimizer_turnover": diagnostics.get("turnover", 0.0),
                }
            )
            allocation_parts.append(local)
            diagnostic_rows.append(diagnostics)
            previous = target.reindex(symbols, fill_value=0.0)

        allocations = pd.concat(allocation_parts, ignore_index=True)
        daily = pd.DataFrame(diagnostic_rows)
        risk = pd.concat(risk_parts, ignore_index=True) if risk_parts else pd.DataFrame()
        optimized = int(daily["status"].isin(["converged", "max_iterations"]).sum())
        failed = int(daily["status"].isin(["no_candidates", "failed"]).sum())
        carried = int(daily["status"].eq("carried").sum())
        diagnostics = OptimizationDiagnostics(
            dates=len(daily),
            optimized_dates=optimized,
            carried_dates=carried,
            failed_dates=failed,
            success_share=float(optimized / max(optimized + failed, 1)),
            average_ex_ante_volatility=float(pd.to_numeric(daily["ex_ante_volatility"], errors="coerce").mean()),
            maximum_ex_ante_volatility=float(pd.to_numeric(daily["ex_ante_volatility"], errors="coerce").max()),
            average_effective_positions=float(pd.to_numeric(daily["effective_positions"], errors="coerce").mean()),
            average_turnover=float(pd.to_numeric(daily["turnover"], errors="coerce").mean()),
            maximum_turnover=float(pd.to_numeric(daily["turnover"], errors="coerce").max()),
            maximum_symbol_weight=float(pd.to_numeric(daily["maximum_symbol_weight"], errors="coerce").max()),
            maximum_sector_weight=float(pd.to_numeric(daily["maximum_sector_weight"], errors="coerce").max()),
            average_covariance_condition_number=float(pd.to_numeric(daily["covariance_condition_number"], errors="coerce").replace([np.inf, -np.inf], np.nan).mean()),
        )
        return allocations, daily, risk, diagnostics


def save_optimizer_outputs(
    allocations: pd.DataFrame,
    daily: pd.DataFrame,
    risk: pd.DataFrame,
    diagnostics: OptimizationDiagnostics,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    allocations.to_csv(output_dir / "optimized_target_weights.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(output_dir / "optimizer_daily_diagnostics.csv", index=False, encoding="utf-8-sig")
    risk.to_csv(output_dir / "risk_attribution.csv", index=False, encoding="utf-8-sig")
    (output_dir / "optimizer_summary.json").write_text(
        json.dumps(diagnostics.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )