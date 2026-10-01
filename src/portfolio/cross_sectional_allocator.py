from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class AllocationDiagnostics:
    dates: int
    average_selected: float
    median_selected: float
    maximum_selected: int
    over_limit_dates: int
    average_gross_exposure: float
    average_turnover: float
    maximum_turnover: float
    turnover_violation_dates: int
    configured_turnover_limit: float
    maximum_symbol_weight: float
    maximum_sector_weight: float
    sector_constraint_applied: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "dates": int(self.dates),
            "average_selected": float(self.average_selected),
            "median_selected": float(self.median_selected),
            "maximum_selected": int(self.maximum_selected),
            "over_limit_dates": int(self.over_limit_dates),
            "average_gross_exposure": float(self.average_gross_exposure),
            "average_turnover": float(self.average_turnover),
            "maximum_turnover": float(self.maximum_turnover),
            "turnover_violation_dates": int(self.turnover_violation_dates),
            "configured_turnover_limit": float(self.configured_turnover_limit),
            "maximum_symbol_weight": float(self.maximum_symbol_weight),
            "maximum_sector_weight": float(self.maximum_sector_weight),
            "sector_constraint_applied": bool(self.sector_constraint_applied),
        }


def _normalize_symbol(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def load_security_master(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "sector"])
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        frame = pd.read_csv(path)
    if "symbol" not in frame.columns:
        return pd.DataFrame(columns=["symbol", "sector"])
    if "sector" not in frame.columns:
        for candidate in ["industry", "申万行业", "行业"]:
            if candidate in frame.columns:
                frame = frame.rename(columns={candidate: "sector"})
                break
    if "sector" not in frame.columns:
        frame["sector"] = "UNKNOWN"
    frame["symbol"] = _normalize_symbol(frame["symbol"])
    frame["sector"] = frame["sector"].astype(str).fillna("UNKNOWN")
    return frame[["symbol", "sector"]].drop_duplicates("symbol")


class CrossSectionalAllocator:
    """Long-only allocator with final hard turnover and concentration controls."""

    def __init__(self, config: dict[str, Any], project_root: Path) -> None:
        self.config = config
        master_path = Path(config.get("security_master_csv", "data/raw/security_master.csv"))
        if not master_path.is_absolute():
            master_path = project_root / master_path
        self.security_master = load_security_master(master_path)
        self.sector_constraint_available = (
            not self.security_master.empty
            and self.security_master["sector"].nunique() > 1
            and bool(config.get("use_sector_constraint", True))
        )

    @staticmethod
    def _safe_numeric(group: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
        if column not in group.columns:
            return pd.Series(default, index=group.index, dtype=float)
        return pd.to_numeric(group[column], errors="coerce").fillna(default)

    def _attach_sector(self, group: pd.DataFrame) -> pd.DataFrame:
        local = group.copy()
        local["symbol"] = _normalize_symbol(local["symbol"])
        if "sector" in local.columns:
            existing_sector = local["sector"].astype("string")
            valid = ~existing_sector.fillna("").str.upper().isin(
                {"", "UNKNOWN", "NAN", "NONE"}
            )
            if valid.mean() >= 0.95:
                local["sector"] = existing_sector.fillna("UNKNOWN").astype(str)
                return local.drop(
                    columns=["sector_x", "sector_y", "sector_master", "industry"],
                    errors="ignore",
                )
        existing = None
        for column in ["sector", "sector_x", "sector_y", "sector_master", "industry"]:
            if column in local.columns:
                values = local[column].astype("string")
                existing = values if existing is None else existing.fillna(values)
        local = local.drop(
            columns=["sector", "sector_x", "sector_y", "sector_master", "industry"],
            errors="ignore",
        )
        local = local.merge(
            self.security_master.rename(columns={"sector": "sector_master"}),
            on="symbol",
            how="left",
        )
        if existing is None:
            local["sector"] = local["sector_master"]
        else:
            existing = pd.Series(existing.to_numpy(), index=local.index, dtype="string")
            invalid = existing.fillna("").str.upper().isin({"", "UNKNOWN", "NAN", "NONE"})
            local["sector"] = existing.where(~invalid, local["sector_master"])
        local["sector"] = local["sector"].fillna("UNKNOWN").astype(str)
        return local.drop(columns=["sector_master"], errors="ignore")

    def _gross_target(self, group: pd.DataFrame) -> float:
        maximum = float(self.config.get("max_gross_exposure", 0.90))
        cash_buffer = float(self.config.get("cash_buffer", 0.10))
        regime_map = self.config.get("regime_gross_exposure", {})
        regime = (
            str(group["market_regime"].mode().iloc[0])
            if "market_regime" in group.columns and not group.empty
            else "neutral"
        )
        regime_gross = float(regime_map.get(regime, maximum))
        return float(np.clip(min(maximum, regime_gross, 1.0 - cash_buffer), 0.0, 1.0))

    @staticmethod
    def _cap_and_redistribute(
        weights: pd.Series,
        cap: float,
        total: float,
        max_iterations: int = 50,
    ) -> pd.Series:
        result = weights.clip(lower=0.0).copy()
        if result.sum() <= 0 or total <= 0:
            return result * 0.0
        result = result / result.sum() * total
        cap = max(float(cap), 0.0)
        for _ in range(max_iterations):
            over = result > cap + 1e-12
            if not over.any():
                break
            excess = float((result[over] - cap).sum())
            result.loc[over] = cap
            under = ~over
            room = (cap - result.loc[under]).clip(lower=0.0)
            if excess <= 1e-12 or room.sum() <= 1e-12:
                break
            allocation = excess * room / room.sum()
            result.loc[under] += np.minimum(allocation, room)
        if result.sum() > total + 1e-12:
            result *= total / result.sum()
        return result

    def _sector_series(self, group: pd.DataFrame, index: pd.Index) -> pd.Series:
        local = self._attach_sector(group)
        sector = local.drop_duplicates("symbol").set_index("symbol")["sector"]
        sector = sector.astype(str).reindex(index).fillna("UNKNOWN")
        return sector.where(
            ~sector.str.upper().isin({"UNKNOWN", "NAN", "NONE", ""}),
            "UNKNOWN_" + sector.index.astype(str),
        )

    def _apply_sector_cap(
        self,
        group: pd.DataFrame,
        weights: pd.Series,
        total: float,
    ) -> pd.Series:
        if not self.sector_constraint_available or weights.sum() <= 0:
            return weights
        max_sector = float(self.config.get("max_weight_per_sector", 0.28))
        result = weights.clip(lower=0.0).copy()
        sector = self._sector_series(group, result.index)
        for _ in range(50):
            sector_weights = result.groupby(sector).sum()
            violating = sector_weights[sector_weights > max_sector + 1e-12]
            if violating.empty:
                break
            freed = 0.0
            for sector_name, sector_weight in violating.items():
                members = result.index[sector.eq(sector_name)]
                before = float(result.loc[members].sum())
                result.loc[members] *= max_sector / float(sector_weight)
                freed += before - float(result.loc[members].sum())
            if freed <= 1e-12:
                break
            sector_room = max_sector - result.groupby(sector).transform("sum")
            eligible = result.index[(sector_room > 1e-12) & (result > 0)]
            if len(eligible) == 0:
                break
            base = result.loc[eligible]
            addition = freed * base / max(float(base.sum()), 1e-12)
            result.loc[eligible] += np.minimum(addition, sector_room.loc[eligible])
        if result.sum() > total + 1e-12:
            result *= total / result.sum()
        return result

    def _proposed_weights(self, group: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
        local = self._attach_sector(group)
        score = self._safe_numeric(local, "adaptive_score")
        confidence = self._safe_numeric(local, "adaptive_confidence")
        predicted = self._safe_numeric(local, "predicted_excess_return")
        tradable = self._safe_numeric(local, "is_tradable_sample", 1.0).gt(0)
        eligible = (
            tradable
            & score.ge(float(self.config.get("minimum_adaptive_score", 0.10)))
            & confidence.ge(float(self.config.get("minimum_adaptive_confidence", 0.20)))
            & predicted.ge(float(self.config.get("minimum_predicted_excess_return", 0.0)))
        )
        candidates = local.loc[eligible].copy()
        candidates["cross_sectional_rank"] = candidates["adaptive_score"].rank(
            method="first", ascending=False
        )
        rank_map = candidates.set_index("symbol")["cross_sectional_rank"]
        local["cross_sectional_rank"] = local["symbol"].map(rank_map)
        top_k = max(int(self.config.get("top_k", 8)), 1)
        candidates = candidates.nsmallest(top_k, "cross_sectional_rank")

        weights = pd.Series(0.0, index=local["symbol"].unique(), dtype=float)
        if candidates.empty:
            return weights, local

        attractiveness = (
            self._safe_numeric(candidates, "adaptive_score").clip(lower=0.0)
            * self._safe_numeric(candidates, "adaptive_confidence").clip(0.0, 1.0)
        )
        risk = self._safe_numeric(candidates, "risk_score").clip(0.0, 1.0)
        volatility = self._safe_numeric(candidates, "volatility_z").clip(lower=0.0)
        inverse_risk = 1.0 / (
            1.0
            + float(self.config.get("inverse_risk_strength", 0.65)) * risk
            + 0.25 * volatility
        )
        raw = attractiveness * inverse_risk
        if raw.sum() <= 0:
            raw = pd.Series(1.0, index=candidates.index)
        raw = raw / raw.sum()
        equal = pd.Series(1.0 / len(raw), index=raw.index)
        blend = float(np.clip(self.config.get("equal_weight_blend", 0.35), 0.0, 1.0))
        raw = (1.0 - blend) * raw + blend * equal

        gross = self._gross_target(local)
        proposed = pd.Series(raw.to_numpy(dtype=float), index=candidates["symbol"].to_numpy())
        proposed = self._cap_and_redistribute(
            proposed,
            cap=float(self.config.get("max_weight_per_symbol", 0.10)),
            total=gross,
        )
        proposed = self._apply_sector_cap(candidates, proposed, gross)
        weights.loc[proposed.index] = proposed
        return weights, local

    def _enforce_active_limit(
        self,
        target: pd.Series,
        proposed: pd.Series,
        enriched: pd.DataFrame,
    ) -> pd.Series:
        maximum = max(int(self.config.get("max_active_positions", 12)), 1)
        active = target[target.gt(0)]
        if len(active) <= maximum:
            return target
        lookup = enriched.drop_duplicates("symbol").set_index("symbol")
        adaptive = pd.to_numeric(
            lookup.get("adaptive_score", pd.Series(0.0, index=lookup.index)),
            errors="coerce",
        ).fillna(0.0)
        priority = (
            target
            + 0.50 * proposed.reindex(target.index, fill_value=0.0)
            + 0.01 * adaptive.reindex(target.index, fill_value=0.0).clip(lower=0.0)
        )
        keep = priority.loc[active.index].nlargest(maximum).index
        result = target.copy()
        result.loc[~result.index.isin(keep)] = 0.0
        return result

    def _hard_turnover_projection(
        self,
        previous: pd.Series,
        desired: pd.Series,
        proposed: pd.Series,
        enriched: pd.DataFrame,
        maximum_turnover: float,
    ) -> pd.Series:
        """Project the final target onto the turnover budget as the last operation.

        The active set is fixed before interpolation. Full exits consume budget
        first; remaining adjustments are scaled. No normalization or sector
        transformation is allowed after this method, so the limit is truly hard.
        """

        if maximum_turnover <= 0:
            return desired
        tolerance = float(self.config.get("turnover_tolerance", 1e-10))
        maximum_active = max(int(self.config.get("max_active_positions", 12)), 1)
        previous = previous.reindex(desired.index, fill_value=0.0).clip(lower=0.0)
        desired = desired.reindex(previous.index, fill_value=0.0).clip(lower=0.0)
        proposed = proposed.reindex(previous.index, fill_value=0.0).clip(lower=0.0)

        lookup = enriched.drop_duplicates("symbol").set_index("symbol")
        adaptive = pd.to_numeric(
            lookup.get("adaptive_score", pd.Series(0.0, index=lookup.index)),
            errors="coerce",
        ).fillna(0.0).reindex(previous.index, fill_value=0.0)
        priority = desired + 0.35 * previous + 0.20 * proposed + 0.01 * adaptive.clip(lower=0.0)
        union = previous.index[(previous.gt(0) | desired.gt(0))]
        keep = list(priority.loc[union].nlargest(min(maximum_active, len(union))).index)

        def forced_exit_weight(selected: list[str]) -> float:
            return float(previous.loc[~previous.index.isin(selected)].sum())

        # If desired new names force too much liquidation, retain the largest old
        # positions and postpone entries until the turnover budget permits them.
        old_ranked = list(previous[previous.gt(0)].sort_values(ascending=False).index)
        while forced_exit_weight(keep) > maximum_turnover + tolerance:
            missing_old = [symbol for symbol in old_ranked if symbol not in keep]
            replaceable = [symbol for symbol in keep if previous.get(symbol, 0.0) <= 0]
            if not missing_old or not replaceable:
                keep = old_ranked[:maximum_active]
                break
            remove = min(replaceable, key=lambda symbol: priority.get(symbol, 0.0))
            keep.remove(remove)
            keep.append(missing_old[0])

        desired_limited = desired.copy()
        desired_limited.loc[~desired_limited.index.isin(keep)] = 0.0
        forced_exit = previous.loc[~previous.index.isin(keep)]
        forced_turnover = float(forced_exit.sum())
        if forced_turnover > maximum_turnover + tolerance:
            return previous.copy()

        result = previous.copy()
        result.loc[~result.index.isin(keep)] = 0.0
        inside = pd.Index(keep)
        inside_delta = desired_limited.loc[inside] - previous.loc[inside]
        remaining_budget = max(maximum_turnover - forced_turnover, 0.0)
        inside_turnover = float(inside_delta.abs().sum())
        scale = 1.0 if inside_turnover <= remaining_budget + tolerance else remaining_budget / max(inside_turnover, 1e-12)
        result.loc[inside] = previous.loc[inside] + scale * inside_delta
        result = result.clip(lower=0.0)

        actual = float((result - previous).abs().sum())
        if actual > maximum_turnover + max(tolerance, 1e-9):
            # Numerical safety only; interpolation preserves all convex caps.
            delta = result - previous
            result = previous + delta * (maximum_turnover / actual)
        return result.clip(lower=0.0)

    def allocate(self, decisions: pd.DataFrame) -> tuple[pd.DataFrame, AllocationDiagnostics]:
        frame = decisions.copy()
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
        frame["symbol"] = _normalize_symbol(frame["symbol"])
        frame = frame.dropna(subset=["datetime", "symbol"]).sort_values(["datetime", "symbol"])
        frame = self._attach_sector(frame)
        all_symbols = sorted(frame["symbol"].unique())
        dates = sorted(frame["datetime"].unique())
        previous = pd.Series(0.0, index=all_symbols, dtype=float)
        records: list[dict[str, Any]] = []
        turnover_values: list[float] = []
        selected_values: list[int] = []
        gross_values: list[float] = []
        symbol_weight_values: list[float] = []
        sector_weight_values: list[float] = []

        rebalance_n = max(int(self.config.get("rebalance_every_n_dates", 5)), 1)
        smoothing = float(np.clip(self.config.get("position_smoothing", 0.65), 0.0, 1.0))
        exit_decay = float(np.clip(self.config.get("exit_decay", 0.45), 0.0, 1.0))
        strict_top_k = bool(self.config.get("strict_top_k", False))
        maximum_turnover = max(float(self.config.get("max_turnover_per_rebalance", 0.50)), 0.0)
        rebalance_band = max(float(self.config.get("rebalance_band", 0.01)), 0.0)
        minimum_weight = max(float(self.config.get("minimum_weight", 0.01)), 0.0)
        minimum_exit_weight = max(float(self.config.get("minimum_exit_weight", 0.005)), 0.0)
        max_active = max(int(self.config.get("max_active_positions", 12)), 1)
        tolerance = float(self.config.get("turnover_tolerance", 1e-10))

        for index, date in enumerate(dates):
            group = frame.loc[frame["datetime"].eq(date)].copy()
            proposed, enriched = self._proposed_weights(group)
            proposed = proposed.reindex(all_symbols, fill_value=0.0)
            is_rebalance = index % rebalance_n == 0
            if not is_rebalance:
                target = previous.copy()
            else:
                if strict_top_k:
                    target = pd.Series(0.0, index=all_symbols, dtype=float)
                    active_new = proposed.gt(0)
                    target.loc[active_new] = previous.loc[active_new] + smoothing * (
                        proposed.loc[active_new] - previous.loc[active_new]
                    )
                else:
                    target = previous.copy()
                    retained = proposed.gt(0)
                    exiting = (~retained) & previous.gt(0)
                    target.loc[retained] = previous.loc[retained] + smoothing * (
                        proposed.loc[retained] - previous.loc[retained]
                    )
                    target.loc[exiting] = previous.loc[exiting] * exit_decay

                target[target.abs() < minimum_exit_weight] = 0.0
                target = self._enforce_active_limit(target, proposed, enriched)
                delta = target - previous
                delta.loc[delta.abs() < rebalance_band] = 0.0
                target = previous + delta
                target[target.abs() < minimum_weight] = 0.0
                target = self._enforce_active_limit(target, proposed, enriched)

                maximum_gross = self._gross_target(group)
                target = target.clip(
                    lower=0.0,
                    upper=float(self.config.get("max_weight_per_symbol", 0.10)),
                )
                target = self._apply_sector_cap(enriched, target, maximum_gross)
                target = self._enforce_active_limit(target, proposed, enriched)
                if target.sum() > maximum_gross:
                    target *= maximum_gross / target.sum()

                # This must be the final transformation.
                target = self._hard_turnover_projection(
                    previous,
                    target,
                    proposed,
                    enriched,
                    maximum_turnover,
                )

            delta = target - previous
            turnover = float(delta.abs().sum())
            selected = int(target.gt(1e-12).sum())
            gross = float(target.sum())
            turnover_values.append(turnover)
            selected_values.append(selected)
            gross_values.append(gross)
            symbol_weight_values.append(float(target.max()) if len(target) else 0.0)
            sectors = self._sector_series(enriched, target.index)
            sector_weight_values.append(float(target.groupby(sectors).sum().max()) if target.sum() > 0 else 0.0)

            lookup = enriched.drop_duplicates("symbol").set_index("symbol")
            for symbol in all_symbols:
                row = lookup.loc[symbol] if symbol in lookup.index else None
                records.append(
                    {
                        "datetime": pd.Timestamp(date),
                        "symbol": symbol,
                        "target_weight": float(target.get(symbol, 0.0)),
                        "previous_weight": float(previous.get(symbol, 0.0)),
                        "weight_change": float(delta.get(symbol, 0.0)),
                        "portfolio_turnover": turnover,
                        "portfolio_gross_exposure": gross,
                        "is_rebalance_date": int(is_rebalance),
                        "adaptive_score": float(row.get("adaptive_score", 0.0)) if row is not None else 0.0,
                        "adaptive_confidence": float(row.get("adaptive_confidence", 0.0)) if row is not None else 0.0,
                        "predicted_excess_return": float(row.get("predicted_excess_return", 0.0)) if row is not None else 0.0,
                        "market_regime": str(row.get("market_regime", "neutral")) if row is not None else "neutral",
                        "sector": str(row.get("sector", "UNKNOWN")) if row is not None else "UNKNOWN",
                        "cross_sectional_rank": float(row.get("cross_sectional_rank", np.nan)) if row is not None else np.nan,
                    }
                )
            previous = target

        allocations = pd.DataFrame(records).sort_values(["datetime", "symbol"]).reset_index(drop=True)
        diagnostics = AllocationDiagnostics(
            dates=len(dates),
            average_selected=float(np.mean(selected_values)) if selected_values else 0.0,
            median_selected=float(np.median(selected_values)) if selected_values else 0.0,
            maximum_selected=int(np.max(selected_values)) if selected_values else 0,
            over_limit_dates=int(sum(value > max_active for value in selected_values)),
            average_gross_exposure=float(np.mean(gross_values)) if gross_values else 0.0,
            average_turnover=float(np.mean(turnover_values)) if turnover_values else 0.0,
            maximum_turnover=float(np.max(turnover_values)) if turnover_values else 0.0,
            turnover_violation_dates=int(
                sum(value > maximum_turnover + max(tolerance, 1e-9) for value in turnover_values)
            ),
            configured_turnover_limit=maximum_turnover,
            maximum_symbol_weight=float(np.max(symbol_weight_values)) if symbol_weight_values else 0.0,
            maximum_sector_weight=float(np.max(sector_weight_values)) if sector_weight_values else 0.0,
            sector_constraint_applied=self.sector_constraint_available,
        )
        return allocations, diagnostics