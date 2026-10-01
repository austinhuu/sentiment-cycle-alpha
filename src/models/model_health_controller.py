from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STATE_SEVERITY = {
    "healthy": 0,
    "cautious": 1,
    "fallback": 2,
    "defensive": 3,
}


@dataclass
class ModelHealthDiagnostics:
    dates: int
    state_counts: dict[str, int]
    state_shares: dict[str, float]
    transitions: int
    fallback_dates: int
    defensive_dates: int
    recent_adaptive_ic: float
    recent_baseline_ic: float
    recent_relative_ic: float
    minimum_health_score: float
    average_health_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "dates": int(self.dates),
            "state_counts": {key: int(value) for key, value in self.state_counts.items()},
            "state_shares": {key: float(value) for key, value in self.state_shares.items()},
            "transitions": int(self.transitions),
            "fallback_dates": int(self.fallback_dates),
            "defensive_dates": int(self.defensive_dates),
            "recent_adaptive_ic": float(self.recent_adaptive_ic),
            "recent_baseline_ic": float(self.recent_baseline_ic),
            "recent_relative_ic": float(self.recent_relative_ic),
            "minimum_health_score": float(self.minimum_health_score),
            "average_health_score": float(self.average_health_score),
        }


def _safe_numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _daily_rank_ic(frame: pd.DataFrame, score: str, target: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for date, group in frame.groupby("datetime", sort=True):
        x = _safe_numeric(group, score, np.nan)
        y = _safe_numeric(group, target, np.nan)
        valid = x.notna() & y.notna()
        value = np.nan
        if valid.sum() >= 3 and x[valid].nunique() >= 2 and y[valid].nunique() >= 2:
            value = x[valid].corr(y[valid], method="spearman")
        records.append({"datetime": pd.Timestamp(date), "rank_ic": value})
    return pd.DataFrame(records)


def _attach_available_date(
    daily: pd.DataFrame,
    all_dates: list[pd.Timestamp],
    horizon: int,
) -> pd.DataFrame:
    position = {date: index for index, date in enumerate(all_dates)}
    available: list[pd.Timestamp | pd.NaT] = []
    for date in daily["datetime"]:
        index = position.get(pd.Timestamp(date))
        if index is None or index + horizon >= len(all_dates):
            available.append(pd.NaT)
        else:
            available.append(all_dates[index + horizon])
    result = daily.copy()
    result["available_date"] = available
    return result


def _rolling_available_ic(
    daily: pd.DataFrame,
    dates: list[pd.Timestamp],
    window: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    available = daily.dropna(subset=["available_date"]).sort_values("available_date")
    for current in dates:
        known = available.loc[available["available_date"].le(current), "rank_ic"].dropna()
        if window > 0:
            known = known.tail(window)
        records.append(
            {
                "datetime": current,
                "rolling_ic": float(known.mean()) if len(known) else np.nan,
                "rolling_ic_observations": int(len(known)),
            }
        )
    return pd.DataFrame(records)


class ModelHealthController:
    """No-lookahead adaptive/fallback controller for V5 production decisions."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def _baseline(self, frame: pd.DataFrame) -> pd.DataFrame:
        technical_weight = float(self.config.get("technical_weight", 0.88))
        sentiment_weight = float(self.config.get("sentiment_weight", 0.12))
        total = max(technical_weight + sentiment_weight, 1e-12)
        technical_weight /= total
        sentiment_weight /= total
        technical = _safe_numeric(frame, "technical_score")
        sentiment = _safe_numeric(frame, "sentiment_score")
        technical_confidence = _safe_numeric(frame, "technical_confidence", 0.5)
        sentiment_confidence = _safe_numeric(frame, "sentiment_confidence", 0.5)
        baseline_score = (technical_weight * technical + sentiment_weight * sentiment).clip(-1.0, 1.0)
        baseline_confidence = (
            technical_weight * technical_confidence
            + sentiment_weight * sentiment_confidence
        ).clip(0.0, 1.0)
        scale = float(self.config.get("fallback_prediction_scale", 0.025))
        return pd.DataFrame(
            {
                "fallback_score": baseline_score,
                "fallback_confidence": baseline_confidence,
                "fallback_predicted_excess_return": baseline_score * scale,
            },
            index=frame.index,
        )

    def _candidate_state(
        self,
        validation_ic: float,
        adaptive_ic: float,
        baseline_ic: float,
        observations: int,
        fold_fallback: bool,
    ) -> tuple[str, float, str]:
        healthy_validation = float(self.config.get("healthy_validation_ic", 0.025))
        cautious_validation = float(self.config.get("cautious_validation_ic", 0.010))
        fallback_validation = float(self.config.get("fallback_validation_ic", 0.0))
        cautious_rolling = float(self.config.get("rolling_ic_cautious", -0.005))
        fallback_rolling = float(self.config.get("rolling_ic_fallback", -0.015))
        defensive_rolling = float(self.config.get("rolling_ic_defensive", -0.030))
        relative_fallback = float(self.config.get("relative_ic_fallback", -0.010))
        minimum_observations = int(self.config.get("minimum_ic_observations", 40))

        validation_component = float(np.clip((validation_ic + 0.02) / 0.08, 0.0, 1.0))
        adaptive_value = adaptive_ic if np.isfinite(adaptive_ic) else 0.0
        baseline_value = baseline_ic if np.isfinite(baseline_ic) else 0.0
        rolling_component = float(np.clip((adaptive_value + 0.04) / 0.08, 0.0, 1.0))
        relative = adaptive_value - baseline_value
        relative_component = float(np.clip((relative + 0.03) / 0.06, 0.0, 1.0))
        observation_component = float(np.clip(observations / max(minimum_observations, 1), 0.0, 1.0))
        health_score = (
            0.35 * validation_component
            + 0.35 * rolling_component
            + 0.20 * relative_component
            + 0.10 * observation_component
        )

        if fold_fallback:
            return "fallback", min(health_score, 0.34), "fold_validation_fallback"
        if observations >= minimum_observations and adaptive_value <= defensive_rolling:
            return "defensive", min(health_score, 0.19), "rolling_ic_defensive"
        if validation_ic < fallback_validation:
            return "fallback", min(health_score, 0.34), "validation_ic_negative"
        if observations >= minimum_observations and (
            adaptive_value < fallback_rolling or relative < relative_fallback
        ):
            return "fallback", min(health_score, 0.34), "rolling_or_relative_ic_weak"
        if validation_ic < cautious_validation:
            return "cautious", min(health_score, 0.54), "validation_ic_low"
        if observations >= minimum_observations and adaptive_value < cautious_rolling:
            return "cautious", min(health_score, 0.54), "rolling_ic_cautious"
        if validation_ic >= healthy_validation and (
            observations < minimum_observations or adaptive_value >= 0.0
        ):
            return "healthy", max(health_score, 0.56), "healthy"
        return "cautious", float(np.clip(health_score, 0.35, 0.70)), "mixed_evidence"

    def apply(self, decisions: pd.DataFrame) -> tuple[pd.DataFrame, ModelHealthDiagnostics]:
        frame = decisions.copy()
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
        frame = frame.dropna(subset=["datetime", "symbol"]).sort_values(
            ["datetime", "symbol"]
        ).reset_index(drop=True)
        baseline = self._baseline(frame)
        for column in baseline:
            frame[column] = baseline[column]

        target_horizon = int(self.config.get("target_horizon", 5))
        target = f"future_return_{target_horizon}"
        dates = [pd.Timestamp(value) for value in sorted(frame["datetime"].unique())]
        adaptive_daily = _daily_rank_ic(frame, "predicted_excess_return", target)
        baseline_daily = _daily_rank_ic(frame, "fallback_score", target)
        adaptive_daily = _attach_available_date(adaptive_daily, dates, target_horizon)
        baseline_daily = _attach_available_date(baseline_daily, dates, target_horizon)
        window = int(self.config.get("rolling_ic_dates", 120))
        adaptive_roll = _rolling_available_ic(adaptive_daily, dates, window).rename(
            columns={
                "rolling_ic": "rolling_adaptive_ic",
                "rolling_ic_observations": "rolling_adaptive_ic_observations",
            }
        )
        baseline_roll = _rolling_available_ic(baseline_daily, dates, window).rename(
            columns={
                "rolling_ic": "rolling_fallback_ic",
                "rolling_ic_observations": "rolling_fallback_ic_observations",
            }
        )
        daily = adaptive_roll.merge(baseline_roll, on="datetime", how="outer")

        daily_validation = frame.groupby("datetime")["fusion_validation_ic"].median() if "fusion_validation_ic" in frame else pd.Series(0.0, index=dates)
        daily_fallback = frame.groupby("datetime")["fusion_fallback_used"].max() if "fusion_fallback_used" in frame else pd.Series(0.0, index=dates)
        daily = daily.merge(
            daily_validation.rename("fold_validation_ic"),
            left_on="datetime",
            right_index=True,
            how="left",
        ).merge(
            daily_fallback.rename("fold_fallback_used"),
            left_on="datetime",
            right_index=True,
            how="left",
        )
        daily["fold_validation_ic"] = pd.to_numeric(daily["fold_validation_ic"], errors="coerce").fillna(0.0)
        daily["fold_fallback_used"] = pd.to_numeric(daily["fold_fallback_used"], errors="coerce").fillna(0).astype(int)

        minimum_days = max(int(self.config.get("state_minimum_days", 20)), 0)
        current_state = "cautious"
        last_change = -minimum_days
        states: list[str] = []
        health_scores: list[float] = []
        reasons: list[str] = []
        for index, row in daily.sort_values("datetime").reset_index(drop=True).iterrows():
            candidate, score, reason = self._candidate_state(
                validation_ic=float(row.get("fold_validation_ic", 0.0)),
                adaptive_ic=float(row.get("rolling_adaptive_ic", np.nan)),
                baseline_ic=float(row.get("rolling_fallback_ic", np.nan)),
                observations=int(row.get("rolling_adaptive_ic_observations", 0)),
                fold_fallback=bool(row.get("fold_fallback_used", 0)),
            )
            candidate_severity = STATE_SEVERITY[candidate]
            current_severity = STATE_SEVERITY[current_state]
            if candidate_severity > current_severity:
                current_state = candidate
                last_change = index
            elif candidate_severity < current_severity and index - last_change >= minimum_days:
                current_state = candidate
                last_change = index
            states.append(current_state)
            health_scores.append(score)
            reasons.append(reason if current_state == candidate else "state_hysteresis")
        daily = daily.sort_values("datetime").reset_index(drop=True)
        daily["model_health_state"] = states
        daily["model_health_score"] = health_scores
        daily["model_health_reason"] = reasons
        daily["rolling_relative_ic"] = daily["rolling_adaptive_ic"] - daily["rolling_fallback_ic"]

        frame = frame.merge(daily, on="datetime", how="left")
        frame["adaptive_score_model"] = _safe_numeric(frame, "adaptive_score")
        frame["adaptive_confidence_model"] = _safe_numeric(frame, "adaptive_confidence")
        frame["predicted_excess_return_model"] = _safe_numeric(
            frame, "predicted_excess_return"
        )
        adaptive_weights = self.config.get("adaptive_weight_by_state", {})
        confidence_multipliers = self.config.get("confidence_multiplier_by_state", {})
        frame["model_health_adaptive_weight"] = frame["model_health_state"].map(
            adaptive_weights
        ).fillna(0.0).astype(float)
        frame["model_health_confidence_multiplier"] = frame["model_health_state"].map(
            confidence_multipliers
        ).fillna(0.8).astype(float)
        weight = frame["model_health_adaptive_weight"].clip(0.0, 1.0)
        frame["production_score"] = (
            weight * frame["adaptive_score_model"]
            + (1.0 - weight) * frame["fallback_score"]
        ).clip(-1.0, 1.0)
        frame["production_confidence"] = (
            weight * frame["adaptive_confidence_model"]
            + (1.0 - weight) * frame["fallback_confidence"]
        ) * frame["model_health_confidence_multiplier"]
        frame["production_confidence"] = frame["production_confidence"].clip(0.0, 1.0)
        frame["production_predicted_excess_return"] = (
            weight * frame["predicted_excess_return_model"]
            + (1.0 - weight) * frame["fallback_predicted_excess_return"]
        )
        frame["adaptive_score"] = frame["production_score"]
        frame["adaptive_confidence"] = frame["production_confidence"]
        frame["predicted_excess_return"] = frame[
            "production_predicted_excess_return"
        ]
        frame["adaptive_signal"] = np.select(
            [frame["adaptive_score"] >= 0.20, frame["adaptive_score"] <= -0.20],
            ["BUY", "SELL"],
            default="HOLD",
        )

        state_by_date = daily["model_health_state"]
        counts = state_by_date.value_counts().to_dict()
        date_count = max(len(daily), 1)
        transitions = int(state_by_date.ne(state_by_date.shift()).sum() - 1) if len(daily) else 0
        diagnostics = ModelHealthDiagnostics(
            dates=len(daily),
            state_counts={state: int(counts.get(state, 0)) for state in STATE_SEVERITY},
            state_shares={state: float(counts.get(state, 0) / date_count) for state in STATE_SEVERITY},
            transitions=max(transitions, 0),
            fallback_dates=int(counts.get("fallback", 0)),
            defensive_dates=int(counts.get("defensive", 0)),
            recent_adaptive_ic=float(daily["rolling_adaptive_ic"].dropna().tail(1).iloc[0]) if daily["rolling_adaptive_ic"].notna().any() else np.nan,
            recent_baseline_ic=float(daily["rolling_fallback_ic"].dropna().tail(1).iloc[0]) if daily["rolling_fallback_ic"].notna().any() else np.nan,
            recent_relative_ic=float(daily["rolling_relative_ic"].dropna().tail(1).iloc[0]) if daily["rolling_relative_ic"].notna().any() else np.nan,
            minimum_health_score=float(daily["model_health_score"].min()) if len(daily) else np.nan,
            average_health_score=float(daily["model_health_score"].mean()) if len(daily) else np.nan,
        )
        return frame, diagnostics


def save_model_health_outputs(
    decisions: pd.DataFrame,
    diagnostics: ModelHealthDiagnostics,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(
        output_dir / "production_decisions.csv", index=False, encoding="utf-8-sig"
    )
    daily_columns = [
        "datetime",
        "model_health_state",
        "model_health_score",
        "model_health_reason",
        "fold_validation_ic",
        "rolling_adaptive_ic",
        "rolling_fallback_ic",
        "rolling_relative_ic",
        "rolling_adaptive_ic_observations",
    ]
    daily = decisions[[column for column in daily_columns if column in decisions]].drop_duplicates("datetime")
    daily.to_csv(
        output_dir / "model_health_daily.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "model_health_diagnostics.json").write_text(
        json.dumps(diagnostics.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )