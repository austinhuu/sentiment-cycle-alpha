from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REGIMES = [
    "bull_trend",
    "bear_trend",
    "high_volatility",
    "range_bound",
    "neutral",
]


@dataclass
class RidgeState:
    requested_feature_set: str
    effective_feature_set: str
    feature_names: list[str]
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    coefficients: np.ndarray
    intercept: float
    train_rows: int
    train_dates: int
    train_start: str
    train_end: str
    target_mean: float
    target_std: float
    train_ic: float
    ridge_alpha: float
    validation_ic: float
    selected_feature_ic: dict[str, float] = field(default_factory=dict)
    news_enabled: bool = False
    fundamental_enabled: bool = False
    fallback_used: bool = False
    fallback_reason: str = ""
    model_mode: str = "adaptive"
    confidence_quantile_levels: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    confidence_quantile_values: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    model_quality: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_feature_set": self.requested_feature_set,
            "effective_feature_set": self.effective_feature_set,
            "feature_names": self.feature_names,
            "medians": self.medians.tolist(),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "coefficients": self.coefficients.tolist(),
            "intercept": float(self.intercept),
            "train_rows": int(self.train_rows),
            "train_dates": int(self.train_dates),
            "train_start": self.train_start,
            "train_end": self.train_end,
            "target_mean": float(self.target_mean),
            "target_std": float(self.target_std),
            "train_ic": float(self.train_ic),
            "ridge_alpha": float(self.ridge_alpha),
            "validation_ic": float(self.validation_ic),
            "selected_feature_ic": {
                key: float(value) for key, value in self.selected_feature_ic.items()
            },
            "news_enabled": bool(self.news_enabled),
            "fundamental_enabled": bool(self.fundamental_enabled),
            "fallback_used": bool(self.fallback_used),
            "fallback_reason": self.fallback_reason,
            "model_mode": self.model_mode,
            "confidence_quantile_levels": self.confidence_quantile_levels.tolist(),
            "confidence_quantile_values": self.confidence_quantile_values.tolist(),
            "model_quality": float(self.model_quality),
        }


class AdaptiveFusionModel:
    """Purged nested-Ridge fusion with low-IC fallback and fold calibration."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.alpha = float(config.get("ridge_alpha", 12.0))
        self.target_horizon = int(config.get("target_horizon", 5))
        self.prediction_clip = abs(float(config.get("prediction_clip", 0.08)))
        self.temperature = max(float(config.get("score_temperature", 1.75)), 0.10)
        self.state: RidgeState | None = None

    @staticmethod
    def _safe_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
        if column not in frame.columns:
            return pd.Series(0.0, index=frame.index, dtype=float)
        return pd.to_numeric(frame[column], errors="coerce")

    @staticmethod
    def _spearman(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
        xs = pd.Series(x).reset_index(drop=True)
        ys = pd.Series(y).reset_index(drop=True)
        valid = xs.notna() & ys.notna()
        if int(valid.sum()) < 3 or xs.loc[valid].nunique() < 2 or ys.loc[valid].nunique() < 2:
            return 0.0
        value = xs.loc[valid].corr(ys.loc[valid], method="spearman")
        return float(value) if pd.notna(value) else 0.0

    @staticmethod
    def _daily_rank_ic(
        dates: pd.Series,
        prediction: np.ndarray,
        target: np.ndarray,
    ) -> float:
        local = pd.DataFrame(
            {
                "datetime": pd.to_datetime(dates, errors="coerce").to_numpy(),
                "prediction": prediction,
                "target": target,
            }
        ).dropna()
        values: list[float] = []
        for _, group in local.groupby("datetime", sort=False):
            if len(group) < 3 or group["prediction"].nunique() < 2:
                continue
            value = group["prediction"].corr(group["target"], method="spearman")
            if pd.notna(value):
                values.append(float(value))
        return float(np.mean(values)) if values else 0.0

    def build_features(
        self,
        frame: pd.DataFrame,
        feature_set: str | None = None,
        include_regime_interactions: bool | None = None,
        include_event_interactions: bool | None = None,
    ) -> pd.DataFrame:
        sets = self.config.get("feature_sets", {})
        selected = feature_set or str(self.config.get("feature_set", "full_no_regime"))
        base_names = [str(value) for value in sets.get(selected, [])]
        if not base_names:
            raise ValueError(f"缺少特征集合：{selected}")

        columns: dict[str, pd.Series] = {
            column: self._safe_numeric(frame, column) for column in base_names
        }
        for prefix in [
            "technical",
            "sentiment",
            "fundamental",
            "news",
            "annual_fundamental",
            "quarterly_fundamental",
        ]:
            score_col = f"{prefix}_score"
            conf_col = f"{prefix}_confidence"
            if score_col in base_names or conf_col in base_names:
                score = self._safe_numeric(frame, score_col).fillna(0.0)
                confidence = self._safe_numeric(frame, conf_col).fillna(0.0)
                columns[f"{prefix}_score_x_confidence"] = score * confidence

        if "risk_score" in base_names:
            columns["risk_inverse"] = 1.0 - self._safe_numeric(
                frame, "risk_score"
            ).fillna(0.0).clip(0.0, 1.0)

        has_news = any(name.startswith("news_") for name in base_names)
        has_fundamental = any("fundamental" in name for name in base_names)
        if has_news:
            columns["news_active"] = self._safe_numeric(frame, "news_active").fillna(0.0).clip(0.0, 1.0)
        if has_fundamental:
            columns["fundamental_active"] = self._safe_numeric(frame, "fundamental_active").fillna(0.0).clip(0.0, 1.0)
            columns["annual_fundamental_active"] = self._safe_numeric(
                frame, "annual_fundamental_active"
            ).fillna(0.0).clip(0.0, 1.0)
            columns["quarterly_fundamental_active"] = self._safe_numeric(
                frame, "quarterly_fundamental_active"
            ).fillna(0.0).clip(0.0, 1.0)

        use_regime = (
            bool(self.config.get("include_regime_interactions", False))
            if include_regime_interactions is None
            else bool(include_regime_interactions)
        )
        regime = frame.get(
            "market_regime", pd.Series("neutral", index=frame.index)
        ).astype(str)
        if use_regime:
            technical = self._safe_numeric(frame, "technical_score").fillna(0.0)
            fundamental = self._safe_numeric(frame, "fundamental_score").fillna(0.0)
            news = self._safe_numeric(frame, "news_score").fillna(0.0)
            risk = self._safe_numeric(frame, "risk_score").fillna(0.0)
            for value in REGIMES:
                indicator = regime.eq(value).astype(float)
                columns[f"regime_{value}"] = indicator
                columns[f"technical_x_{value}"] = technical * indicator
                if has_fundamental:
                    columns[f"fundamental_x_{value}"] = fundamental * indicator
                if has_news:
                    columns[f"news_x_{value}"] = news * indicator
                if "risk_score" in base_names:
                    columns[f"risk_x_{value}"] = risk * indicator

        event_config = self.config.get("event_regime_config", {})
        use_event = (
            bool(self.config.get("include_event_interactions", False))
            if include_event_interactions is None
            else bool(include_event_interactions)
        ) and bool(event_config.get("enabled", False))
        if use_event and has_news:
            severity = self._safe_numeric(frame, "news_event_severity").fillna(0.0)
            news = self._safe_numeric(frame, "news_score").fillna(0.0)
            fundamental = self._safe_numeric(frame, "fundamental_score").fillna(0.0)
            technical = self._safe_numeric(frame, "technical_score").fillna(0.0)
            columns["news_score_x_severity"] = news * severity
            if has_fundamental:
                columns["fundamental_x_news"] = fundamental * news
            columns["technical_x_news"] = technical * news

        return pd.DataFrame(columns, index=frame.index).replace([np.inf, -np.inf], np.nan)

    def build_target(self, frame: pd.DataFrame) -> pd.Series:
        column = f"future_return_{self.target_horizon}"
        if column not in frame.columns:
            raise ValueError(f"决策文件缺少目标列：{column}")
        target = pd.to_numeric(frame[column], errors="coerce")
        if bool(self.config.get("use_cross_sectional_excess_target", True)):
            dates = pd.to_datetime(frame["datetime"], errors="coerce")
            target = target - target.groupby(dates).transform("mean")
        quantile = min(max(float(self.config.get("target_winsor_quantile", 0.01)), 0.0), 0.20)
        valid = target.dropna()
        if not valid.empty and quantile > 0:
            lower, upper = valid.quantile([quantile, 1.0 - quantile])
            target = target.clip(lower, upper)
        return target

    def _select_features(
        self,
        frame: pd.DataFrame,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> tuple[list[str], dict[str, float], bool, bool]:
        valid = target.notna()
        local_features = features.loc[valid]
        local_target = target.loc[valid]
        news_coverage = float(
            self._safe_numeric(frame.loc[valid], "news_active").fillna(0.0).gt(0).mean()
        )
        fundamental_coverage = float(
            self._safe_numeric(frame.loc[valid], "fundamental_active").fillna(0.0).gt(0).mean()
        )
        news_enabled = news_coverage >= float(self.config.get("minimum_news_active_coverage", 0.03))
        fundamental_enabled = fundamental_coverage >= float(
            self.config.get("minimum_fundamental_active_coverage", 0.30)
        )
        minimum_nonzero = float(self.config.get("minimum_nonzero_feature_coverage", 0.02))
        minimum_ic = float(self.config.get("minimum_abs_univariate_ic", 0.005))
        mandatory = {str(value) for value in self.config.get("mandatory_features", [])}
        maximum = max(int(self.config.get("maximum_selected_features", 18)), 1)

        feature_ic: dict[str, float] = {}
        candidates: list[str] = []
        for column in local_features.columns:
            lower = column.lower()
            if not news_enabled and ("news" in lower or lower.startswith("event_")):
                continue
            if not fundamental_enabled and "fundamental" in lower:
                continue
            values = pd.to_numeric(local_features[column], errors="coerce")
            nonzero = float(values.fillna(0.0).abs().gt(1e-12).mean())
            std = float(values.std(ddof=0)) if values.notna().any() else 0.0
            if not np.isfinite(std) or std < 1e-8:
                continue
            if nonzero < minimum_nonzero and column not in mandatory:
                continue
            ic = self._spearman(values, local_target)
            feature_ic[column] = ic
            if abs(ic) >= minimum_ic or column in mandatory:
                candidates.append(column)

        selected = [column for column in candidates if column in mandatory]
        ranked = sorted(
            [column for column in candidates if column not in mandatory],
            key=lambda name: abs(feature_ic.get(name, 0.0)),
            reverse=True,
        )
        selected = list(dict.fromkeys(selected + ranked))[:maximum]
        if not selected:
            selected = sorted(feature_ic, key=lambda name: abs(feature_ic[name]), reverse=True)[:maximum]
        if not selected:
            raise ValueError("训练特征全部无效，无法拟合自适应融合模型。")
        return selected, {name: feature_ic[name] for name in selected}, news_enabled, fundamental_enabled

    @staticmethod
    def _fit_ridge_arrays(
        x: np.ndarray,
        y: np.ndarray,
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        medians = np.nanmedian(x, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        filled = np.where(np.isnan(x), medians, x)
        means = np.mean(filled, axis=0)
        scales = np.std(filled, axis=0)
        scales = np.where(scales < 1e-8, 1.0, scales)
        standardized = (filled - means) / scales
        intercept = float(np.mean(y))
        centered = y - intercept
        matrix = standardized.T @ standardized + float(alpha) * np.eye(standardized.shape[1])
        vector = standardized.T @ centered
        try:
            coefficients = np.linalg.solve(matrix, vector)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.pinv(matrix) @ vector
        return medians, means, scales, coefficients, intercept

    @staticmethod
    def _predict_arrays(
        x: np.ndarray,
        medians: np.ndarray,
        means: np.ndarray,
        scales: np.ndarray,
        coefficients: np.ndarray,
        intercept: float,
    ) -> np.ndarray:
        filled = np.where(np.isnan(x), medians, x)
        standardized = (filled - means) / scales
        return intercept + standardized @ coefficients

    def _choose_alpha(
        self,
        dates: pd.Series,
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[float, float]:
        if not bool(self.config.get("nested_alpha_selection", True)):
            return self.alpha, 0.0
        converted = pd.to_datetime(dates, errors="coerce")
        unique_dates = np.array(sorted(converted.dropna().unique()))
        minimum_validation = int(self.config.get("minimum_validation_dates", 60))
        fraction = float(self.config.get("validation_fraction", 0.20))
        validation_count = max(minimum_validation, int(len(unique_dates) * fraction))
        purge_count = max(int(self.config.get("purge_days", self.target_horizon)), 0) + max(
            int(self.config.get("embargo_days", 1)), 0
        )
        if len(unique_dates) <= validation_count + purge_count + 60:
            return self.alpha, 0.0
        validation_dates = unique_dates[-validation_count:]
        inner_train_dates = unique_dates[: -(validation_count + purge_count)]
        validation_mask = converted.isin(set(validation_dates)).to_numpy()
        train_mask = converted.isin(set(inner_train_dates)).to_numpy()
        if int(train_mask.sum()) < 200 or int(validation_mask.sum()) < 100:
            return self.alpha, 0.0

        grid = [float(value) for value in self.config.get(
            "ridge_alpha_grid", [12.0, 24.0, 48.0, 96.0, 192.0]
        )]
        best_alpha = self.alpha
        best_ic = -np.inf
        for alpha in grid:
            medians, means, scales, coefficients, intercept = self._fit_ridge_arrays(
                x[train_mask], y[train_mask], alpha
            )
            prediction = self._predict_arrays(
                x[validation_mask], medians, means, scales, coefficients, intercept
            )
            ic = self._daily_rank_ic(
                converted.loc[validation_mask], prediction, y[validation_mask]
            )
            if ic > best_ic + 1e-12 or (
                abs(ic - best_ic) <= 1e-12 and alpha > best_alpha
            ):
                best_alpha = alpha
                best_ic = ic
        return best_alpha, float(best_ic if np.isfinite(best_ic) else 0.0)

    def _score_predictions(self, dates: pd.Series, prediction: np.ndarray) -> np.ndarray:
        local = pd.DataFrame(
            {
                "datetime": pd.to_datetime(dates, errors="coerce").to_numpy(),
                "prediction": prediction,
            }
        )

        def score_group(values: pd.Series) -> pd.Series:
            std = values.std(ddof=0)
            z = values * 0.0 if not np.isfinite(std) or std < 1e-12 else (values - values.mean()) / std
            return np.tanh(z / self.temperature).clip(-1.0, 1.0)

        return local["prediction"].groupby(local["datetime"]).transform(score_group).to_numpy(dtype=float)

    def _fit_candidate(
        self,
        frame: pd.DataFrame,
        requested_feature_set: str,
        effective_feature_set: str,
        include_regime_interactions: bool,
        include_event_interactions: bool,
    ) -> RidgeState:
        features = self.build_features(
            frame,
            feature_set=effective_feature_set,
            include_regime_interactions=include_regime_interactions,
            include_event_interactions=include_event_interactions,
        )
        target = self.build_target(frame)
        valid = target.notna()
        if int(valid.sum()) == 0:
            raise ValueError("训练集没有可用的未来收益标签。")
        selected, feature_ic, news_enabled, fundamental_enabled = self._select_features(
            frame, features, target
        )
        selected_features = features.loc[valid, selected].copy()
        y = target.loc[valid].to_numpy(dtype=float)
        x = selected_features.to_numpy(dtype=float)
        dates = pd.to_datetime(frame.loc[valid, "datetime"], errors="coerce")
        chosen_alpha, validation_ic = self._choose_alpha(dates, x, y)
        medians, means, scales, coefficients, intercept = self._fit_ridge_arrays(
            x, y, chosen_alpha
        )
        predictions = self._predict_arrays(x, medians, means, scales, coefficients, intercept)
        train_ic = self._daily_rank_ic(dates, predictions, y)
        train_scores = np.abs(self._score_predictions(dates, predictions))
        quantile_count = max(int(self.config.get("confidence_calibration_quantiles", 101)), 11)
        quantile_levels = np.linspace(0.0, 1.0, quantile_count)
        quantile_values = np.quantile(train_scores[np.isfinite(train_scores)], quantile_levels)
        quality_target = max(float(self.config.get("confidence_quality_target_ic", 0.06)), 1e-6)
        quality_floor = float(np.clip(self.config.get("confidence_quality_floor", 0.20), 0.0, 1.0))
        quality_source = max(validation_ic, 0.25 * max(train_ic, 0.0), 0.0)
        model_quality = float(np.clip(quality_floor + (1.0 - quality_floor) * quality_source / quality_target, quality_floor, 1.0))
        return RidgeState(
            requested_feature_set=requested_feature_set,
            effective_feature_set=effective_feature_set,
            feature_names=selected,
            medians=medians,
            means=means,
            scales=scales,
            coefficients=coefficients,
            intercept=intercept,
            train_rows=int(len(selected_features)),
            train_dates=int(dates.nunique()),
            train_start=str(dates.min().date()) if dates.notna().any() else "",
            train_end=str(dates.max().date()) if dates.notna().any() else "",
            target_mean=float(np.mean(y)),
            target_std=float(np.std(y)),
            train_ic=float(train_ic),
            ridge_alpha=float(chosen_alpha),
            validation_ic=float(validation_ic),
            selected_feature_ic=feature_ic,
            news_enabled=news_enabled,
            fundamental_enabled=fundamental_enabled,
            confidence_quantile_levels=quantile_levels,
            confidence_quantile_values=quantile_values,
            model_quality=model_quality,
        )

    def fit(
        self,
        frame: pd.DataFrame,
        feature_set: str | None = None,
        include_regime_interactions: bool | None = None,
        include_event_interactions: bool | None = None,
    ) -> RidgeState:
        requested = feature_set or str(self.config.get("feature_set", "full_no_regime"))
        use_regime = bool(self.config.get("include_regime_interactions", False)) if include_regime_interactions is None else bool(include_regime_interactions)
        use_event = bool(self.config.get("include_event_interactions", False)) if include_event_interactions is None else bool(include_event_interactions)
        primary = self._fit_candidate(frame, requested, requested, use_regime, use_event)
        selected = primary
        threshold = float(self.config.get("minimum_validation_ic", 0.01))
        fallback_set = str(self.config.get("fallback_feature_set", "technical_sentiment"))
        fallback_available = bool(
            self.config.get("feature_sets", {}).get(fallback_set, [])
        )
        if (
            bool(self.config.get("fallback_enabled", True))
            and requested != fallback_set
            and primary.validation_ic < threshold
            and fallback_available
        ):
            fallback = self._fit_candidate(frame, requested, fallback_set, False, False)
            fallback_minimum = float(self.config.get("fallback_minimum_validation_ic", 0.0))
            if fallback.validation_ic >= fallback_minimum or fallback.validation_ic >= primary.validation_ic:
                fallback.fallback_used = True
                fallback.fallback_reason = (
                    f"primary_validation_ic={primary.validation_ic:.6f}<threshold={threshold:.6f}"
                )
                fallback.model_mode = f"fallback_{fallback_set}"
                selected = fallback
            else:
                primary.fallback_reason = (
                    f"fallback_rejected:{fallback.validation_ic:.6f}<primary:{primary.validation_ic:.6f}"
                )
        elif (
            bool(self.config.get("fallback_enabled", True))
            and requested != fallback_set
            and primary.validation_ic < threshold
            and not fallback_available
        ):
            primary.fallback_reason = f"fallback_feature_set_missing:{fallback_set}"
        self.state = selected
        return selected

    @staticmethod
    def _percentile_from_reference(values: np.ndarray, reference: np.ndarray, levels: np.ndarray) -> np.ndarray:
        reference = np.asarray(reference, dtype=float)
        levels = np.asarray(levels, dtype=float)
        valid = np.isfinite(reference) & np.isfinite(levels)
        reference = reference[valid]
        levels = levels[valid]
        if len(reference) < 2:
            return np.clip(values, 0.0, 1.0)
        unique_reference, unique_index = np.unique(reference, return_index=True)
        unique_levels = levels[unique_index]
        if len(unique_reference) < 2:
            return np.where(values > unique_reference[0], 1.0, 0.5)
        return np.interp(values, unique_reference, unique_levels, left=0.0, right=1.0)

    def predict(
        self,
        frame: pd.DataFrame,
        feature_set: str | None = None,
        include_regime_interactions: bool | None = None,
        include_event_interactions: bool | None = None,
    ) -> pd.DataFrame:
        if self.state is None:
            raise RuntimeError("AdaptiveFusionModel must be fit before predict.")
        effective = self.state.effective_feature_set
        use_regime = bool(include_regime_interactions) and not self.state.fallback_used
        use_event = bool(include_event_interactions) and not self.state.fallback_used
        features = self.build_features(
            frame,
            feature_set=effective,
            include_regime_interactions=use_regime,
            include_event_interactions=use_event,
        ).reindex(columns=self.state.feature_names)
        x = features.to_numpy(dtype=float)
        prediction = self._predict_arrays(
            x,
            self.state.medians,
            self.state.means,
            self.state.scales,
            self.state.coefficients,
            self.state.intercept,
        )
        prediction = np.clip(prediction, -self.prediction_clip, self.prediction_clip)
        dates = pd.to_datetime(frame["datetime"], errors="coerce")
        score = self._score_predictions(dates, prediction)
        percentile = self._percentile_from_reference(
            np.abs(score),
            self.state.confidence_quantile_values,
            self.state.confidence_quantile_levels,
        )
        floor = float(self.config.get("confidence_floor", 0.10))
        ceiling = float(self.config.get("confidence_ceiling", 0.95))
        confidence = np.clip(
            floor + (ceiling - floor) * percentile * self.state.model_quality,
            0.0,
            1.0,
        )

        result = pd.DataFrame(index=frame.index)
        result["predicted_excess_return"] = prediction
        result["adaptive_score"] = score
        result["adaptive_confidence_raw"] = np.abs(score)
        result["adaptive_confidence_percentile"] = percentile
        result["adaptive_confidence"] = confidence
        result["adaptive_signal"] = np.select(
            [result["adaptive_score"] >= 0.20, result["adaptive_score"] <= -0.20],
            ["BUY", "SELL"],
            default="HOLD",
        )
        result["fusion_model_mode"] = self.state.model_mode
        result["fusion_effective_feature_set"] = self.state.effective_feature_set
        result["fusion_fallback_used"] = int(self.state.fallback_used)
        result["fusion_model_quality"] = self.state.model_quality
        return result


def _purge_training_dates(
    train: pd.DataFrame,
    purge_days: int,
    embargo_days: int,
) -> tuple[pd.DataFrame, list[str]]:
    unique_dates = sorted(pd.to_datetime(train["datetime"], errors="coerce").dropna().unique())
    count = max(int(purge_days), 0) + max(int(embargo_days), 0)
    if count <= 0 or len(unique_dates) <= count:
        return train, []
    removed = unique_dates[-count:]
    purged = train.loc[~pd.to_datetime(train["datetime"]).isin(removed)].copy()
    return purged, [str(pd.Timestamp(value).date()) for value in removed]


def run_walk_forward_fusion(
    decisions: pd.DataFrame,
    config: dict[str, Any],
    feature_set: str | None = None,
    include_regime_interactions: bool | None = None,
    include_event_interactions: bool | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frame = decisions.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame = frame.dropna(subset=["datetime", "symbol"]).sort_values(
        ["datetime", "symbol"]
    ).reset_index(drop=True)

    first_test_year = int(config.get("first_test_year", 2023))
    training_window_years = int(config.get("training_window_years", 3))
    expanding = bool(config.get("expanding_window", False))
    minimum_rows = int(config.get("minimum_training_rows", 1000))
    minimum_dates = int(config.get("minimum_training_dates", 180))
    target_horizon = int(config.get("target_horizon", 5))
    purge_days = int(config.get("purge_days", target_horizon))
    embargo_days = int(config.get("embargo_days", 1))

    output_parts: list[pd.DataFrame] = []
    coefficient_rows: list[dict[str, Any]] = []
    fold_metadata: list[dict[str, Any]] = []
    max_year = int(frame["datetime"].dt.year.max())

    for test_year in range(first_test_year, max_year + 1):
        test_start = pd.Timestamp(test_year, 1, 1)
        test_end = pd.Timestamp(test_year + 1, 1, 1)
        train_start = frame["datetime"].min() if expanding else pd.Timestamp(
            test_year - training_window_years, 1, 1
        )
        train = frame.loc[
            (frame["datetime"] >= train_start) & (frame["datetime"] < test_start)
        ].copy()
        train, purged_dates = _purge_training_dates(train, purge_days, embargo_days)
        test = frame.loc[
            (frame["datetime"] >= test_start) & (frame["datetime"] < test_end)
        ].copy()
        if test.empty:
            continue

        target_column = f"future_return_{target_horizon}"
        train_valid = train[target_column].notna() if target_column in train else pd.Series(False, index=train.index)
        valid_dates = int(train.loc[train_valid, "datetime"].nunique())
        if int(train_valid.sum()) < minimum_rows or valid_dates < minimum_dates:
            raise RuntimeError(
                f"{test_year}折训练样本不足：rows={int(train_valid.sum())}, dates={valid_dates}"
            )

        model = AdaptiveFusionModel(config)
        state = model.fit(
            train,
            feature_set=feature_set,
            include_regime_interactions=include_regime_interactions,
            include_event_interactions=include_event_interactions,
        )
        predictions = model.predict(
            test,
            feature_set=feature_set,
            include_regime_interactions=include_regime_interactions,
            include_event_interactions=include_event_interactions,
        )
        fold = test.copy()
        for column in predictions.columns:
            fold[column] = predictions[column].to_numpy()
        fold["test_year"] = test_year
        fold["fusion_train_start"] = state.train_start
        fold["fusion_train_end"] = state.train_end
        fold["fusion_train_ic"] = state.train_ic
        fold["fusion_validation_ic"] = state.validation_ic
        fold["fusion_ridge_alpha"] = state.ridge_alpha
        fold["fusion_feature_set"] = state.requested_feature_set
        output_parts.append(fold)

        for feature, coefficient, scale in zip(
            state.feature_names, state.coefficients, state.scales
        ):
            coefficient_rows.append(
                {
                    "test_year": test_year,
                    "feature": feature,
                    "standardized_coefficient": float(coefficient),
                    "raw_scale": float(scale),
                    "absolute_coefficient": float(abs(coefficient)),
                    "univariate_train_ic": float(state.selected_feature_ic.get(feature, 0.0)),
                    "selected_ridge_alpha": float(state.ridge_alpha),
                    "validation_ic": float(state.validation_ic),
                    "requested_feature_set": state.requested_feature_set,
                    "effective_feature_set": state.effective_feature_set,
                    "fallback_used": int(state.fallback_used),
                    "model_mode": state.model_mode,
                }
            )
        fold_metadata.append(
            {
                "test_year": test_year,
                "test_start": str(test_start.date()),
                "test_end_exclusive": str(test_end.date()),
                "purge_days": purge_days,
                "embargo_days": embargo_days,
                "purged_training_dates": purged_dates,
                **state.to_dict(),
            }
        )

    if not output_parts:
        raise RuntimeError("没有生成任何V4.6 Walk-forward融合结果。")
    output = pd.concat(output_parts, ignore_index=True).sort_values(
        ["datetime", "symbol"]
    ).reset_index(drop=True)
    coefficients = pd.DataFrame(coefficient_rows)
    metadata = {
        "version": "4.6",
        "feature_set": feature_set or config.get("feature_set", "full_no_regime"),
        "target_horizon": target_horizon,
        "purge_days": purge_days,
        "embargo_days": embargo_days,
        "folds": fold_metadata,
    }
    return output, coefficients, metadata


def save_fusion_outputs(
    decisions: pd.DataFrame,
    coefficients: pd.DataFrame,
    metadata: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(output_dir / "adaptive_decisions.csv", index=False, encoding="utf-8-sig")
    coefficients.to_csv(output_dir / "adaptive_coefficients.csv", index=False, encoding="utf-8-sig")
    (output_dir / "adaptive_model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )