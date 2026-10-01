from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DriftReport:
    feature_drift: pd.DataFrame
    rolling_ic: pd.DataFrame
    coefficient_drift: pd.DataFrame
    regime_drift: pd.DataFrame
    model_mode_summary: pd.DataFrame
    alerts: list[dict[str, Any]]
    summary: dict[str, Any]


def _safe_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _active_coverage(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    return float(values.fillna(0.0).abs().gt(1e-12).mean()) if len(values) else 0.0


def _psi(baseline: pd.Series, recent: pd.Series, bins: int) -> float:
    base = pd.to_numeric(baseline, errors="coerce").dropna()
    new = pd.to_numeric(recent, errors="coerce").dropna()
    if len(base) < 20 or len(new) < 20:
        return np.nan
    edges = np.unique(base.quantile(np.linspace(0.0, 1.0, max(bins, 2) + 1)).to_numpy())
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    base_counts = pd.cut(base, bins=edges, include_lowest=True).value_counts(sort=False)
    new_counts = pd.cut(new, bins=edges, include_lowest=True).value_counts(sort=False)
    base_pct = base_counts / max(base_counts.sum(), 1)
    new_pct = new_counts / max(new_counts.sum(), 1)
    epsilon = 1e-6
    return float(
        (((new_pct + epsilon) - (base_pct + epsilon))
         * np.log((new_pct + epsilon) / (base_pct + epsilon))).sum()
    )


def _jensen_shannon(p: pd.Series, q: pd.Series) -> float:
    labels = sorted(set(p.index).union(q.index))
    p_values = p.reindex(labels, fill_value=0.0).to_numpy(dtype=float)
    q_values = q.reindex(labels, fill_value=0.0).to_numpy(dtype=float)
    if p_values.sum() <= 0 or q_values.sum() <= 0:
        return 0.0
    p_values /= p_values.sum()
    q_values /= q_values.sum()
    midpoint = 0.5 * (p_values + q_values)
    epsilon = 1e-12
    return float(
        0.5 * np.sum(p_values * np.log((p_values + epsilon) / (midpoint + epsilon)))
        + 0.5 * np.sum(q_values * np.log((q_values + epsilon) / (midpoint + epsilon)))
    )


def _rolling_spearman(
    frame: pd.DataFrame,
    score_column: str,
    target_column: str,
    window_dates: int,
) -> pd.DataFrame:
    local = frame[["datetime", score_column, target_column]].copy()
    local["datetime"] = pd.to_datetime(local["datetime"], errors="coerce")
    local[score_column] = pd.to_numeric(local[score_column], errors="coerce")
    local[target_column] = pd.to_numeric(local[target_column], errors="coerce")
    rows: list[tuple[pd.Timestamp, float]] = []
    for date, group in local.groupby("datetime", sort=True):
        if group[score_column].notna().sum() < 3 or group[target_column].notna().sum() < 3:
            value = np.nan
        else:
            value = group[score_column].corr(group[target_column], method="spearman")
        rows.append((date, value))
    daily = pd.Series({date: value for date, value in rows}).sort_index()
    rolling = daily.rolling(window_dates, min_periods=max(20, window_dates // 4)).mean()
    return pd.DataFrame(
        {
            "datetime": daily.index,
            "daily_ic": daily.to_numpy(),
            "rolling_ic": rolling.to_numpy(),
            "score_column": score_column,
        }
    )


def _deduplicate_alerts(alerts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    severity_rank = {"normal": 0, "warning": 1, "critical": 2}
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for item in alerts:
        key = (str(item.get("type", "")), str(item.get("feature", "")))
        previous = selected.get(key)
        if previous is None or severity_rank.get(str(item.get("severity")), 0) > severity_rank.get(str(previous.get("severity")), 0):
            selected[key] = item
    return list(selected.values())


def build_drift_report(
    decisions: pd.DataFrame,
    coefficients: pd.DataFrame,
    config: dict[str, Any],
) -> DriftReport:
    frame = decisions.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame = frame.dropna(subset=["datetime"]).sort_values("datetime")

    baseline_end_year = int(config.get("baseline_end_year", 2022))
    recent_window_days = int(config.get("recent_window_days", 252))
    baseline = frame.loc[frame["datetime"].dt.year <= baseline_end_year].copy()
    unique_dates = sorted(frame["datetime"].dropna().unique())
    recent_dates = set(unique_dates[-recent_window_days:])
    recent = frame.loc[frame["datetime"].isin(recent_dates)].copy()
    if baseline.empty and unique_dates:
        earliest_dates = set(unique_dates[: min(recent_window_days, len(unique_dates))])
        baseline = frame.loc[frame["datetime"].isin(earliest_dates)].copy()

    monitored_features = [
        "technical_score",
        "technical_confidence",
        "sentiment_score",
        "fundamental_score",
        "fundamental_confidence",
        "annual_fundamental_score",
        "annual_fundamental_confidence",
        "quarterly_fundamental_score",
        "quarterly_fundamental_confidence",
        "news_score",
        "news_confidence",
        "risk_score",
        "market_stress_percentile",
        "adaptive_score",
        "adaptive_confidence",
        "predicted_excess_return",
    ]
    if bool(config.get("monitor_raw_confidence", False)):
        monitored_features.append("adaptive_confidence_raw")
    bins = int(config.get("psi_bins", 10))
    psi_warning = float(config.get("psi_warning", 0.10))
    psi_critical = float(config.get("psi_critical", 0.25))
    minimum_coverage = float(config.get("minimum_feature_coverage", 0.70))
    minimum_active = float(config.get("minimum_active_coverage", 0.03))
    feature_rows: list[dict[str, Any]] = []
    alerts: list[dict[str, Any]] = []

    for feature in monitored_features:
        if feature not in frame.columns:
            continue
        base_series = _safe_numeric(baseline, feature)
        recent_series = _safe_numeric(recent, feature)
        base_coverage = float(base_series.notna().mean()) if len(base_series) else 0.0
        recent_coverage = float(recent_series.notna().mean()) if len(recent_series) else 0.0
        base_active = _active_coverage(base_series)
        recent_active = _active_coverage(recent_series)
        sparse = max(base_active, recent_active) < minimum_active
        psi_value = np.nan if sparse else _psi(base_series, recent_series, bins)
        status = "insufficient_data" if sparse else "normal"
        if not sparse:
            if pd.notna(psi_value) and psi_value >= psi_critical:
                status = "critical"
            elif pd.notna(psi_value) and psi_value >= psi_warning:
                status = "warning"
            if recent_coverage < minimum_coverage:
                status = "critical"
        feature_rows.append(
            {
                "feature": feature,
                "psi": psi_value,
                "baseline_coverage": base_coverage,
                "recent_coverage": recent_coverage,
                "baseline_active_coverage": base_active,
                "recent_active_coverage": recent_active,
                "baseline_mean": float(base_series.mean()) if base_series.notna().any() else np.nan,
                "recent_mean": float(recent_series.mean()) if recent_series.notna().any() else np.nan,
                "status": status,
            }
        )
        if status in {"warning", "critical"}:
            alerts.append(
                {
                    "type": "feature_drift",
                    "severity": status,
                    "feature": feature,
                    "value": psi_value,
                    "message": f"{feature}的PSI或覆盖率出现异常",
                }
            )

    feature_drift = pd.DataFrame(feature_rows)
    target_column = "future_return_5"
    ic_frames: list[pd.DataFrame] = []
    window = int(config.get("ic_window_days", 120))
    ic_warning_drop = float(config.get("ic_warning_drop", 0.05))
    for score_column in [
        "technical_score",
        "fundamental_score",
        "news_score",
        "adaptive_score",
        "predicted_excess_return",
    ]:
        if score_column not in frame.columns or target_column not in frame.columns:
            continue
        if _active_coverage(frame[score_column]) < minimum_active:
            continue
        ic_frame = _rolling_spearman(frame, score_column, target_column, window)
        ic_frames.append(ic_frame)
        valid = ic_frame["rolling_ic"].dropna()
        if len(valid) >= 2:
            baseline_ic = float(valid.iloc[: max(1, len(valid) // 3)].mean())
            recent_ic = float(valid.iloc[-1])
            if recent_ic < baseline_ic - ic_warning_drop:
                alerts.append(
                    {
                        "type": "ic_decay",
                        "severity": "warning",
                        "feature": score_column,
                        "value": recent_ic,
                        "message": f"{score_column}滚动IC由约{baseline_ic:.3f}下降到{recent_ic:.3f}",
                    }
                )
    rolling_ic = pd.concat(ic_frames, ignore_index=True) if ic_frames else pd.DataFrame()

    coefficient_rows: list[dict[str, Any]] = []
    relative_warning = float(config.get("coefficient_change_warning", 0.50))
    minimum_abs = float(config.get("minimum_abs_coefficient", 0.001))
    minimum_change = float(config.get("minimum_absolute_coefficient_change", 0.002))
    required = {"test_year", "feature", "standardized_coefficient"}
    if not coefficients.empty and required.issubset(coefficients.columns):
        coefficient_frame = coefficients.copy()
        coefficient_frame["test_year"] = pd.to_numeric(coefficient_frame["test_year"], errors="coerce")
        coefficient_frame["standardized_coefficient"] = pd.to_numeric(
            coefficient_frame["standardized_coefficient"], errors="coerce"
        )
        coefficient_frame = coefficient_frame.dropna(subset=["test_year", "feature", "standardized_coefficient"])
        coefficient_frame = coefficient_frame.drop_duplicates(["test_year", "feature"], keep="last")
        for feature, group in coefficient_frame.groupby("feature"):
            previous: float | None = None
            for row in group.sort_values("test_year").itertuples(index=False):
                value = float(row.standardized_coefficient)
                absolute_change = np.nan if previous is None else abs(value - previous)
                relative_change = np.nan
                if previous is not None:
                    relative_change = absolute_change / max(abs(previous) + abs(value), 1e-4)
                coefficient_rows.append(
                    {
                        "feature": feature,
                        "test_year": int(row.test_year),
                        "coefficient": value,
                        "absolute_change": absolute_change,
                        "relative_change": relative_change,
                    }
                )
                meaningful = (
                    previous is not None
                    and max(abs(previous), abs(value)) >= minimum_abs
                    and absolute_change >= minimum_change
                    and relative_change >= relative_warning
                )
                if meaningful:
                    alerts.append(
                        {
                            "type": "coefficient_drift",
                            "severity": "warning",
                            "feature": feature,
                            "value": relative_change,
                            "message": f"{feature}的折间标准化系数发生显著变化",
                        }
                    )
                previous = value
    coefficient_drift = pd.DataFrame(coefficient_rows)

    regime_rows: list[dict[str, Any]] = []
    if "market_regime" in baseline.columns and "market_regime" in recent.columns:
        base_dist = baseline["market_regime"].astype(str).value_counts(normalize=True)
        recent_dist = recent["market_regime"].astype(str).value_counts(normalize=True)
        js = _jensen_shannon(base_dist, recent_dist)
        for label in sorted(set(base_dist.index).union(recent_dist.index)):
            regime_rows.append(
                {
                    "market_regime": label,
                    "baseline_share": float(base_dist.get(label, 0.0)),
                    "recent_share": float(recent_dist.get(label, 0.0)),
                    "jensen_shannon_total": js,
                }
            )
        if js >= float(config.get("regime_js_warning", 0.10)):
            alerts.append(
                {
                    "type": "regime_distribution_drift",
                    "severity": "warning",
                    "feature": "market_regime",
                    "value": js,
                    "message": "近期市场状态分布与基准期明显不同",
                }
            )
    regime_drift = pd.DataFrame(regime_rows)

    mode_rows: list[dict[str, Any]] = []
    mode_column = "fusion_model_mode" if "fusion_model_mode" in frame.columns else None
    if mode_column is not None:
        for period_name, period_frame in [("baseline", baseline), ("recent", recent)]:
            distribution = period_frame[mode_column].astype(str).value_counts(normalize=True)
            for mode, share in distribution.items():
                mode_rows.append(
                    {
                        "period": period_name,
                        "model_mode": str(mode),
                        "share": float(share),
                        "rows": int(period_frame[mode_column].astype(str).eq(str(mode)).sum()),
                    }
                )
    model_mode_summary = pd.DataFrame(mode_rows)

    alerts = _deduplicate_alerts(alerts)
    summary = {
        "baseline_rows": int(len(baseline)),
        "recent_rows": int(len(recent)),
        "baseline_end_year": baseline_end_year,
        "recent_window_dates": int(len(recent_dates)),
        "insufficient_data_feature_count": int(
            feature_drift["status"].eq("insufficient_data").sum()
        ) if not feature_drift.empty else 0,
        "alert_count": int(len(alerts)),
        "critical_alert_count": int(sum(item["severity"] == "critical" for item in alerts)),
        "warning_alert_count": int(sum(item["severity"] == "warning" for item in alerts)),
        "system_status": (
            "critical"
            if any(item["severity"] == "critical" for item in alerts)
            else "warning" if alerts else "normal"
        ),
        "recent_fallback_share": float(
            pd.to_numeric(recent.get("fusion_fallback_used", 0), errors="coerce")
            .fillna(0.0)
            .gt(0)
            .mean()
        ) if len(recent) else 0.0,
        "recent_mean_calibrated_confidence": float(
            pd.to_numeric(recent.get("adaptive_confidence", np.nan), errors="coerce").mean()
        ) if len(recent) else np.nan,
    }
    return DriftReport(
        feature_drift=feature_drift,
        rolling_ic=rolling_ic,
        coefficient_drift=coefficient_drift,
        regime_drift=regime_drift,
        model_mode_summary=model_mode_summary,
        alerts=alerts,
        summary=summary,
    )


def save_drift_report(report: DriftReport, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report.feature_drift.to_csv(output_dir / "feature_drift_psi.csv", index=False, encoding="utf-8-sig")
    report.rolling_ic.to_csv(output_dir / "rolling_agent_ic.csv", index=False, encoding="utf-8-sig")
    report.coefficient_drift.to_csv(output_dir / "coefficient_drift.csv", index=False, encoding="utf-8-sig")
    report.regime_drift.to_csv(output_dir / "regime_distribution_drift.csv", index=False, encoding="utf-8-sig")
    report.model_mode_summary.to_csv(output_dir / "model_mode_summary.csv", index=False, encoding="utf-8-sig")
    (output_dir / "drift_alerts.json").write_text(
        json.dumps(report.alerts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "drift_summary.json").write_text(
        json.dumps(report.summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )