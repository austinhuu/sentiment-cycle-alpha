from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ACTION_MAP = {"BUY": 1.0, "HOLD": 0.0, "SELL": -1.0}


@dataclass
class LLMCalibrationResult:
    merged_samples: pd.DataFrame
    bucket_table: pd.DataFrame
    report: dict[str, Any]
    adjusted_decisions: pd.DataFrame


def _spearman(x: pd.Series, y: pd.Series) -> float:
    x_values = pd.to_numeric(x, errors="coerce")
    y_values = pd.to_numeric(y, errors="coerce")
    valid = x_values.notna() & y_values.notna()
    if (
        valid.sum() < 3
        or x_values[valid].nunique() < 2
        or y_values[valid].nunique() < 2
    ):
        return 0.0
    value = x_values[valid].corr(y_values[valid], method="spearman")
    return float(value) if pd.notna(value) else 0.0


def _normalize_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def calibrate_llm_results(
    decisions: pd.DataFrame,
    llm_results: pd.DataFrame,
    config: dict[str, Any],
) -> LLMCalibrationResult:
    decision_frame = decisions.copy()
    llm = llm_results.copy()
    decision_frame["datetime"] = _normalize_datetime(decision_frame["datetime"])
    llm["datetime"] = _normalize_datetime(llm["datetime"])
    decision_frame["symbol"] = decision_frame["symbol"].astype(str).str.upper()
    llm["symbol"] = llm["symbol"].astype(str).str.upper()

    horizon = int(config.get("target_horizon", 5))
    target_column = f"future_return_{horizon}"
    keep = [
        "datetime",
        "symbol",
        target_column,
        "adaptive_score",
        "adaptive_confidence",
        "predicted_excess_return",
    ]
    for column in keep:
        if column not in decision_frame.columns:
            decision_frame[column] = np.nan
    merged = llm.merge(
        decision_frame[keep].drop_duplicates(["datetime", "symbol"]),
        on=["datetime", "symbol"],
        how="left",
    )
    merged["llm_action"] = merged["llm_action"].astype(str).str.upper()
    merged["llm_signed_action"] = merged["llm_action"].map(ACTION_MAP).fillna(0.0)
    merged["llm_confidence"] = pd.to_numeric(
        merged["llm_confidence"], errors="coerce"
    ).fillna(0.0).clip(0.0, 1.0)
    merged["llm_signed_score"] = (
        merged["llm_signed_action"] * merged["llm_confidence"]
    )
    merged[target_column] = pd.to_numeric(
        merged[target_column], errors="coerce"
    )
    valid = merged.dropna(subset=[target_column]).copy()

    directional = valid.loc[valid["llm_signed_action"].ne(0)].copy()
    if directional.empty:
        directional_accuracy = 0.0
    else:
        directional_accuracy = float(
            (
                np.sign(directional[target_column])
                == np.sign(directional["llm_signed_action"])
            ).mean()
        )
    ic = _spearman(valid["llm_signed_score"], valid[target_column]) if not valid.empty else 0.0
    minimum_samples = int(config.get("minimum_samples", 100))
    minimum_accuracy = float(config.get("minimum_directional_accuracy", 0.53))
    minimum_abs_ic = float(config.get("minimum_absolute_ic", 0.03))
    passed = bool(
        len(valid) >= minimum_samples
        and directional_accuracy >= minimum_accuracy
        and abs(ic) >= minimum_abs_ic
    )

    bucket_count = max(int(config.get("confidence_bucket_count", 5)), 2)
    if valid["llm_confidence"].nunique() >= 2:
        valid["confidence_bucket"] = pd.qcut(
            valid["llm_confidence"],
            q=min(bucket_count, valid["llm_confidence"].nunique()),
            duplicates="drop",
        ).astype(str)
    else:
        valid["confidence_bucket"] = "single_bucket"
    bucket_table = valid.groupby(
        ["llm_action", "confidence_bucket"],
        dropna=False,
        as_index=False,
    ).agg(
        samples=(target_column, "count"),
        mean_forward_return=(target_column, "mean"),
        median_forward_return=(target_column, "median"),
        positive_ratio=(target_column, lambda s: float((s > 0).mean())),
        mean_llm_confidence=("llm_confidence", "mean"),
    )

    max_score_adjustment = abs(float(config.get("maximum_score_adjustment", 0.03)))
    max_conf_adjustment = abs(float(config.get("maximum_confidence_adjustment", 0.03)))
    adjusted = decision_frame.copy()
    adjusted["llm_calibrated_score_adjustment"] = 0.0
    adjusted["llm_calibrated_confidence_adjustment"] = 0.0
    adjusted["llm_calibration_applied"] = 0

    influence_enabled = bool(config.get("default_influence_enabled", False))
    if passed and influence_enabled and not valid.empty:
        slope_denominator = float((valid["llm_signed_score"] ** 2).sum())
        slope = (
            float(
                (
                    valid["llm_signed_score"]
                    * valid[target_column]
                ).sum()
                / slope_denominator
            )
            if slope_denominator > 1e-12
            else 0.0
        )
        valid["calibrated_score_adjustment"] = (
            valid["llm_signed_score"] * np.sign(slope) * max_score_adjustment
        ).clip(-max_score_adjustment, max_score_adjustment)
        valid["calibrated_confidence_adjustment"] = (
            (valid["llm_confidence"] - 0.5)
            * 2.0
            * max_conf_adjustment
        ).clip(-max_conf_adjustment, max_conf_adjustment)
        adjustments = valid[
            [
                "datetime",
                "symbol",
                "calibrated_score_adjustment",
                "calibrated_confidence_adjustment",
            ]
        ].drop_duplicates(["datetime", "symbol"], keep="last")
        adjusted = adjusted.merge(
            adjustments,
            on=["datetime", "symbol"],
            how="left",
            suffixes=("", "_new"),
        )
        adjusted["llm_calibrated_score_adjustment"] = adjusted[
            "calibrated_score_adjustment"
        ].fillna(0.0)
        adjusted["llm_calibrated_confidence_adjustment"] = adjusted[
            "calibrated_confidence_adjustment"
        ].fillna(0.0)
        adjusted["llm_calibration_applied"] = adjusted[
            "calibrated_score_adjustment"
        ].notna().astype(int)
        adjusted["adaptive_score"] = (
            pd.to_numeric(adjusted["adaptive_score"], errors="coerce").fillna(0.0)
            + adjusted["llm_calibrated_score_adjustment"]
        ).clip(-1.0, 1.0)
        adjusted["adaptive_confidence"] = (
            pd.to_numeric(
                adjusted["adaptive_confidence"], errors="coerce"
            ).fillna(0.0)
            + adjusted["llm_calibrated_confidence_adjustment"]
        ).clip(0.0, 1.0)
        adjusted = adjusted.drop(
            columns=[
                "calibrated_score_adjustment",
                "calibrated_confidence_adjustment",
            ],
            errors="ignore",
        )

    report = {
        "target_horizon": horizon,
        "available_llm_rows": int(len(llm)),
        "realized_samples": int(len(valid)),
        "directional_samples": int(len(directional)),
        "directional_accuracy": directional_accuracy,
        "spearman_ic": ic,
        "minimum_samples": minimum_samples,
        "minimum_directional_accuracy": minimum_accuracy,
        "minimum_absolute_ic": minimum_abs_ic,
        "statistical_gate_passed": passed,
        "influence_enabled": influence_enabled,
        "adjustment_applied": bool(passed and influence_enabled),
        "reason": (
            "calibration_passed"
            if passed
            else "insufficient_or_unreliable_llm_history"
        ),
    }
    return LLMCalibrationResult(
        merged_samples=merged,
        bucket_table=bucket_table,
        report=report,
        adjusted_decisions=adjusted,
    )


def save_llm_calibration(
    result: LLMCalibrationResult,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.merged_samples.to_csv(
        output_dir / "llm_calibration_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )
    result.bucket_table.to_csv(
        output_dir / "llm_confidence_buckets.csv",
        index=False,
        encoding="utf-8-sig",
    )
    result.adjusted_decisions.to_csv(
        output_dir / "adaptive_decisions_llm_calibrated.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_dir / "llm_calibration_report.json").write_text(
        json.dumps(result.report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )