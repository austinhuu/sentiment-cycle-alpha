from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.akshare_fundamental_downloader import (
    _statutory_deadline,
    _validated_available_date,
)
from src.data.akshare_text_downloader import classify_event
from src.models.adaptive_fusion import run_walk_forward_fusion


def test_point_in_time_deadlines_and_event_neutrality() -> None:
    assert _statutory_deadline(pd.Timestamp("2024-12-31")) == pd.Timestamp(
        "2025-04-30"
    )
    available, source = _validated_available_date(
        "2026-04-20",
        pd.Timestamp("2024-03-31"),
        lag_days=1,
        grace_days=60,
    )
    assert source == "statutory_deadline_fallback"
    assert available == pd.Timestamp("2024-05-01")

    neutral = classify_event("关于召开董事会的公告")
    positive = classify_event("关于拟回购股份方案的公告")
    assert neutral["event_score"] == 0.0
    assert neutral["event_confidence"] == 0.0
    assert positive["event_score"] > 0
    assert positive["is_material_event"] == 1


def test_adaptive_fusion_is_out_of_sample() -> None:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-31", "2023-12-31", freq="ME")
    symbols = [f"{index:06d}.SZ" for index in range(1, 5)]
    rows = []
    for date in dates:
        for symbol_index, symbol in enumerate(symbols):
            technical = rng.normal()
            future = 0.01 * technical + rng.normal(scale=0.02)
            rows.append(
                {
                    "datetime": date,
                    "symbol": symbol,
                    "technical_score": np.tanh(technical),
                    "technical_confidence": 0.8,
                    "momentum_z": technical,
                    "ma_gap_short_long_z": technical * 0.5,
                    "return_5_z": technical * 0.3,
                    "risk_score": 0.2,
                    "market_regime": "neutral",
                    "future_return_5": future,
                }
            )
    frame = pd.DataFrame(rows)
    config = {
        "target_horizon": 5,
        "first_test_year": 2022,
        "training_window_years": 2,
        "minimum_training_rows": 40,
        "minimum_training_dates": 12,
        "ridge_alpha": 5.0,
        "target_winsor_quantile": 0.01,
        "prediction_clip": 0.08,
        "score_temperature": 1.5,
        "feature_set": "technical_only",
        "include_regime_interactions": False,
        "include_event_interactions": False,
        "feature_sets": {
            "technical_only": [
                "technical_score",
                "technical_confidence",
                "momentum_z",
                "ma_gap_short_long_z",
                "return_5_z",
            ]
        },
    }
    fused, coefficients, metadata = run_walk_forward_fusion(frame, config)
    assert set(fused["test_year"].unique()) == {2022, 2023}
    assert pd.to_datetime(fused["fusion_train_end"]).dt.year.max() <= 2022
    assert not coefficients.empty
    assert len(metadata["folds"]) == 2