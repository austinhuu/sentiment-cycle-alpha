from __future__ import annotations

import numpy as np
import pandas as pd

from src.v7.transition_overlay import build_causal_transition_overlay


def _synthetic() -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", "2025-06-30")
    phases = ["EBB", "REPAIR", "FERMENTATION", "CLIMAX", "DIVERGENCE", "ICE_POINT"]
    rows = []
    for i, dt in enumerate(dates):
        phase = phases[(i // 11) % len(phases)]
        score = np.sin(i / 13.0) * 0.6
        for j in range(8):
            rows.append({
                "datetime": dt,
                "symbol": f"S{j}",
                "cycle_phase": phase,
                "cycle_score": score,
                "cycle_breadth": 0.5 + 0.2 * np.sin(i / 7.0),
                "cycle_volume_ratio": 1.0 + 0.1 * np.cos(i / 9.0),
                "cycle_cross_vol": 0.02 + 0.005 * np.sin(i / 5.0),
                "future_return_5": 0.003 * np.sin(i / 17.0) + j * 0.00001,
            })
    return pd.DataFrame(rows)


def test_transition_overlay_is_bounded_and_daily() -> None:
    frame = _synthetic()
    out = build_causal_transition_overlay(frame)
    assert out["datetime"].is_unique
    assert len(out) == frame["datetime"].nunique()
    assert out["transition_cycle_multiplier"].between(0.55, 1.0).all()
    assert {"transition", "transition_edge_return", "transition_confidence"}.issubset(out.columns)


def test_test_year_labels_do_not_change_same_year_overlay() -> None:
    frame = _synthetic()
    base = build_causal_transition_overlay(frame)
    changed = frame.copy()
    mask = pd.to_datetime(changed["datetime"]).dt.year == 2025
    changed.loc[mask, "future_return_5"] = changed.loc[mask, "future_return_5"] + 0.50
    altered = build_causal_transition_overlay(changed)
    a = base.loc[pd.to_datetime(base["datetime"]).dt.year == 2025, "transition_cycle_multiplier"].to_numpy()
    b = altered.loc[pd.to_datetime(altered["datetime"]).dt.year == 2025, "transition_cycle_multiplier"].to_numpy()
    np.testing.assert_allclose(a, b, atol=0.0, rtol=0.0)
