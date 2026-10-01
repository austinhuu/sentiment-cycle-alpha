from __future__ import annotations

import pandas as pd

from src.v7.cycle import build_sentiment_cycle
from src.v7.event_structurer import build_structured_event_score
from src.v7.momentum import build_momentum_breakout


def _features() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=90, freq="B")
    rows = []
    for j, symbol in enumerate(["A", "B", "C", "D"]):
        for i, dt in enumerate(dates):
            rows.append({
                "datetime": dt, "symbol": symbol, "return_1": ((i + j) % 7 - 3) / 100,
                "return_5": ((i + 2*j) % 9 - 4) / 50, "volume_ratio": 1 + ((i+j) % 5)/10,
                "momentum_z": ((i+j)%7-3)/2, "volume_ratio_z": ((i+j)%5-2)/2,
                "ma_gap_short_long_z": ((i+j)%9-4)/2, "price_position": ((i+j)%10)/10,
                "breakout_up": float((i+j)%11 == 0),
            })
    return pd.DataFrame(rows)


def test_cycle_is_bounded_and_phased() -> None:
    result = build_sentiment_cycle(_features(), {"z_min_periods": 10, "z_window": 30})
    assert result["cycle_score"].dropna().between(-1, 1).all()
    assert set(result["cycle_phase"].dropna()).issubset({"ICE_POINT", "REPAIR", "FERMENTATION", "CLIMAX", "DIVERGENCE", "EBB"})


def test_momentum_is_bounded() -> None:
    result = build_momentum_breakout(_features())
    assert result["momentum_breakout_score"].between(-1, 1).all()


def test_event_structurer_neutral_when_inactive() -> None:
    d = pd.DataFrame({"datetime": ["2024-01-01"], "symbol": ["A"], "news_score": [0.8], "news_confidence": [0.9], "news_event_severity": [1.0], "news_material_event_count": [1], "news_active": [0], "dominant_event_type": ["x"]})
    result = build_structured_event_score(d)
    assert result.loc[0, "structured_event_score"] == 0.0
