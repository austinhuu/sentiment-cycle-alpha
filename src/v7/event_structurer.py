from __future__ import annotations

import numpy as np
import pandas as pd


def build_structured_event_score(decisions: pd.DataFrame) -> pd.DataFrame:
    """Convert the existing point-in-time news/event fields into a bounded structured score.

    This is deterministic and therefore backtestable. A live LLM parser can populate the same
    schema later, but the LLM is intentionally not allowed to emit BUY/SELL directly.
    """
    d = decisions.copy()
    score = pd.to_numeric(d.get("news_score", 0.0), errors="coerce").fillna(0.0)
    conf = pd.to_numeric(d.get("news_confidence", 0.0), errors="coerce").fillna(0.0)
    severity = pd.to_numeric(d.get("news_event_severity", 0.0), errors="coerce").fillna(0.0).clip(0, 1)
    material = pd.to_numeric(d.get("news_material_event_count", 0.0), errors="coerce").fillna(0.0)
    active = pd.to_numeric(d.get("news_active", 0.0), errors="coerce").fillna(0.0)
    event_score = (score * (0.55 + 0.45 * conf) * (0.75 + 0.25 * severity)).clip(-1, 1)
    event_score = event_score.where(active.gt(0), 0.0)
    confidence = (0.50 * conf + 0.30 * severity + 0.20 * (material > 0).astype(float)).clip(0, 1)
    return pd.DataFrame({
        "datetime": pd.to_datetime(d["datetime"]),
        "symbol": d["symbol"].astype(str),
        "structured_event_score": event_score,
        "structured_event_confidence": confidence,
        "structured_event_active": active.astype(int),
        "structured_event_type": d.get("dominant_event_type", "routine_or_unclassified"),
    })
