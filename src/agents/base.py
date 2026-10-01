from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentView:
    agent: str
    direction: str
    score: float
    confidence: float
    evidence: list[dict[str, Any]] = field(default_factory=list)
    risks: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "direction": self.direction,
            "score": float(self.score),
            "confidence": float(self.confidence),
            "evidence": self.evidence,
            "risks": self.risks,
        }


def direction_from_score(score: float, neutral_band: float = 0.08) -> str:
    if score > neutral_band:
        return "bullish"
    if score < -neutral_band:
        return "bearish"
    return "neutral"


def clip(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))
