from __future__ import annotations

import math
from typing import Any

from .base import AgentView, clip, direction_from_score


class SentimentAgent:
    name = "sentiment_agent"

    @staticmethod
    def _safe(row: dict[str, Any], key: str, default: float = 0.0) -> float:
        try:
            value = float(row.get(key, default))
            return value if math.isfinite(value) else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _z(value: float) -> float:
        return clip(value / 3.0)

    def analyze(self, row: dict[str, Any]) -> AgentView:
        breadth = self._safe(row, "market_breadth", 0.5)
        breadth_z = self._safe(row, "market_breadth_z")
        return_z = self._safe(row, "market_return_z")
        volume_z = self._safe(row, "market_volume_ratio_z")
        cross_vol_z = self._safe(row, "cross_section_volatility_z")
        symbol_count = self._safe(row, "valid_symbol_count", 0.0)

        breadth_component = 0.55 * self._z(breadth_z) + 0.45 * clip((breadth - 0.5) * 2.0)
        return_component = self._z(return_z)
        volume_component = self._z(volume_z)
        stress_penalty = max(self._z(cross_vol_z), 0.0)
        score = clip(
            0.48 * breadth_component
            + 0.34 * return_component
            + 0.12 * volume_component
            - 0.10 * stress_penalty
        )

        evidence: list[dict[str, Any]] = []
        risks: list[dict[str, Any]] = []
        if breadth >= 0.60 and breadth_z >= 0:
            evidence.append({"feature": "market_breadth", "value": breadth, "message": "情绪参考池上涨家数占比较高"})
        elif breadth <= 0.40 and breadth_z <= 0:
            risks.append({"feature": "market_breadth", "value": breadth, "message": "情绪参考池上涨家数占比较低"})
        if return_z >= 0.6:
            evidence.append({"feature": "market_return_z", "value": return_z, "message": "市场横截面收益强于历史常态"})
        elif return_z <= -0.6:
            risks.append({"feature": "market_return_z", "value": return_z, "message": "市场横截面收益弱于历史常态"})
        if volume_z >= 0.8:
            evidence.append({"feature": "market_volume_ratio_z", "value": volume_z, "message": "市场整体成交活跃度偏高"})
        if cross_vol_z >= 1.0:
            risks.append({"feature": "cross_section_volatility_z", "value": cross_vol_z, "message": "市场分化和横截面波动偏高"})

        coverage = clip(symbol_count / 25.0, 0.0, 1.0)
        confidence = clip(0.30 + 0.42 * coverage + 0.22 * abs(score), 0.0, 1.0)
        return AgentView(
            agent=self.name,
            direction=direction_from_score(score),
            score=score,
            confidence=confidence,
            evidence=evidence[:4],
            risks=risks[:4],
        )
