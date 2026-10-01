from __future__ import annotations

import math
from typing import Any

from .base import AgentView, clip, direction_from_score


class TechnicalAgent:
    name = "technical_agent"

    @staticmethod
    def _safe(row: dict[str, Any], key: str, default: float = 0.0) -> float:
        try:
            value = float(row.get(key, default))
            return value if math.isfinite(value) else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _z_component(value: float) -> float:
        # z-score已截断至[-3, 3]，用3归一化，不再依赖主观的80/45倍数。
        return clip(value / 3.0)

    def analyze(self, row: dict[str, Any]) -> AgentView:
        trend_z = self._safe(row, "ma_gap_short_long_z")
        mid_trend_z = self._safe(row, "ma_gap_mid_long_z")
        momentum_z = self._safe(row, "momentum_z")
        return_5_z = self._safe(row, "return_5_z")
        volume_z = self._safe(row, "volume_ratio_z")
        volatility_z = self._safe(row, "volatility_z")
        rsi = self._safe(row, "rsi", 50.0)
        breakout_up = self._safe(row, "breakout_up")
        breakout_down = self._safe(row, "breakout_down")
        price_position = self._safe(row, "price_position", 0.5)

        trend_component = 0.65 * self._z_component(trend_z) + 0.35 * self._z_component(mid_trend_z)
        momentum_component = 0.65 * self._z_component(momentum_z) + 0.35 * self._z_component(return_5_z)
        rsi_component = clip((rsi - 50.0) / 25.0)
        volume_component = self._z_component(volume_z)
        breakout_component = clip(breakout_up - breakout_down)
        position_component = clip((price_position - 0.5) * 2.0)
        volatility_penalty = max(self._z_component(volatility_z), 0.0)
        overheat_penalty = clip((rsi - 72.0) / 18.0, 0.0, 1.0)

        score = clip(
            0.30 * trend_component
            + 0.25 * momentum_component
            + 0.12 * rsi_component
            + 0.10 * volume_component
            + 0.14 * breakout_component
            + 0.09 * position_component
            - 0.08 * volatility_penalty
            - 0.08 * overheat_penalty
        )

        evidence: list[dict[str, Any]] = []
        risks: list[dict[str, Any]] = []
        if trend_z >= 0.5:
            evidence.append({"feature": "ma_gap_short_long_z", "value": trend_z, "message": "均线趋势强于该股历史常态"})
        elif trend_z <= -0.5:
            risks.append({"feature": "ma_gap_short_long_z", "value": trend_z, "message": "均线趋势弱于该股历史常态"})
        if momentum_z >= 0.5:
            evidence.append({"feature": "momentum_z", "value": momentum_z, "message": "近期动量处于历史偏强区间"})
        elif momentum_z <= -0.5:
            risks.append({"feature": "momentum_z", "value": momentum_z, "message": "近期动量处于历史偏弱区间"})
        if volume_z >= 0.7:
            evidence.append({"feature": "volume_ratio_z", "value": volume_z, "message": "成交活跃度显著高于历史水平"})
        elif volume_z <= -1.0:
            risks.append({"feature": "volume_ratio_z", "value": volume_z, "message": "成交活跃度显著低于历史水平"})
        if rsi >= 72:
            risks.append({"feature": "rsi", "value": rsi, "message": "RSI偏高，存在短期过热风险"})
        elif rsi <= 28:
            evidence.append({"feature": "rsi", "value": rsi, "message": "RSI偏低，存在超跌修复可能"})
        if breakout_up > 0:
            evidence.append({"feature": "breakout_up", "value": 1, "message": "连续价格向上突破前期区间"})
        if breakout_down > 0:
            risks.append({"feature": "breakout_down", "value": 1, "message": "连续价格向下跌破前期区间"})
        if volatility_z >= 1.2:
            risks.append({"feature": "volatility_z", "value": volatility_z, "message": "波动率处于该股历史高位"})

        components = [trend_component, momentum_component, rsi_component, volume_component, breakout_component]
        nonzero = [x for x in components if abs(x) > 0.05]
        agreement = abs(sum(1 if x > 0 else -1 for x in nonzero)) / max(len(nonzero), 1)
        coverage_keys = ["ma_gap_short_long_z", "momentum_z", "return_5_z", "volume_ratio_z", "volatility_z", "rsi"]
        coverage = sum(math.isfinite(self._safe(row, key, float("nan"))) for key in coverage_keys) / len(coverage_keys)
        confidence = clip(0.30 + 0.28 * coverage + 0.22 * abs(score) + 0.20 * agreement, 0.0, 1.0)

        return AgentView(
            agent=self.name,
            direction=direction_from_score(score),
            score=score,
            confidence=confidence,
            evidence=evidence[:5],
            risks=risks[:5],
        )
