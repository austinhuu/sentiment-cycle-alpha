from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .base import clip


@dataclass
class MarketRegimeView:
    """市场状态识别智能体的结构化输出。"""

    regime: str
    confidence: float
    trend_score: float
    momentum_score: float
    breadth_score: float
    volatility_score: float
    evidence: list[dict[str, Any]] = field(default_factory=list)
    risks: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "confidence": float(self.confidence),
            "trend_score": float(self.trend_score),
            "momentum_score": float(self.momentum_score),
            "breadth_score": float(self.breadth_score),
            "volatility_score": float(self.volatility_score),
            "evidence": self.evidence,
            "risks": self.risks,
        }


class MarketRegimeAgent:
    """基于市场代理序列识别牛市、熊市、高波动和震荡状态。"""

    name = "market_regime_agent"

    @staticmethod
    def _safe(
        row: dict[str, Any],
        key: str,
        default: float = 0.0,
    ) -> float:
        try:
            value = float(row.get(key, default))
            return value if math.isfinite(value) else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _cfg(
        config: dict[str, Any],
        key: str,
        default: float,
    ) -> float:
        try:
            return float(config.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _positive_margin(
        value: float,
        threshold: float,
        scale: float,
    ) -> float:
        return clip(
            (value - threshold) / max(scale, 1e-8),
            0.0,
            1.0,
        )

    @staticmethod
    def _negative_margin(
        value: float,
        threshold: float,
        scale: float,
    ) -> float:
        return clip(
            (threshold - value) / max(scale, 1e-8),
            0.0,
            1.0,
        )

    def analyze(
        self,
        row: dict[str, Any],
        config: dict[str, Any],
    ) -> MarketRegimeView:
        if not bool(config.get("enabled", True)):
            return MarketRegimeView(
                regime="neutral",
                confidence=1.0,
                trend_score=0.0,
                momentum_score=0.0,
                breadth_score=0.0,
                volatility_score=0.0,
                evidence=[
                    {
                        "feature": "market_regime_enabled",
                        "value": False,
                        "message": "市场状态识别已关闭，使用中性状态",
                    }
                ],
            )

        trend_gap = self._safe(row, "market_trend_gap", 0.0)
        momentum = self._safe(row, "market_momentum", 0.0)
        breadth = self._safe(row, "market_breadth_smooth", 0.5)
        volatility_percentile = self._safe(
            row,
            "market_volatility_percentile",
            0.5,
        )
        volatility_z = self._safe(
            row,
            "market_volatility_z",
            0.0,
        )
        drawdown = min(
            self._safe(row, "market_drawdown", 0.0),
            0.0,
        )

        bull_gap = self._cfg(config, "bull_trend_gap_min", 0.015)
        bull_momentum = self._cfg(config, "bull_momentum_min", 0.020)
        bull_breadth = self._cfg(config, "bull_breadth_min", 0.55)

        bear_gap = self._cfg(config, "bear_trend_gap_max", -0.015)
        bear_momentum = self._cfg(config, "bear_momentum_max", -0.020)
        bear_breadth = self._cfg(config, "bear_breadth_max", 0.45)

        high_vol_pct = self._cfg(
            config,
            "high_volatility_percentile",
            0.80,
        )
        high_vol_z = self._cfg(
            config,
            "high_volatility_zscore",
            1.00,
        )
        high_vol_drawdown = self._cfg(
            config,
            "high_volatility_drawdown",
            -0.08,
        )

        range_gap = self._cfg(config, "range_trend_abs_max", 0.010)
        range_momentum = self._cfg(
            config,
            "range_momentum_abs_max",
            0.025,
        )
        range_breadth_min = self._cfg(
            config,
            "range_breadth_min",
            0.45,
        )
        range_breadth_max = self._cfg(
            config,
            "range_breadth_max",
            0.55,
        )
        range_volatility_max = self._cfg(
            config,
            "range_volatility_percentile_max",
            0.75,
        )

        high_vol_condition = (
            volatility_percentile >= high_vol_pct
            and (
                volatility_z >= high_vol_z
                or drawdown <= high_vol_drawdown
            )
        )
        bull_condition = (
            trend_gap >= bull_gap
            and momentum >= bull_momentum
            and breadth >= bull_breadth
        )
        bear_condition = (
            trend_gap <= bear_gap
            and momentum <= bear_momentum
            and breadth <= bear_breadth
        )
        range_condition = (
            abs(trend_gap) <= range_gap
            and abs(momentum) <= range_momentum
            and range_breadth_min <= breadth <= range_breadth_max
            and volatility_percentile <= range_volatility_max
        )

        evidence: list[dict[str, Any]] = []
        risks: list[dict[str, Any]] = []

        trend_score = clip(trend_gap / max(abs(bull_gap), 1e-8))
        momentum_score = clip(momentum / max(abs(bull_momentum), 1e-8))
        breadth_score = clip((breadth - 0.5) / 0.15)
        volatility_score = clip(
            0.60 * volatility_percentile
            + 0.25 * clip((volatility_z + 1.0) / 3.0, 0.0, 1.0)
            + 0.15 * clip(abs(drawdown) / 0.15, 0.0, 1.0),
            0.0,
            1.0,
        )

        if high_vol_condition:
            regime = "high_volatility"
            confidence = clip(
                0.35
                + 0.35
                * self._positive_margin(
                    volatility_percentile,
                    high_vol_pct,
                    1.0 - high_vol_pct,
                )
                + 0.20
                * self._positive_margin(
                    volatility_z,
                    high_vol_z,
                    2.0,
                )
                + 0.10
                * self._negative_margin(
                    drawdown,
                    high_vol_drawdown,
                    0.12,
                ),
                0.0,
                1.0,
            )
            risks.extend(
                [
                    {
                        "feature": "market_volatility_percentile",
                        "value": volatility_percentile,
                        "message": "市场波动率处于历史高分位",
                    },
                    {
                        "feature": "market_drawdown",
                        "value": drawdown,
                        "message": "市场净值代理出现较深回撤",
                    },
                ]
            )
        elif bull_condition:
            regime = "bull_trend"
            confidence = clip(
                0.35
                + 0.25
                * self._positive_margin(
                    trend_gap,
                    bull_gap,
                    0.04,
                )
                + 0.25
                * self._positive_margin(
                    momentum,
                    bull_momentum,
                    0.08,
                )
                + 0.15
                * self._positive_margin(
                    breadth,
                    bull_breadth,
                    0.30,
                ),
                0.0,
                1.0,
            )
            evidence.extend(
                [
                    {
                        "feature": "market_trend_gap",
                        "value": trend_gap,
                        "message": "市场快速均线显著高于慢速均线",
                    },
                    {
                        "feature": "market_momentum",
                        "value": momentum,
                        "message": "市场中期动量为正",
                    },
                    {
                        "feature": "market_breadth_smooth",
                        "value": breadth,
                        "message": "上涨股票覆盖面较广",
                    },
                ]
            )
        elif bear_condition:
            regime = "bear_trend"
            confidence = clip(
                0.35
                + 0.25
                * self._negative_margin(
                    trend_gap,
                    bear_gap,
                    0.04,
                )
                + 0.25
                * self._negative_margin(
                    momentum,
                    bear_momentum,
                    0.08,
                )
                + 0.15
                * self._negative_margin(
                    breadth,
                    bear_breadth,
                    0.30,
                ),
                0.0,
                1.0,
            )
            risks.extend(
                [
                    {
                        "feature": "market_trend_gap",
                        "value": trend_gap,
                        "message": "市场快速均线低于慢速均线",
                    },
                    {
                        "feature": "market_momentum",
                        "value": momentum,
                        "message": "市场中期动量为负",
                    },
                    {
                        "feature": "market_breadth_smooth",
                        "value": breadth,
                        "message": "市场上涨覆盖面较窄",
                    },
                ]
            )
        elif range_condition:
            regime = "range_bound"
            confidence = clip(
                0.40
                + 0.20 * (1.0 - abs(trend_gap) / max(range_gap, 1e-8))
                + 0.20
                * (1.0 - abs(momentum) / max(range_momentum, 1e-8))
                + 0.20 * (1.0 - abs(breadth - 0.5) / 0.10),
                0.0,
                1.0,
            )
            evidence.append(
                {
                    "feature": "range_bound_conditions",
                    "value": confidence,
                    "message": "趋势、动量和市场宽度均接近中性区间",
                }
            )
        else:
            regime = "neutral"
            confidence = clip(
                0.35
                + 0.20 * min(abs(trend_score), 1.0)
                + 0.20 * min(abs(momentum_score), 1.0)
                + 0.15 * min(abs(breadth_score), 1.0)
                + 0.10 * (1.0 - volatility_score),
                0.0,
                1.0,
            )
            evidence.append(
                {
                    "feature": "mixed_market_conditions",
                    "value": confidence,
                    "message": "市场趋势、动量与宽度未形成一致状态",
                }
            )

        minimum_confidence = self._cfg(
            config,
            "minimum_regime_confidence",
            0.35,
        )
        if confidence < minimum_confidence:
            regime = "neutral"
            evidence.append(
                {
                    "feature": "regime_confidence",
                    "value": confidence,
                    "message": "市场状态置信度不足，回退至中性状态",
                }
            )

        return MarketRegimeView(
            regime=regime,
            confidence=confidence,
            trend_score=trend_score,
            momentum_score=momentum_score,
            breadth_score=breadth_score,
            volatility_score=volatility_score,
            evidence=evidence[:4],
            risks=risks[:4],
        )