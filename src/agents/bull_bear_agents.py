from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import AgentView, clip
from .market_regime_agent import MarketRegimeView


@dataclass
class DebateSide:
    side: str
    initial_score: float
    confidence: float
    arguments: list[dict[str, Any]]
    challenges: list[str]
    opposing_evidence_strength: float
    source_scores: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "initial_score": float(self.initial_score),
            "confidence": float(self.confidence),
            "arguments": self.arguments,
            "challenges": self.challenges,
            "opposing_evidence_strength": float(self.opposing_evidence_strength),
            "source_scores": self.source_scores,
        }


def _weighted_support(
    views: list[tuple[AgentView, float]],
    bullish: bool,
) -> tuple[float, float, dict[str, float]]:
    support = 0.0
    opposition = 0.0
    source_scores: dict[str, float] = {}
    total_weight = 0.0
    for view, weight in views:
        effective_weight = weight * max(float(view.confidence), 0.15)
        signed = float(view.score) if bullish else -float(view.score)
        support += effective_weight * max(signed, 0.0)
        opposition += effective_weight * max(-signed, 0.0)
        total_weight += effective_weight
        source_scores[view.agent] = float(view.score)
    if total_weight <= 0:
        return 0.0, 0.0, source_scores
    return (
        clip(support / total_weight, 0.0, 1.0),
        clip(opposition / total_weight, 0.0, 1.0),
        source_scores,
    )


class BullAgent:
    def build(
        self,
        technical: AgentView,
        sentiment: AgentView,
        fundamental: AgentView,
        news: AgentView,
        market_regime: MarketRegimeView,
    ) -> DebateSide:
        views = [
            (technical, 0.58),
            (fundamental, 0.18),
            (news, 0.12),
            (sentiment, 0.12),
        ]
        support, opposition, source_scores = _weighted_support(views, True)
        arguments = (
            technical.evidence[:3]
            + fundamental.evidence[:2]
            + news.evidence[:2]
            + sentiment.evidence[:1]
            + market_regime.evidence[:1]
        )
        challenges: list[str] = []
        if technical.risks:
            challenges.append("空头质询：技术趋势存在反向、过热或波动证据")
        if fundamental.score < -0.08 and fundamental.confidence > 0:
            challenges.append("空头质询：已披露财务质量或增长指标偏弱")
        if news.score < -0.08 and news.confidence > 0:
            challenges.append("空头质询：近期公告或新闻包含负面事件")
        if market_regime.regime == "bear_trend":
            challenges.append("空头质询：市场处于熊市趋势环境")
        if not arguments:
            challenges.append("多头证据不足，无法形成高质量开仓论据")
        confidence = clip(
            0.45 * technical.confidence
            + 0.18 * fundamental.confidence
            + 0.12 * news.confidence
            + 0.10 * sentiment.confidence
            + 0.15 * support,
            0.0,
            1.0,
        )
        return DebateSide(
            "bull",
            support,
            confidence,
            arguments,
            challenges,
            opposition,
            source_scores,
        )


class BearAgent:
    def build(
        self,
        technical: AgentView,
        sentiment: AgentView,
        fundamental: AgentView,
        news: AgentView,
        market_regime: MarketRegimeView,
    ) -> DebateSide:
        views = [
            (technical, 0.58),
            (fundamental, 0.18),
            (news, 0.12),
            (sentiment, 0.12),
        ]
        support, opposition, source_scores = _weighted_support(views, False)
        arguments = (
            technical.risks[:3]
            + fundamental.risks[:2]
            + news.risks[:2]
            + sentiment.risks[:1]
            + market_regime.risks[:1]
        )
        challenges: list[str] = []
        if technical.evidence:
            challenges.append("多头质询：趋势、动量或突破仍存在正向证据")
        if fundamental.score > 0.08 and fundamental.confidence > 0:
            challenges.append("多头质询：财务质量与增长证据支持长期韧性")
        if news.score > 0.08 and news.confidence > 0:
            challenges.append("多头质询：近期公告或新闻包含积极事件")
        if market_regime.regime in {"bull_trend", "high_volatility"}:
            challenges.append("多头质询：当前市场环境仍可能支持趋势或反弹")
        if not arguments:
            challenges.append("空头证据不足，无法形成高质量退出论据")
        confidence = clip(
            0.45 * technical.confidence
            + 0.18 * fundamental.confidence
            + 0.12 * news.confidence
            + 0.10 * sentiment.confidence
            + 0.15 * support,
            0.0,
            1.0,
        )
        return DebateSide(
            "bear",
            support,
            confidence,
            arguments,
            challenges,
            opposition,
            source_scores,
        )