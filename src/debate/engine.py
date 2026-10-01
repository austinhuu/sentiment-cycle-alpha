from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agents.base import clip
from src.agents.bull_bear_agents import DebateSide


def _messages(items: list[dict[str, Any]], limit: int = 2) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item.get("message", item)).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
        if len(result) >= limit:
            break
    return result


@dataclass
class DebateResult:
    initial_bull_score: float
    initial_bear_score: float
    bull_challenge_penalty: float
    bear_challenge_penalty: float
    revised_bull_score: float
    revised_bear_score: float
    initial_net_score: float
    revised_net_score: float
    revision_delta: float
    disagreement: float
    confidence: float
    winner: str
    accepted_points: list[str]
    rejected_points: list[str]
    unresolved_conflicts: list[str]

    # 向后兼容旧代码/旧输出字段。
    @property
    def bull_score(self) -> float:
        return self.revised_bull_score

    @property
    def bear_score(self) -> float:
        return self.revised_bear_score

    @property
    def net_score(self) -> float:
        return self.revised_net_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_bull_score": float(self.initial_bull_score),
            "initial_bear_score": float(self.initial_bear_score),
            "bull_challenge_penalty": float(self.bull_challenge_penalty),
            "bear_challenge_penalty": float(self.bear_challenge_penalty),
            "revised_bull_score": float(self.revised_bull_score),
            "revised_bear_score": float(self.revised_bear_score),
            "initial_net_score": float(self.initial_net_score),
            "revised_net_score": float(self.revised_net_score),
            "revision_delta": float(self.revision_delta),
            "disagreement": float(self.disagreement),
            "confidence": float(self.confidence),
            "winner": self.winner,
            "accepted_points": self.accepted_points,
            "rejected_points": self.rejected_points,
            "unresolved_conflicts": self.unresolved_conflicts,
        }


class DebateEngine:
    """三阶段结构化辩论：独立陈述、交叉质询、观点修正。"""

    def debate(self, bull: DebateSide, bear: DebateSide) -> DebateResult:
        initial_bull = clip(bull.initial_score * bull.confidence, 0.0, 1.0)
        initial_bear = clip(bear.initial_score * bear.confidence, 0.0, 1.0)
        initial_net = clip(initial_bull - initial_bear)

        # 每一方受到“对方有效证据”和自身未解决挑战的惩罚。
        bull_penalty = clip(
            0.62 * bull.opposing_evidence_strength
            + 0.06 * min(len(bull.challenges), 3)
            + 0.03 * min(len(bear.arguments), 4),
            0.0,
            0.65,
        )
        bear_penalty = clip(
            0.62 * bear.opposing_evidence_strength
            + 0.06 * min(len(bear.challenges), 3)
            + 0.03 * min(len(bull.arguments), 4),
            0.0,
            0.65,
        )

        revised_bull = clip(initial_bull * (1.0 - bull_penalty), 0.0, 1.0)
        revised_bear = clip(initial_bear * (1.0 - bear_penalty), 0.0, 1.0)
        revised_net = clip(revised_bull - revised_bear)
        revision_delta = clip(revised_net - initial_net)

        total = revised_bull + revised_bear
        disagreement = 0.0 if total <= 0 else clip(1.0 - abs(revised_bull - revised_bear) / total, 0.0, 1.0)
        confidence = clip(
            0.30
            + 0.42 * abs(revised_net)
            + 0.20 * (1.0 - disagreement)
            + 0.08 * (1.0 - min(bull_penalty, bear_penalty)),
            0.0,
            1.0,
        )

        if revised_net > 0.06:
            winner = "bull"
            accepted = _messages(bull.arguments, 3)
            rejected = _messages(bear.arguments, 2)
        elif revised_net < -0.06:
            winner = "bear"
            accepted = _messages(bear.arguments, 3)
            rejected = _messages(bull.arguments, 2)
        else:
            winner = "draw"
            accepted = []
            rejected = []

        unresolved: list[str] = []
        if disagreement > 0.55:
            unresolved.append("交叉质询后多空证据仍接近，方向存在较大分歧")
        unresolved.extend(bull.challenges[:1])
        unresolved.extend(bear.challenges[:1])

        return DebateResult(
            initial_bull_score=initial_bull,
            initial_bear_score=initial_bear,
            bull_challenge_penalty=bull_penalty,
            bear_challenge_penalty=bear_penalty,
            revised_bull_score=revised_bull,
            revised_bear_score=revised_bear,
            initial_net_score=initial_net,
            revised_net_score=revised_net,
            revision_delta=revision_delta,
            disagreement=disagreement,
            confidence=confidence,
            winner=winner,
            accepted_points=accepted,
            rejected_points=rejected,
            unresolved_conflicts=unresolved[:3],
        )
