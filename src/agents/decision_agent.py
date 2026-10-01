from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.debate.engine import DebateResult

from .base import AgentView, clip
from .market_regime_agent import MarketRegimeView
from .risk_agent import RiskAssessment


@dataclass
class FinalDecision:
    signal: str
    base_score: float
    fundamental_adjustment: float
    news_adjustment: float
    alternative_adjustment: float
    debate_adjustment: float
    pre_risk_score: float
    final_score: float
    confidence: float
    sentiment_multiplier: float
    news_position_multiplier: float
    effective_risk_multiplier: float
    regime_position_multiplier: float
    target_position: float
    risk_action: str
    pre_risk_buy_condition: bool
    market_regime: str
    regime_confidence: float
    effective_buy_threshold: float
    effective_sell_threshold: float
    effective_minimum_confidence: float
    regime_minimum_holding_days: int
    regime_cooldown_days: int
    reasons: list[str]
    risks: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal": self.signal,
            "base_score": float(self.base_score),
            "fundamental_adjustment": float(self.fundamental_adjustment),
            "news_adjustment": float(self.news_adjustment),
            "alternative_adjustment": float(self.alternative_adjustment),
            "debate_adjustment": float(self.debate_adjustment),
            "pre_risk_score": float(self.pre_risk_score),
            "final_score": float(self.final_score),
            "confidence": float(self.confidence),
            "sentiment_multiplier": float(self.sentiment_multiplier),
            "news_position_multiplier": float(self.news_position_multiplier),
            "effective_risk_multiplier": float(self.effective_risk_multiplier),
            "regime_position_multiplier": float(self.regime_position_multiplier),
            "target_position": float(self.target_position),
            "risk_action": self.risk_action,
            "pre_risk_buy_condition": bool(self.pre_risk_buy_condition),
            "market_regime": self.market_regime,
            "regime_confidence": float(self.regime_confidence),
            "effective_buy_threshold": float(self.effective_buy_threshold),
            "effective_sell_threshold": float(self.effective_sell_threshold),
            "effective_minimum_confidence": float(self.effective_minimum_confidence),
            "regime_minimum_holding_days": int(self.regime_minimum_holding_days),
            "regime_cooldown_days": int(self.regime_cooldown_days),
            "reasons": self.reasons,
            "risks": self.risks,
        }


class DecisionAgent:
    """V4静态基线决策智能体。

    该文件保留一个可解释的静态基线，供V4自适应融合和消融实验比较。
    中性基本面/新闻严格不参与得分、置信度和仓位融合。
    """

    @staticmethod
    def _float(config: dict[str, Any], key: str, default: float) -> float:
        try:
            return float(config.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _list(value: Any) -> list[Any]:
        if value is None:
            return []
        return value if isinstance(value, list) else [value]

    @staticmethod
    def _message(item: Any) -> str:
        if isinstance(item, dict):
            return str(
                item.get("message", item.get("interpretation", item))
            ).strip()
        return str(item).strip()

    @classmethod
    def _dedupe(cls, items: list[Any], limit: int = 6) -> list[str]:
        result: list[str] = []
        seen_features: set[str] = set()
        seen_messages: set[str] = set()
        for item in items:
            feature = (
                str(item.get("feature", "")).strip()
                if isinstance(item, dict)
                else ""
            )
            message = cls._message(item)
            if not message or message in seen_messages:
                continue
            if feature and feature in seen_features:
                continue
            if feature:
                seen_features.add(feature)
            seen_messages.add(message)
            result.append(message)
            if len(result) >= limit:
                break
        return result

    @classmethod
    def _sentiment_multiplier(
        cls,
        score: float,
        config: dict[str, Any],
    ) -> float:
        minimum = max(
            0.0,
            cls._float(config, "sentiment_min_multiplier", 0.70),
        )
        maximum = max(
            minimum,
            cls._float(config, "sentiment_max_multiplier", 1.05),
        )
        normalized = (clip(score, -1.0, 1.0) + 1.0) / 2.0
        return clip(
            minimum + (maximum - minimum) * normalized,
            minimum,
            maximum,
        )

    @staticmethod
    def _bounded_adjustment(
        view: AgentView,
        config: dict[str, Any],
        default_weight: float,
        default_maximum: float,
        default_minimum_confidence: float,
    ) -> float:
        confidence_threshold = float(
            config.get(
                "minimum_confidence_to_influence",
                default_minimum_confidence,
            )
        )
        minimum_abs_score = abs(
            float(config.get("minimum_abs_score_to_influence", 0.03))
        )
        if (
            float(view.confidence) < confidence_threshold
            or abs(float(view.score)) < minimum_abs_score
        ):
            return 0.0
        weight = float(config.get("score_weight", default_weight))
        maximum = abs(
            float(config.get("max_score_adjustment", default_maximum))
        )
        raw = float(view.score) * float(view.confidence) * weight
        return clip(raw, -maximum, maximum)

    @staticmethod
    def _news_position_multiplier(
        news: AgentView,
        config: dict[str, Any],
    ) -> float:
        minimum = max(
            0.0,
            float(config.get("position_multiplier_min", 0.92)),
        )
        maximum = max(
            minimum,
            float(config.get("position_multiplier_max", 1.05)),
        )
        minimum_confidence = float(
            config.get("minimum_confidence_to_influence", 0.40)
        )
        minimum_abs_score = abs(
            float(config.get("minimum_abs_score_to_influence", 0.03))
        )
        if (
            news.confidence < minimum_confidence
            or abs(news.score) < minimum_abs_score
        ):
            return 1.0
        normalized = (clip(news.score, -1.0, 1.0) + 1.0) / 2.0
        raw = minimum + (maximum - minimum) * normalized
        # 置信度低时自动回到1，不让弱新闻信号过度影响仓位。
        return 1.0 + (raw - 1.0) * clip(news.confidence, 0.0, 1.0)

    @staticmethod
    def _weighted_confidence(
        technical: AgentView,
        sentiment: AgentView,
        fundamental: AgentView,
        news: AgentView,
        agent_config: dict[str, Any],
        fundamental_config: dict[str, Any],
        news_config: dict[str, Any],
    ) -> float:
        candidates = [
            (
                float(technical.confidence),
                float(agent_config.get("technical_confidence_weight", 0.75)),
                True,
            ),
            (
                float(sentiment.confidence),
                float(agent_config.get("sentiment_confidence_weight", 0.07)),
                True,
            ),
            (
                float(fundamental.confidence),
                float(fundamental_config.get("confidence_weight", 0.08)),
                (
                    fundamental.confidence
                    >= float(
                        fundamental_config.get(
                            "minimum_confidence_to_influence", 0.40
                        )
                    )
                    and abs(fundamental.score)
                    >= abs(
                        float(
                            fundamental_config.get(
                                "minimum_abs_score_to_influence", 0.03
                            )
                        )
                    )
                ),
            ),
            (
                float(news.confidence),
                float(news_config.get("confidence_weight", 0.06)),
                (
                    news.confidence
                    >= float(
                        news_config.get(
                            "minimum_confidence_to_influence", 0.40
                        )
                    )
                    and abs(news.score)
                    >= abs(
                        float(
                            news_config.get(
                                "minimum_abs_score_to_influence", 0.03
                            )
                        )
                    )
                ),
            ),
        ]
        numerator = 0.0
        denominator = 0.0
        for confidence, weight, active in candidates:
            if not active or weight <= 0:
                continue
            numerator += clip(confidence, 0.0, 1.0) * weight
            denominator += weight
        return clip(numerator / denominator if denominator > 0 else 0.0, 0.0, 1.0)

    @staticmethod
    def _regime_multiplier(
        market_regime: MarketRegimeView,
        regime_config: dict[str, Any],
    ) -> float:
        if not bool(regime_config.get("use_for_trading", False)):
            return 1.0
        multipliers = regime_config.get("lightweight_position_multipliers", {})
        raw = float(multipliers.get(market_regime.regime, 1.0))
        strength = clip(
            float(regime_config.get("position_adjustment_strength", 0.15)),
            0.0,
            1.0,
        )
        return max(0.0, 1.0 + (raw - 1.0) * strength)

    def decide(
        self,
        technical: AgentView,
        sentiment: AgentView,
        fundamental: AgentView,
        news: AgentView,
        debate: DebateResult,
        risk: RiskAssessment,
        market_regime: MarketRegimeView,
        config: dict[str, Any],
        regime_config: dict[str, Any],
        fundamental_config: dict[str, Any],
        news_config: dict[str, Any],
        is_tradable_sample: bool = True,
    ) -> FinalDecision:
        base_score = clip(float(technical.score))
        fundamental_adjustment = self._bounded_adjustment(
            fundamental,
            fundamental_config,
            default_weight=0.10,
            default_maximum=0.08,
            default_minimum_confidence=0.35,
        )
        news_adjustment = self._bounded_adjustment(
            news,
            news_config,
            default_weight=0.07,
            default_maximum=0.05,
            default_minimum_confidence=0.30,
        )
        alternative_adjustment = clip(
            fundamental_adjustment + news_adjustment,
            -0.12,
            0.12,
        )
        debate_adjustment = 0.0
        pre_risk_score = clip(base_score + alternative_adjustment)
        final_score = pre_risk_score

        confidence = self._weighted_confidence(
            technical,
            sentiment,
            fundamental,
            news,
            config,
            fundamental_config,
            news_config,
        )

        buy_threshold = self._float(config, "buy_threshold", 0.25)
        sell_threshold = self._float(config, "sell_threshold", -0.15)
        minimum_confidence = self._float(
            config,
            "minimum_confidence",
            0.72,
        )
        max_position = clip(
            self._float(config, "max_position_per_symbol", 0.15),
            0.0,
            1.0,
        )

        pre_risk_buy_condition = bool(
            is_tradable_sample
            and final_score >= buy_threshold
            and confidence >= minimum_confidence
        )

        if not is_tradable_sample:
            provisional_signal = "HOLD"
        elif confidence < minimum_confidence:
            provisional_signal = "HOLD"
        elif final_score >= buy_threshold:
            provisional_signal = "BUY"
        elif final_score <= sell_threshold:
            provisional_signal = "SELL"
        else:
            provisional_signal = "HOLD"

        risk_action = "none"
        if risk.force_exit:
            signal = "SELL"
            risk_action = "force_exit"
        elif provisional_signal == "BUY" and risk.veto_new_entry:
            signal = "HOLD"
            risk_action = "veto_new_entry"
        else:
            signal = provisional_signal

        sentiment_multiplier = self._sentiment_multiplier(
            sentiment.score,
            config,
        )
        news_position_multiplier = self._news_position_multiplier(
            news,
            news_config,
        )
        effective_risk_multiplier = clip(
            float(risk.position_multiplier),
            0.0,
            1.0,
        )
        regime_position_multiplier = self._regime_multiplier(
            market_regime,
            regime_config,
        )

        target_position = 0.0
        if signal == "BUY":
            target_position = (
                max_position
                * confidence
                * sentiment_multiplier
                * news_position_multiplier
                * effective_risk_multiplier
                * regime_position_multiplier
            )
        target_position = clip(target_position, 0.0, max_position)

        reason_items: list[Any] = []
        if not is_tradable_sample:
            reason_items.append("样本仍处于上市或数据起点预热期，暂不交易")
        elif signal == "BUY":
            reason_items.extend(self._list(technical.evidence)[:3])
            reason_items.extend(self._list(fundamental.evidence)[:2])
            reason_items.extend(self._list(news.evidence)[:2])
            reason_items.extend(self._list(sentiment.evidence)[:1])
            reason_items.extend(self._list(market_regime.evidence)[:1])
        elif signal == "SELL":
            reason_items.extend(self._list(technical.risks)[:3])
            reason_items.extend(self._list(fundamental.risks)[:2])
            reason_items.extend(self._list(news.risks)[:2])
            reason_items.extend(self._list(market_regime.risks)[:1])
        else:
            reason_items.extend(self._list(debate.unresolved_conflicts)[:2])
            reason_items.extend(self._list(risk.reasons)[:2])
            reason_items.extend(self._list(fundamental.evidence)[:1])
            reason_items.extend(self._list(news.evidence)[:1])

        reasons = self._dedupe(reason_items, 6) or [
            "综合得分或置信度未达到交易阈值"
        ]

        risk_items: list[Any] = []
        risk_items.extend(self._list(technical.risks)[:2])
        risk_items.extend(self._list(fundamental.risks)[:2])
        risk_items.extend(self._list(news.risks)[:2])
        risk_items.extend(self._list(sentiment.risks)[:1])
        risk_items.extend(self._list(market_regime.risks)[:1])
        risk_items.extend(self._list(risk.reasons)[:2])
        if float(debate.disagreement) >= 0.55:
            risk_items.append(
                {
                    "feature": "debate_disagreement",
                    "message": "多空证据仍存在较大分歧；当前版本仅记录，不修改回测",
                }
            )
        risks = self._dedupe(risk_items, 6) or ["未发现显著异常风险"]

        return FinalDecision(
            signal=signal,
            base_score=base_score,
            fundamental_adjustment=fundamental_adjustment,
            news_adjustment=news_adjustment,
            alternative_adjustment=alternative_adjustment,
            debate_adjustment=debate_adjustment,
            pre_risk_score=pre_risk_score,
            final_score=final_score,
            confidence=confidence,
            sentiment_multiplier=sentiment_multiplier,
            news_position_multiplier=news_position_multiplier,
            effective_risk_multiplier=effective_risk_multiplier,
            regime_position_multiplier=regime_position_multiplier,
            target_position=target_position,
            risk_action=risk_action,
            pre_risk_buy_condition=pre_risk_buy_condition,
            market_regime=market_regime.regime,
            regime_confidence=market_regime.confidence,
            effective_buy_threshold=buy_threshold,
            effective_sell_threshold=sell_threshold,
            effective_minimum_confidence=minimum_confidence,
            regime_minimum_holding_days=3,
            regime_cooldown_days=2,
            reasons=reasons,
            risks=risks,
        )