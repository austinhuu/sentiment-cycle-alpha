from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _clip(
    series: pd.Series,
    lower: float = -1.0,
    upper: float = 1.0,
) -> pd.Series:
    return series.clip(lower=lower, upper=upper)


def _numeric_column(
    frame: pd.DataFrame,
    column: str,
    default: float,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _boolean_column(
    frame: pd.DataFrame,
    column: str,
    default: bool = False,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    values = frame[column]
    if values.dtype == bool:
        return values.fillna(default)
    return (
        values.astype(str)
        .str.strip()
        .str.lower()
        .isin(["1", "true", "yes", "y", "t", "on"])
    )


def _sentiment_multiplier(
    sentiment_score: pd.Series,
    config: dict[str, Any],
) -> pd.Series:
    minimum = max(
        0.0,
        float(config.get("sentiment_min_multiplier", 0.70)),
    )
    maximum = max(
        minimum,
        float(config.get("sentiment_max_multiplier", 1.05)),
    )
    normalized = _clip((sentiment_score + 1.0) / 2.0, 0.0, 1.0)
    return (
        minimum + (maximum - minimum) * normalized
    ).clip(lower=minimum, upper=maximum)


def _bounded_adjustment(
    score: pd.Series,
    confidence: pd.Series,
    config: dict[str, Any],
    default_weight: float,
    default_maximum: float,
    default_minimum_confidence: float,
) -> pd.Series:
    weight = float(config.get("score_weight", default_weight))
    maximum = abs(float(config.get("max_score_adjustment", default_maximum)))
    threshold = float(
        config.get(
            "minimum_confidence_to_influence",
            default_minimum_confidence,
        )
    )
    raw = score * confidence * weight
    return raw.where(confidence >= threshold, 0.0).clip(
        lower=-maximum,
        upper=maximum,
    )


def _weighted_confidence(
    frame: pd.DataFrame,
    agent_config: dict[str, Any],
    fundamental_config: dict[str, Any],
    news_config: dict[str, Any],
    use_sentiment: bool,
    use_fundamental: bool,
    use_news: bool,
) -> pd.Series:
    technical = _numeric_column(frame, "technical_confidence", 0.0)
    sentiment = _numeric_column(frame, "sentiment_confidence", 0.0)
    fundamental = _numeric_column(frame, "fundamental_confidence", 0.0)
    news = _numeric_column(frame, "news_confidence", 0.0)

    numerator = technical * float(
        agent_config.get("technical_confidence_weight", 0.75)
    )
    denominator = pd.Series(
        float(agent_config.get("technical_confidence_weight", 0.75)),
        index=frame.index,
    )

    if use_sentiment:
        weight = float(agent_config.get("sentiment_confidence_weight", 0.07))
        numerator = numerator + sentiment * weight
        denominator = denominator + weight
    if use_fundamental:
        weight = float(fundamental_config.get("confidence_weight", 0.10))
        active = fundamental > 0
        numerator = numerator + fundamental * weight
        denominator = denominator + active.astype(float) * weight
    if use_news:
        weight = float(news_config.get("confidence_weight", 0.08))
        active = news > 0
        numerator = numerator + news * weight
        denominator = denominator + active.astype(float) * weight

    return (numerator / denominator.replace(0.0, np.nan)).fillna(0.0).clip(0.0, 1.0)


def build_variant_decisions(
    components: pd.DataFrame,
    agent_config: dict[str, Any],
    variant: str = "full_v34",
    overrides: dict[str, float] | None = None,
    regime_config: dict[str, Any] | None = None,
    fundamental_config: dict[str, Any] | None = None,
    news_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """基于相同底层组件构造V3.4消融版本。"""

    aliases = {
        "full": "full_v34",
        "full_v31": "regime_v31",
        "full_v22": "technical_sentiment_risk",
    }
    variant = aliases.get(variant, variant)

    supported = {
        "moving_average",
        "technical_only",
        "technical_fundamental",
        "technical_news",
        "technical_fundamental_news",
        "technical_sentiment_risk",
        "regime_v31",
        "full_v34",
    }
    if variant not in supported:
        raise ValueError(f"未知消融版本：{variant}")

    config = dict(agent_config)
    config.update(overrides or {})
    regime_config = regime_config or {}
    fundamental_config = fundamental_config or {}
    news_config = news_config or {}
    frame = components.copy()

    technical_score = _numeric_column(frame, "technical_score", 0.0)
    sentiment_score = _numeric_column(frame, "sentiment_score", 0.0)
    fundamental_score = _numeric_column(frame, "fundamental_score", 0.0)
    fundamental_confidence = _numeric_column(
        frame,
        "fundamental_confidence",
        0.0,
    )
    news_score = _numeric_column(frame, "news_score", 0.0)
    news_confidence = _numeric_column(frame, "news_confidence", 0.0)
    risk_multiplier = _numeric_column(
        frame,
        "position_multiplier",
        1.0,
    ).clip(0.0, 1.0)
    tradable = _boolean_column(
        frame,
        "is_tradable_sample",
        default=True,
    )

    use_sentiment = variant in {
        "technical_sentiment_risk",
        "regime_v31",
        "full_v34",
    }
    use_risk = variant in {
        "technical_sentiment_risk",
        "regime_v31",
        "full_v34",
    }
    use_fundamental = variant in {
        "technical_fundamental",
        "technical_fundamental_news",
        "full_v34",
    }
    use_news = variant in {
        "technical_news",
        "technical_fundamental_news",
        "full_v34",
    }

    if variant == "moving_average":
        final_score = _clip(
            _numeric_column(frame, "ma_gap_short_long_z", 0.0) / 3.0
        )
        confidence = _clip(
            0.55 + 0.25 * final_score.abs(),
            0.0,
            1.0,
        )
    else:
        fundamental_adjustment = (
            _bounded_adjustment(
                fundamental_score,
                fundamental_confidence,
                fundamental_config,
                0.10,
                0.08,
                0.35,
            )
            if use_fundamental
            else pd.Series(0.0, index=frame.index)
        )
        news_adjustment = (
            _bounded_adjustment(
                news_score,
                news_confidence,
                news_config,
                0.07,
                0.05,
                0.30,
            )
            if use_news
            else pd.Series(0.0, index=frame.index)
        )
        final_score = _clip(
            technical_score
            + (fundamental_adjustment + news_adjustment).clip(-0.12, 0.12)
        )
        confidence = _weighted_confidence(
            frame,
            config,
            fundamental_config,
            news_config,
            use_sentiment=use_sentiment,
            use_fundamental=use_fundamental,
            use_news=use_news,
        )

    buy_threshold = float(config.get("buy_threshold", 0.25))
    sell_threshold = float(config.get("sell_threshold", -0.15))
    minimum_confidence = float(config.get("minimum_confidence", 0.72))
    max_position = float(config.get("max_position_per_symbol", 0.15))

    signal = np.where(
        ~tradable.to_numpy(),
        "HOLD",
        np.where(
            confidence.to_numpy() < minimum_confidence,
            "HOLD",
            np.where(
                final_score.to_numpy() >= buy_threshold,
                "BUY",
                np.where(
                    final_score.to_numpy() <= sell_threshold,
                    "SELL",
                    "HOLD",
                ),
            ),
        ),
    ).astype(object)

    sentiment_multiplier = (
        _sentiment_multiplier(sentiment_score, config)
        if use_sentiment
        else pd.Series(1.0, index=frame.index)
    )
    effective_risk = (
        risk_multiplier
        if use_risk
        else pd.Series(1.0, index=frame.index)
    )

    if variant == "regime_v31":
        multipliers = regime_config.get(
            "lightweight_position_multipliers",
            {},
        )
        regime_series = frame.get(
            "market_regime",
            pd.Series("neutral", index=frame.index),
        ).astype(str)
        regime_position = regime_series.map(
            lambda value: float(multipliers.get(value, 1.0))
        )
    else:
        regime_position = pd.Series(1.0, index=frame.index)

    if use_news:
        minimum = float(news_config.get("position_multiplier_min", 0.92))
        maximum = float(news_config.get("position_multiplier_max", 1.05))
        normalized_news = (news_score.clip(-1.0, 1.0) + 1.0) / 2.0
        raw_news_multiplier = minimum + (maximum - minimum) * normalized_news
        news_position = 1.0 + (
            raw_news_multiplier - 1.0
        ) * news_confidence.clip(0.0, 1.0)
    else:
        news_position = pd.Series(1.0, index=frame.index)

    target_position = np.where(
        signal == "BUY",
        max_position
        * confidence.to_numpy()
        * sentiment_multiplier.to_numpy()
        * effective_risk.to_numpy()
        * regime_position.to_numpy()
        * news_position.to_numpy(),
        0.0,
    )

    frame["signal"] = signal
    frame["target_position"] = np.clip(
        target_position,
        0.0,
        max_position,
    )
    frame["base_score"] = technical_score
    frame["final_score"] = final_score
    frame["confidence"] = confidence
    frame["sentiment_multiplier"] = sentiment_multiplier
    frame["news_position_multiplier"] = news_position
    frame["effective_risk_multiplier"] = effective_risk
    frame["regime_position_multiplier"] = regime_position
    frame["effective_buy_threshold"] = buy_threshold
    frame["effective_sell_threshold"] = sell_threshold
    frame["effective_minimum_confidence"] = minimum_confidence
    frame["regime_minimum_holding_days"] = 3
    frame["regime_cooldown_days"] = 2
    frame["risk_action"] = "none"
    frame["pre_risk_buy_condition"] = (
        tradable
        & (confidence >= minimum_confidence)
        & (final_score >= buy_threshold)
    )
    frame["variant"] = variant
    return frame