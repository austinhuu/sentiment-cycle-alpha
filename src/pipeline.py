from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.agents.bull_bear_agents import BearAgent, BullAgent
from src.agents.decision_agent import DecisionAgent
from src.agents.fundamental_agent import FundamentalAgent
from src.agents.market_regime_agent import MarketRegimeAgent
from src.agents.news_agent import NewsAgent
from src.agents.risk_agent import RiskAgent
from src.agents.sentiment_agent import SentimentAgent
from src.agents.technical_agent import TechnicalAgent
from src.config import PROJECT_ROOT
from src.data.alternative_data import merge_alternative_data
from src.debate.engine import DebateEngine
from src.features.engineer import add_features
from src.utils.io import write_jsonl


class MultiAgentPipeline:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.technical_agent = TechnicalAgent()
        self.sentiment_agent = SentimentAgent()
        self.fundamental_agent = FundamentalAgent()
        self.news_agent = NewsAgent()
        self.market_regime_agent = MarketRegimeAgent()
        self.bull_agent = BullAgent()
        self.bear_agent = BearAgent()
        self.debate_engine = DebateEngine()
        self.risk_agent = RiskAgent()
        self.decision_agent = DecisionAgent()

    def run(
        self,
        clean_data: pd.DataFrame,
        output_dir: Path,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        data_config = self.config["data"]
        feature_config = self.config["features"]
        agent_config = self.config["agents"]
        risk_config = self.config.get("risk", {})
        regime_config = self.config.get("market_regime", {})
        alternative_config = self.config.get("alternative_data", {})
        fundamental_config = self.config.get("fundamental_agent", {})
        news_config = self.config.get("news_agent", {})

        sentiment_pool = data_config.get("sentiment_pool", [])
        trading_pool = {
            str(symbol).strip().upper()
            for symbol in data_config.get("trading_pool", [])
        }

        features = add_features(
            clean_data,
            feature_config,
            sentiment_pool=sentiment_pool,
        )
        features = merge_alternative_data(
            features,
            PROJECT_ROOT,
            alternative_config,
        )
        features["history_count"] = features.groupby("symbol").cumcount() + 1

        minimum_history = int(feature_config.get("min_history_bars", 140))
        eligible = features.loc[
            features["history_count"] >= minimum_history
        ].copy()

        if trading_pool:
            eligible = eligible.loc[
                eligible["symbol"].isin(trading_pool)
            ].copy()

        required_features = [
            "ma_gap_short_long_z",
            "momentum_z",
            "rsi",
            "atr_pct_z",
            "volatility_z",
            "volume_ratio_z",
            "market_breadth_z",
            "market_return_z",
            "volatility_percentile",
            "atr_percentile",
            "drawdown_percentile",
            "market_stress_percentile",
            "market_trend_gap",
            "market_momentum",
            "market_breadth_smooth",
            "market_volatility_z",
            "market_volatility_percentile",
            "market_drawdown",
        ]
        eligible = eligible.dropna(subset=required_features)
        if eligible.empty:
            raise RuntimeError(
                "特征预热后没有可决策数据。请下载更长时间范围或检查交易池。"
            )

        future_horizons = [
            int(value)
            for value in feature_config.get("future_horizons", [1, 5, 20])
        ]

        records: list[dict[str, Any]] = []
        logs: list[dict[str, Any]] = []

        for row in eligible.to_dict(orient="records"):
            technical = self.technical_agent.analyze(row)
            sentiment = self.sentiment_agent.analyze(row)
            fundamental = self.fundamental_agent.analyze(row)
            news = self.news_agent.analyze(row)
            market_regime = self.market_regime_agent.analyze(
                row,
                regime_config,
            )
            bull = self.bull_agent.build(
                technical,
                sentiment,
                fundamental,
                news,
                market_regime,
            )
            bear = self.bear_agent.build(
                technical,
                sentiment,
                fundamental,
                news,
                market_regime,
            )
            debate = self.debate_engine.debate(bull, bear)
            risk = self.risk_agent.assess(row, risk_config)
            is_tradable_sample = bool(
                row.get("is_tradable_sample", True)
            )

            decision = self.decision_agent.decide(
                technical=technical,
                sentiment=sentiment,
                fundamental=fundamental,
                news=news,
                debate=debate,
                risk=risk,
                market_regime=market_regime,
                config=agent_config,
                regime_config=regime_config,
                fundamental_config=fundamental_config,
                news_config=news_config,
                is_tradable_sample=is_tradable_sample,
            )

            record: dict[str, Any] = {
                "datetime": row["datetime"],
                "symbol": row["symbol"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "signal_open": row["signal_open"],
                "signal_close": row["signal_close"],
                "volume": row["volume"],
                "amount": row.get("amount"),
                "return_1": row["return_1"],
                "raw_close_return_1": row.get("raw_close_return_1"),
                "possible_raw_price_gap": row.get(
                    "possible_raw_price_gap",
                    0,
                ),
                "days_since_listing": row.get("days_since_listing", 0),
                "is_tradable_sample": is_tradable_sample,
                "ma_gap_short_long_z": row.get("ma_gap_short_long_z"),
                "ma_gap_mid_long_z": row.get("ma_gap_mid_long_z"),
                "momentum_z": row.get("momentum_z"),
                "return_5_z": row.get("return_5_z"),
                "volume_ratio_z": row.get("volume_ratio_z"),
                "volatility_z": row.get("volatility_z"),
                "atr_pct_z": row.get("atr_pct_z"),
                "rolling_drawdown": row.get("rolling_drawdown"),
                "volatility_percentile": row.get("volatility_percentile"),
                "atr_percentile": row.get("atr_percentile"),
                "drawdown_percentile": row.get("drawdown_percentile"),
                "market_stress_percentile": row.get(
                    "market_stress_percentile"
                ),
                "market_breadth": row.get("market_breadth"),
                "market_breadth_z": row.get("market_breadth_z"),
                "market_return": row.get("market_return"),
                "market_return_z": row.get("market_return_z"),
                "market_proxy_close": row.get("market_proxy_close"),
                "market_trend_gap": row.get("market_trend_gap"),
                "market_momentum": row.get("market_momentum"),
                "market_breadth_smooth": row.get("market_breadth_smooth"),
                "market_volatility": row.get("market_volatility"),
                "market_volatility_z": row.get("market_volatility_z"),
                "market_volatility_percentile": row.get(
                    "market_volatility_percentile"
                ),
                "market_drawdown": row.get("market_drawdown"),
                "valid_symbol_count": row.get("valid_symbol_count"),
                "market_regime": market_regime.regime,
                "regime_confidence": market_regime.confidence,
                "regime_trend_score": market_regime.trend_score,
                "regime_momentum_score": market_regime.momentum_score,
                "regime_breadth_score": market_regime.breadth_score,
                "regime_volatility_score": market_regime.volatility_score,
                "technical_score": technical.score,
                "technical_confidence": technical.confidence,
                "sentiment_score": sentiment.score,
                "sentiment_confidence": sentiment.confidence,
                "fundamental_score": fundamental.score,
                "fundamental_confidence": fundamental.confidence,
                "fundamental_active": row.get("fundamental_active", 0),
                "fundamental_growth_acceleration": row.get(
                    "fundamental_growth_acceleration", 0.0
                ),
                "fundamental_quality_change": row.get(
                    "fundamental_quality_change", 0.0
                ),
                "fundamental_cash_quality": row.get(
                    "fundamental_cash_quality", 0.0
                ),
                "annual_fundamental_score": row.get("annual_fundamental_score", 0.0),
                "annual_fundamental_confidence": row.get("annual_fundamental_confidence", 0.0),
                "annual_fundamental_active": row.get("annual_fundamental_active", 0),
                "annual_fundamental_growth_acceleration": row.get(
                    "annual_fundamental_growth_acceleration", 0.0
                ),
                "annual_fundamental_quality_change": row.get(
                    "annual_fundamental_quality_change", 0.0
                ),
                "annual_fundamental_cash_quality": row.get(
                    "annual_fundamental_cash_quality", 0.0
                ),
                "annual_available_date": row.get("annual_available_date"),
                "annual_fundamental_age_days": row.get("annual_fundamental_age_days"),
                "annual_report_period": row.get("annual_report_period"),
                "quarterly_fundamental_score": row.get("quarterly_fundamental_score", 0.0),
                "quarterly_fundamental_confidence": row.get(
                    "quarterly_fundamental_confidence", 0.0
                ),
                "quarterly_fundamental_active": row.get("quarterly_fundamental_active", 0),
                "quarterly_fundamental_growth_acceleration": row.get(
                    "quarterly_fundamental_growth_acceleration", 0.0
                ),
                "quarterly_fundamental_quality_change": row.get(
                    "quarterly_fundamental_quality_change", 0.0
                ),
                "quarterly_fundamental_cash_quality": row.get(
                    "quarterly_fundamental_cash_quality", 0.0
                ),
                "quarterly_available_date": row.get("quarterly_available_date"),
                "quarterly_fundamental_age_days": row.get(
                    "quarterly_fundamental_age_days"
                ),
                "quarterly_report_period": row.get("quarterly_report_period"),
                "fundamental_bundle_components": row.get(
                    "fundamental_bundle_components", "0+0"
                ),
                "available_date_source": row.get("available_date_source", ""),
                "availability_lag_days": row.get("availability_lag_days"),
                "fundamental_available_date": row.get(
                    "fundamental_available_date"
                ),
                "fundamental_age_days": row.get("fundamental_age_days"),
                "report_period": row.get("report_period"),
                "eps": row.get("eps"),
                "revenue_yoy": row.get("revenue_yoy"),
                "net_profit_yoy": row.get("net_profit_yoy"),
                "roe": row.get("roe"),
                "gross_margin": row.get("gross_margin"),
                "ocf_per_share": row.get("ocf_per_share"),
                "news_score": news.score,
                "news_confidence": news.confidence,
                "news_active": row.get("news_active", 0),
                "news_material_event_count": row.get(
                    "news_material_event_count", 0
                ),
                "news_neutral_event_count": row.get(
                    "news_neutral_event_count", 0
                ),
                "news_event_severity": row.get("news_event_severity", 0.0),
                "news_event_half_life_days": row.get(
                    "news_event_half_life_days"
                ),
                "news_decay_factor": row.get("news_decay_factor", 0.0),
                "dominant_event_type": row.get(
                    "dominant_event_type", "routine_or_unclassified"
                ),
                "event_type_breakdown": row.get("event_type_breakdown", "{}"),
                "news_available_date": row.get("news_available_date"),
                "news_age_days": row.get("news_age_days"),
                "news_event_count": row.get("news_event_count", 0),
                "latest_news_title": row.get("latest_news_title", ""),
                "initial_bull_score": debate.initial_bull_score,
                "initial_bear_score": debate.initial_bear_score,
                "bull_challenge_penalty": debate.bull_challenge_penalty,
                "bear_challenge_penalty": debate.bear_challenge_penalty,
                "revised_bull_score": debate.revised_bull_score,
                "revised_bear_score": debate.revised_bear_score,
                "debate_initial_net_score": debate.initial_net_score,
                "debate_net_score": debate.revised_net_score,
                "debate_revision_delta": debate.revision_delta,
                "debate_disagreement": debate.disagreement,
                "debate_confidence": debate.confidence,
                "risk_score": risk.risk_score,
                "risk_component_peak": risk.component_peak,
                "risk_level": risk.risk_level,
                "risk_veto_new_entry": risk.veto_new_entry,
                "risk_force_exit": risk.force_exit,
                "position_multiplier": risk.position_multiplier,
                "signal": decision.signal,
                "base_score": decision.base_score,
                "fundamental_adjustment": decision.fundamental_adjustment,
                "news_adjustment": decision.news_adjustment,
                "alternative_adjustment": decision.alternative_adjustment,
                "debate_adjustment": decision.debate_adjustment,
                "pre_risk_score": decision.pre_risk_score,
                "final_score": decision.final_score,
                "confidence": decision.confidence,
                "sentiment_multiplier": decision.sentiment_multiplier,
                "news_position_multiplier": decision.news_position_multiplier,
                "effective_risk_multiplier": (
                    decision.effective_risk_multiplier
                ),
                "regime_position_multiplier": (
                    decision.regime_position_multiplier
                ),
                "target_position": decision.target_position,
                "risk_action": decision.risk_action,
                "pre_risk_buy_condition": (
                    decision.pre_risk_buy_condition
                ),
                "effective_buy_threshold": (
                    decision.effective_buy_threshold
                ),
                "effective_sell_threshold": (
                    decision.effective_sell_threshold
                ),
                "effective_minimum_confidence": (
                    decision.effective_minimum_confidence
                ),
                "regime_minimum_holding_days": (
                    decision.regime_minimum_holding_days
                ),
                "regime_cooldown_days": decision.regime_cooldown_days,
                "reasons": json.dumps(
                    decision.reasons,
                    ensure_ascii=False,
                ),
                "risks": json.dumps(
                    decision.risks,
                    ensure_ascii=False,
                ),
            }

            for horizon in future_horizons:
                record[f"future_return_{horizon}"] = row.get(
                    f"future_return_{horizon}"
                )

            records.append(record)
            logs.append(
                {
                    "datetime": row["datetime"],
                    "symbol": row["symbol"],
                    "market_regime_view": market_regime.to_dict(),
                    "technical_view": technical.to_dict(),
                    "sentiment_view": sentiment.to_dict(),
                    "fundamental_view": fundamental.to_dict(),
                    "news_view": news.to_dict(),
                    "bull_argument": bull.to_dict(),
                    "bear_argument": bear.to_dict(),
                    "debate_result": debate.to_dict(),
                    "risk_assessment": risk.to_dict(),
                    "final_decision": decision.to_dict(),
                }
            )

        decisions = pd.DataFrame(records).sort_values(
            ["datetime", "symbol"]
        ).reset_index(drop=True)

        # Stable key used by LLM audit, calibration and post-trade monitoring.
        decisions["decision_id"] = (
            pd.to_datetime(decisions["datetime"]).dt.strftime("%Y%m%d")
            + "_"
            + decisions["symbol"].astype(str)
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        features.to_csv(
            output_dir / "processed_features.csv",
            index=False,
            encoding="utf-8-sig",
        )
        decisions.to_csv(
            output_dir / "agent_decisions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        write_jsonl(output_dir / "debate_logs.jsonl", logs)

        return features, decisions