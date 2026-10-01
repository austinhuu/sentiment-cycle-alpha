from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .base import clip


@dataclass
class RiskAssessment:
    """风险智能体的结构化输出。"""

    risk_score: float
    risk_level: str
    position_multiplier: float
    veto_new_entry: bool
    force_exit: bool
    component_peak: float
    reasons: list[str]

    @property
    def veto(self) -> bool:
        """兼容旧代码中的 risk.veto。"""

        return self.veto_new_entry

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_score": float(self.risk_score),
            "risk_level": self.risk_level,
            "position_multiplier": float(
                self.position_multiplier
            ),
            "veto_new_entry": bool(
                self.veto_new_entry
            ),
            "force_exit": bool(self.force_exit),
            "component_peak": float(
                self.component_peak
            ),
            "reasons": self.reasons,
        }


class RiskAgent:
    """V2.2独立风险管理智能体。

    设计原则：
    1. 风险不修改技术方向得分；
    2. 默认关闭硬性禁止开仓和强制退出；
    3. 只通过连续仓位倍率控制风险；
    4. 保留硬性动作接口，便于后续消融和研究验证。
    """

    @staticmethod
    def _safe(
        row: dict[str, Any],
        key: str,
        default: float = 0.50,
    ) -> float:
        try:
            value = float(row.get(key, default))
            return value if math.isfinite(value) else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _config_float(
        config: dict[str, Any],
        key: str,
        default: float,
    ) -> float:
        try:
            return float(config.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _config_bool(
        config: dict[str, Any],
        key: str,
        default: bool,
    ) -> bool:
        value = config.get(key, default)

        if isinstance(value, bool):
            return value

        return str(value).strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }

    def assess(
        self,
        row: dict[str, Any],
        config: dict[str, Any],
    ) -> RiskAssessment:
        volatility_percentile = clip(
            self._safe(
                row,
                "volatility_percentile",
                0.50,
            ),
            0.0,
            1.0,
        )
        atr_percentile = clip(
            self._safe(
                row,
                "atr_percentile",
                0.50,
            ),
            0.0,
            1.0,
        )
        drawdown_percentile = clip(
            self._safe(
                row,
                "drawdown_percentile",
                0.50,
            ),
            0.0,
            1.0,
        )
        market_stress_percentile = clip(
            self._safe(
                row,
                "market_stress_percentile",
                0.50,
            ),
            0.0,
            1.0,
        )
        rolling_drawdown = min(
            self._safe(
                row,
                "rolling_drawdown",
                0.0,
            ),
            0.0,
        )

        component_peak = max(
            volatility_percentile,
            atr_percentile,
            drawdown_percentile,
            market_stress_percentile,
        )

        composite = (
            0.35 * volatility_percentile
            + 0.30 * atr_percentile
            + 0.20 * drawdown_percentile
            + 0.15 * market_stress_percentile
        )

        risk_score = clip(
            0.65 * composite
            + 0.35 * component_peak,
            0.0,
            1.0,
        )

        enable_veto = self._config_bool(
            config,
            "enable_veto_new_entry",
            False,
        )
        enable_force_exit = self._config_bool(
            config,
            "enable_force_exit",
            False,
        )

        veto_component_percentile = (
            self._config_float(
                config,
                "veto_component_percentile",
                0.90,
            )
        )
        veto_score_threshold = (
            self._config_float(
                config,
                "veto_score_threshold",
                0.78,
            )
        )
        force_exit_component_percentile = (
            self._config_float(
                config,
                "force_exit_component_percentile",
                0.99,
            )
        )
        force_exit_score_threshold = (
            self._config_float(
                config,
                "force_exit_score_threshold",
                0.90,
            )
        )
        force_exit_drawdown = self._config_float(
            config,
            "force_exit_drawdown",
            -0.15,
        )

        minimum_multiplier = clip(
            self._config_float(
                config,
                "minimum_position_multiplier",
                0.60,
            ),
            0.0,
            1.0,
        )
        penalty_start = clip(
            self._config_float(
                config,
                "position_penalty_start",
                0.55,
            ),
            0.0,
            0.99,
        )

        raw_veto_condition = bool(
            component_peak
            >= veto_component_percentile
            or risk_score >= veto_score_threshold
        )
        raw_force_exit_condition = bool(
            component_peak
            >= force_exit_component_percentile
            and risk_score
            >= force_exit_score_threshold
            and rolling_drawdown
            <= force_exit_drawdown
        )

        veto_new_entry = bool(
            enable_veto and raw_veto_condition
        )
        force_exit = bool(
            enable_force_exit
            and raw_force_exit_condition
        )

        # 风险在penalty_start以下时完全不降仓；超过阈值后，
        # 平滑下降至minimum_position_multiplier。
        if risk_score <= penalty_start:
            position_multiplier = 1.0
        else:
            progress = clip(
                (risk_score - penalty_start)
                / max(1.0 - penalty_start, 1e-8),
                0.0,
                1.0,
            )
            position_multiplier = (
                1.0
                - progress
                * (1.0 - minimum_multiplier)
            )

        position_multiplier = clip(
            position_multiplier,
            minimum_multiplier,
            1.0,
        )

        if risk_score >= 0.90:
            risk_level = "high"
        elif risk_score >= 0.75:
            risk_level = "medium_high"
        elif risk_score >= 0.55:
            risk_level = "medium"
        else:
            risk_level = "low"

        reasons: list[str] = []

        if volatility_percentile >= 0.85:
            reasons.append(
                "个股波动率位于自身历史高分位"
            )
        if atr_percentile >= 0.85:
            reasons.append(
                "ATR占价格比例位于自身历史高分位"
            )
        if drawdown_percentile >= 0.85:
            reasons.append(
                "近期回撤严重程度位于历史高分位"
            )
        if market_stress_percentile >= 0.85:
            reasons.append(
                "市场横截面波动处于历史高分位"
            )
        if risk_score > penalty_start:
            reasons.append(
                "风险得分超过连续降仓起点，目标仓位已平滑下调"
            )
        if veto_new_entry:
            reasons.append(
                "风险环境达到禁止新开仓条件"
            )
        elif raw_veto_condition and not enable_veto:
            reasons.append(
                "风险条件较高，但V2.2已关闭硬性禁止开仓"
            )
        if force_exit:
            reasons.append(
                "极端风险与深度回撤同时出现，触发强制退出"
            )
        elif (
            raw_force_exit_condition
            and not enable_force_exit
        ):
            reasons.append(
                "极端风险条件成立，但V2.2已关闭硬性强制退出"
            )
        if not reasons:
            reasons.append(
                "当前风险水平未达到明显降仓阈值"
            )

        return RiskAssessment(
            risk_score=risk_score,
            risk_level=risk_level,
            position_multiplier=position_multiplier,
            veto_new_entry=veto_new_entry,
            force_exit=force_exit,
            component_peak=component_peak,
            reasons=reasons[:6],
        )
