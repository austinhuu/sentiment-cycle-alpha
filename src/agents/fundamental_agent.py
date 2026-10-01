from __future__ import annotations

from typing import Any

from .base import AgentView, clip, direction_from_score


class FundamentalAgent:
    """Point-in-time annual + interim fundamental evidence agent."""

    name = "fundamental_agent"

    @staticmethod
    def _number(row: dict[str, Any], key: str) -> float | None:
        value = row.get(key)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed

    @staticmethod
    def _append_component(
        evidence: list[dict[str, Any]],
        risks: list[dict[str, Any]],
        feature: str,
        value: float | None,
        message: str,
    ) -> None:
        if value is None:
            return
        item = {"feature": feature, "value": value, "message": message}
        (evidence if value >= 0 else risks).append(item)

    def analyze(self, row: dict[str, Any]) -> AgentView:
        score = clip(self._number(row, "fundamental_score") or 0.0)
        confidence = clip(
            self._number(row, "fundamental_confidence") or 0.0,
            0.0,
            1.0,
        )
        active = int(self._number(row, "fundamental_active") or 0)
        if active <= 0 or confidence <= 0.0 or abs(score) < 0.03:
            return AgentView(
                self.name,
                "neutral",
                0.0,
                0.0,
                [],
                [
                    {
                        "feature": "fundamental_active",
                        "value": 0,
                        "message": "当前没有足够新鲜且具有方向性的年度/季度财务证据",
                    }
                ],
            )

        evidence: list[dict[str, Any]] = []
        risks: list[dict[str, Any]] = []
        summary = {
            "feature": "fundamental_score",
            "value": score,
            "message": f"年度质量与最新季度变化联合得分为{score:.3f}",
        }
        (evidence if score > 0 else risks).append(summary)

        annual_active = int(self._number(row, "annual_fundamental_active") or 0)
        quarterly_active = int(self._number(row, "quarterly_fundamental_active") or 0)
        if annual_active:
            annual_score = self._number(row, "annual_fundamental_score")
            self._append_component(
                evidence,
                risks,
                "annual_fundamental_score",
                annual_score,
                f"最近年报变化得分为{(annual_score or 0.0):.3f}",
            )
            annual_period = row.get("annual_report_period")
            if annual_period is not None:
                evidence.append(
                    {
                        "feature": "annual_report_period",
                        "value": str(annual_period),
                        "message": f"年报通道最近可用报告期为{annual_period}",
                    }
                )
        if quarterly_active:
            quarterly_score = self._number(row, "quarterly_fundamental_score")
            self._append_component(
                evidence,
                risks,
                "quarterly_fundamental_score",
                quarterly_score,
                f"最近季度变化得分为{(quarterly_score or 0.0):.3f}",
            )
            quarterly_period = row.get("quarterly_report_period")
            if quarterly_period is not None:
                evidence.append(
                    {
                        "feature": "quarterly_report_period",
                        "value": str(quarterly_period),
                        "message": f"季度通道最近可用报告期为{quarterly_period}",
                    }
                )

        components = [
            ("fundamental_growth_acceleration", "盈利与收入增速变化"),
            ("fundamental_quality_change", "盈利质量变化"),
            ("fundamental_cash_quality", "现金流质量变化"),
        ]
        for feature, label in components:
            value = self._number(row, feature)
            self._append_component(
                evidence,
                risks,
                feature,
                value,
                f"{label}联合标准化得分为{(value or 0.0):.3f}",
            )

        age = self._number(row, "fundamental_age_days")
        if age is not None:
            item = {
                "feature": "fundamental_age_days",
                "value": age,
                "message": f"最近一条联合披露证据距当前约{int(max(age, 0))}天",
            }
            (evidence if age <= 120 else risks).append(item)

        return AgentView(
            self.name,
            direction_from_score(score),
            score,
            confidence,
            evidence[:6],
            risks[:6],
        )