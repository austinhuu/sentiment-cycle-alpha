from __future__ import annotations

from typing import Any

from .base import AgentView, clip, direction_from_score


class NewsAgent:
    """V4 material-event news agent.

    Routine or unclassified announcements are strictly neutral: score=0 and
    confidence=0. They cannot lower the combined confidence or change a trade.
    """

    name = "news_agent"

    @staticmethod
    def _number(row: dict[str, Any], key: str) -> float | None:
        try:
            return float(row.get(key))
        except (TypeError, ValueError):
            return None

    def analyze(self, row: dict[str, Any]) -> AgentView:
        score = clip(self._number(row, "news_sentiment_score") or 0.0)
        confidence = clip(
            self._number(row, "news_sentiment_confidence") or 0.0,
            0.0,
            1.0,
        )
        active = int(self._number(row, "news_active") or 0)
        material_count = int(
            self._number(row, "news_material_event_count") or 0
        )
        total_count = int(self._number(row, "news_event_count") or 0)
        severity = self._number(row, "news_event_severity") or 0.0
        age = self._number(row, "news_age_days")
        category = str(
            row.get("dominant_event_type", "routine_or_unclassified") or
            "routine_or_unclassified"
        )
        title = str(row.get("latest_news_title", "") or "").strip()

        if (
            active <= 0
            or material_count <= 0
            or confidence <= 0.0
            or abs(score) < 0.03
        ):
            return AgentView(
                self.name,
                "neutral",
                0.0,
                0.0,
                [],
                [
                    {
                        "feature": "news_active",
                        "value": 0,
                        "message": "近期没有具有明确方向的重大公告或新闻事件",
                    }
                ],
            )

        evidence: list[dict[str, Any]] = []
        risks: list[dict[str, Any]] = []
        summary = {
            "feature": "news_sentiment_score",
            "value": score,
            "message": (
                f"{material_count}条重大事件的衰减方向得分为{score:.3f}，"
                f"主事件类型为{category}"
            ),
        }
        (evidence if score > 0 else risks).append(summary)

        event_item = {
            "feature": "news_event_severity",
            "value": severity,
            "message": f"主事件严重度为{severity:.2f}，总事件数为{total_count}",
        }
        (evidence if score > 0 else risks).append(event_item)
        if title:
            item = {
                "feature": "latest_news_title",
                "value": title,
                "message": f"最近事件：{title}",
            }
            (evidence if score > 0 else risks).append(item)
        if age is not None:
            item = {
                "feature": "news_age_days",
                "value": age,
                "message": f"主事件距当前约{int(max(age, 0))}天",
            }
            (evidence if age <= 7 else risks).append(item)

        return AgentView(
            self.name,
            direction_from_score(score),
            score,
            confidence,
            evidence[:4],
            risks[:4],
        )