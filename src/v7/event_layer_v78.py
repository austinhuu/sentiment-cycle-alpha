from __future__ import annotations

import re
import numpy as np
import pandas as pd

POSITIVE = ["增长", "突破", "中标", "获批", "回购", "增持", "扩产", "合作", "订单", "上调", "盈利", "国产替代"]
NEGATIVE = ["减持", "处罚", "调查", "亏损", "下调", "风险", "诉讼", "终止", "延迟", "下滑", "制裁", "限制"]
POLICY = ["政策", "规划", "工信部", "国务院", "发改委", "补贴", "专项", "产业基金"]
INDUSTRY = ["半导体", "芯片", "晶圆", "先进制程", "设备", "材料", "封测", "存储"]


def _contains(text: str, terms: list[str]) -> int:
    return sum(1 for t in terms if t in text)


def structure_recent_news(news: pd.DataFrame) -> pd.DataFrame:
    """Deterministic schema compatible with a future LLM parser.

    The parser does not emit BUY/SELL. It produces direction, relevance, surprise proxy,
    persistence, scope, confidence and a bounded event score. LLM enrichment can replace these
    fields later without changing the downstream interface.
    """
    if news is None or news.empty:
        return pd.DataFrame()
    d = news.copy()
    title = d.get("新闻标题", pd.Series("", index=d.index)).fillna("").astype(str)
    body = d.get("新闻内容", pd.Series("", index=d.index)).fillna("").astype(str)
    text = title + " " + body
    pos = text.map(lambda x: _contains(x, POSITIVE))
    neg = text.map(lambda x: _contains(x, NEGATIVE))
    policy = text.map(lambda x: _contains(x, POLICY))
    industry = text.map(lambda x: _contains(x, INDUSTRY))
    direction = np.tanh((pos - neg) / 2.0)
    relevance = np.clip(0.35 + 0.18 * industry + 0.12 * policy, 0, 1)
    persistence = np.clip(0.35 + 0.15 * policy + 0.10 * industry, 0, 1)
    surprise = np.clip(0.25 + 0.10 * (pos + neg), 0, 1)
    confidence = np.clip(0.30 + 0.12 * (pos + neg + policy + industry), 0, 1)
    event_type = np.where(policy > 0, "industry_policy", np.where(industry > 0, "industry_event", "company_news"))
    score = np.clip(direction * relevance * (0.7 + 0.3 * persistence) * confidence, -1, 1)
    return pd.DataFrame({
        "published_at": pd.to_datetime(d.get("published_at", d.get("发布时间")), errors="coerce"),
        "query_keyword": d.get("query_keyword", ""),
        "event_type": event_type,
        "direction": direction,
        "relevance": relevance,
        "surprise": surprise,
        "persistence": persistence,
        "confidence": confidence,
        "structured_event_score_v78": score,
        "headline": title.str.replace(r"\s+", " ", regex=True).str.slice(0, 200),
    })
