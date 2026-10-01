from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.openai_config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL


LOGGER = logging.getLogger(__name__)


DEBATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "bull_thesis": {"type": "string"},
        "bear_thesis": {"type": "string"},
        "accepted_evidence_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "rejected_evidence_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "unresolved_conflicts": {
            "type": "array",
            "items": {"type": "string"},
        },
        "recommended_action": {
            "type": "string",
            "enum": ["BUY", "HOLD", "SELL"],
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "confidence_adjustment": {
            "type": "number",
            "minimum": -0.05,
            "maximum": 0.05,
        },
        "summary": {"type": "string"},
    },
    "required": [
        "bull_thesis",
        "bear_thesis",
        "accepted_evidence_ids",
        "rejected_evidence_ids",
        "unresolved_conflicts",
        "recommended_action",
        "confidence",
        "confidence_adjustment",
        "summary",
    ],
}


@dataclass
class LLMDebateResult:
    bull_thesis: str
    bear_thesis: str
    accepted_evidence_ids: list[str]
    rejected_evidence_ids: list[str]
    unresolved_conflicts: list[str]
    recommended_action: str
    confidence: float
    confidence_adjustment: float
    summary: str
    model: str
    cache_key: str
    raw_response_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "bull_thesis": self.bull_thesis,
            "bear_thesis": self.bear_thesis,
            "accepted_evidence_ids": self.accepted_evidence_ids,
            "rejected_evidence_ids": self.rejected_evidence_ids,
            "unresolved_conflicts": self.unresolved_conflicts,
            "recommended_action": self.recommended_action,
            "confidence": float(self.confidence),
            "confidence_adjustment": float(self.confidence_adjustment),
            "summary": self.summary,
            "model": self.model,
            "cache_key": self.cache_key,
            "raw_response_id": self.raw_response_id,
        }


class LLMDebateAdapter:
    """V3.4基于DeepSeek OpenAI兼容Chat Completions API的结构化多空辩论适配器。

    LLM只能引用输入中存在的evidence_id。默认由单独脚本调用，
    不在历史pipeline中逐行请求，也不自动改变回测结果。
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.model = str(config.get("model", OPENAI_MODEL)).strip() or OPENAI_MODEL
        self.api_key = str(OPENAI_API_KEY).strip()
        if not self.api_key or "在这里填写" in self.api_key:
            raise RuntimeError(
                "请先在 config/openai_config.py 中填写 OPENAI_API_KEY。"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "缺少openai，请运行：python -m pip install openai"
            ) from exc
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=OPENAI_BASE_URL,
            timeout=180.0,
            max_retries=0,
        )
        self.max_retries = max(int(config.get("max_retries", 3)), 1)
        self.pause = max(float(config.get("request_pause_seconds", 0.5)), 0.0)
        self.reasoning_effort = str(config.get("reasoning_effort", "low"))
        self.max_output_tokens = int(config.get("max_output_tokens", 1800))
        self.store = bool(config.get("store_responses", False))

    @staticmethod
    def _extract_evidence(view: dict[str, Any], prefix: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for category in ("evidence", "risks"):
            for index, item in enumerate(view.get(category, []) or []):
                if isinstance(item, dict):
                    message = str(
                        item.get("message", item.get("interpretation", item))
                    )
                    feature = str(item.get("feature", ""))
                    value = item.get("value")
                else:
                    message = str(item)
                    feature = ""
                    value = None
                items.append(
                    {
                        "evidence_id": f"{prefix}_{category}_{index + 1}",
                        "source": prefix,
                        "category": category,
                        "feature": feature,
                        "value": value,
                        "message": message,
                    }
                )
        return items

    @classmethod
    def build_context(cls, log_record: dict[str, Any]) -> dict[str, Any]:
        evidence: list[dict[str, Any]] = []
        view_mapping = {
            "technical": log_record.get("technical_view", {}),
            "sentiment": log_record.get("sentiment_view", {}),
            "fundamental": log_record.get("fundamental_view", {}),
            "news": log_record.get("news_view", {}),
            "regime": log_record.get("market_regime_view", {}),
            "risk": log_record.get("risk_assessment", {}),
        }
        for prefix, view in view_mapping.items():
            if isinstance(view, dict):
                evidence.extend(cls._extract_evidence(view, prefix))

        decision = log_record.get("final_decision", {}) or {}
        debate = log_record.get("debate_result", {}) or {}
        return {
            "datetime": str(log_record.get("datetime")),
            "symbol": str(log_record.get("symbol")),
            "quantitative_decision": {
                "signal": decision.get("signal"),
                "final_score": decision.get("final_score"),
                "confidence": decision.get("confidence"),
                "target_position": decision.get("target_position"),
            },
            "structured_debate": {
                "winner": debate.get("winner"),
                "disagreement": debate.get("disagreement"),
                "accepted_points": debate.get("accepted_points", []),
                "unresolved_conflicts": debate.get(
                    "unresolved_conflicts", []
                ),
            },
            "evidence": evidence,
        }

    @staticmethod
    def build_prompt(context: dict[str, Any]) -> str:
        output_example = {
            "bull_thesis": "多头论点",
            "bear_thesis": "空头论点",
            "accepted_evidence_ids": [],
            "rejected_evidence_ids": [],
            "unresolved_conflicts": [],
            "recommended_action": "HOLD",
            "confidence": 0.5,
            "confidence_adjustment": 0.0,
            "summary": "审计总结",
        }
        return (
            "你是量化交易公司的审慎辩论主持人。"
            "只能使用输入JSON里的证据，不得补充公司事实、新闻、价格或财务数字。"
            "必须引用evidence_id；若证据不足，应推荐HOLD。"
            "你的任务是分别形成最强多头与空头论点，交叉检查证据，"
            "并给出一个仅用于审计的建议。不要假装这是确定性预测。"
            "必须只输出一个合法JSON对象，不得输出Markdown代码块或额外文字。"
            f"\nJSON输出示例：{json.dumps(output_example, ensure_ascii=False)}"
            f"\n输入JSON：{json.dumps(context, ensure_ascii=False, default=str)}"
        )

    @staticmethod
    def _cache_key(context: dict[str, Any], model: str) -> str:
        raw = json.dumps(
            {"model": model, "context": context},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start >= 0 and end > start:
                return json.loads(cleaned[start : end + 1])
            raise

    @staticmethod
    def validate_response(
        payload: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        required = set(DEBATE_SCHEMA["required"])
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"LLM辩论结果缺少字段：{sorted(missing)}")

        valid_ids = {
            str(item["evidence_id"])
            for item in context.get("evidence", [])
        }
        accepted = [
            str(value)
            for value in payload.get("accepted_evidence_ids", [])
            if str(value) in valid_ids
        ]
        rejected = [
            str(value)
            for value in payload.get("rejected_evidence_ids", [])
            if str(value) in valid_ids
        ]
        payload["accepted_evidence_ids"] = accepted
        payload["rejected_evidence_ids"] = rejected
        payload["recommended_action"] = str(
            payload.get("recommended_action", "HOLD")
        ).upper()
        if payload["recommended_action"] not in {"BUY", "HOLD", "SELL"}:
            payload["recommended_action"] = "HOLD"
        payload["confidence"] = max(
            0.0,
            min(1.0, float(payload.get("confidence", 0.0))),
        )
        payload["confidence_adjustment"] = max(
            -0.05,
            min(0.05, float(payload.get("confidence_adjustment", 0.0))),
        )
        return payload

    def _request_once(self, prompt: str):
        return self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是审慎的量化交易辩论审计器。"
                        "你必须只输出合法JSON，并严格遵守用户给出的字段。"
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=self.max_output_tokens,
            stream=False,
            extra_body={"thinking": {"type": "disabled"}},
        )

    def debate(self, context: dict[str, Any]) -> LLMDebateResult:
        prompt = self.build_prompt(context)
        cache_key = self._cache_key(context, self.model)
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._request_once(prompt)
                payload = self._extract_json(
                    response.choices[0].message.content or ""
                )
                payload = self.validate_response(payload, context)
                return LLMDebateResult(
                    bull_thesis=str(payload["bull_thesis"]),
                    bear_thesis=str(payload["bear_thesis"]),
                    accepted_evidence_ids=list(
                        payload["accepted_evidence_ids"]
                    ),
                    rejected_evidence_ids=list(
                        payload["rejected_evidence_ids"]
                    ),
                    unresolved_conflicts=[
                        str(value)
                        for value in payload["unresolved_conflicts"]
                    ],
                    recommended_action=str(payload["recommended_action"]),
                    confidence=float(payload["confidence"]),
                    confidence_adjustment=float(
                        payload["confidence_adjustment"]
                    ),
                    summary=str(payload["summary"]),
                    model=self.model,
                    cache_key=cache_key,
                    raw_response_id=getattr(response, "id", None),
                )
            except Exception as exc:
                last_error = exc
                LOGGER.warning(
                    "LLM debate failed attempt %d/%d: %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(min(2.0**attempt, 12.0))
        if last_error is not None:
            raise last_error
        raise RuntimeError("LLM debate failed without a detailed error")


def load_jsonl_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError:
                continue
            key = item.get("cache_key")
            if key:
                result[str(key)] = item
    return result


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")