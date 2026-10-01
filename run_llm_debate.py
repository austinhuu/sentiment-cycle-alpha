from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.debate.llm_adapter import (
    LLMDebateAdapter,
    append_jsonl,
    load_jsonl_cache,
)


LOGGER = logging.getLogger(__name__)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                records.append(json.loads(text))
    return records


def _select(
    logs: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    mode = str(config.get("selection_mode", "high_conflict"))
    maximum = max(int(config.get("max_requests", 50)), 1)
    latest_n_dates = max(int(config.get("latest_n_dates", 20)), 1)
    minimum_disagreement = float(config.get("minimum_disagreement", 0.50))

    frame = pd.DataFrame(
        [
            {
                "index": index,
                "datetime": pd.to_datetime(record.get("datetime")),
                "symbol": record.get("symbol"),
                "signal": (record.get("final_decision", {}) or {}).get(
                    "signal"
                ),
                "confidence": float(
                    (record.get("final_decision", {}) or {}).get(
                        "confidence", 0.0
                    )
                ),
                "disagreement": float(
                    (record.get("debate_result", {}) or {}).get(
                        "disagreement", 0.0
                    )
                ),
            }
            for index, record in enumerate(logs)
        ]
    )
    if frame.empty:
        return []

    latest_dates = sorted(frame["datetime"].dropna().unique())[-latest_n_dates:]
    recent = frame.loc[frame["datetime"].isin(latest_dates)].copy()

    if mode == "latest":
        selected = recent.sort_values(
            ["datetime", "confidence"], ascending=[False, False]
        )
    elif mode == "buy_candidates":
        selected = recent.loc[recent["signal"] == "BUY"].sort_values(
            ["datetime", "confidence"], ascending=[False, False]
        )
    else:
        selected = recent.loc[
            recent["disagreement"] >= minimum_disagreement
        ].sort_values(
            ["disagreement", "confidence"], ascending=[False, False]
        )
        if selected.empty:
            selected = recent.sort_values(
                ["disagreement", "confidence"], ascending=[False, False]
            )

    return [logs[int(index)] for index in selected.head(maximum)["index"]]


def main() -> None:
    config = load_config()
    llm_config = config.get("llm_debate", {})
    if not bool(llm_config.get("enabled", True)):
        raise RuntimeError("llm_debate.enabled=false，未启用LLM辩论。")

    log_path = PROJECT_ROOT / "outputs" / "debate_logs.jsonl"
    if not log_path.exists():
        raise FileNotFoundError(
            "缺少 outputs/debate_logs.jsonl，请先运行 run_pipeline.py。"
        )

    cache_path = Path(
        llm_config.get("cache_path", "data/cache/llm_debate_cache.jsonl")
    )
    if not cache_path.is_absolute():
        cache_path = PROJECT_ROOT / cache_path

    output_dir = Path(
        llm_config.get("output_dir", "outputs/llm_debate")
    )
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logs = _read_jsonl(log_path)
    selected = _select(logs, llm_config)
    adapter = LLMDebateAdapter(llm_config)
    cache = load_jsonl_cache(cache_path)

    results: list[dict[str, Any]] = []
    for number, record in enumerate(selected, start=1):
        context = adapter.build_context(record)
        cache_key = adapter._cache_key(context, adapter.model)
        if cache_key in cache:
            result = dict(cache[cache_key])
            result["from_cache"] = True
        else:
            LOGGER.info(
                "LLM debate %d/%d | %s | %s",
                number,
                len(selected),
                context.get("datetime"),
                context.get("symbol"),
            )
            llm_result = adapter.debate(context)
            result = llm_result.to_dict()
            result["datetime"] = context.get("datetime")
            result["symbol"] = context.get("symbol")
            result["quantitative_decision"] = context.get(
                "quantitative_decision"
            )
            result["from_cache"] = False
            append_jsonl(cache_path, result)
            time.sleep(
                max(float(llm_config.get("request_pause_seconds", 0.5)), 0.0)
            )
        results.append(result)

    output_jsonl = output_dir / "llm_debate_results.jsonl"
    output_csv = output_dir / "llm_debate_results.csv"
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")

    flattened = []
    for result in results:
        quant = result.get("quantitative_decision", {}) or {}
        flattened.append(
            {
                "datetime": result.get("datetime"),
                "symbol": result.get("symbol"),
                "quant_signal": quant.get("signal"),
                "quant_confidence": quant.get("confidence"),
                "llm_action": result.get("recommended_action"),
                "llm_confidence": result.get("confidence"),
                "confidence_adjustment": result.get(
                    "confidence_adjustment"
                ),
                "summary": result.get("summary"),
                "model": result.get("model"),
                "from_cache": result.get("from_cache"),
            }
        )
    pd.DataFrame(flattened).to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"LLM debate rows: {len(results):,}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()