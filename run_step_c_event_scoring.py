from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config.openai_config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
from src.data.focus_news_fetcher_v7 import FocusNewsConfig, FocusNewsFetcher, normalize_eastmoney_news
from src.v7.llm_event_structurer_v7 import (
    EventScoringConfig,
    LLMEventStructurer,
    aggregate_event_state,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _load_existing(path: Path, symbol: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input news CSV not found: {path}")
    d = pd.read_csv(path)
    if {"新闻标题", "发布时间"}.intersection(d.columns):
        d = normalize_eastmoney_news(d, symbol)
    else:
        rename = {
            "event_time": "published_at",
            "新闻标题": "title",
            "新闻内容": "content",
            "文章来源": "source",
            "新闻链接": "url",
        }
        d = d.rename(columns=rename)
        if "symbol" in d.columns:
            d = d.loc[d["symbol"].astype(str).eq(symbol)].copy()
        for col, default in {"title": "", "content": "", "source": "", "url": ""}.items():
            if col not in d.columns:
                d[col] = default
        d["published_at"] = pd.to_datetime(d["published_at"], errors="coerce")
        d["symbol"] = symbol
        d["source_type"] = d.get("source_type", "existing_text_events")
        d = d.dropna(subset=["published_at", "title"])
    return d.sort_values("published_at").reset_index(drop=True)


def _clean_api_key(value: str) -> str:
    key = str(value or "").strip()
    placeholders = ("在这里", "填写", "your", "api key", "api_key")
    if not key or any(token in key.lower() for token in placeholders):
        raise RuntimeError(
            "Step C requires a real DeepSeek API key. Fill config/openai_config.py -> OPENAI_API_KEY first."
        )
    return key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V7 Final Step C: Prompt Engineering -> Structured Event Score")
    parser.add_argument("--symbol", default="688981.SH")
    parser.add_argument("--company-name", default="中芯国际")
    parser.add_argument("--source", choices=["eastmoney", "csv"], default="eastmoney")
    parser.add_argument("--input-csv", default="data/raw/text_events.csv")
    parser.add_argument("--max-age-days", type=int, default=180)
    parser.add_argument("--max-events", type=int, default=12)
    parser.add_argument("--model", default=str(OPENAI_MODEL))
    parser.add_argument("--base-url", default=str(OPENAI_BASE_URL))
    parser.add_argument("--output-dir", default="outputs/v7_event_llm")
    parser.add_argument("--cache-dir", default="data/v7_event_llm/cache")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "eastmoney":
        news_path = PROJECT_ROOT / "data/v7_event_llm/focus_news_recent.csv"
        news = FocusNewsFetcher(
            FocusNewsConfig(
                symbol=args.symbol,
                max_age_days=max(args.max_age_days, 0),
                output_csv=news_path,
            )
        ).fetch()
        source_label = "AKShare/Eastmoney stock_news_em"
    else:
        news_path = _resolve(args.input_csv)
        news = _load_existing(news_path, args.symbol)
        source_label = f"CSV:{news_path}"

    if news.empty:
        raise RuntimeError("Step C received no usable news/events")

    if args.max_age_days > 0:
        cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=args.max_age_days)
        news = news.loc[pd.to_datetime(news["published_at"]) >= cutoff].copy()
    news = news.sort_values("published_at").tail(max(int(args.max_events), 1)).reset_index(drop=True)
    if news.empty:
        raise RuntimeError("Step C has no events after date filtering")

    api_key = _clean_api_key(OPENAI_API_KEY)
    structurer = LLMEventStructurer(
        api_key=api_key,
        config=EventScoringConfig(
            model=str(args.model),
            base_url=str(args.base_url),
            max_retries=3,
            request_pause_seconds=0.25,
            max_output_tokens=1200,
            cache_dir=_resolve(args.cache_dir),
        ),
    )

    records: list[dict] = []
    cache_hits = 0
    print(f"[V7 Final Step C] source={source_label} candidate_events={len(news)} model={args.model}")
    for idx, row in news.iterrows():
        structured, meta = structurer.structure(
            row.to_dict(),
            symbol=args.symbol,
            company_name=args.company_name,
        )
        cache_hits += int(bool(meta.get("cache_hit")))
        rec = {
            "published_at": pd.to_datetime(row["published_at"]),
            "symbol": args.symbol,
            "title": str(row.get("title", "")),
            "source": str(row.get("source", "")),
            "source_type": str(row.get("source_type", "")),
            **structured.to_dict(),
            "model": str(meta.get("model", args.model)),
            "cache_key": str(meta.get("cache_key", "")),
            "cache_hit": bool(meta.get("cache_hit", False)),
            "response_id": meta.get("response_id"),
        }
        records.append(rec)
        print(
            f"[V7 Final Step C] {idx + 1}/{len(news)} "
            f"{rec['published_at']} type={rec['event_type']} "
            f"dir={rec['direction']:.2f} score={rec['structured_event_score']:.3f} "
            f"conf={rec['confidence']:.2f} cache={rec['cache_hit']}"
        )

    structured_df = pd.DataFrame(records).sort_values("published_at").reset_index(drop=True)
    state = aggregate_event_state(structured_df, as_of=pd.Timestamp.now(), max_active_age_days=30.0)
    state.update(
        {
            "version": "V7 Final Step C",
            "symbol": args.symbol,
            "company_name": args.company_name,
            "quality": "REAL_LLM_STRUCTURED_EVENTS",
            "source": source_label,
            "model": str(args.model),
            "prompt_engineering": True,
            "llm_direct_trade_action_allowed": False,
            "score_computed_by": "deterministic_python_formula",
            "processed_event_count": int(len(structured_df)),
            "cache_hits": int(cache_hits),
        }
    )

    structured_path = output_dir / "structured_events_llm.csv"
    state_path = output_dir / "event_state.json"
    instruction_path = output_dir / "event_risk_instruction.csv"
    structured_df.to_csv(structured_path, index=False, encoding="utf-8-sig")
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "as_of": pd.Timestamp.now().isoformat(),
                "symbol": args.symbol,
                "event_gate": state["event_gate"],
                "event_active": state["event_active"],
                "decayed_event_score": state["decayed_event_score"],
                "event_confidence": state["event_confidence"],
                "event_risk_multiplier": state["event_risk_multiplier"],
                "rule": "final_position = quant_position * event_risk_multiplier; positive events never lever upward",
            }
        ]
    ).to_csv(instruction_path, index=False, encoding="utf-8-sig")

    print("[V7 Final Step C] COMPLETE")
    print(
        f"event_gate={state['event_gate']} event_active={state['event_active']} "
        f"processed={len(structured_df)} cache_hits={cache_hits}"
    )
    print(
        f"decayed_event_score={state['decayed_event_score']:.4f} "
        f"event_confidence={state['event_confidence']:.3f} "
        f"event_risk_multiplier={state['event_risk_multiplier']:.4f}"
    )
    print(f"dominant_event_type={state['dominant_event_type']}")
    print(f"output={output_dir}")


if __name__ == "__main__":
    main()
