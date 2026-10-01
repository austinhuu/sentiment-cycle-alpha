from __future__ import annotations

from src.config import PROJECT_ROOT, load_config
from src.data.loader import load_market_data
from src.data.validator import validate_market_data
from src.pipeline import MultiAgentPipeline


def main() -> None:
    config = load_config()
    raw, input_path = load_market_data(config)
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned, report = validate_market_data(raw)
    report.to_csv(
        output_dir / "data_quality_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    features, decisions = MultiAgentPipeline(config).run(cleaned, output_dir)

    fundamental_coverage = (
        decisions["fundamental_active"].fillna(0).gt(0).mean()
        if "fundamental_active" in decisions.columns
        else decisions["fundamental_confidence"].fillna(0).gt(0).mean()
        if "fundamental_confidence" in decisions.columns
        else 0.0
    )
    news_coverage = (
        decisions["news_active"].fillna(0).gt(0).mean()
        if "news_active" in decisions.columns
        else decisions["news_confidence"].fillna(0).gt(0).mean()
        if "news_confidence" in decisions.columns
        else 0.0
    )

    summary = (
        f"Input file: {input_path}\n"
        f"Clean rows: {len(cleaned):,}\n"
        f"Feature rows: {len(features):,}\n"
        f"Decision rows: {len(decisions):,}\n"
        f"Trading symbols: {decisions['symbol'].nunique()}\n"
        f"Sentiment coverage median: {decisions['valid_symbol_count'].median():.0f}\n"
        f"Fundamental coverage: {fundamental_coverage:.2%}\n"
        f"News coverage: {news_coverage:.2%}\n"
        f"BUY count: {(decisions['signal'] == 'BUY').sum():,}\n"
        f"SELL count: {(decisions['signal'] == 'SELL').sum():,}\n"
        f"HOLD count: {(decisions['signal'] == 'HOLD').sum():,}\n"
        f"Risk entry veto count: {(decisions['risk_action'] == 'veto_new_entry').sum():,}\n"
        f"Risk force exit count: {(decisions['risk_action'] == 'force_exit').sum():,}\n"
        "LLM debate: separate audit stage; run python run_llm_debate.py\n"
    )
    (output_dir / "run_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()