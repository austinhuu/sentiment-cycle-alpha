from __future__ import annotations

import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.v7 import V76ResearchPipeline


def _read_selected(path, wanted):
    header = pd.read_csv(path, nrows=0).columns
    use = [c for c in wanted if c in header]
    return pd.read_csv(path, usecols=use, low_memory=False)


def main() -> None:
    config = load_config()
    features_path = PROJECT_ROOT / "outputs" / "processed_features.csv"
    decisions_path = PROJECT_ROOT / "outputs" / "agent_decisions.csv"
    master_path = PROJECT_ROOT / "data" / "raw" / "security_master.csv"
    missing = [p for p in [features_path, decisions_path, master_path] if not p.is_file()]
    if missing:
        print(f"[V7.6] upstream outputs missing: {missing}", flush=True)
        print("[V7.6] building V6 pipeline outputs first...", flush=True)
        from src.data.loader import load_market_data
        from src.data.validator import validate_market_data
        from src.pipeline import MultiAgentPipeline
        raw, _ = load_market_data(config)
        cleaned, report = validate_market_data(raw)
        (PROJECT_ROOT / "outputs").mkdir(parents=True, exist_ok=True)
        report.to_csv(PROJECT_ROOT / "outputs" / "data_quality_report.csv", index=False, encoding="utf-8-sig")
        MultiAgentPipeline(config).run(cleaned, PROJECT_ROOT / "outputs")
        if not master_path.is_file():
            raise FileNotFoundError("security_master.csv is required; run python run_build_security_master.py")

    print("[V7.6] loading selected V6 upstream columns...", flush=True)
    feature_cols = [
        "datetime", "symbol", "return_1", "return_5", "volume_ratio",
        "momentum_z", "volume_ratio_z", "ma_gap_short_long_z", "price_position", "breakout_up",
    ]
    decision_cols = [
        "datetime", "symbol", "future_return_1", "future_return_5", "future_return_20",
        "final_score", "confidence", "risk_score", "effective_risk_multiplier",
        "news_score", "news_confidence", "news_active", "news_material_event_count",
        "news_event_severity", "dominant_event_type",
    ]
    features = _read_selected(features_path, feature_cols)
    decisions = _read_selected(decisions_path, decision_cols)
    master = pd.read_csv(master_path, encoding="utf-8-sig")
    output_dir = PROJECT_ROOT / "outputs" / "v7_6"
    _, summary = V76ResearchPipeline(config).run(features, decisions, master, output_dir)
    full = summary["metrics"]["full:v7_full"]
    base = summary["metrics"]["full:v6_score_baseline"]
    hold = summary["metrics"]["holdout_2025:v7_full"]
    print("[V7.6] COMPLETE", flush=True)
    print(f"rows={summary['rows']:,} dates={summary['dates']} symbols={summary['symbols']}")
    print(f"V7 annual={full['annual_return']:.2%} sharpe={full['sharpe']:.3f} maxDD={full['max_drawdown']:.2%} IC5={full['ic5_mean_ic']:.4f}")
    print(f"baseline annual={base['annual_return']:.2%} sharpe={base['sharpe']:.3f}")
    print(f"2025 holdout V7 annual={hold['annual_return']:.2%} sharpe={hold['sharpe']:.3f} maxDD={hold['max_drawdown']:.2%}")
    print(f"output={output_dir}")


if __name__ == "__main__":
    main()
