from __future__ import annotations

import json

import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.monitoring.production_health import (
    build_production_health_report,
    save_production_health_outputs,
)


def main() -> None:
    config = load_config()
    production_dir = PROJECT_ROOT / "outputs" / "v5" / "production"
    decisions_path = production_dir / "production_decisions.csv"
    portfolio_path = production_dir / "portfolio_equity.csv"
    if not decisions_path.exists() or not portfolio_path.exists():
        raise FileNotFoundError("请先运行 python run_v5_walk_forward.py")
    decisions = pd.read_csv(decisions_path, low_memory=False)
    portfolio = pd.read_csv(portfolio_path)
    health_config = config.get("v5_health_monitor", {})
    daily, summary, alerts = build_production_health_report(
        decisions,
        portfolio,
        health_config,
    )
    output_dir = PROJECT_ROOT / health_config.get("output_dir", "outputs/v5/health")
    save_production_health_outputs(daily, summary, alerts, output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if alerts:
        print(json.dumps(alerts, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()