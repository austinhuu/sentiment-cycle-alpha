from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import PROJECT_ROOT
from src.robustness.scenario_runner import run_production_scenario


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    scenario = json.loads(Path(args.scenario).read_text(encoding="utf-8"))
    decisions_path = PROJECT_ROOT / scenario.get(
        "decisions_path", "outputs/v5/production/production_decisions.csv"
    )
    decisions = pd.read_csv(decisions_path, low_memory=False)
    symbols = scenario.get("symbols")
    if symbols:
        symbol_set = {str(value).upper() for value in symbols}
        decisions = decisions.loc[
            decisions["symbol"].astype(str).str.upper().isin(symbol_set)
        ].copy()
    result = run_production_scenario(
        decisions,
        scenario["portfolio_config"],
        scenario["backtest_config"],
        scenario["risk_budget_config"],
        PROJECT_ROOT,
    )
    output: dict[str, Any] = {
        "name": scenario.get("name", "scenario"),
        "parameter": scenario.get("parameter", "scenario"),
        "value": scenario.get("value"),
        "symbol_count": int(decisions["symbol"].nunique()),
        "metrics": result["metrics"],
        "exposure_matched_metrics": result["exposure_matched_metrics"],
        "equal_weight_metrics": result["equal_weight_metrics"],
        "allocation_diagnostics": result["allocation_diagnostics"],
        "risk_diagnostics": result["risk_diagnostics"],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()