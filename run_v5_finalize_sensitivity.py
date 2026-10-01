from __future__ import annotations

import json

import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.robustness.sensitivity import save_sensitivity_outputs


def main() -> None:
    config = load_config()
    sensitivity = config.get("v5_sensitivity", {})
    output_dir = PROJECT_ROOT / sensitivity.get("output_dir", "outputs/v5/sensitivity")
    cache_dir = output_dir / "scenario_cache"
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for entry in manifest["scenarios"]:
        result = json.loads(open(entry["result_path"], encoding="utf-8").read())
        scenario = json.loads(open(entry["scenario_path"], encoding="utf-8").read())
        metrics = result["metrics"]
        exposure = result["exposure_matched_metrics"]
        rows.append(
            {
                "scenario": result["name"],
                "parameter": result["parameter"],
                "value": result["value"],
                "cost_multiplier": scenario.get("cost_multiplier", 1.0),
                **{key: value for key, value in metrics.items() if isinstance(value, (int, float))},
                "exposure_matched_total_return": float(exposure.get("total_return", 0.0)),
                "exposure_matched_sharpe_ratio": float(exposure.get("sharpe_ratio", 0.0)),
                "excess_total_return": float(metrics.get("total_return", 0.0)) - float(exposure.get("total_return", 0.0)),
                "allocation_maximum_turnover": float(result["allocation_diagnostics"].get("maximum_turnover", 0.0)),
                "risk_average_multiplier": float(result["risk_diagnostics"].get("average_multiplier", 1.0)),
            }
        )
    results = pd.DataFrame(rows)
    non_base = results.loc[~results["scenario"].eq("base")]
    summary = {
        "scenarios": int(len(results)),
        "positive_return_share": float(non_base["total_return"].gt(0).mean()),
        "positive_sharpe_share": float(non_base["sharpe_ratio"].gt(0).mean()),
        "outperformance_share": float(non_base["excess_total_return"].gt(0).mean()),
        "worst_total_return": float(results["total_return"].min()),
        "worst_sharpe_ratio": float(results["sharpe_ratio"].min()),
        "worst_max_drawdown": float(results["max_drawdown"].min()),
        "median_total_return": float(results["total_return"].median()),
        "parameter_group_summary": results.groupby("parameter").agg(
            scenarios=("scenario", "count"),
            median_total_return=("total_return", "median"),
            minimum_total_return=("total_return", "min"),
            median_sharpe=("sharpe_ratio", "median"),
            minimum_sharpe=("sharpe_ratio", "min"),
        ).reset_index().to_dict(orient="records"),
    }
    save_sensitivity_outputs(results, summary, output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

