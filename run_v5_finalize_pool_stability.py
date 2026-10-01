from __future__ import annotations

import json

import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.robustness.pool_stability import save_pool_stability_outputs


def main() -> None:
    config = load_config()
    pool_config = config.get("v5_pool_stability", {})
    output_dir = PROJECT_ROOT / pool_config.get(
        "output_dir", "outputs/v5/pool_stability"
    )
    cache_dir = output_dir / "scenario_cache"
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for entry in manifest["scenarios"]:
        result = json.loads(open(entry["result_path"], encoding="utf-8").read())
        metrics = result["metrics"]
        exposure = result["exposure_matched_metrics"]
        rows.append(
            {
                "pool": result["name"],
                "symbol_count": int(result["symbol_count"]),
                "total_return": float(metrics.get("total_return", 0.0)),
                "annualized_return": float(metrics.get("annualized_return", 0.0)),
                "sharpe_ratio": float(metrics.get("sharpe_ratio", 0.0)),
                "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
                "calmar_ratio": float(metrics.get("calmar_ratio", 0.0)),
                "average_gross_exposure": float(metrics.get("average_gross_exposure", 0.0)),
                "total_transaction_cost": float(metrics.get("total_transaction_cost", 0.0)),
                "exposure_matched_total_return": float(exposure.get("total_return", 0.0)),
                "exposure_matched_sharpe_ratio": float(exposure.get("sharpe_ratio", 0.0)),
                "excess_total_return": float(metrics.get("total_return", 0.0)) - float(exposure.get("total_return", 0.0)),
            }
        )
    results = pd.DataFrame(rows)
    summary = {
        "pools": int(len(results)),
        "positive_return_share": float(results["total_return"].gt(0).mean()),
        "positive_sharpe_share": float(results["sharpe_ratio"].gt(0).mean()),
        "outperformance_share": float(results["excess_total_return"].gt(0).mean()),
        "minimum_total_return": float(results["total_return"].min()),
        "minimum_sharpe_ratio": float(results["sharpe_ratio"].min()),
        "worst_max_drawdown": float(results["max_drawdown"].min()),
        "median_total_return": float(results["total_return"].median()),
        "median_sharpe_ratio": float(results["sharpe_ratio"].median()),
    }
    save_pool_stability_outputs(
        results, summary, manifest["pools"], output_dir
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()