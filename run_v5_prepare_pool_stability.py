from __future__ import annotations

import json
from copy import deepcopy

import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.robustness.pool_stability import build_pool_definitions


def main() -> None:
    config = load_config()
    pool_config = config.get("v5_pool_stability", {})
    output_dir = PROJECT_ROOT / pool_config.get(
        "output_dir", "outputs/v5/pool_stability"
    )
    cache_dir = output_dir / "scenario_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for path in cache_dir.glob("*.json"):
        path.unlink()
    decisions_path = PROJECT_ROOT / "outputs" / "v5" / "production" / "production_decisions.csv"
    decisions = pd.read_csv(decisions_path, low_memory=False)
    pools = build_pool_definitions(decisions, pool_config, PROJECT_ROOT)
    entries = []
    for index, (name, symbols) in enumerate(pools.items(), start=1):
        scenario = {
            "name": name,
            "parameter": "stock_pool",
            "value": name,
            "symbols": symbols,
            "portfolio_config": deepcopy(config.get("portfolio", {})),
            "backtest_config": deepcopy(config.get("backtest", {})),
            "risk_budget_config": deepcopy(
                config.get("v5_production", {}).get("dynamic_risk_budget", {})
            ),
        }
        scenario_path = cache_dir / f"scenario_{index:03d}.json"
        result_path = cache_dir / f"result_{index:03d}.json"
        scenario_path.write_text(
            json.dumps(scenario, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        entries.append(
            {
                "index": index,
                "name": name,
                "symbol_count": len(symbols),
                # Store project-relative POSIX paths. This avoids mojibake and
                # invalid JSON escape sequences when Windows PowerShell 5 reads
                # a manifest located under a Chinese directory name.
                "scenario_path": scenario_path.relative_to(PROJECT_ROOT).as_posix(),
                "result_path": result_path.relative_to(PROJECT_ROOT).as_posix(),
            }
        )
    manifest = {
        "type": "pool_stability",
        "count": len(entries),
        "pools": pools,
        "scenarios": entries,
    }
    manifest_path = cache_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(str(manifest_path), flush=True)


if __name__ == "__main__":
    main()