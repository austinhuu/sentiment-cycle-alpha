from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT, load_config


def set_nested(configs: dict[str, dict[str, Any]], path: str, value: Any) -> None:
    parts = path.split(".")
    target = configs[parts[0]]
    for part in parts[1:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def get_nested(configs: dict[str, dict[str, Any]], path: str) -> Any:
    value: Any = configs
    for part in path.split("."):
        value = value[part]
    return value


def main() -> None:
    config = load_config()
    sensitivity = config.get("v5_sensitivity", {})
    output_dir = PROJECT_ROOT / sensitivity.get("output_dir", "outputs/v5/sensitivity")
    cache_dir = output_dir / "scenario_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for path in cache_dir.glob("*.json"):
        path.unlink()
    base = {
        "portfolio": deepcopy(config.get("portfolio", {})),
        "backtest": deepcopy(config.get("backtest", {})),
        "v5_production": {
            "dynamic_risk_budget": deepcopy(
                config.get("v5_production", {}).get("dynamic_risk_budget", {})
            )
        },
    }
    scenarios: list[dict[str, Any]] = []

    def append(name: str, parameter: str, value: Any, configs: dict[str, Any], cost: float = 1.0) -> None:
        backtest = deepcopy(configs["backtest"])
        for key in [
            "buy_commission_rate",
            "sell_commission_rate",
            "stamp_duty_rate",
            "slippage_rate",
        ]:
            backtest[key] = float(backtest.get(key, 0.0)) * cost
        scenarios.append(
            {
                "name": name,
                "parameter": parameter,
                "value": value,
                "portfolio_config": configs["portfolio"],
                "backtest_config": backtest,
                "risk_budget_config": configs["v5_production"]["dynamic_risk_budget"],
                "cost_multiplier": cost,
            }
        )

    append("base", "base", "base", deepcopy(base))
    for parameter, values in sensitivity.get("parameters", {}).items():
        base_value = get_nested(base, parameter)
        for value in values:
            if value == base_value:
                continue
            local = deepcopy(base)
            set_nested(local, parameter, value)
            append(f"{parameter}={value}", parameter, value, local)
    for multiplier in sensitivity.get("cost_multipliers", [1.0]):
        if float(multiplier) == 1.0:
            continue
        append(
            f"transaction_cost_multiplier={multiplier}",
            "transaction_cost_multiplier",
            multiplier,
            deepcopy(base),
            float(multiplier),
        )

    entries = []
    for index, scenario in enumerate(scenarios, start=1):
        scenario_path = cache_dir / f"scenario_{index:03d}.json"
        result_path = cache_dir / f"result_{index:03d}.json"
        scenario_path.write_text(
            json.dumps(scenario, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        entries.append(
            {
                "index": index,
                "name": scenario["name"],
                # Store project-relative POSIX paths. This avoids mojibake and
                # invalid JSON escape sequences when Windows PowerShell 5 reads
                # a manifest located under a Chinese directory name.
                "scenario_path": scenario_path.relative_to(PROJECT_ROOT).as_posix(),
                "result_path": result_path.relative_to(PROJECT_ROOT).as_posix(),
            }
        )
    manifest = {"type": "sensitivity", "count": len(entries), "scenarios": entries}
    manifest_path = cache_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(str(manifest_path), flush=True)


if __name__ == "__main__":
    main()