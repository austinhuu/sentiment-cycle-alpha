from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.robustness.scenario_runner import run_production_scenario


def _set_nested(configs: dict[str, dict[str, Any]], path: str, value: Any) -> None:
    parts = path.split(".")
    if len(parts) < 2:
        raise ValueError(f"参数路径必须包含配置组：{path}")
    group = parts[0]
    if group not in configs:
        raise KeyError(f"未知配置组：{group}")
    target: dict[str, Any] = configs[group]
    for part in parts[1:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            child = {}
            target[part] = child
        target = child
    target[parts[-1]] = value


def _base_value(configs: dict[str, dict[str, Any]], path: str) -> Any:
    parts = path.split(".")
    value: Any = configs[parts[0]]
    for part in parts[1:]:
        value = value[part]
    return value


def _numeric_metrics(values: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in values.items()
        if isinstance(value, (int, float, np.integer, np.floating))
    }


def run_parameter_sensitivity(
    decisions: pd.DataFrame,
    portfolio_config: dict[str, Any],
    backtest_config: dict[str, Any],
    risk_budget_config: dict[str, Any],
    sensitivity_config: dict[str, Any],
    project_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    base_configs = {
        "portfolio": deepcopy(portfolio_config),
        "backtest": deepcopy(backtest_config),
        "v5_production": {"dynamic_risk_budget": deepcopy(risk_budget_config)},
    }
    scenarios: list[tuple[str, str, Any, dict[str, dict[str, Any]], float]] = []
    scenarios.append(("base", "base", "base", deepcopy(base_configs), 1.0))
    for parameter, values in sensitivity_config.get("parameters", {}).items():
        base = _base_value(base_configs, parameter)
        for value in values:
            if value == base:
                continue
            configs = deepcopy(base_configs)
            _set_nested(configs, parameter, value)
            name = f"{parameter}={value}"
            scenarios.append((name, parameter, value, configs, 1.0))
    for multiplier in sensitivity_config.get("cost_multipliers", [1.0]):
        if float(multiplier) == 1.0:
            continue
        configs = deepcopy(base_configs)
        name = f"transaction_cost_multiplier={multiplier}"
        scenarios.append((name, "transaction_cost_multiplier", multiplier, configs, float(multiplier)))

    rows: list[dict[str, Any]] = []
    for index, (name, parameter, value, configs, cost_multiplier) in enumerate(scenarios, start=1):
        print(f"[V5] Sensitivity {index}/{len(scenarios)}: {name}", flush=True)
        local_backtest = deepcopy(configs["backtest"])
        for key in [
            "buy_commission_rate",
            "sell_commission_rate",
            "stamp_duty_rate",
            "slippage_rate",
        ]:
            local_backtest[key] = float(local_backtest.get(key, 0.0)) * cost_multiplier
        result = run_production_scenario(
            decisions,
            configs["portfolio"],
            local_backtest,
            configs["v5_production"]["dynamic_risk_budget"],
            project_root,
        )
        metrics = _numeric_metrics(result["metrics"])
        exposure = result["exposure_matched_metrics"]
        rows.append(
            {
                "scenario": name,
                "parameter": parameter,
                "value": value,
                "cost_multiplier": cost_multiplier,
                **metrics,
                "exposure_matched_total_return": float(exposure.get("total_return", 0.0)),
                "exposure_matched_sharpe_ratio": float(exposure.get("sharpe_ratio", 0.0)),
                "excess_total_return": float(metrics.get("total_return", 0.0))
                - float(exposure.get("total_return", 0.0)),
                "allocation_maximum_turnover": float(
                    result["allocation_diagnostics"].get("maximum_turnover", 0.0)
                ),
                "risk_average_multiplier": float(
                    result["risk_diagnostics"].get("average_multiplier", 1.0)
                ),
            }
        )
    results = pd.DataFrame(rows)
    non_base = results.loc[~results["scenario"].eq("base")]
    summary = {
        "scenarios": int(len(results)),
        "positive_return_share": float(non_base["total_return"].gt(0).mean()) if len(non_base) else 1.0,
        "positive_sharpe_share": float(non_base["sharpe_ratio"].gt(0).mean()) if len(non_base) else 1.0,
        "outperformance_share": float(non_base["excess_total_return"].gt(0).mean()) if len(non_base) else 1.0,
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
    return results, summary


def save_sensitivity_outputs(
    results: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(
        output_dir / "parameter_sensitivity_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_dir / "parameter_sensitivity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )