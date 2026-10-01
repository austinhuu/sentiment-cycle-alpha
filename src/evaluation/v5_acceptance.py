from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def build_v5_acceptance(
    project_root: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    production_dir = project_root / "outputs" / "v5" / "production"
    metrics = _read_json(production_dir / "performance_metrics.json")
    bootstrap = _read_json(project_root / "outputs" / "v5" / "bootstrap" / "bootstrap_summary.json")
    monte_carlo = _read_json(project_root / "outputs" / "v5" / "monte_carlo" / "monte_carlo_summary.json")
    sensitivity = _read_json(project_root / "outputs" / "v5" / "sensitivity" / "parameter_sensitivity_summary.json")
    pool = _read_json(project_root / "outputs" / "v5" / "pool_stability" / "pool_stability_summary.json")
    health = _read_json(project_root / "outputs" / "v5" / "health" / "production_health_summary.json")

    checks = [
        ("production_sharpe", float(metrics.get("sharpe_ratio", np.nan)), ">=", float(config.get("minimum_sharpe", 0.60))),
        ("production_max_drawdown", float(metrics.get("max_drawdown", np.nan)), ">=", float(config.get("maximum_drawdown_floor", -0.22))),
        ("bootstrap_positive_probability", float(bootstrap.get("probability_positive_total_return", np.nan)), ">=", float(config.get("minimum_bootstrap_positive_probability", 0.70))),
        ("bootstrap_outperformance_probability", float(bootstrap.get("probability_outperform_benchmark", np.nan)), ">=", float(config.get("minimum_bootstrap_outperformance_probability", 0.55))),
        ("monte_carlo_drawdown_breach_probability", float(monte_carlo.get("probability_drawdown_worse_than_threshold", np.nan)), "<=", float(config.get("maximum_monte_carlo_drawdown_breach_probability", 0.20))),
        ("sensitivity_positive_share", float(sensitivity.get("positive_return_share", np.nan)), ">=", float(config.get("minimum_sensitivity_positive_share", 0.70))),
        ("pool_positive_share", float(pool.get("positive_return_share", np.nan)), ">=", float(config.get("minimum_pool_positive_share", 0.80))),
        ("pool_positive_sharpe_share", float(pool.get("positive_sharpe_share", np.nan)), ">=", float(config.get("minimum_pool_sharpe_positive_share", 0.70))),
        ("defensive_state_share", float(health.get("recent_defensive_share", np.nan)), "<=", float(config.get("maximum_defensive_state_share", 0.25))),
    ]
    rows: list[dict[str, Any]] = []
    for name, value, operator, threshold in checks:
        available = bool(np.isfinite(value))
        passed = False
        if available:
            passed = value >= threshold if operator == ">=" else value <= threshold
        rows.append(
            {
                "criterion": name,
                "value": value if available else np.nan,
                "operator": operator,
                "threshold": threshold,
                "available": int(available),
                "passed": int(passed),
            }
        )
    table = pd.DataFrame(rows)
    available = table.loc[table["available"].eq(1)]
    missing = table.loc[table["available"].eq(0), "criterion"].tolist()
    pass_ratio = float(available["passed"].mean()) if len(available) else 0.0
    minimum_pass_ratio = float(config.get("minimum_pass_ratio", 0.75))
    critical_names = {
        "production_max_drawdown",
        "bootstrap_positive_probability",
        "monte_carlo_drawdown_breach_probability",
    }
    critical_failed = table.loc[
        table["criterion"].isin(critical_names) & table["available"].eq(1) & table["passed"].eq(0)
    ]
    if missing:
        status = "incomplete"
    elif pass_ratio >= minimum_pass_ratio and critical_failed.empty:
        status = "pass"
    elif pass_ratio >= 0.60:
        status = "conditional"
    else:
        status = "fail"
    summary = {
        "status": status,
        "pass_ratio": pass_ratio,
        "minimum_pass_ratio": minimum_pass_ratio,
        "available_criteria": int(len(available)),
        "passed_criteria": int(available["passed"].sum()) if len(available) else 0,
        "failed_criteria": int((available["passed"] == 0).sum()) if len(available) else 0,
        "critical_failed": critical_failed["criterion"].tolist(),
        "missing_criteria": missing,
        "production_metrics": metrics,
        "recommendation": (
            "V5稳健性验收通过，可以进入V6组合优化与准生产阶段。"
            if status == "pass"
            else "V5结果不完整，请先运行所有稳健性脚本。"
            if status == "incomplete"
            else "V5部分通过，应根据失败指标继续降低风险或简化模型。"
            if status == "conditional"
            else "V5未通过，不应进入V6。"
        ),
    }
    return table, summary


def save_v5_acceptance(
    table: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(
        output_dir / "v5_acceptance_criteria.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "v5_acceptance_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )