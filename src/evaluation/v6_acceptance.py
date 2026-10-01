from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def build_v6_acceptance(
    project_root: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    v5 = _read_json(project_root / "outputs/v5/acceptance/v5_acceptance_summary.json")
    v5_metrics = v5.get("production_metrics", {})
    metrics = _read_json(project_root / "outputs/v6/backtest/performance_metrics.json")
    optimizer = _read_json(project_root / "outputs/v6/optimization/optimizer_summary.json")
    orders = _read_json(project_root / "outputs/v6/execution/latest_order_plan_summary.json")
    snapshot = _read_json(project_root / "outputs/v6/production_snapshot/latest/snapshot_summary.json")

    v5_sharpe = float(v5_metrics.get("sharpe_ratio", np.nan))
    v5_drawdown = float(v5_metrics.get("max_drawdown", np.nan))
    checks = [
        ("v5_acceptance_passed", 1.0 if v5.get("status") == "pass" else 0.0, ">=", 1.0),
        ("v6_sharpe", float(metrics.get("sharpe_ratio", np.nan)), ">=", float(config.get("minimum_sharpe", 0.75))),
        ("v6_max_drawdown", float(metrics.get("max_drawdown", np.nan)), ">=", float(config.get("maximum_drawdown_floor", -0.18))),
        ("sharpe_vs_v5_delta", float(metrics.get("sharpe_ratio", np.nan)) - v5_sharpe, ">=", float(config.get("minimum_sharpe_delta_vs_v5", -0.05))),
        ("drawdown_vs_v5_delta", float(metrics.get("max_drawdown", np.nan)) - v5_drawdown, ">=", float(config.get("minimum_drawdown_delta_vs_v5", -0.02))),
        ("optimizer_success_share", float(optimizer.get("success_share", np.nan)), ">=", float(config.get("minimum_optimizer_success_share", 0.95))),
        ("average_effective_positions", float(optimizer.get("average_effective_positions", np.nan)), ">=", float(config.get("minimum_effective_positions", 5.0))),
        ("maximum_turnover", float(optimizer.get("maximum_turnover", np.nan)), "<=", float(config.get("maximum_turnover", 0.50))),
        ("maximum_symbol_weight", float(optimizer.get("maximum_symbol_weight", np.nan)), "<=", float(config.get("maximum_symbol_weight", 0.10))),
        ("maximum_sector_weight", float(optimizer.get("maximum_sector_weight", np.nan)), "<=", float(config.get("maximum_sector_weight", 0.28))),
        ("maximum_ex_ante_volatility", float(optimizer.get("maximum_ex_ante_volatility", np.nan)), "<=", float(config.get("maximum_ex_ante_volatility", 0.14))),
        ("order_plan_created", 1.0 if orders else np.nan, ">=", 1.0),
        ("snapshot_ready", 1.0 if snapshot.get("ready") else 0.0 if snapshot else np.nan, ">=", 1.0),
    ]
    rows: list[dict[str, Any]] = []
    for criterion, value, operator, threshold in checks:
        available = bool(np.isfinite(value))
        passed = False
        if available:
            passed = value >= threshold if operator == ">=" else value <= threshold
        rows.append(
            {
                "criterion": criterion,
                "value": value if available else np.nan,
                "operator": operator,
                "threshold": threshold,
                "available": int(available),
                "passed": int(passed),
            }
        )
    table = pd.DataFrame(rows)
    missing = table.loc[table["available"].eq(0), "criterion"].tolist()
    available = table.loc[table["available"].eq(1)]
    pass_ratio = float(available["passed"].mean()) if len(available) else 0.0
    critical = {
        "v5_acceptance_passed",
        "v6_max_drawdown",
        "optimizer_success_share",
        "maximum_turnover",
        "maximum_symbol_weight",
        "maximum_sector_weight",
    }
    critical_failed = table.loc[
        table["criterion"].isin(critical)
        & table["available"].eq(1)
        & table["passed"].eq(0),
        "criterion",
    ].tolist()
    minimum_pass_ratio = float(config.get("minimum_pass_ratio", 0.80))
    if missing:
        status = "incomplete"
    elif pass_ratio >= minimum_pass_ratio and not critical_failed:
        status = "pass"
    elif pass_ratio >= 0.65 and not critical_failed:
        status = "conditional"
    else:
        status = "fail"
    recommendation = {
        "pass": "V6组合优化与准生产验收通过，研究主线正式完成；可选择进入V7模拟盘/实盘部署。",
        "conditional": "V6部分通过，可保留研究成果，但进入模拟盘前应修正未通过指标。",
        "fail": "V6未通过，应回退到V5生产组合或降低优化器复杂度。",
        "incomplete": "V6结果不完整，请先运行优化、回测、订单计划和生产快照。",
    }[status]
    summary = {
        "status": status,
        "pass_ratio": pass_ratio,
        "minimum_pass_ratio": minimum_pass_ratio,
        "available_criteria": int(len(available)),
        "passed_criteria": int(available["passed"].sum()) if len(available) else 0,
        "failed_criteria": int((available["passed"] == 0).sum()) if len(available) else 0,
        "critical_failed": critical_failed,
        "missing_criteria": missing,
        "v5_status": v5.get("status", "missing"),
        "v5_metrics": v5_metrics,
        "v6_metrics": metrics,
        "optimizer_summary": optimizer,
        "recommendation": recommendation,
    }
    return table, summary


def save_v6_acceptance(
    table: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "v6_acceptance_criteria.csv", index=False, encoding="utf-8-sig")
    (output_dir / "v6_acceptance_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )