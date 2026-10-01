from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _percentage(value: Any) -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "N/A"


def _number(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "N/A"


def build_v5_report(project_root: Path) -> str:
    root = project_root / "outputs" / "v5"
    production = _read_json(root / "production" / "v5_production_summary.json")
    bootstrap = _read_json(root / "bootstrap" / "bootstrap_summary.json")
    monte = _read_json(root / "monte_carlo" / "monte_carlo_summary.json")
    sensitivity = _read_json(root / "sensitivity" / "parameter_sensitivity_summary.json")
    pool = _read_json(root / "pool_stability" / "pool_stability_summary.json")
    health = _read_json(root / "health" / "production_health_summary.json")
    acceptance = _read_json(root / "acceptance" / "v5_acceptance_summary.json")
    metrics = production.get("portfolio_metrics", {})
    exposure_metrics = production.get("exposure_matched_metrics", {})
    portfolio_path = root / "production" / "portfolio_equity.csv"
    yearly_rows: list[str] = []
    if portfolio_path.exists():
        frame = pd.read_csv(portfolio_path)
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
        frame["year"] = frame["datetime"].dt.year
        for year, group in frame.groupby("year"):
            strategy = (1.0 + pd.to_numeric(group["strategy_return"], errors="coerce").fillna(0)).prod() - 1.0
            benchmark = (1.0 + pd.to_numeric(group["exposure_matched_return"], errors="coerce").fillna(0)).prod() - 1.0
            yearly_rows.append(f"| {int(year)} | {_percentage(strategy)} | {_percentage(benchmark)} | {_percentage(strategy - benchmark)} |")
    yearly_table = "\n".join(yearly_rows) if yearly_rows else "| N/A | N/A | N/A | N/A |"
    return f"""# V5 Adaptive Robustness System Report

## 1. Production performance

| Metric | V5 production | Exposure-matched |
|---|---:|---:|
| Total return | {_percentage(metrics.get('total_return'))} | {_percentage(exposure_metrics.get('total_return'))} |
| Annualized return | {_percentage(metrics.get('annualized_return'))} | {_percentage(exposure_metrics.get('annualized_return'))} |
| Sharpe | {_number(metrics.get('sharpe_ratio'))} | {_number(exposure_metrics.get('sharpe_ratio'))} |
| Max drawdown | {_percentage(metrics.get('max_drawdown'))} | {_percentage(exposure_metrics.get('max_drawdown'))} |
| Calmar | {_number(metrics.get('calmar_ratio'))} | {_number(exposure_metrics.get('calmar_ratio'))} |

## 2. Yearly sample-out performance

| Year | Strategy | Exposure-matched | Excess |
|---|---:|---:|---:|
{yearly_table}

## 3. Model health and risk budget

- Health status: **{health.get('system_status', 'not_run')}**
- Recent fallback share: {_percentage(health.get('recent_fallback_share'))}
- Recent defensive share: {_percentage(health.get('recent_defensive_share'))}
- Recent adaptive IC: {_number(health.get('recent_adaptive_ic'))}
- Average dynamic risk multiplier: {_number(metrics.get('average_dynamic_risk_budget_multiplier'))}
- Risk overlay active days: {metrics.get('risk_overlay_active_days', 'N/A')}

## 4. Bootstrap inference

- Probability of positive total return: {_percentage(bootstrap.get('probability_positive_total_return'))}
- Probability of outperforming exposure-matched benchmark: {_percentage(bootstrap.get('probability_outperform_benchmark'))}
- Probability of Sharpe outperforming benchmark: {_percentage(bootstrap.get('probability_sharpe_above_benchmark'))}
- Probability of breaching bootstrap drawdown threshold: {_percentage(bootstrap.get('probability_drawdown_worse_than_threshold'))}

## 5. Monte Carlo stress

- Probability of positive return under stress: {_percentage(monte.get('probability_positive_total_return'))}
- Probability of outperforming benchmark under stress: {_percentage(monte.get('probability_outperform_benchmark'))}
- Probability of drawdown threshold breach: {_percentage(monte.get('probability_drawdown_worse_than_threshold'))}
- Model-failure scenarios represented: {_percentage(monte.get('probability_model_failure_scenario'))}

## 6. Parameter and pool stability

- Sensitivity positive-return share: {_percentage(sensitivity.get('positive_return_share'))}
- Sensitivity outperformance share: {_percentage(sensitivity.get('outperformance_share'))}
- Pool positive-return share: {_percentage(pool.get('positive_return_share'))}
- Pool outperformance share: {_percentage(pool.get('outperformance_share'))}
- Worst pool return: {_percentage(pool.get('minimum_total_return'))}

## 7. Formal acceptance

- Status: **{acceptance.get('status', 'not_run')}**
- Passed criteria ratio: {_percentage(acceptance.get('pass_ratio'))}
- Critical failed criteria: {', '.join(acceptance.get('critical_failed', [])) or 'None'}

{acceptance.get('recommendation', 'Run every V5 stage before drawing a final conclusion.')}
"""


def save_v5_report(project_root: Path) -> Path:
    output_path = project_root / "outputs" / "v5" / "V5_ROBUSTNESS_REPORT.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_v5_report(project_root), encoding="utf-8")
    return output_path