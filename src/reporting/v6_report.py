from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def _number(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def build_v6_report(project_root: Path) -> str:
    acceptance = _read_json(project_root / "outputs/v6/acceptance/v6_acceptance_summary.json")
    v6 = acceptance.get("v6_metrics", {})
    v5 = acceptance.get("v5_metrics", {})
    optimizer = acceptance.get("optimizer_summary", {})
    orders = _read_json(project_root / "outputs/v6/execution/latest_order_plan_summary.json")
    snapshot = _read_json(project_root / "outputs/v6/production_snapshot/latest/snapshot_summary.json")
    lines = [
        "# V6组合优化与准生产最终报告",
        "",
        f"**验收状态：{acceptance.get('status', 'incomplete')}**",
        "",
        acceptance.get("recommendation", ""),
        "",
        "## 1. V5与V6表现对比",
        "",
        "| 指标 | V5生产组合 | V6优化组合 |",
        "|---|---:|---:|",
        f"| 总收益率 | {_pct(v5.get('total_return'))} | {_pct(v6.get('total_return'))} |",
        f"| 年化收益率 | {_pct(v5.get('annualized_return'))} | {_pct(v6.get('annualized_return'))} |",
        f"| 夏普比率 | {_number(v5.get('sharpe_ratio'))} | {_number(v6.get('sharpe_ratio'))} |",
        f"| 最大回撤 | {_pct(v5.get('max_drawdown'))} | {_pct(v6.get('max_drawdown'))} |",
        f"| Calmar | {_number(v5.get('calmar_ratio'))} | {_number(v6.get('calmar_ratio'))} |",
        "",
        "## 2. 优化器诊断",
        "",
        f"- 优化成功比例：{_pct(optimizer.get('success_share'))}",
        f"- 平均事前波动率：{_pct(optimizer.get('average_ex_ante_volatility'))}",
        f"- 最大事前波动率：{_pct(optimizer.get('maximum_ex_ante_volatility'))}",
        f"- 平均有效持仓数：{_number(optimizer.get('average_effective_positions'), 2)}",
        f"- 最大换手：{_number(optimizer.get('maximum_turnover'))}",
        f"- 最大单股权重：{_pct(optimizer.get('maximum_symbol_weight'))}",
        f"- 最大行业权重：{_pct(optimizer.get('maximum_sector_weight'))}",
        "",
        "## 3. 最新纸面订单计划",
        "",
        f"- 信号日期：{orders.get('signal_date', 'N/A')}",
        f"- 买入订单数：{orders.get('buy_orders', 'N/A')}",
        f"- 卖出订单数：{orders.get('sell_orders', 'N/A')}",
        f"- 被阻止订单数：{orders.get('blocked_orders', 'N/A')}",
        f"- 预计交易成本：{orders.get('estimated_transaction_cost', 'N/A')}",
        "",
        "## 4. 准生产快照",
        "",
        f"- 模式：{snapshot.get('mode', 'paper_research')}",
        f"- 快照就绪：{snapshot.get('ready', False)}",
        f"- Run ID：{snapshot.get('run_id', 'N/A')}",
        "",
        "## 5. 项目最终定位",
        "",
        "V6完成后，研究系统已经覆盖数据Point-in-Time、智能体信号、无泄露Walk-forward、模型健康降级、动态风险预算、Bootstrap/Monte Carlo、参数与股票池稳健性、约束组合优化、订单计划、产物哈希和准生产验收。",
        "",
        "V7属于可选部署阶段，主要工作是连接QMT/VeighNa、实时行情、订单状态、断线重连和模拟盘监控，不再属于研究主线的必需部分。",
    ]
    return "\n".join(lines)


def save_v6_report(report: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")