from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.portfolio.cross_sectional_allocator import load_security_master
from src.robustness.scenario_runner import run_production_scenario


def _normalize_symbol(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def _liquidity_ranking(decisions: pd.DataFrame, column: str) -> pd.DataFrame:
    frame = decisions.copy()
    frame["symbol"] = _normalize_symbol(frame["symbol"])
    if column not in frame.columns:
        column = "amount" if "amount" in frame.columns else "volume"
    values = pd.to_numeric(frame.get(column), errors="coerce")
    frame["_liquidity"] = values
    ranking = (
        frame.groupby("symbol", as_index=False)["_liquidity"]
        .median()
        .sort_values("_liquidity", ascending=False)
        .reset_index(drop=True)
    )
    ranking["liquidity_rank"] = np.arange(1, len(ranking) + 1)
    return ranking


def _sector_balanced_pool(
    ranking: pd.DataFrame,
    master: pd.DataFrame,
    size: int,
) -> list[str]:
    merged = ranking.merge(master, on="symbol", how="left")
    merged["sector"] = merged["sector"].fillna("UNKNOWN")
    buckets = {
        sector: group.sort_values("liquidity_rank")["symbol"].tolist()
        for sector, group in merged.groupby("sector")
    }
    sectors = sorted(buckets, key=lambda value: (len(buckets[value]), value), reverse=True)
    selected: list[str] = []
    position = 0
    while len(selected) < size:
        added = False
        for sector in sectors:
            values = buckets[sector]
            if position < len(values):
                symbol = values[position]
                if symbol not in selected:
                    selected.append(symbol)
                    added = True
                    if len(selected) >= size:
                        break
        if not added:
            break
        position += 1
    return selected[:size]


def _hash_half(symbols: list[str], parity: int) -> list[str]:
    return [
        symbol
        for symbol in symbols
        if int(hashlib.sha256(symbol.encode("utf-8")).hexdigest(), 16) % 2 == parity
    ]


def build_pool_definitions(
    decisions: pd.DataFrame,
    config: dict[str, Any],
    project_root: Path,
) -> dict[str, list[str]]:
    symbols = sorted(_normalize_symbol(decisions["symbol"]).unique())
    ranking = _liquidity_ranking(decisions, str(config.get("liquidity_column", "amount")))
    master_path = project_root / "data" / "raw" / "security_master.csv"
    master = load_security_master(master_path)
    pools: dict[str, list[str]] = {"full_pool": symbols}
    minimum = max(int(config.get("minimum_symbols", 12)), 2)
    for size in config.get("liquidity_pool_sizes", [16, 24]):
        size = min(max(int(size), minimum), len(symbols))
        pools[f"liquidity_top_{size}"] = ranking.head(size)["symbol"].tolist()
    for size in config.get("sector_balanced_pool_sizes", [16, 24]):
        size = min(max(int(size), minimum), len(symbols))
        pools[f"sector_balanced_{size}"] = _sector_balanced_pool(ranking, master, size)
    if bool(config.get("include_hash_halves", True)):
        for parity, name in [(0, "hash_half_a"), (1, "hash_half_b")]:
            values = _hash_half(symbols, parity)
            if len(values) >= minimum:
                pools[name] = values
    return {name: values for name, values in pools.items() if len(values) >= minimum}


def run_pool_stability(
    decisions: pd.DataFrame,
    portfolio_config: dict[str, Any],
    backtest_config: dict[str, Any],
    risk_budget_config: dict[str, Any],
    pool_config: dict[str, Any],
    project_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, list[str]]]:
    pools = build_pool_definitions(decisions, pool_config, project_root)
    rows: list[dict[str, Any]] = []
    for index, (name, symbols) in enumerate(pools.items(), start=1):
        print(f"[V5] Pool stability {index}/{len(pools)}: {name} ({len(symbols)})", flush=True)
        local = decisions.loc[_normalize_symbol(decisions["symbol"]).isin(symbols)].copy()
        result = run_production_scenario(
            local,
            portfolio_config,
            backtest_config,
            risk_budget_config,
            project_root,
        )
        metrics = result["metrics"]
        exposure = result["exposure_matched_metrics"]
        rows.append(
            {
                "pool": name,
                "symbol_count": len(symbols),
                "date_count": int(pd.to_datetime(local["datetime"], errors="coerce").nunique()),
                "total_return": float(metrics.get("total_return", 0.0)),
                "annualized_return": float(metrics.get("annualized_return", 0.0)),
                "sharpe_ratio": float(metrics.get("sharpe_ratio", 0.0)),
                "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
                "calmar_ratio": float(metrics.get("calmar_ratio", 0.0)),
                "average_gross_exposure": float(metrics.get("average_gross_exposure", 0.0)),
                "total_transaction_cost": float(metrics.get("total_transaction_cost", 0.0)),
                "exposure_matched_total_return": float(exposure.get("total_return", 0.0)),
                "exposure_matched_sharpe_ratio": float(exposure.get("sharpe_ratio", 0.0)),
                "excess_total_return": float(metrics.get("total_return", 0.0))
                - float(exposure.get("total_return", 0.0)),
            }
        )
    results = pd.DataFrame(rows)
    summary = {
        "pools": int(len(results)),
        "positive_return_share": float(results["total_return"].gt(0).mean()) if len(results) else 0.0,
        "positive_sharpe_share": float(results["sharpe_ratio"].gt(0).mean()) if len(results) else 0.0,
        "outperformance_share": float(results["excess_total_return"].gt(0).mean()) if len(results) else 0.0,
        "minimum_total_return": float(results["total_return"].min()) if len(results) else np.nan,
        "minimum_sharpe_ratio": float(results["sharpe_ratio"].min()) if len(results) else np.nan,
        "worst_max_drawdown": float(results["max_drawdown"].min()) if len(results) else np.nan,
        "median_total_return": float(results["total_return"].median()) if len(results) else np.nan,
        "median_sharpe_ratio": float(results["sharpe_ratio"].median()) if len(results) else np.nan,
    }
    return results, summary, pools


def save_pool_stability_outputs(
    results: pd.DataFrame,
    summary: dict[str, Any],
    pools: dict[str, list[str]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(
        output_dir / "pool_stability_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_dir / "pool_stability_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "pool_definitions.json").write_text(
        json.dumps(pools, ensure_ascii=False, indent=2), encoding="utf-8"
    )