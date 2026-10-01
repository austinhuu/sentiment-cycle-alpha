from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from src.backtest.portfolio_engine_v6 import (
    PortfolioBacktestEngineV6,
    save_portfolio_backtest_v6,
)
from src.config import PROJECT_ROOT, load_config
from src.optimization.robust_optimizer import (
    RobustPortfolioOptimizer,
    save_optimizer_outputs,
)


def main() -> None:
    config = load_config()
    optimizer_config = dict(config.get("v6_optimizer", {}))
    decisions_path = PROJECT_ROOT / optimizer_config.get(
        "decisions_csv", "outputs/v5/production/production_decisions.csv"
    )
    prior_path = PROJECT_ROOT / optimizer_config.get(
        "prior_allocations_csv", "outputs/v5/production/risk_adjusted_target_weights.csv"
    )
    if not decisions_path.exists():
        raise FileNotFoundError(
            f"Missing V5 production decisions: {decisions_path}. Run run_v5_walk_forward.py first."
        )
    if not prior_path.exists():
        raise FileNotFoundError(
            f"Missing V5 risk-adjusted weights: {prior_path}. Run run_v5_walk_forward.py first."
        )
    decisions = pd.read_csv(decisions_path, low_memory=False)
    prior = pd.read_csv(prior_path, low_memory=False)
    output_dir = PROJECT_ROOT / optimizer_config.get(
        "output_dir", "outputs/v6/optimization"
    )
    print("[V6] Estimating robust covariance and optimizing portfolio...", flush=True)
    optimizer = RobustPortfolioOptimizer(optimizer_config, PROJECT_ROOT)
    allocations, daily, risk, diagnostics = optimizer.optimize(decisions, prior)
    save_optimizer_outputs(allocations, daily, risk, diagnostics, output_dir)
    print(json.dumps(diagnostics.to_dict(), ensure_ascii=False, indent=2), flush=True)

    print("[V6] Running optimizer portfolio backtest...", flush=True)
    backtest_dir = PROJECT_ROOT / config.get("v6_backtest", {}).get(
        "output_dir", "outputs/v6/backtest"
    )
    backtest_config = {
        **config.get("backtest", {}),
        **config.get("portfolio", {}),
        **config.get("v6_backtest", {}),
    }
    result = PortfolioBacktestEngineV6(backtest_config).run(
        decisions, allocations, daily
    )
    save_portfolio_backtest_v6(result, backtest_dir)
    plt.figure(figsize=(11, 6))
    plt.plot(result.portfolio["datetime"], result.portfolio["equity"], label="V6 Optimized")
    plt.plot(
        result.portfolio["datetime"],
        result.portfolio["exposure_matched_equity"],
        label="Exposure-Matched",
    )
    plt.plot(
        result.portfolio["datetime"],
        result.portfolio["equal_weight_equity"],
        label="Equal-Weight",
    )
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.title("V6 Robust Optimizer Backtest")
    plt.legend()
    plt.tight_layout()
    plt.savefig(backtest_dir / "v6_equity_curve.png", dpi=180)
    plt.close()
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()