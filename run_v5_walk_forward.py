from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from src.backtest.portfolio_engine_v5 import (
    PortfolioBacktestEngineV5,
    save_portfolio_backtest_v5,
)
from src.config import PROJECT_ROOT, load_config
from src.models.adaptive_fusion import run_walk_forward_fusion, save_fusion_outputs
from src.models.model_health_controller import (
    ModelHealthController,
    save_model_health_outputs,
)
from src.portfolio.cross_sectional_allocator import CrossSectionalAllocator
from src.risk.dynamic_risk_budget import (
    DynamicRiskBudgetOverlay,
    save_dynamic_risk_outputs,
)


def main() -> None:
    config = load_config()
    decisions_path = PROJECT_ROOT / "outputs" / "agent_decisions.csv"
    if not decisions_path.exists():
        raise FileNotFoundError(
            "未找到outputs/agent_decisions.csv，请先运行 python run_pipeline.py"
        )
    raw = pd.read_csv(decisions_path, low_memory=False)
    adaptive_config = dict(config.get("adaptive_fusion", {}))
    adaptive_config["event_regime_config"] = config.get("event_regime", {})
    production_config = config.get("v5_production", {})
    output_dir = PROJECT_ROOT / production_config.get(
        "output_dir", "outputs/v5/production"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[V5] Running purged adaptive fusion...", flush=True)
    fused, coefficients, metadata = run_walk_forward_fusion(raw, adaptive_config)
    save_fusion_outputs(fused, coefficients, metadata, output_dir / "adaptive_fusion")

    print("[V5] Applying model-health controller...", flush=True)
    controller = ModelHealthController(production_config.get("model_health", {}))
    production_decisions, health_diagnostics = controller.apply(fused)
    save_model_health_outputs(production_decisions, health_diagnostics, output_dir)

    portfolio_config = dict(config.get("portfolio", {}))
    print("[V5] Building base cross-sectional portfolio...", flush=True)
    allocator = CrossSectionalAllocator(portfolio_config, PROJECT_ROOT)
    base_allocations, allocation_diagnostics = allocator.allocate(production_decisions)
    base_allocations.to_csv(
        output_dir / "base_target_weights.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "base_allocation_diagnostics.json").write_text(
        json.dumps(allocation_diagnostics.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[V5] Applying dynamic risk budget...", flush=True)
    risk_config = dict(production_config.get("dynamic_risk_budget", {}))
    risk_config["minimum_adaptive_score"] = portfolio_config.get(
        "minimum_adaptive_score", 0.10
    )
    risk_overlay = DynamicRiskBudgetOverlay(risk_config)
    allocations, risk_diagnostics, risk_daily = risk_overlay.apply(
        production_decisions,
        base_allocations,
    )
    save_dynamic_risk_outputs(
        allocations,
        risk_daily,
        risk_diagnostics,
        output_dir,
    )

    print("[V5] Running production backtest...", flush=True)
    backtest_config = {**config.get("backtest", {}), **portfolio_config}
    result = PortfolioBacktestEngineV5(backtest_config).run(
        production_decisions,
        allocations,
    )
    save_portfolio_backtest_v5(result, output_dir)

    portfolio = result.portfolio
    plt.figure(figsize=(11, 6))
    plt.plot(portfolio["datetime"], portfolio["equity"], label="V5 Production")
    plt.plot(
        portfolio["datetime"],
        portfolio["exposure_matched_equity"],
        label="Exposure-Matched",
    )
    plt.plot(
        portfolio["datetime"],
        portfolio["equal_weight_equity"],
        label="Equal-Weight",
    )
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.title("V5 Adaptive Robustness Production Portfolio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "v5_portfolio_equity_curve.png", dpi=160)
    plt.close()

    summary = {
        "version": "5.0",
        "portfolio_metrics": result.metrics,
        "exposure_matched_metrics": result.exposure_matched_metrics,
        "equal_weight_metrics": result.equal_weight_metrics,
        "model_health": health_diagnostics.to_dict(),
        "base_allocation": allocation_diagnostics.to_dict(),
        "dynamic_risk_budget": risk_diagnostics.to_dict(),
    }
    (output_dir / "v5_production_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[V5] Production walk-forward completed.", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()