from __future__ import annotations

import json

import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.execution.order_planner import OrderPlanner, save_order_plan


def main() -> None:
    config = load_config()
    execution_config = dict(config.get("v6_execution", {}))
    decisions_path = PROJECT_ROOT / execution_config.get(
        "decisions_csv", "outputs/v5/production/production_decisions.csv"
    )
    allocations_path = PROJECT_ROOT / execution_config.get(
        "allocations_csv", "outputs/v6/optimization/optimized_target_weights.csv"
    )
    if not decisions_path.exists() or not allocations_path.exists():
        raise FileNotFoundError(
            "Missing V6 inputs. Run run_v6_optimize.py before building the order plan."
        )
    decisions = pd.read_csv(decisions_path, low_memory=False)
    allocations = pd.read_csv(allocations_path, low_memory=False)
    planner = OrderPlanner(execution_config, PROJECT_ROOT)
    orders, summary = planner.build(decisions, allocations)
    output_dir = PROJECT_ROOT / execution_config.get(
        "output_dir", "outputs/v6/execution"
    )
    save_order_plan(orders, summary, output_dir)
    print(orders.to_string(index=False), flush=True)
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()