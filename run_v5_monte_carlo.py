from __future__ import annotations

import json

import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.robustness.monte_carlo import run_monte_carlo_stress, save_monte_carlo_outputs


def main() -> None:
    config = load_config()
    portfolio_path = PROJECT_ROOT / "outputs" / "v5" / "production" / "portfolio_equity.csv"
    if not portfolio_path.exists():
        raise FileNotFoundError("请先运行 python run_v5_walk_forward.py")
    portfolio = pd.read_csv(portfolio_path)
    monte_config = config.get("v5_monte_carlo", {})
    samples, summary = run_monte_carlo_stress(portfolio, monte_config)
    output_dir = PROJECT_ROOT / monte_config.get("output_dir", "outputs/v5/monte_carlo")
    save_monte_carlo_outputs(samples, summary, output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()