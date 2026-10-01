from __future__ import annotations

import json

import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.evaluation.llm_calibration import (
    calibrate_llm_results,
    save_llm_calibration,
)


def main() -> None:
    config = load_config()
    calibration_config = config.get("llm_calibration", {})
    adaptive_dir = PROJECT_ROOT / config.get("adaptive_fusion", {}).get(
        "output_dir", "outputs/v4/adaptive_fusion"
    )
    decisions_path = adaptive_dir / "adaptive_decisions.csv"
    llm_path = PROJECT_ROOT / calibration_config.get(
        "llm_results_csv", "outputs/llm_debate/llm_debate_results.csv"
    )
    if not decisions_path.exists():
        raise FileNotFoundError(
            "未找到adaptive_decisions.csv，请先运行 python run_adaptive_fusion.py"
        )
    if not llm_path.exists():
        raise FileNotFoundError(
            "未找到LLM结果，请先运行 python run_llm_debate.py"
        )
    decisions = pd.read_csv(decisions_path)
    llm_results = pd.read_csv(llm_path)
    result = calibrate_llm_results(
        decisions,
        llm_results,
        calibration_config,
    )
    output_dir = PROJECT_ROOT / calibration_config.get(
        "output_dir", "outputs/v4/llm_calibration"
    )
    save_llm_calibration(result, output_dir)
    print(json.dumps(result.report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()