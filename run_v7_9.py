from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.v7 import V79ResearchPipeline


def _ensure_v78_final() -> pd.DataFrame:
    path = PROJECT_ROOT / "outputs" / "v7_8" / "v7_8_signals.csv"

    if not path.is_file():
        raise FileNotFoundError(
            "缺少冻结的 V7.8 Final signals。请先在同一个主项目中跑通 run_v7_8.py。"
        )

    print(
        f"[V7.9] reusing frozen V7.8 Final signals: {path}",
        flush=True,
    )

    return pd.read_csv(path, low_memory=False)


def _print_metrics(label: str, metrics: dict) -> None:
    if not metrics:
        return

    print(
        f"{label} "
        f"annual={float(metrics.get('annual_return', 0.0)):.2%} "
        f"sharpe={float(metrics.get('sharpe', 0.0)):.3f} "
        f"maxDD={float(metrics.get('max_drawdown', 0.0)):.2%} "
        f"trades={int(metrics.get('trade_count', 0))}"
    )


def main() -> None:
    config = load_config()
    v78 = _ensure_v78_final()

    data_dir = PROJECT_ROOT / "data" / "v7_9"
    output_dir = PROJECT_ROOT / "outputs" / "v7_9"

    _, summary = V79ResearchPipeline(config).run(
        v78,
        data_dir,
        output_dir,
    )

    legacy = summary.get("metrics_v7_9_full", {})
    entry_only = summary.get(
        "metrics_v7_9_entry_confirmation_only",
        {},
    )
    entry_no_momentum = summary.get(
        "metrics_v7_9_entry_confirmation_no_momentum",
        {},
    )
    baseline = summary.get("metrics_v7_8_open_baseline", {})

    print("[V7.9] COMPLETE")
    print(f"focus_symbol={summary['focus_symbol']}")

    print(
        f"intraday_gate={summary['intraday_data_gate_passed']} "
        f"quality={summary['intraday_quality']} "
        f"dates={summary['intraday_dates']} "
        f"median_bars/day={summary['median_bars_per_day']:.1f} "
        f"daily_signal_coverage={summary['daily_signal_coverage']:.1%}"
    )

    print(
        "T+1=True "
        "next_bar_execution=True "
        "daily_model=V7.8_Final_frozen"
    )

    print(
        "research_change=asymmetric_entry_confirmation "
        "selection_period=2020-2024 "
        "2025=robustness_only"
    )

    _print_metrics(
        "V7.9 legacy full",
        legacy,
    )

    _print_metrics(
        "V7.9 entry-confirmation-only",
        entry_only,
    )

    _print_metrics(
        "V7.9 entry-confirmation-no-momentum",
        entry_no_momentum,
    )

    _print_metrics(
        "V7.8 open baseline",
        baseline,
    )

    if not summary["intraday_data_gate_passed"]:
        print(
            "[V7.9] WARNING: intraday data gate is CLOSED. "
            "Current performance is smoke/engineering-only."
        )
        print(
            "[V7.9] Use real historical 5min bars before making any strategy conclusion."
        )

    print(f"output={output_dir}")


if __name__ == "__main__":
    main()
