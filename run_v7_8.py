from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.v7 import V76ResearchPipeline, V77ResearchPipeline
from src.v7.research_pipeline_v78 import (
    FINAL_V78_VARIANT,
    V78ResearchPipeline,
)


def _read_selected(
    path: Path,
    wanted: list[str],
) -> pd.DataFrame:
    header = pd.read_csv(
        path,
        nrows=0,
    ).columns

    use = [
        column
        for column in wanted
        if column in header
    ]

    return pd.read_csv(
        path,
        usecols=use,
        low_memory=False,
    )


def _ensure_v77_signals(
    config: dict,
) -> pd.DataFrame:
    """
    V7.8 always starts from the frozen V7.7 research output.

    Priority:
        1. Reuse V7.7 signals
        2. Rebuild V7.7 from V7.6
        3. Rebuild V6 -> V7.6 -> V7.7
    """

    v77_path = (
        PROJECT_ROOT
        / "outputs"
        / "v7_7"
        / "v7_7_signals.csv"
    )

    if v77_path.is_file():
        print(
            f"[V7.8] reusing frozen V7.7 signals: "
            f"{v77_path}",
            flush=True,
        )

        return pd.read_csv(
            v77_path,
            low_memory=False,
        )

    v76_path = (
        PROJECT_ROOT
        / "outputs"
        / "v7_6"
        / "v7_signals.csv"
    )

    if v76_path.is_file():
        print(
            f"[V7.8] rebuilding V7.7 from frozen "
            f"V7.6 signals: {v76_path}",
            flush=True,
        )

        v76 = pd.read_csv(
            v76_path,
            low_memory=False,
        )

    else:
        features_path = (
            PROJECT_ROOT
            / "outputs"
            / "processed_features.csv"
        )

        decisions_path = (
            PROJECT_ROOT
            / "outputs"
            / "agent_decisions.csv"
        )

        master_path = (
            PROJECT_ROOT
            / "data"
            / "raw"
            / "security_master.csv"
        )

        missing = [
            path
            for path in [
                features_path,
                decisions_path,
                master_path,
            ]
            if not path.is_file()
        ]

        if missing:
            print(
                f"[V7.8] upstream outputs missing: "
                f"{missing}",
                flush=True,
            )

            print(
                "[V7.8] rebuilding V6 pipeline first...",
                flush=True,
            )

            from src.data.loader import (
                load_market_data,
            )

            from src.data.validator import (
                validate_market_data,
            )

            from src.pipeline import (
                MultiAgentPipeline,
            )

            raw, _ = load_market_data(
                config
            )

            cleaned, report = (
                validate_market_data(
                    raw
                )
            )

            (
                PROJECT_ROOT
                / "outputs"
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

            report.to_csv(
                PROJECT_ROOT
                / "outputs"
                / "data_quality_report.csv",
                index=False,
                encoding="utf-8-sig",
            )

            MultiAgentPipeline(
                config
            ).run(
                cleaned,
                PROJECT_ROOT
                / "outputs",
            )

        feature_cols = [
            "datetime",
            "symbol",
            "return_1",
            "return_5",
            "volume_ratio",
            "momentum_z",
            "volume_ratio_z",
            "ma_gap_short_long_z",
            "price_position",
            "breakout_up",
        ]

        decision_cols = [
            "datetime",
            "symbol",
            "future_return_1",
            "future_return_5",
            "future_return_20",
            "final_score",
            "confidence",
            "risk_score",
            "effective_risk_multiplier",
            "news_score",
            "news_confidence",
            "news_active",
            "news_material_event_count",
            "news_event_severity",
            "dominant_event_type",
        ]

        features = _read_selected(
            features_path,
            feature_cols,
        )

        decisions = _read_selected(
            decisions_path,
            decision_cols,
        )

        master = pd.read_csv(
            master_path,
            encoding="utf-8-sig",
        )

        v76, _ = (
            V76ResearchPipeline(
                config
            ).run(
                features,
                decisions,
                master,
                PROJECT_ROOT
                / "outputs"
                / "v7_6",
            )
        )

    v77, _ = (
        V77ResearchPipeline(
            config
        ).run(
            v76,
            PROJECT_ROOT
            / "outputs"
            / "v7_7",
        )
    )

    return v77


def _metric(
    summary: dict,
    period: str,
    variant: str,
) -> dict:
    key = (
        f"{period}:{variant}"
    )

    if key not in summary["metrics"]:
        raise KeyError(
            f"Missing V7.8 metric: {key}"
        )

    return summary[
        "metrics"
    ][key]


def main() -> None:
    config = load_config()

    # ========================================================
    # Frozen V7.7 input
    # ========================================================

    v77 = _ensure_v77_signals(
        config
    )

    data_dir = (
        PROJECT_ROOT
        / "data"
        / "v7_8"
    )

    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "v7_8"
    )

    _, summary = (
        V78ResearchPipeline(
            config
        ).run(
            v77,
            data_dir,
            output_dir,
        )
    )

    selected = str(
        summary.get(
            "selected_variant",
            FINAL_V78_VARIANT,
        )
    )

    # ========================================================
    # V7.8 Final challenger metrics
    # ========================================================

    final_full = _metric(
        summary,
        "full",
        selected,
    )

    final_development = _metric(
        summary,
        "development_2020_2024",
        selected,
    )

    final_holdout = _metric(
        summary,
        "holdout_2025",
        selected,
    )

    # ========================================================
    # Frozen V7.7 champion metrics
    # ========================================================

    frozen_full = _metric(
        summary,
        "full",
        "v7_7_frozen",
    )

    frozen_development = _metric(
        summary,
        "development_2020_2024",
        "v7_7_frozen",
    )

    frozen_holdout = _metric(
        summary,
        "holdout_2025",
        "v7_7_frozen",
    )

    # ========================================================
    # Console report
    # ========================================================

    print(
        "[V7.8 FINAL] COMPLETE",
        flush=True,
    )

    print(
        f"rows={summary['rows']:,} "
        f"dates={summary['dates']} "
        f"symbols={summary['symbols']}"
    )

    print(
        f"market_gate="
        f"{summary['market_data_gate_passed']} "
        f"coverage="
        f"{summary['market_date_coverage']:.1%} "
        f"median_market_count="
        f"{summary['median_market_count']:.0f}"
    )

    print(
        f"sector_gate="
        f"{summary['sector_data_gate_passed']} "
        f"coverage="
        f"{summary['historical_board_date_coverage']:.1%}"
    )

    print(
        f"event_gate="
        f"{summary['event_data_gate_passed']} "
        f"coverage="
        f"{summary['event_date_coverage']:.1%}"
    )

    print(
        f"selected_variant={selected}"
    )

    print(
        "selection_period="
        "development_2020_2024"
    )

    print(
        "champion=v7_7_frozen "
        "challenger="
        "v7_8_market_return_only"
    )

    # ========================================================
    # V7.8
    # ========================================================

    print(
        f"V7.8 final full "
        f"annual="
        f"{final_full['annual_return']:.2%} "
        f"sharpe="
        f"{final_full['sharpe']:.3f} "
        f"maxDD="
        f"{final_full['max_drawdown']:.2%}"
    )

    print(
        f"V7.8 development 2020-2024 "
        f"annual="
        f"{final_development['annual_return']:.2%} "
        f"sharpe="
        f"{final_development['sharpe']:.3f} "
        f"maxDD="
        f"{final_development['max_drawdown']:.2%}"
    )

    print(
        f"V7.8 holdout 2025 "
        f"annual="
        f"{final_holdout['annual_return']:.2%} "
        f"sharpe="
        f"{final_holdout['sharpe']:.3f} "
        f"maxDD="
        f"{final_holdout['max_drawdown']:.2%}"
    )

    # ========================================================
    # V7.7
    # ========================================================

    print(
        f"V7.7 frozen full "
        f"annual="
        f"{frozen_full['annual_return']:.2%} "
        f"sharpe="
        f"{frozen_full['sharpe']:.3f} "
        f"maxDD="
        f"{frozen_full['max_drawdown']:.2%}"
    )

    print(
        f"V7.7 development 2020-2024 "
        f"annual="
        f"{frozen_development['annual_return']:.2%} "
        f"sharpe="
        f"{frozen_development['sharpe']:.3f} "
        f"maxDD="
        f"{frozen_development['max_drawdown']:.2%}"
    )

    print(
        f"V7.7 holdout 2025 "
        f"annual="
        f"{frozen_holdout['annual_return']:.2%} "
        f"sharpe="
        f"{frozen_holdout['sharpe']:.3f} "
        f"maxDD="
        f"{frozen_holdout['max_drawdown']:.2%}"
    )

    # ========================================================
    # Gate explanation
    # ========================================================

    if not summary[
        "market_data_gate_passed"
    ]:
        print(
            "[V7.8] market gate is closed: "
            "final overlay falls back to "
            "frozen V7.7 exposure."
        )

    if not summary[
        "sector_data_gate_passed"
    ]:
        print(
            "[V7.8] sector gate is closed: "
            "frozen V7.7 sector context "
            "is retained."
        )

    else:
        print(
            "[V7.8] historical sector data "
            "is available as a challenger "
            "diagnostic; it does not replace "
            "the frozen final variant."
        )

    if not summary[
        "event_data_gate_passed"
    ]:
        print(
            "[V7.8] event gate is closed: "
            "recent news remains diagnostic only."
        )

    else:
        print(
            "[V7.8] event data passed its gate "
            "but remains diagnostic in the "
            "frozen V7.8 Final model."
        )

    print(
        f"output={output_dir}"
    )


if __name__ == "__main__":
    main()