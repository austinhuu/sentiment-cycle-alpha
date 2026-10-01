from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.v79_intraday_downloader import (
    REAL_5MIN_QUALITY,
    USER_5MIN_QUALITY,
    V79IntradayDataManager,
)
from .intraday_backtest_v79 import (
    build_focus_daily_targets,
    simulate_focus_strategy,
)
from .intraday_confirmation_v79 import build_confirmation_score
from .intraday_features_v79 import build_intraday_features


# -----------------------------------------------------------------------------
# V7.9 research variants
# -----------------------------------------------------------------------------
#
# The original V7.9 full model is retained unchanged as the historical baseline.
# The two new structural experiments test a narrower architecture:
#
#   V7.8 daily model -> reductions / exits
#   V7.9 5min model  -> entry timing only
#
# No threshold or factor weight is re-optimized here.
# -----------------------------------------------------------------------------

V79_VARIANTS: dict[str, dict[str, Any]] = {
    "v7_9_full": {
        "disabled": set(),
        "mode": "confirmation",
    },
    "v7_9_no_vwap": {
        "disabled": {"vwap"},
        "mode": "confirmation",
    },
    "v7_9_no_momentum": {
        "disabled": {"momentum_5", "momentum_15"},
        "mode": "confirmation",
    },
    "v7_9_no_rvol": {
        "disabled": {"rvol"},
        "mode": "confirmation",
    },
    "v7_9_no_breakout": {
        "disabled": {"breakout"},
        "mode": "confirmation",
    },
    "v7_9_no_volatility_control": {
        "disabled": {"volatility"},
        "mode": "confirmation",
    },
    "v7_9_entry_confirmation_only": {
        "disabled": set(),
        "mode": "entry_confirmation_only",
    },
    "v7_9_entry_confirmation_no_momentum": {
        "disabled": {"momentum_5", "momentum_15"},
        "mode": "entry_confirmation_only",
    },
}


class V79ResearchPipeline:
    """V7.9: 5-minute execution research on frozen V7.8 daily decisions.

    The V7.8 daily alpha and daily target are never re-fitted here.

    V7.9 now evaluates two architectures:

    1. Legacy symmetric confirmation
       5-minute information may time entries, reductions and exits.

    2. Asymmetric entry confirmation
       V7.8 owns reductions/exits at the next-session first bar open, while
       V7.9 5-minute information only decides when to establish/add exposure.

    Every intraday signal is executed no earlier than the next bar open and
    A-share T+1 sellability is enforced by the simulator.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.v79 = dict(config.get("v7_9", {}))

    def run(
        self,
        v78_signals: pd.DataFrame,
        data_dir: Path,
        output_dir: Path,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        manager = V79IntradayDataManager(data_dir)
        bars = manager.load_all()
        manifest = manager.read_manifest()

        if bars.empty:
            raise RuntimeError(
                "V7.9 没有找到 5min 数据。先运行 run_v7_9_data.py 下载/注册数据；"
                "也可以用 --source demo 做工程 smoke test。"
            )

        focus = str(
            self.v79.get(
                "focus_symbol",
                self.config.get("v7_8", {}).get(
                    "focus_symbol",
                    "688981.SH",
                ),
            )
        ).upper()

        bars = bars.loc[bars["symbol"].eq(focus)].copy()

        if bars.empty:
            raise RuntimeError(f"5min 数据中没有 focus symbol {focus}。")

        features = build_intraday_features(
            bars,
            dict(self.v79.get("features", {})),
        )

        if features.empty:
            raise RuntimeError("V7.9 5min 数据经过交易时段清洗后为空。")

        # ------------------------------------------------------------------
        # Research data gate
        # ------------------------------------------------------------------

        symbol_manifest = pd.DataFrame()

        if not manifest.empty and "symbol" in manifest.columns:
            symbol_manifest = manifest.loc[
                manifest["symbol"].astype(str).str.upper().eq(focus)
            ]

        quality = (
            str(symbol_manifest.iloc[-1]["quality"])
            if not symbol_manifest.empty
            else "UNKNOWN"
        )

        accepted = set(
            self.v79.get("gates", {}).get(
                "accepted_intraday_quality",
                [REAL_5MIN_QUALITY, USER_5MIN_QUALITY],
            )
        )

        dates = pd.DatetimeIndex(features["trade_date"].dropna().unique())
        bars_per_day = features.groupby("trade_date").size()
        median_bars = float(bars_per_day.median()) if len(bars_per_day) else 0.0

        target_cfg = {
            **dict(self.config.get("v7", {}).get("backtest", {})),
            **dict(self.v79.get("daily_target", {})),
        }

        targets = build_focus_daily_targets(
            v78_signals,
            dates,
            focus,
            target_cfg,
        )

        if targets.empty:
            raise RuntimeError("无法把 V7.8 日频信号映射到 V7.9 5min 交易日。")

        overlap = float(targets["signal_date"].notna().mean())
        gate_cfg = dict(self.v79.get("gates", {}))

        point_in_time = True

        if not symbol_manifest.empty and "point_in_time" in symbol_manifest.columns:
            point_in_time = (
                str(symbol_manifest.iloc[-1]["point_in_time"])
                .strip()
                .lower()
                in {"true", "1", "yes"}
            )

        data_gate = bool(
            quality in accepted
            and point_in_time
            and overlap
            >= float(gate_cfg.get("min_daily_signal_coverage", 0.80))
            and median_bars
            >= float(gate_cfg.get("min_median_bars_per_day", 40))
        )

        confirmation_cfg = dict(self.v79.get("confirmation", {}))
        backtest_cfg = dict(self.v79.get("backtest", {}))

        rows: list[dict[str, Any]] = []

        exported_bars: dict[str, pd.DataFrame] = {}
        exported_results: dict[str, Any] = {}

        # ------------------------------------------------------------------
        # V7.8 open baseline
        # ------------------------------------------------------------------

        baseline_features = features.copy()
        baseline_features["confirmation_score"] = 0.0

        baseline = simulate_focus_strategy(
            baseline_features,
            targets,
            backtest_cfg,
            mode="open_baseline",
        )

        rows.append(
            {
                "period": "full",
                "variant": "v7_8_open_baseline",
                **baseline.metrics,
            }
        )

        baseline.daily.to_csv(
            output_dir / "backtest_v7_8_open_baseline.csv",
            index=False,
            encoding="utf-8-sig",
        )

        baseline.trades.to_csv(
            output_dir / "trades_v7_8_open_baseline.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # Also report the baseline by period for apples-to-apples comparison.
        for period_name, start, end in [
            ("historical_2020_2024", None, pd.Timestamp("2025-01-01")),
            ("robustness_2025", pd.Timestamp("2025-01-01"), None),
        ]:
            local_bars = baseline_features.copy()
            local_targets = targets.copy()

            if start is not None:
                local_bars = local_bars.loc[local_bars["trade_date"] >= start]
                local_targets = local_targets.loc[local_targets["trade_date"] >= start]

            if end is not None:
                local_bars = local_bars.loc[local_bars["trade_date"] < end]
                local_targets = local_targets.loc[local_targets["trade_date"] < end]

            if local_bars.empty:
                continue

            local = simulate_focus_strategy(
                local_bars,
                local_targets,
                backtest_cfg,
                mode="open_baseline",
            )

            rows.append(
                {
                    "period": period_name,
                    "variant": "v7_8_open_baseline",
                    **local.metrics,
                }
            )

        # ------------------------------------------------------------------
        # V7.9 ablation + structural experiments
        # ------------------------------------------------------------------

        for name, spec in V79_VARIANTS.items():
            disabled = set(spec.get("disabled", set()))
            mode = str(spec.get("mode", "confirmation"))

            scored = build_confirmation_score(
                features,
                confirmation_cfg,
                disabled=disabled,
            )

            result = simulate_focus_strategy(
                scored,
                targets,
                backtest_cfg,
                mode=mode,
            )

            rows.append(
                {
                    "period": "full",
                    "variant": name,
                    **result.metrics,
                }
            )

            exported_results[name] = result
            exported_bars[name] = result.bars.copy()

            if name in {
                "v7_9_full",
                "v7_9_entry_confirmation_only",
                "v7_9_entry_confirmation_no_momentum",
            }:
                result.daily.to_csv(
                    output_dir / f"backtest_{name}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

                result.trades.to_csv(
                    output_dir / f"trades_{name}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

            # Period reports remain descriptive. Parameter selection must use
            # 2020-2024 only; 2025 is robustness-only.
            for period_name, start, end in [
                ("historical_2020_2024", None, pd.Timestamp("2025-01-01")),
                ("robustness_2025", pd.Timestamp("2025-01-01"), None),
            ]:
                local_bars = scored.copy()
                local_targets = targets.copy()

                if start is not None:
                    local_bars = local_bars.loc[local_bars["trade_date"] >= start]
                    local_targets = local_targets.loc[
                        local_targets["trade_date"] >= start
                    ]

                if end is not None:
                    local_bars = local_bars.loc[local_bars["trade_date"] < end]
                    local_targets = local_targets.loc[
                        local_targets["trade_date"] < end
                    ]

                if local_bars.empty:
                    continue

                local = simulate_focus_strategy(
                    local_bars,
                    local_targets,
                    backtest_cfg,
                    mode=mode,
                )

                rows.append(
                    {
                        "period": period_name,
                        "variant": name,
                        **local.metrics,
                    }
                )

        # ------------------------------------------------------------------
        # Outputs
        # ------------------------------------------------------------------

        ablation = pd.DataFrame(rows)
        ablation.to_csv(
            output_dir / "ablation_v7_9.csv",
            index=False,
            encoding="utf-8-sig",
        )

        development = ablation.loc[
            ablation["period"].eq("historical_2020_2024")
        ].copy()

        development.to_csv(
            output_dir / "ablation_v7_9_development_only.csv",
            index=False,
            encoding="utf-8-sig",
        )

        feature_export_cols = [
            "datetime",
            "symbol",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "bar_index",
            "bars_in_day",
            "vwap",
            "vwap_strength_score",
            "momentum_5",
            "momentum_15",
            "momentum_5_score",
            "momentum_15_score",
            "rvol",
            "rvol_score",
            "opening_range_high",
            "opening_range_low",
            "breakout_score",
            "intraday_volatility",
            "volatility_control_score",
        ]

        features[
            [c for c in feature_export_cols if c in features.columns]
        ].to_csv(
            output_dir / "intraday_features_v7_9.csv",
            index=False,
            encoding="utf-8-sig",
        )

        targets.to_csv(
            output_dir / "daily_targets_v7_9.csv",
            index=False,
            encoding="utf-8-sig",
        )

        decision_export_cols = feature_export_cols + [
            "confirmation_score",
            "confirmation_variant",
            "simulation_mode",
            "target_weight",
            "position_weight",
            "sellable_weight",
            "decision",
            "executed_side",
            "cash",
            "shares",
            "equity",
        ]

        if "v7_9_full" in exported_bars:
            legacy_bars = exported_bars["v7_9_full"]
            legacy_bars[
                [c for c in decision_export_cols if c in legacy_bars.columns]
            ].to_csv(
                output_dir / "intraday_decisions_v7_9.csv",
                index=False,
                encoding="utf-8-sig",
            )

        if "v7_9_entry_confirmation_only" in exported_bars:
            entry_bars = exported_bars["v7_9_entry_confirmation_only"]
            entry_bars[
                [c for c in decision_export_cols if c in entry_bars.columns]
            ].to_csv(
                output_dir / "intraday_decisions_v7_9_entry_confirmation_only.csv",
                index=False,
                encoding="utf-8-sig",
            )

        def _metric(variant: str) -> dict[str, Any]:
            row = ablation.loc[
                (ablation["period"] == "full")
                & (ablation["variant"] == variant)
            ]
            return row.iloc[0].to_dict() if not row.empty else {}

        summary = {
            "version": "7.9",
            "focus_symbol": focus,
            "intraday_data_gate_passed": data_gate,
            "intraday_quality": quality,
            "intraday_dates": int(len(dates)),
            "median_bars_per_day": median_bars,
            "daily_signal_coverage": float(overlap),
            "t_plus_one_enforced": True,
            "next_bar_execution": True,
            "daily_model_frozen": "V7.8 Final",
            "parameter_status": "STRUCTURAL_DIAGNOSTIC_NOT_TUNED",
            "research_selection_period": "historical_2020_2024",
            "robustness_period": "2025",
            "metrics_v7_9_full": _metric("v7_9_full"),
            "metrics_v7_9_entry_confirmation_only": _metric(
                "v7_9_entry_confirmation_only"
            ),
            "metrics_v7_9_entry_confirmation_no_momentum": _metric(
                "v7_9_entry_confirmation_no_momentum"
            ),
            "metrics_v7_8_open_baseline": _metric("v7_8_open_baseline"),
        }

        (output_dir / "v7_9_metrics.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self._write_report(summary, ablation, output_dir)

        # Backward-compatible return: legacy full bars. Structural candidate bars
        # are exported separately and summarized in v7_9_metrics.json.
        full_bars = exported_bars.get("v7_9_full", pd.DataFrame())

        return full_bars, summary

    @staticmethod
    def _write_report(
        summary: dict[str, Any],
        ablation: pd.DataFrame,
        output_dir: Path,
    ) -> None:
        lines = [
            "# V7.9 — 5min Intraday Confirmation",
            "",
            "## Frozen architecture",
            "",
            "- V7.8 daily alpha and daily target logic are frozen.",
            "- Signals computed at a 5-minute bar close execute no earlier than the next bar open.",
            "- A-share T+1 is enforced: same-day purchases cannot be sold on the same day.",
            "- No VeighNa dependency is used.",
            "",
            "## Structural experiment",
            "",
            "- `v7_9_full`: legacy symmetric 5min confirmation for ENTER / REDUCE / EXIT.",
            "- `v7_9_entry_confirmation_only`: V7.8 controls daily reductions/exits at the first bar open; 5min confirmation controls only increases in exposure.",
            "- `v7_9_entry_confirmation_no_momentum`: same asymmetric structure with 5min/15min momentum disabled.",
            "- No thresholds or factor weights are tuned in this structural test.",
            "- Model selection must use 2020–2024 only. 2025 is robustness-only.",
            "",
            "## Data quality",
            "",
            f"- Intraday gate: `{summary['intraday_data_gate_passed']}`",
            f"- Quality: `{summary['intraday_quality']}`",
            f"- Trading dates: {summary['intraday_dates']}",
            f"- Median bars/day: {summary['median_bars_per_day']:.1f}",
            f"- Daily-signal coverage: {summary['daily_signal_coverage']:.1%}",
            "",
        ]

        if not summary["intraday_data_gate_passed"]:
            lines += [
                "> The current run is an engineering/smoke result only. Do not interpret performance until the real 5-minute data gate passes.",
                "",
            ]

        lines += [
            "## Ablation",
            "",
            ablation.to_markdown(index=False),
        ]

        (output_dir / "V7_9_REPORT.md").write_text(
            "\n".join(lines),
            encoding="utf-8",
        )
