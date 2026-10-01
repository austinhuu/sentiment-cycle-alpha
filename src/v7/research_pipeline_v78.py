from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtest import cross_section_ic, run_rank_portfolio
from .event_layer_v78 import structure_recent_news
from .market_sentiment_v78 import build_real_market_sentiment
from .sector_linkage_v78 import (
    build_historical_board_linkage,
    build_real_semiconductor_linkage,
)
from .transition_overlay import build_causal_transition_overlay


# ============================================================
# V7.8 FINAL FREEZE
# ============================================================
# This selection was made using the 2020-2024 development period.
# 2025 is robustness-only and must NOT be used to change this choice.
#
# Development winner:
#   V7.7 frozen alpha
#   + Real full-A market-return transition overlay
#
# Other market components, sector data and event data remain
# challengers / diagnostics only.
# ============================================================

FINAL_V78_VARIANT = "v7_8_market_return_only"
FINAL_V78_SCORE_COL = "v7_alpha"
FINAL_V78_OVERLAY_COL = "v7_8_overlay_return"


def _read_optional(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _manifest_latest(
    data_dir: Path,
    dataset: str,
) -> dict[str, Any]:
    path = data_dir / "data_manifest.csv"

    if not path.is_file():
        return {}

    try:
        manifest = pd.read_csv(path, low_memory=False)
    except Exception:
        return {}

    if "dataset" not in manifest.columns:
        return {}

    rows = manifest.loc[
        manifest["dataset"].astype(str) == dataset
    ]

    if rows.empty:
        return {}

    return rows.iloc[-1].to_dict()


def _as_bool(
    value: Any,
    default: bool = True,
) -> bool:
    if value is None:
        return default

    if isinstance(value, float) and np.isnan(value):
        return default

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def _phase_from_score(
    score: pd.Series,
) -> pd.Series:
    """
    Convert a continuous market sentiment score into the same
    V7.x A-share sentiment-cycle semantics.

    The phase definition itself is fixed and is not fitted on 2025.
    """

    s = (
        pd.to_numeric(score, errors="coerce")
        .fillna(0.0)
        .clip(-1.0, 1.0)
    )

    momentum = s.diff(3) / 3.0

    phase = np.select(
        [
            (s <= -0.35) & (momentum <= 0),
            (s <= -0.15) & (momentum > 0),
            (s > -0.15) & (s < 0.45) & (momentum > 0),
            (s >= 0.45) & (momentum >= -0.01),
            (s >= 0.15) & (momentum < -0.01),
        ],
        [
            "ICE_POINT",
            "REPAIR",
            "FERMENTATION",
            "CLIMAX",
            "DIVERGENCE",
        ],
        default="EBB",
    )

    return pd.Series(
        phase,
        index=score.index,
        dtype="object",
    )


def _score_from_market_component(
    market: pd.DataFrame,
    name: str,
) -> pd.Series:
    """
    Build the pre-declared V7.8 real-market component score.

    IMPORTANT:
    These definitions are fixed research specifications.

    They are NOT fitted using the 2025 holdout period.

    Available components:
        breadth
        return
        volume
        dispersion
        extreme_breadth
        full
    """

    zero = pd.Series(
        0.0,
        index=market.index,
        dtype=float,
    )

    if name == "breadth":
        raw = pd.to_numeric(
            market.get("breadth_z", zero),
            errors="coerce",
        ).fillna(0.0)

        return np.tanh(raw / 1.5)

    if name == "return":
        raw = pd.to_numeric(
            market.get("market_equal_return_z", zero),
            errors="coerce",
        ).fillna(0.0)

        return np.tanh(raw / 1.5)

    if name == "volume":
        raw = pd.to_numeric(
            market.get("amount_ratio_20_z", zero),
            errors="coerce",
        ).fillna(0.0)

        return np.tanh(raw / 1.5)

    if name == "dispersion":
        raw = pd.to_numeric(
            market.get("cross_section_vol_z", zero),
            errors="coerce",
        ).fillna(0.0)

        # High cross-sectional dispersion is treated as
        # a market-risk warning in this diagnostic.
        return -np.tanh(raw / 1.5)

    if name == "extreme_breadth":
        strong_up = np.tanh(
            pd.to_numeric(
                market.get("strong_up_rate_z", zero),
                errors="coerce",
            ).fillna(0.0)
            / 1.5
        )

        strong_down = np.tanh(
            pd.to_numeric(
                market.get("strong_down_rate_z", zero),
                errors="coerce",
            ).fillna(0.0)
            / 1.5
        )

        limit_balance = np.tanh(
            pd.to_numeric(
                market.get("limit_balance_z", zero),
                errors="coerce",
            ).fillna(0.0)
            / 1.5
        )

        return (
            0.40 * strong_up
            - 0.35 * strong_down
            + 0.25 * limit_balance
        ).clip(-1.0, 1.0)

    if name == "full":
        return (
            pd.to_numeric(
                market.get("real_sentiment_score", zero),
                errors="coerce",
            )
            .fillna(0.0)
            .clip(-1.0, 1.0)
        )

    raise ValueError(
        f"Unknown market component: {name}"
    )


class V78ResearchPipeline:
    """
    V7.8 Final research layer.

    ============================================================
    FINAL MODEL
    ============================================================

    V7.7 frozen alpha
        +
    Real full-A market-return transition overlay

    ============================================================

    Research discipline:

    1. V7.7 remains frozen.

    2. The V7.8 Final model was selected only using
       2020-2024 development results.

    3. 2025 is robustness-only.

    4. Real full-A breadth / volume / dispersion /
       extreme breadth / full sentiment remain diagnostics
       and formal ablation variants.

    5. Historical sector data may be evaluated as a challenger,
       but cannot silently replace V7.8 Final.

    6. Current semiconductor membership is survivorship-biased
       for historical research and is diagnostic only.

    7. Recent event/news data remain diagnostic unless a genuine
       historical PIT event dataset becomes available.

    8. No VeighNa/execution dependency exists in V7.8.
    """

    def __init__(
        self,
        config: dict[str, Any],
    ) -> None:
        self.config = config
        self.v78 = dict(
            config.get("v7_8", {})
        )

    def _build_market_multiplier(
        self,
        out: pd.DataFrame,
        market: pd.DataFrame,
        component: str,
        fallback_col: str = "transition_cycle_multiplier",
    ) -> pd.Series:
        """
        Re-run the causal V7.7 transition overlay using one
        real full-A market component.

        Training remains walk-forward:
        only prior information is available to each date.
        """

        fallback_source = out.get(
            fallback_col,
            0.78,
        )

        fallback = (
            pd.to_numeric(
                fallback_source,
                errors="coerce",
            )
            .fillna(0.78)
        )

        if market.empty:
            return fallback

        daily = market[
            ["datetime"]
        ].copy()

        daily["component_score"] = (
            _score_from_market_component(
                market,
                component,
            )
        )

        daily["component_phase"] = (
            _phase_from_score(
                daily["component_score"]
            )
        )

        temp = out.copy()

        temp = temp.merge(
            daily,
            on="datetime",
            how="left",
        )

        previous_cycle_score = pd.to_numeric(
            temp.get(
                "cycle_score",
                pd.Series(
                    np.nan,
                    index=temp.index,
                ),
            ),
            errors="coerce",
        )

        temp["cycle_score"] = (
            pd.to_numeric(
                temp["component_score"],
                errors="coerce",
            )
            .fillna(previous_cycle_score)
        )

        if "cycle_phase" in temp.columns:
            temp["cycle_phase"] = (
                temp["component_phase"]
                .fillna(temp["cycle_phase"])
            )
        else:
            temp["cycle_phase"] = (
                temp["component_phase"]
            )

        transition_cfg = dict(
            self.config
            .get("v7_7", {})
            .get("transition", {})
        )

        transition = (
            build_causal_transition_overlay(
                temp,
                target_col="future_return_5",
                cfg=transition_cfg,
            )
        )

        merged = (
            out[["datetime"]]
            .merge(
                transition[
                    [
                        "datetime",
                        "transition_cycle_multiplier",
                    ]
                ],
                on="datetime",
                how="left",
            )
        )

        multiplier = (
            pd.to_numeric(
                merged[
                    "transition_cycle_multiplier"
                ],
                errors="coerce",
            )
            .fillna(
                fallback.reset_index(drop=True)
            )
        )

        multiplier.index = out.index

        return multiplier

    def run(
        self,
        v77_signals: pd.DataFrame,
        data_dir: Path,
        output_dir: Path,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        data_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ====================================================
        # Frozen V7.7 input
        # ====================================================

        out = v77_signals.copy()

        out["datetime"] = pd.to_datetime(
            out["datetime"],
            errors="coerce",
        )

        out["symbol"] = (
            out["symbol"]
            .astype(str)
            .str.upper()
        )

        out = (
            out
            .dropna(
                subset=[
                    "datetime",
                    "symbol",
                ]
            )
            .sort_values(
                [
                    "datetime",
                    "symbol",
                ]
            )
            .reset_index(drop=True)
        )

        # ====================================================
        # Load optional V7.8 datasets
        # ====================================================

        all_a = _read_optional(
            data_dir
            / "all_a_market_summary.csv"
        )

        market_dataset_name = (
            "all_a_market_summary"
        )

        if all_a.empty:
            all_a = _read_optional(
                data_dir
                / "all_a_daily.csv"
            )

            market_dataset_name = (
                "all_a_daily"
            )

        limits = _read_optional(
            data_dir
            / "recent_limit_pools.csv"
        )

        semiconductor_current = (
            _read_optional(
                data_dir
                / "semiconductor_daily.csv"
            )
        )

        semiconductor_board = (
            _read_optional(
                data_dir
                / "semiconductor_board_daily.csv"
            )
        )

        recent_news = _read_optional(
            data_dir
            / "focus_news_recent.csv"
        )

        focus = str(
            self.v78.get(
                "focus_symbol",
                self.config
                .get("v7", {})
                .get(
                    "focus_symbol",
                    "688981.SH",
                ),
            )
        ).upper()

        market_cfg = dict(
            self.v78.get(
                "market_sentiment",
                {},
            )
        )

        sector_cfg = dict(
            self.v78.get(
                "sector",
                {},
            )
        )

        gate_cfg = dict(
            self.v78.get(
                "gates",
                {},
            )
        )

        # ====================================================
        # Real full-A sentiment
        # ====================================================

        market = (
            build_real_market_sentiment(
                all_a,
                exact_limit_pool=limits,
                z_window=int(
                    market_cfg.get(
                        "z_window",
                        120,
                    )
                ),
                min_periods=int(
                    market_cfg.get(
                        "z_min_periods",
                        40,
                    )
                ),
            )
        )

        # ====================================================
        # Semiconductor diagnostics
        # ====================================================

        current_peer_sector = (
            build_real_semiconductor_linkage(
                semiconductor_current,
                focus_symbol=focus,
                corr_window=int(
                    sector_cfg.get(
                        "corr_window",
                        20,
                    )
                ),
                z_window=int(
                    sector_cfg.get(
                        "z_window",
                        120,
                    )
                ),
                z_min_periods=int(
                    sector_cfg.get(
                        "z_min_periods",
                        40,
                    )
                ),
            )
        )

        focus_cols = [
            c
            for c in [
                "datetime",
                "return_1",
                "return_5",
            ]
            if c in out.columns
        ]

        focus_daily = (
            out.loc[
                out["symbol"].eq(focus),
                focus_cols,
            ]
            .copy()
        )

        historical_sector = (
            build_historical_board_linkage(
                semiconductor_board,
                focus_daily,
                corr_window=int(
                    sector_cfg.get(
                        "corr_window",
                        20,
                    )
                ),
                z_window=int(
                    sector_cfg.get(
                        "z_window",
                        120,
                    )
                ),
                z_min_periods=int(
                    sector_cfg.get(
                        "z_min_periods",
                        40,
                    )
                ),
            )
        )

        structured_news = (
            structure_recent_news(
                recent_news
            )
        )

        # ====================================================
        # Independent data gates
        # ====================================================

        market_manifest = _manifest_latest(
            data_dir,
            market_dataset_name,
        )

        current_sector_manifest = (
            _manifest_latest(
                data_dir,
                "semiconductor_daily",
            )
        )

        board_manifest = (
            _manifest_latest(
                data_dir,
                "semiconductor_board_daily",
            )
        )

        news_manifest = (
            _manifest_latest(
                data_dir,
                "focus_news_recent",
            )
        )

        market_survivorship = _as_bool(
            market_manifest.get(
                "survivorship_flag"
            ),
            default=True,
        )

        current_sector_survivorship = (
            _as_bool(
                current_sector_manifest.get(
                    "survivorship_flag"
                ),
                default=True,
            )
        )

        board_survivorship = _as_bool(
            board_manifest.get(
                "survivorship_flag"
            ),
            default=False,
        )

        signal_dates = pd.Index(
            out[
                "datetime"
            ]
            .dt.normalize()
            .dropna()
            .unique()
        )

        market_datetime = pd.to_datetime(
            market.get(
                "datetime",
                pd.Series(
                    dtype="datetime64[ns]"
                ),
            ),
            errors="coerce",
        )

        market_dates = pd.Index(
            market_datetime
            .dropna()
            .dt.normalize()
            .unique()
        )

        market_coverage = float(
            len(
                signal_dates.intersection(
                    market_dates
                )
            )
            / max(
                1,
                len(signal_dates),
            )
        )

        if not market.empty:
            median_market_count = float(
                pd.to_numeric(
                    market.get(
                        "market_count"
                    ),
                    errors="coerce",
                ).median()
            )
        else:
            median_market_count = 0.0

        accepted_market_quality = set(
            gate_cfg.get(
                "accepted_market_quality",
                [
                    "REAL_POINT_IN_TIME",
                ],
            )
        )

        market_quality = str(
            market_manifest.get(
                "quality",
                "unknown",
            )
        )

        market_gate = bool(
            not market.empty
            and market_coverage
            >= float(
                gate_cfg.get(
                    "min_market_date_coverage",
                    0.80,
                )
            )
            and median_market_count
            >= int(
                gate_cfg.get(
                    "min_market_count",
                    1500,
                )
            )
            and not market_survivorship
            and market_quality
            in accepted_market_quality
        )

        # ----------------------------------------------------
        # Historical sector gate
        # ----------------------------------------------------

        board_datetime = pd.to_datetime(
            historical_sector.get(
                "datetime",
                pd.Series(
                    dtype="datetime64[ns]"
                ),
            ),
            errors="coerce",
        )

        board_dates = pd.Index(
            board_datetime
            .dropna()
            .dt.normalize()
            .unique()
        )

        board_coverage = float(
            len(
                signal_dates.intersection(
                    board_dates
                )
            )
            / max(
                1,
                len(signal_dates),
            )
        )

        accepted_sector_quality = set(
            gate_cfg.get(
                "accepted_sector_quality",
                [
                    "REAL_HISTORICAL_BOARD_INDEX",
                ],
            )
        )

        board_quality = str(
            board_manifest.get(
                "quality",
                "unknown",
            )
        )

        sector_gate = bool(
            not historical_sector.empty
            and board_coverage
            >= float(
                gate_cfg.get(
                    "min_sector_date_coverage",
                    0.80,
                )
            )
            and not board_survivorship
            and board_quality
            in accepted_sector_quality
        )

        # ----------------------------------------------------
        # Event gate
        # ----------------------------------------------------

        event_datetime = pd.to_datetime(
            structured_news.get(
                "published_at",
                pd.Series(
                    dtype="datetime64[ns]"
                ),
            ),
            errors="coerce",
        )

        event_dates = pd.Index(
            event_datetime
            .dropna()
            .dt.normalize()
            .unique()
        )

        event_coverage = float(
            len(
                signal_dates.intersection(
                    event_dates
                )
            )
            / max(
                1,
                len(signal_dates),
            )
        )

        event_quality = str(
            news_manifest.get(
                "quality",
                "unknown",
            )
        )

        accepted_event_quality = set(
            gate_cfg.get(
                "accepted_event_quality",
                [
                    "REAL_HISTORICAL_POINT_IN_TIME",
                ],
            )
        )

        event_gate = bool(
            not structured_news.empty
            and event_coverage
            >= float(
                gate_cfg.get(
                    "min_event_date_coverage",
                    0.20,
                )
            )
            and event_quality
            in accepted_event_quality
        )

        # ====================================================
        # Merge real-market diagnostics
        # ====================================================

        if not market.empty:
            market_merge_cols = [
                c
                for c in market.columns
                if c
                != "market_equal_return"
            ]

            out = out.merge(
                market[
                    market_merge_cols
                ],
                on="datetime",
                how="left",
            )

        else:
            out[
                "real_sentiment_score"
            ] = np.nan

            out[
                "real_cycle_phase"
            ] = pd.NA

        # ====================================================
        # Merge current-membership sector diagnostics
        # ====================================================

        if not current_peer_sector.empty:
            out = out.merge(
                current_peer_sector,
                on="datetime",
                how="left",
            )

        else:
            out[
                "real_sector_strength_score"
            ] = np.nan

            out[
                "real_sector_stock_link_score"
            ] = np.nan

            out[
                "real_sector_peer_count"
            ] = np.nan

        # ====================================================
        # Merge historical sector challenger
        # ====================================================

        if not historical_sector.empty:
            out = out.merge(
                historical_sector,
                on="datetime",
                how="left",
            )

        else:
            out[
                "historical_board_strength_score"
            ] = np.nan

            out[
                "historical_board_stock_link_score"
            ] = np.nan

        # ====================================================
        # REAL MARKET OVERLAYS
        # ====================================================

        frozen_cycle = (
            pd.to_numeric(
                out.get(
                    "transition_cycle_multiplier",
                    0.78,
                ),
                errors="coerce",
            )
            .fillna(0.78)
        )

        # Default = frozen V7.7
        out[
            "v7_8_cycle_multiplier"
        ] = frozen_cycle

        ablation_components = [
            "breadth",
            "return",
            "volume",
            "dispersion",
            "extreme_breadth",
            "full",
        ]

        # Build every candidate for diagnostics / formal ablation
        for component in ablation_components:
            cycle_col = (
                f"v7_8_cycle_{component}"
            )

            out[
                cycle_col
            ] = frozen_cycle

            if market_gate:
                out[
                    cycle_col
                ] = (
                    self._build_market_multiplier(
                        out,
                        market,
                        component,
                    )
                )

        # ====================================================
        # V7.8 FINAL FREEZE
        # ====================================================
        #
        # Development winner = market_return_only.
        #
        # Do NOT replace this with the full sentiment score
        # just because additional data becomes available.
        #
        # 2025 is not allowed to change this selection.
        # ====================================================

        if market_gate:
            out[
                "v7_8_cycle_multiplier"
            ] = out[
                "v7_8_cycle_return"
            ]

        out[
            "real_transition_cycle_multiplier"
        ] = (
            out[
                "v7_8_cycle_multiplier"
            ]
            if market_gate
            else np.nan
        )

        out[
            "v7_8_selected_overlay"
        ] = FINAL_V78_VARIANT

        # ====================================================
        # Sector context
        # ====================================================

        final_score = (
            pd.to_numeric(
                out.get(
                    "final_score"
                ),
                errors="coerce",
            )
            .fillna(0.0)
        )

        frozen_sector = (
            pd.to_numeric(
                out.get(
                    "sector_context_score"
                ),
                errors="coerce",
            )
            .fillna(0.0)
        )

        focus_mask = (
            out["symbol"].eq(focus)
        )

        historical_strength = (
            pd.to_numeric(
                out.get(
                    "historical_board_strength_score"
                ),
                errors="coerce",
            )
        )

        safe_sector = (
            frozen_sector.copy()
        )

        # Historical real sector data can be evaluated as
        # a challenger when the gate passes.
        if sector_gate:
            safe_sector = safe_sector.where(
                ~(
                    focus_mask
                    & historical_strength.notna()
                ),
                historical_strength,
            )

        current_strength = (
            pd.to_numeric(
                out.get(
                    "real_sector_strength_score"
                ),
                errors="coerce",
            )
        )

        # Current membership is always diagnostic.
        current_membership_experimental = (
            safe_sector.where(
                ~(
                    focus_mask
                    & current_strength.notna()
                ),
                current_strength,
            )
        )

        out[
            "v7_8_sector_context"
        ] = safe_sector

        out[
            "v7_8_sector_context_current_membership_diagnostic"
        ] = current_membership_experimental

        # Historical-sector challenger
        out["v7_8_alpha"] = (
            0.80 * final_score
            + 0.20 * safe_sector
        ).clip(
            -1.0,
            1.0,
        )

        # Current-membership diagnostic challenger
        out[
            "v7_8_alpha_current_membership_diagnostic"
        ] = (
            0.80 * final_score
            + 0.20
            * current_membership_experimental
        ).clip(
            -1.0,
            1.0,
        )

        # ====================================================
        # FINAL SCORE
        # ====================================================
        #
        # The development winner used frozen V7.7 alpha.
        #
        # Therefore even if historical sector data later becomes
        # available, it must NOT silently replace V7.8 Final.
        # ====================================================

        out[
            "v7_8_final_score"
        ] = (
            pd.to_numeric(
                out.get(
                    FINAL_V78_SCORE_COL
                ),
                errors="coerce",
            )
            .fillna(0.0)
        )

        # ====================================================
        # Dynamic position multipliers
        # ====================================================

        non_cycle = (
            pd.to_numeric(
                out.get(
                    "overlay_no_cycle"
                ),
                errors="coerce",
            )
            .fillna(1.0)
        )

        # Final V7.8 multiplier
        out[
            "v7_8_dynamic_position_multiplier"
        ] = (
            non_cycle
            * out[
                "v7_8_cycle_multiplier"
            ]
        ).clip(
            0.20,
            1.00,
        )

        # Preserve all ablation overlays
        for component in ablation_components:
            out[
                f"v7_8_overlay_{component}"
            ] = (
                non_cycle
                * out[
                    f"v7_8_cycle_{component}"
                ]
            ).clip(
                0.20,
                1.00,
            )

        # ====================================================
        # Event diagnostics
        # ====================================================

        if not structured_news.empty:
            structured_news = (
                structured_news.copy()
            )

            structured_news[
                "datetime"
            ] = (
                pd.to_datetime(
                    structured_news[
                        "published_at"
                    ],
                    errors="coerce",
                )
                .dt.normalize()
            )

            daily_news = (
                structured_news
                .groupby(
                    "datetime",
                    as_index=False,
                )
                .agg(
                    recent_event_score_v78=(
                        "structured_event_score_v78",
                        "mean",
                    ),
                    recent_event_confidence_v78=(
                        "confidence",
                        "mean",
                    ),
                    recent_event_count_v78=(
                        "structured_event_score_v78",
                        "size",
                    ),
                )
            )

            out = out.merge(
                daily_news,
                on="datetime",
                how="left",
            )

        if (
            "recent_event_score_v78"
            not in out.columns
        ):
            out[
                "recent_event_score_v78"
            ] = 0.0

        out[
            "recent_event_score_v78"
        ] = (
            pd.to_numeric(
                out[
                    "recent_event_score_v78"
                ],
                errors="coerce",
            )
            .fillna(0.0)
        )

        out[
            "v7_8_event_diagnostic_multiplier"
        ] = (
            1.0
            - 0.15
            * (
                -out[
                    "recent_event_score_v78"
                ]
            ).clip(
                0.0,
                1.0,
            )
        ).clip(
            0.85,
            1.0,
        )

        # IMPORTANT:
        # Event diagnostic multiplier is deliberately NOT
        # multiplied into V7.8 Final.

        # ====================================================
        # Formal ablation
        # ====================================================

        bt_cfg = {
            **dict(
                self.config.get(
                    "backtest",
                    {},
                )
            ),
            **dict(
                self.config
                .get("v7", {})
                .get(
                    "backtest",
                    {},
                )
            ),
        }

        bt_cfg[
            "round_trip_turnover_cost"
        ] = float(
            self.config
            .get("v7", {})
            .get(
                "backtest",
                {},
            )
            .get(
                "round_trip_turnover_cost",
                0.0017,
            )
        )

        # ----------------------------------------------------
        # DO NOT remove variants.
        #
        # They remain essential for research diagnostics.
        # ----------------------------------------------------

        variants: dict[
            str,
            tuple[str, str],
        ] = {
            "v7_7_frozen": (
                "v7_alpha",
                "v7_7_dynamic_position_multiplier",
            ),

            "v7_8_breadth_only": (
                "v7_alpha",
                "v7_8_overlay_breadth",
            ),

            "v7_8_market_return_only": (
                "v7_alpha",
                "v7_8_overlay_return",
            ),

            "v7_8_volume_only": (
                "v7_alpha",
                "v7_8_overlay_volume",
            ),

            "v7_8_dispersion_only": (
                "v7_alpha",
                "v7_8_overlay_dispersion",
            ),

            "v7_8_extreme_breadth_only": (
                "v7_alpha",
                "v7_8_overlay_extreme_breadth",
            ),

            "v7_8_real_market_full": (
                "v7_alpha",
                "v7_8_overlay_full",
            ),

            # Historical-sector challenger
            "v7_8_real_market_plus_sector": (
                "v7_8_alpha",
                "v7_8_overlay_full",
            ),

            # Survivorship-biased current-membership diagnostic
            "v7_8_current_membership_sector_diagnostic": (
                "v7_8_alpha_current_membership_diagnostic",
                "v7_8_overlay_full",
            ),
        }

        periods = {
            "full": out,

            "development_2020_2024": (
                out.loc[
                    out["datetime"]
                    < pd.Timestamp(
                        "2025-01-01"
                    )
                ]
            ),

            "holdout_2025": (
                out.loc[
                    out["datetime"]
                    >= pd.Timestamp(
                        "2025-01-01"
                    )
                ]
            ),
        }

        rows: list[
            dict[str, Any]
        ] = []

        metrics: dict[
            str,
            Any,
        ] = {}

        full_backtests: dict[
            str,
            pd.DataFrame,
        ] = {}

        empty_perf = {
            "total_return": 0.0,
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "mean_turnover": 0.0,
            "total_cost": 0.0,
            "trading_days": 0,
        }

        empty_ic = {
            "mean_ic": 0.0,
            "ic_std": 0.0,
            "ic_ir": 0.0,
            "positive_ic_rate": 0.0,
            "ic_days": 0,
        }

        for (
            period_name,
            period_frame,
        ) in periods.items():

            for (
                variant,
                (
                    score_col,
                    overlay_col,
                ),
            ) in variants.items():

                if period_frame.empty:
                    daily = pd.DataFrame(
                        columns=[
                            "datetime",
                            "net_return",
                            "turnover",
                            "cost",
                        ]
                    )

                    perf = dict(
                        empty_perf
                    )

                    ic5 = dict(
                        empty_ic
                    )

                else:
                    local_cfg = dict(
                        bt_cfg
                    )

                    local_cfg[
                        "use_overlay"
                    ] = True

                    local_cfg[
                        "overlay_col"
                    ] = overlay_col

                    daily, perf = (
                        run_rank_portfolio(
                            period_frame,
                            score_col,
                            local_cfg,
                        )
                    )

                    ic5 = (
                        cross_section_ic(
                            period_frame,
                            score_col,
                            "future_return_5",
                        )
                    )

                record = {
                    "period": period_name,
                    "variant": variant,
                    **perf,
                    **{
                        f"ic5_{key}": value
                        for (
                            key,
                            value,
                        ) in ic5.items()
                    },
                }

                rows.append(
                    record
                )

                metrics[
                    f"{period_name}:{variant}"
                ] = record

                if (
                    period_name
                    == "full"
                ):
                    full_backtests[
                        variant
                    ] = daily

        ablation = pd.DataFrame(
            rows
        )

        # ====================================================
        # Coverage report
        # ====================================================

        current_sector_datetime = (
            pd.to_datetime(
                current_peer_sector.get(
                    "datetime",
                    pd.Series(
                        dtype="datetime64[ns]"
                    ),
                ),
                errors="coerce",
            )
        )

        current_sector_dates = pd.Index(
            current_sector_datetime
            .dropna()
            .dt.normalize()
            .unique()
        )

        current_sector_coverage = float(
            len(
                signal_dates.intersection(
                    current_sector_dates
                )
            )
            / max(
                1,
                len(signal_dates),
            )
        )

        if not current_peer_sector.empty:
            median_peer_count = float(
                pd.to_numeric(
                    current_peer_sector.get(
                        "real_sector_peer_count"
                    ),
                    errors="coerce",
                ).median()
            )
        else:
            median_peer_count = 0.0

        coverage = pd.DataFrame(
            [
                {
                    "dataset": market_dataset_name,
                    "present": not all_a.empty,
                    "date_coverage": market_coverage,
                    "median_market_count": (
                        median_market_count
                    ),
                    "survivorship_flag": (
                        market_survivorship
                    ),
                    "gate": "market_gate",
                    "passed_primary_backtest_gate": (
                        market_gate
                    ),
                    "source": (
                        market_manifest.get(
                            "source",
                            "missing_manifest",
                        )
                    ),
                    "quality": (
                        market_quality
                    ),
                },

                {
                    "dataset": (
                        "semiconductor_board_daily"
                    ),
                    "present": (
                        not historical_sector.empty
                    ),
                    "date_coverage": (
                        board_coverage
                    ),
                    "median_market_count": np.nan,
                    "survivorship_flag": (
                        board_survivorship
                    ),
                    "gate": "sector_gate",
                    "passed_primary_backtest_gate": (
                        sector_gate
                    ),
                    "source": (
                        board_manifest.get(
                            "source",
                            "missing_manifest",
                        )
                    ),
                    "quality": (
                        board_quality
                    ),
                },

                {
                    "dataset": (
                        "semiconductor_daily"
                    ),
                    "present": (
                        not current_peer_sector.empty
                    ),
                    "date_coverage": (
                        current_sector_coverage
                    ),
                    "median_market_count": (
                        median_peer_count
                    ),
                    "survivorship_flag": (
                        current_sector_survivorship
                    ),
                    "gate": "diagnostic_only",
                    "passed_primary_backtest_gate": False,
                    "source": (
                        current_sector_manifest.get(
                            "source",
                            "missing_manifest",
                        )
                    ),
                    "quality": (
                        current_sector_manifest.get(
                            "quality",
                            "unknown",
                        )
                    ),
                },

                {
                    "dataset": (
                        "focus_news_recent"
                    ),
                    "present": (
                        not structured_news.empty
                    ),
                    "date_coverage": (
                        event_coverage
                    ),
                    "median_market_count": np.nan,
                    "survivorship_flag": False,
                    "gate": "event_gate",
                    "passed_primary_backtest_gate": (
                        event_gate
                    ),
                    "source": (
                        news_manifest.get(
                            "source",
                            "Eastmoney recent news",
                        )
                    ),
                    "quality": (
                        event_quality
                    ),
                },
            ]
        )

        # ====================================================
        # Outputs
        # ====================================================

        out.to_csv(
            output_dir
            / "v7_8_signals.csv",
            index=False,
            encoding="utf-8-sig",
        )

        market.to_csv(
            output_dir
            / "real_market_sentiment.csv",
            index=False,
            encoding="utf-8-sig",
        )

        current_peer_sector.to_csv(
            output_dir
            / "real_semiconductor_linkage_current_membership.csv",
            index=False,
            encoding="utf-8-sig",
        )

        historical_sector.to_csv(
            output_dir
            / "historical_semiconductor_sector_linkage.csv",
            index=False,
            encoding="utf-8-sig",
        )

        structured_news.to_csv(
            output_dir
            / "structured_recent_events.csv",
            index=False,
            encoding="utf-8-sig",
        )

        coverage.to_csv(
            output_dir
            / "data_coverage_report.csv",
            index=False,
            encoding="utf-8-sig",
        )

        ablation.to_csv(
            output_dir
            / "ablation_v7_8.csv",
            index=False,
            encoding="utf-8-sig",
        )

        for (
            name,
            backtest,
        ) in full_backtests.items():
            backtest.to_csv(
                output_dir
                / f"backtest_{name}.csv",
                index=False,
                encoding="utf-8-sig",
            )

        development_ablation = (
            ablation.loc[
                ablation[
                    "period"
                ].eq(
                    "development_2020_2024"
                )
            ]
            .copy()
        )

        development_ablation.to_csv(
            output_dir
            / "ablation_v7_8_development_only.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # ====================================================
        # 中芯国际 case study
        # ====================================================

        case_columns = [
            "datetime",
            "symbol",

            "cycle_phase",
            "transition",
            "transition_cycle_multiplier",

            "real_cycle_phase",
            "real_sentiment_score",

            "v7_8_cycle_multiplier",
            "v7_8_selected_overlay",
            "v7_8_final_score",

            "sector_source",
            "sector_context_score",

            "historical_board_strength_score",
            "historical_board_stock_link_score",

            "real_sector_strength_score",
            "real_sector_stock_link_score",
            "real_sector_peer_count",

            "v7_alpha",
            "v7_8_alpha",

            "v7_8_dynamic_position_multiplier",

            "recent_event_score_v78",

            "future_return_1",
            "future_return_5",
            "future_return_20",
        ]

        case = (
            out.loc[
                out["symbol"].eq(focus),
                [
                    c
                    for c in case_columns
                    if c in out.columns
                ],
            ]
            .copy()
        )

        case.to_csv(
            output_dir
            / (
                f"case_study_"
                f"{focus.replace('.', '_')}"
                f"_v7_8.csv"
            ),
            index=False,
            encoding="utf-8-sig",
        )

        # ====================================================
        # Final summary
        # ====================================================

        summary = {
            "version": "7.8",

            "rows": int(
                len(out)
            ),

            "dates": int(
                out[
                    "datetime"
                ].nunique()
            ),

            "symbols": int(
                out[
                    "symbol"
                ].nunique()
            ),

            "focus_symbol": focus,

            # -------------------------
            # Market gate
            # -------------------------

            "market_data_gate_passed": (
                market_gate
            ),

            "market_date_coverage": (
                market_coverage
            ),

            "median_market_count": (
                median_market_count
            ),

            "market_survivorship_flag": (
                market_survivorship
            ),

            # -------------------------
            # Sector gate
            # -------------------------

            "sector_data_gate_passed": (
                sector_gate
            ),

            # Backward-compatible keys
            "historical_board_sector_gate_passed": (
                sector_gate
            ),

            "historical_board_date_coverage": (
                board_coverage
            ),

            "sector_survivorship_flag": (
                current_sector_survivorship
            ),

            # -------------------------
            # Event gate
            # -------------------------

            "event_data_gate_passed": (
                event_gate
            ),

            "event_date_coverage": (
                event_coverage
            ),

            # -------------------------
            # FINAL FREEZE
            # -------------------------

            "selected_variant": (
                FINAL_V78_VARIANT
            ),

            "selected_score_col": (
                FINAL_V78_SCORE_COL
            ),

            "selected_overlay_col": (
                FINAL_V78_OVERLAY_COL
            ),

            "selection_period": (
                "development_2020_2024"
            ),

            "champion": (
                "v7_7_frozen"
            ),

            "challenger": (
                FINAL_V78_VARIANT
            ),

            # -------------------------
            # Metrics
            # -------------------------

            "metrics": metrics,
        }

        (
            output_dir
            / "v7_8_metrics.json"
        ).write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        self._write_report(
            summary,
            coverage,
            ablation,
            output_dir,
        )

        return out, summary

    @staticmethod
    def _write_report(
        summary: dict[str, Any],
        coverage: pd.DataFrame,
        ablation: pd.DataFrame,
        output_dir: Path,
    ) -> None:
        def pick(
            period: str,
            variant: str,
        ) -> pd.Series:
            rows = ablation.loc[
                (
                    ablation["period"]
                    == period
                )
                & (
                    ablation["variant"]
                    == variant
                )
            ]

            if rows.empty:
                raise KeyError(
                    f"Missing ablation row: "
                    f"{period}:{variant}"
                )

            return rows.iloc[0]

        selected = str(
            summary.get(
                "selected_variant",
                FINAL_V78_VARIANT,
            )
        )

        # ====================================================
        # V7.8 Final challenger
        # ====================================================

        final_full = pick(
            "full",
            selected,
        )

        final_development = pick(
            "development_2020_2024",
            selected,
        )

        final_holdout = pick(
            "holdout_2025",
            selected,
        )

        # ====================================================
        # V7.7 frozen champion
        # ====================================================

        frozen_full = pick(
            "full",
            "v7_7_frozen",
        )

        frozen_development = pick(
            "development_2020_2024",
            "v7_7_frozen",
        )

        frozen_holdout = pick(
            "holdout_2025",
            "v7_7_frozen",
        )

        lines = [
            "# V7.8 Final — Real A-Share Market Research Layer",
            "",

            "## Frozen selection",
            "",

            f"- Selected variant: `{selected}`",
            "- Selection window: 2020–2024 development only.",
            "- 2025 is robustness-only and was not used to choose or tune the final variant.",
            "- Champion: V7.7 frozen.",
            "- Challenger: V7.8 market-return-only overlay.",
            "",

            "## Research discipline",
            "",

            "- V7.7 is frozen.",
            "- Real full-A market return is the only V7.8 feature allowed into the final daily exposure overlay.",
            "- Breadth, volume, dispersion, extreme breadth and full sentiment remain diagnostics/ablation variants.",
            "- Historical sector context remains a challenger; current semiconductor membership is diagnostic only.",
            "- Recent news remains diagnostic unless a future historical point-in-time event gate is available.",
            "- V7.8 contains no VeighNa execution dependency.",
            "",

            "## Data gates",
            "",

            coverage.to_markdown(
                index=False
            ),
            "",

            "## Final V7.8 challenger",
            "",

            (
                f"- Full annual: "
                f"{float(final_full['annual_return']):.2%}, "
                f"Sharpe: "
                f"{float(final_full['sharpe']):.3f}, "
                f"MaxDD: "
                f"{float(final_full['max_drawdown']):.2%}"
            ),

            (
                f"- Development 2020–2024 annual: "
                f"{float(final_development['annual_return']):.2%}, "
                f"Sharpe: "
                f"{float(final_development['sharpe']):.3f}, "
                f"MaxDD: "
                f"{float(final_development['max_drawdown']):.2%}"
            ),

            (
                f"- Holdout 2025 annual: "
                f"{float(final_holdout['annual_return']):.2%}, "
                f"Sharpe: "
                f"{float(final_holdout['sharpe']):.3f}, "
                f"MaxDD: "
                f"{float(final_holdout['max_drawdown']):.2%}"
            ),
            "",

            "## Frozen V7.7 champion",
            "",

            (
                f"- Full annual: "
                f"{float(frozen_full['annual_return']):.2%}, "
                f"Sharpe: "
                f"{float(frozen_full['sharpe']):.3f}, "
                f"MaxDD: "
                f"{float(frozen_full['max_drawdown']):.2%}"
            ),

            (
                f"- Development 2020–2024 annual: "
                f"{float(frozen_development['annual_return']):.2%}, "
                f"Sharpe: "
                f"{float(frozen_development['sharpe']):.3f}, "
                f"MaxDD: "
                f"{float(frozen_development['max_drawdown']):.2%}"
            ),

            (
                f"- Holdout 2025 annual: "
                f"{float(frozen_holdout['annual_return']):.2%}, "
                f"Sharpe: "
                f"{float(frozen_holdout['sharpe']):.3f}, "
                f"MaxDD: "
                f"{float(frozen_holdout['max_drawdown']):.2%}"
            ),
            "",

            (
                "Sector and event gates are independent of the "
                "V7.8 Final selection. A valid historical sector "
                "series may still be evaluated as a challenger, "
                "but it cannot silently replace the frozen final model."
            ),
            "",

            "## Formal ablation",
            "",

            ablation.to_markdown(
                index=False
            ),
        ]

        (
            output_dir
            / "V7_8_REPORT.md"
        ).write_text(
            "\n".join(lines),
            encoding="utf-8",
        )