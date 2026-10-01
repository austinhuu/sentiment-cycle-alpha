from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.v78_real_data_downloader import (
    DatasetManifest,
    V78RealDataDownloader,
)
from src.v7.event_layer_v78 import (
    structure_recent_news,
)
from src.v7.market_sentiment_v78 import (
    build_real_market_sentiment,
)
from src.v7.research_pipeline_v78 import (
    FINAL_V78_VARIANT,
    V78ResearchPipeline,
)
from src.v7.sector_linkage_v78 import (
    build_historical_board_linkage,
    build_historical_sector_context,
    build_real_semiconductor_linkage,
)


def _config() -> dict:
    return {
        "backtest": {
            "annualization_bars": 252,
        },

        "v7": {
            "focus_symbol": "688981.SH",

            "backtest": {
                "top_k": 3,
                "min_score": 0.0,
                "max_gross": 1.0,
                "round_trip_turnover_cost": 0.001,
                "rebalance_every": 5,
            },
        },

        "v7_7": {
            "transition": {
                "base_exposure": 0.78,
                "sensitivity": 0.12,
                "min_exposure": 0.55,
                "max_exposure": 1.0,
                "phase_prior_strength": 10.0,
                "transition_prior_strength": 8.0,
                "purge_dates": 3,
                "min_train_dates": 20,
            },
        },

        "v7_8": {
            "focus_symbol": "688981.SH",

            "gates": {
                "min_market_date_coverage": 0.80,
                "min_market_count": 50,

                "accepted_market_quality": [
                    "REAL_POINT_IN_TIME",
                ],

                "min_sector_date_coverage": 0.80,

                "accepted_sector_quality": [
                    "REAL_HISTORICAL_BOARD_INDEX",
                ],

                "min_event_date_coverage": 0.20,

                "accepted_event_quality": [
                    "REAL_HISTORICAL_POINT_IN_TIME",
                ],
            },

            "market_sentiment": {
                "z_window": 30,
                "z_min_periods": 10,
            },

            "sector": {
                "corr_window": 10,
                "z_window": 30,
                "z_min_periods": 10,
            },
        },
    }


def _frozen_signals(
    periods: int = 80,
    symbols: int = 10,
) -> pd.DataFrame:
    dates = pd.bdate_range(
        "2024-01-02",
        periods=periods,
    )

    rows = []

    names = (
        ["688981.SH"]
        + [
            f"600{i:03d}.SH"
            for i in range(
                1,
                symbols,
            )
        ]
    )

    phases = [
        "REPAIR",
        "FERMENTATION",
        "CLIMAX",
        "DIVERGENCE",
        "EBB",
        "ICE_POINT",
    ]

    for i, dt in enumerate(dates):
        market_phase = phases[
            i % len(phases)
        ]

        for j, symbol in enumerate(names):
            score = (
                0.05
                + 0.01 * j
                + 0.01
                * np.sin(
                    i / 6
                )
            )

            rows.append(
                {
                    "datetime": dt,
                    "symbol": symbol,

                    "final_score": score,

                    "sector_context_score": (
                        0.1
                        * np.sin(
                            i / 9
                        )
                    ),

                    "v7_alpha": score,

                    "transition_cycle_multiplier": 0.78,

                    "v7_7_dynamic_position_multiplier": 0.78,

                    "overlay_no_cycle": 1.0,

                    "cycle_phase": market_phase,

                    "cycle_score": (
                        0.1
                        * np.sin(
                            i / 8
                        )
                    ),

                    "transition": "X→Y",

                    "return_1": (
                        0.001
                        * np.sin(
                            i / 5
                            + j
                        )
                    ),

                    "return_5": (
                        0.005
                        * np.sin(
                            i / 7
                            + j
                        )
                    ),

                    "future_return_1": (
                        0.001
                        * np.sin(
                            (i + 1)
                            / 5
                            + j
                        )
                    ),

                    "future_return_5": (
                        0.005
                        * np.sin(
                            (i + 5)
                            / 7
                            + j
                        )
                    ),

                    "future_return_20": (
                        0.01
                        * np.sin(
                            (i + 20)
                            / 10
                            + j
                        )
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def _market_summary(
    dates: pd.DatetimeIndex,
    count: int = 3000,
) -> pd.DataFrame:
    i = np.arange(
        len(dates)
    )

    return pd.DataFrame(
        {
            "datetime": dates,

            "market_count": count,

            "up_count": (
                count
                * (
                    0.50
                    + 0.04
                    * np.sin(
                        i / 8
                    )
                )
            ).astype(int),

            "down_count": (
                count
                * (
                    0.47
                    - 0.03
                    * np.sin(
                        i / 8
                    )
                )
            ).astype(int),

            "flat_count": int(
                count * 0.03
            ),

            "strong_up_count": (
                count
                * (
                    0.03
                    + 0.005
                    * np.sin(
                        i / 5
                    )
                )
            ).astype(int),

            "strong_down_count": (
                count
                * (
                    0.025
                    - 0.004
                    * np.sin(
                        i / 5
                    )
                )
            ).astype(int),

            "approx_limit_up_count": (
                30
                + (
                    5
                    * np.sin(
                        i / 4
                    )
                ).astype(int)
            ),

            "approx_limit_down_count": (
                10
                + (
                    3
                    * np.cos(
                        i / 4
                    )
                ).astype(int)
            ),

            "limit_up_count": (
                30
                + (
                    5
                    * np.sin(
                        i / 4
                    )
                ).astype(int)
            ),

            "limit_down_count": (
                10
                + (
                    3
                    * np.cos(
                        i / 4
                    )
                ).astype(int)
            ),

            "limit_count_source": (
                "APPROX_FROM_DAILY_RETURN"
            ),

            "market_equal_return": (
                0.002
                * np.sin(
                    i / 7
                )
            ),

            "market_median_return": (
                0.001
                * np.sin(
                    i / 7
                )
            ),

            "cross_section_vol": (
                0.02
                + 0.002
                * np.cos(
                    i / 9
                )
            ),

            "total_amount": (
                8e11
                * (
                    1
                    + 0.1
                    * np.sin(
                        i / 6
                    )
                )
            ),

            "breadth": (
                0.50
                + 0.04
                * np.sin(
                    i / 8
                )
            ),

            "strong_up_rate": (
                0.03
                + 0.005
                * np.sin(
                    i / 5
                )
            ),

            "strong_down_rate": (
                0.025
                - 0.004
                * np.sin(
                    i / 5
                )
            ),
        }
    )


def _write_manifest(
    path: Path,
    rows: list[dict],
) -> None:
    pd.DataFrame(
        rows
    ).to_csv(
        path
        / "data_manifest.csv",
        index=False,
    )


def test_real_market_sentiment_builds_breadth_and_phase() -> None:
    dates = pd.bdate_range(
        "2024-01-02",
        periods=70,
    )

    rows = []

    for i, dt in enumerate(dates):
        for j in range(80):
            rows.append(
                {
                    "datetime": dt,
                    "symbol": (
                        f"{j:06d}.SZ"
                    ),

                    "pct_chg": (
                        0.01
                        * np.sin(
                            i / 7.0
                        )
                        + (
                            j - 40
                        )
                        / 10000.0
                    ),

                    "amount": (
                        1_000_000
                        + j * 1000
                    ),
                }
            )

    result = (
        build_real_market_sentiment(
            pd.DataFrame(
                rows
            ),
            z_window=30,
            min_periods=10,
        )
    )

    assert len(result) == len(dates)

    assert {
        "breadth",
        "real_sentiment_score",
        "real_cycle_phase",
        "market_count",
    }.issubset(
        result.columns
    )

    assert int(
        result[
            "market_count"
        ].median()
    ) == 80

    assert (
        result[
            "real_sentiment_score"
        ]
        .dropna()
        .between(
            -1,
            1,
        )
        .all()
    )


def test_semiconductor_linkage_excludes_focus_from_peers() -> None:
    dates = pd.bdate_range(
        "2024-01-02",
        periods=80,
    )

    symbols = [
        "688981.SH",
        "688001.SH",
        "688002.SH",
        "688003.SH",
        "688004.SH",
        "688005.SH",
    ]

    rows = []

    for i, dt in enumerate(dates):
        for j, symbol in enumerate(symbols):
            rows.append(
                {
                    "datetime": dt,
                    "symbol": symbol,

                    "pct_chg": (
                        0.002
                        * np.sin(
                            i / 5
                        )
                        + 0.0002 * j
                    ),

                    "amount": (
                        1_000_000
                        * (
                            1 + j
                        )
                    ),

                    "volume": (
                        100_000
                        * (
                            1 + j
                        )
                    ),
                }
            )

    result = (
        build_real_semiconductor_linkage(
            pd.DataFrame(
                rows
            ),
            z_window=30,
            z_min_periods=10,
        )
    )

    assert not result.empty

    assert int(
        result[
            "real_sector_peer_count"
        ].median()
    ) == 5

    assert (
        result[
            "real_sector_strength_score"
        ]
        .dropna()
        .between(
            -1,
            1,
        )
        .all()
    )

    assert (
        result[
            "real_sector_survivorship_flag"
        ].all()
    )


def test_recent_news_structurer_never_emits_trade_action() -> None:
    news = pd.DataFrame(
        {
            "发布时间": [
                "2026-08-01 10:00:00",
                "2026-08-02 10:00:00",
            ],

            "新闻标题": [
                "半导体产业政策支持扩产",
                "公司遭调查并提示风险",
            ],

            "新闻内容": [
                "工信部发布产业规划，国产替代加速",
                "监管调查，经营下滑风险增加",
            ],

            "query_keyword": [
                "半导体",
                "688981",
            ],
        }
    )

    result = (
        structure_recent_news(
            news
        )
    )

    assert {
        "event_type",
        "direction",
        "relevance",
        "confidence",
        "structured_event_score_v78",
    }.issubset(
        result.columns
    )

    assert (
        "action"
        not in result.columns
    )

    assert (
        result[
            "structured_event_score_v78"
        ]
        .between(
            -1,
            1,
        )
        .all()
    )

    assert (
        result.iloc[0][
            "direction"
        ]
        > 0
    )

    assert (
        result.iloc[1][
            "direction"
        ]
        < 0
    )


def test_historical_sector_context_has_vendor_independent_schema() -> None:
    dates = pd.bdate_range(
        "2024-01-02",
        periods=80,
    )

    board = pd.DataFrame(
        {
            "datetime": dates,

            "pct_chg": (
                0.002
                * np.sin(
                    np.arange(
                        len(dates)
                    )
                    / 5.0
                )
            ),

            "amount": (
                2_000_000
                + np.arange(
                    len(dates)
                )
                * 10_000
            ),
        }
    )

    focus = pd.DataFrame(
        {
            "datetime": dates,

            "return_1": (
                0.0025
                * np.sin(
                    np.arange(
                        len(dates)
                    )
                    / 5.0
                )
                + 0.0002
            ),
        }
    )

    focus[
        "return_5"
    ] = (
        focus[
            "return_1"
        ]
        .rolling(
            5,
            min_periods=1,
        )
        .sum()
    )

    result = (
        build_historical_sector_context(
            board,
            focus,
            source="test historical source",
            quality=(
                "REAL_HISTORICAL_BOARD_INDEX"
            ),
            z_window=30,
            z_min_periods=10,
        )
    )

    assert (
        len(result)
        == len(dates)
    )

    assert {
        "sector_strength_score",
        "sector_stock_link_score",
        "sector_sync_corr",
        "sector_source",
    }.issubset(
        result.columns
    )

    assert (
        result[
            "sector_survivorship_flag"
        ]
        .eq(False)
        .all()
    )


def test_historical_board_wrapper_keeps_backward_compatible_columns() -> None:
    dates = pd.bdate_range(
        "2024-01-02",
        periods=80,
    )

    board = pd.DataFrame(
        {
            "datetime": dates,
            "pct_chg": 0.002,
            "amount": 2_000_000,
        }
    )

    focus = pd.DataFrame(
        {
            "datetime": dates,
            "return_1": 0.0022,
            "return_5": 0.011,
        }
    )

    result = (
        build_historical_board_linkage(
            board,
            focus,
            z_window=30,
            z_min_periods=10,
        )
    )

    assert {
        "historical_board_strength_score",
        "historical_board_stock_link_score",
        "historical_board_data_quality",
    }.issubset(
        result.columns
    )

    assert (
        result[
            "historical_board_data_quality"
        ]
        .eq(
            "REAL_HISTORICAL_BOARD_INDEX"
        )
        .all()
    )


def test_v78_manifest_persists_across_runs(
    tmp_path,
) -> None:
    first = V78RealDataDownloader(
        tmp_path
    )

    first.manifests.append(
        DatasetManifest(
            dataset=(
                "all_a_market_summary"
            ),
            source="test-a",
            rows=10,
            start_date="2024-01-01",
            end_date="2024-01-10",
            quality="REAL_POINT_IN_TIME",
            survivorship_flag=False,
            notes="a",
        )
    )

    first.write_manifest()

    second = V78RealDataDownloader(
        tmp_path
    )

    second.manifests.append(
        DatasetManifest(
            dataset="focus_news_recent",
            source="test-b",
            rows=20,
            start_date="2026-08-01",
            end_date="2026-08-10",
            quality="REAL_RECENT",
            survivorship_flag=False,
            notes="b",
        )
    )

    result = (
        second.write_manifest()
    )

    assert set(
        result[
            "dataset"
        ]
    ) == {
        "all_a_market_summary",
        "focus_news_recent",
    }


def test_market_gate_and_sector_gate_are_independent(
    tmp_path,
) -> None:
    data_dir = (
        tmp_path
        / "data"
    )

    output_dir = (
        tmp_path
        / "out"
    )

    data_dir.mkdir()

    signals = (
        _frozen_signals()
    )

    dates = pd.DatetimeIndex(
        sorted(
            signals[
                "datetime"
            ].unique()
        )
    )

    _market_summary(
        dates,
        count=3000,
    ).to_csv(
        data_dir
        / "all_a_market_summary.csv",
        index=False,
    )

    _write_manifest(
        data_dir,
        [
            {
                "dataset": (
                    "all_a_market_summary"
                ),
                "source": "test PIT",
                "rows": len(dates),
                "start_date": str(
                    dates.min().date()
                ),
                "end_date": str(
                    dates.max().date()
                ),
                "quality": (
                    "REAL_POINT_IN_TIME"
                ),
                "survivorship_flag": False,
                "notes": "test",
            },
        ],
    )

    _, summary = (
        V78ResearchPipeline(
            _config()
        ).run(
            signals,
            data_dir,
            output_dir,
        )
    )

    assert (
        summary[
            "market_data_gate_passed"
        ]
        is True
    )

    assert (
        summary[
            "sector_data_gate_passed"
        ]
        is False
    )

    assert (
        output_dir
        / "ablation_v7_8_development_only.csv"
    ).is_file()


def test_current_membership_never_enters_primary_backtest(
    tmp_path,
) -> None:
    data_dir = (
        tmp_path
        / "data"
    )

    output_dir = (
        tmp_path
        / "out"
    )

    data_dir.mkdir()

    signals = (
        _frozen_signals()
    )

    dates = pd.DatetimeIndex(
        sorted(
            signals[
                "datetime"
            ].unique()
        )
    )

    _market_summary(
        dates,
        count=3000,
    ).to_csv(
        data_dir
        / "all_a_market_summary.csv",
        index=False,
    )

    peer_rows = []

    peer_symbols = [
        "688981.SH",
        "688001.SH",
        "688002.SH",
        "688003.SH",
        "688004.SH",
        "688005.SH",
    ]

    for dt in dates:
        for (
            j,
            symbol,
        ) in enumerate(
            peer_symbols
        ):
            peer_rows.append(
                {
                    "datetime": dt,
                    "symbol": symbol,
                    "pct_chg": (
                        0.002
                        + j * 0.0001
                    ),
                    "amount": 1e7,
                }
            )

    pd.DataFrame(
        peer_rows
    ).to_csv(
        data_dir
        / "semiconductor_daily.csv",
        index=False,
    )

    _write_manifest(
        data_dir,
        [
            {
                "dataset": (
                    "all_a_market_summary"
                ),
                "source": "test PIT",
                "rows": len(dates),
                "start_date": str(
                    dates.min().date()
                ),
                "end_date": str(
                    dates.max().date()
                ),
                "quality": (
                    "REAL_POINT_IN_TIME"
                ),
                "survivorship_flag": False,
                "notes": "test",
            },

            {
                "dataset": (
                    "semiconductor_daily"
                ),
                "source": (
                    "current members"
                ),
                "rows": len(
                    peer_rows
                ),
                "start_date": str(
                    dates.min().date()
                ),
                "end_date": str(
                    dates.max().date()
                ),
                "quality": (
                    "REAL_PRICE_CURRENT_MEMBERSHIP"
                ),
                "survivorship_flag": True,
                "notes": "biased",
            },
        ],
    )

    result, summary = (
        V78ResearchPipeline(
            _config()
        ).run(
            signals,
            data_dir,
            output_dir,
        )
    )

    assert (
        summary[
            "sector_data_gate_passed"
        ]
        is False
    )

    focus = result.loc[
        result[
            "symbol"
        ].eq(
            "688981.SH"
        )
    ]

    np.testing.assert_allclose(
        pd.to_numeric(
            focus[
                "v7_8_sector_context"
            ],
            errors="coerce",
        ),

        pd.to_numeric(
            focus[
                "sector_context_score"
            ],
            errors="coerce",
        ),

        equal_nan=True,
    )

    assert (
        "v7_8_sector_context_current_membership_diagnostic"
        in result.columns
    )


def test_historical_sector_gate_can_activate_without_current_membership(
    tmp_path,
) -> None:
    data_dir = (
        tmp_path
        / "data"
    )

    output_dir = (
        tmp_path
        / "out"
    )

    data_dir.mkdir()

    signals = (
        _frozen_signals()
    )

    dates = pd.DatetimeIndex(
        sorted(
            signals[
                "datetime"
            ].unique()
        )
    )

    _market_summary(
        dates,
        count=3000,
    ).to_csv(
        data_dir
        / "all_a_market_summary.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "datetime": dates,

            "pct_chg": (
                0.003
                * np.sin(
                    np.arange(
                        len(dates)
                    )
                    / 6
                )
            ),

            "amount": (
                5e9
                * (
                    1
                    + 0.05
                    * np.sin(
                        np.arange(
                            len(dates)
                        )
                        / 8
                    )
                )
            ),
        }
    ).to_csv(
        data_dir
        / "semiconductor_board_daily.csv",
        index=False,
    )

    _write_manifest(
        data_dir,
        [
            {
                "dataset": (
                    "all_a_market_summary"
                ),
                "source": "test PIT",
                "rows": len(dates),
                "start_date": str(
                    dates.min().date()
                ),
                "end_date": str(
                    dates.max().date()
                ),
                "quality": (
                    "REAL_POINT_IN_TIME"
                ),
                "survivorship_flag": False,
                "notes": "test",
            },

            {
                "dataset": (
                    "semiconductor_board_daily"
                ),
                "source": (
                    "historical board"
                ),
                "rows": len(dates),
                "start_date": str(
                    dates.min().date()
                ),
                "end_date": str(
                    dates.max().date()
                ),
                "quality": (
                    "REAL_HISTORICAL_BOARD_INDEX"
                ),
                "survivorship_flag": False,
                "notes": "test",
            },
        ],
    )

    _, summary = (
        V78ResearchPipeline(
            _config()
        ).run(
            signals,
            data_dir,
            output_dir,
        )
    )

    assert (
        summary[
            "market_data_gate_passed"
        ]
        is True
    )

    assert (
        summary[
            "sector_data_gate_passed"
        ]
        is True
    )


def test_v78_pipeline_has_no_veighna_output(
    tmp_path,
) -> None:
    data_dir = (
        tmp_path
        / "data"
    )

    output_dir = (
        tmp_path
        / "out"
    )

    data_dir.mkdir()

    signals = (
        _frozen_signals()
    )

    dates = pd.DatetimeIndex(
        sorted(
            signals[
                "datetime"
            ].unique()
        )
    )

    _market_summary(
        dates,
        count=3000,
    ).to_csv(
        data_dir
        / "all_a_market_summary.csv",
        index=False,
    )

    _write_manifest(
        data_dir,
        [
            {
                "dataset": (
                    "all_a_market_summary"
                ),
                "source": "test PIT",
                "rows": len(dates),
                "start_date": str(
                    dates.min().date()
                ),
                "end_date": str(
                    dates.max().date()
                ),
                "quality": (
                    "REAL_POINT_IN_TIME"
                ),
                "survivorship_flag": False,
                "notes": "test",
            },
        ],
    )

    V78ResearchPipeline(
        _config()
    ).run(
        signals,
        data_dir,
        output_dir,
    )

    assert not (
        output_dir
        / "veighna_order_intents_v7_8.csv"
    ).exists()


# ============================================================
# V7.8 FINAL FREEZE TESTS
# ============================================================


def test_v78_final_variant_is_frozen_to_market_return_only() -> None:
    """
    Development selection is frozen.

    2025 robustness results must not be allowed to change it.
    """

    assert (
        FINAL_V78_VARIANT
        == "v7_8_market_return_only"
    )


def test_v78_final_uses_selected_market_return_overlay(
    tmp_path,
) -> None:
    """
    When the real full-A market gate passes:

        V7.8 Final Score
            =
        frozen V7.7 alpha

    and:

        V7.8 Final Position Overlay
            =
        market-return-only overlay
    """

    data_dir = (
        tmp_path
        / "data"
    )

    output_dir = (
        tmp_path
        / "out"
    )

    data_dir.mkdir()

    signals = (
        _frozen_signals(
            periods=90,
        )
    )

    dates = pd.DatetimeIndex(
        sorted(
            signals[
                "datetime"
            ].unique()
        )
    )

    _market_summary(
        dates,
        count=3000,
    ).to_csv(
        data_dir
        / "all_a_market_summary.csv",
        index=False,
    )

    _write_manifest(
        data_dir,
        [
            {
                "dataset": (
                    "all_a_market_summary"
                ),
                "source": "test PIT",
                "rows": len(dates),
                "start_date": str(
                    dates.min().date()
                ),
                "end_date": str(
                    dates.max().date()
                ),
                "quality": (
                    "REAL_POINT_IN_TIME"
                ),
                "survivorship_flag": False,
                "notes": "test",
            },
        ],
    )

    result, summary = (
        V78ResearchPipeline(
            _config()
        ).run(
            signals,
            data_dir,
            output_dir,
        )
    )

    assert (
        summary[
            "selected_variant"
        ]
        == "v7_8_market_return_only"
    )

    assert (
        summary[
            "selected_score_col"
        ]
        == "v7_alpha"
    )

    assert (
        summary[
            "selected_overlay_col"
        ]
        == "v7_8_overlay_return"
    )

    assert (
        summary[
            "selection_period"
        ]
        == "development_2020_2024"
    )

    assert (
        summary[
            "champion"
        ]
        == "v7_7_frozen"
    )

    assert (
        summary[
            "challenger"
        ]
        == "v7_8_market_return_only"
    )

    # Final dynamic position must exactly equal
    # the development-selected return overlay.
    np.testing.assert_allclose(
        pd.to_numeric(
            result[
                "v7_8_dynamic_position_multiplier"
            ],
            errors="coerce",
        ),

        pd.to_numeric(
            result[
                "v7_8_overlay_return"
            ],
            errors="coerce",
        ),

        equal_nan=True,
    )

    # Final stock ranking remains frozen V7.7 alpha.
    np.testing.assert_allclose(
        pd.to_numeric(
            result[
                "v7_8_final_score"
            ],
            errors="coerce",
        ),

        pd.to_numeric(
            result[
                "v7_alpha"
            ],
            errors="coerce",
        ),

        equal_nan=True,
    )

    assert (
        result[
            "v7_8_selected_overlay"
        ]
        .eq(
            "v7_8_market_return_only"
        )
        .all()
    )


def test_v78_final_does_not_require_sector_or_event_gate(
    tmp_path,
) -> None:
    """
    The valid production-research state for the current project is:

        market_gate = True
        sector_gate = False
        event_gate = False

    V7.8 Final must remain fully runnable in this state.
    """

    data_dir = (
        tmp_path
        / "data"
    )

    output_dir = (
        tmp_path
        / "out"
    )

    data_dir.mkdir()

    signals = (
        _frozen_signals(
            periods=90,
        )
    )

    dates = pd.DatetimeIndex(
        sorted(
            signals[
                "datetime"
            ].unique()
        )
    )

    _market_summary(
        dates,
        count=3000,
    ).to_csv(
        data_dir
        / "all_a_market_summary.csv",
        index=False,
    )

    _write_manifest(
        data_dir,
        [
            {
                "dataset": (
                    "all_a_market_summary"
                ),
                "source": "test PIT",
                "rows": len(dates),
                "start_date": str(
                    dates.min().date()
                ),
                "end_date": str(
                    dates.max().date()
                ),
                "quality": (
                    "REAL_POINT_IN_TIME"
                ),
                "survivorship_flag": False,
                "notes": "test",
            },
        ],
    )

    result, summary = (
        V78ResearchPipeline(
            _config()
        ).run(
            signals,
            data_dir,
            output_dir,
        )
    )

    assert (
        summary[
            "market_data_gate_passed"
        ]
        is True
    )

    assert (
        summary[
            "sector_data_gate_passed"
        ]
        is False
    )

    assert (
        summary[
            "event_data_gate_passed"
        ]
        is False
    )

    assert (
        summary[
            "selected_variant"
        ]
        == "v7_8_market_return_only"
    )

    assert not result.empty

    assert (
        output_dir
        / "V7_8_REPORT.md"
    ).is_file()