from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.v79_intraday_downloader import (
    DEMO_5MIN_QUALITY,
    REAL_5MIN_QUALITY,
    IntradayManifest,
    V79IntradayDataManager,
    BaoStockV79MinuteDownloader,
)
from src.v7.intraday_backtest_v79 import build_focus_daily_targets, simulate_focus_strategy
from src.v7.intraday_confirmation_v79 import build_confirmation_score
from src.v7.intraday_features_v79 import build_intraday_features
from src.v7.research_pipeline_v79 import V79ResearchPipeline


def _config() -> dict:
    return {
        "v7": {
            "backtest": {"top_k": 3, "min_score": 0.05, "max_gross": 1.0},
        },
        "v7_8": {"focus_symbol": "688981.SH"},
        "v7_9": {
            "focus_symbol": "688981.SH",
            "gates": {
                "accepted_intraday_quality": [REAL_5MIN_QUALITY, "USER_PROVIDED_5MIN"],
                "min_daily_signal_coverage": 0.8,
                "min_median_bars_per_day": 40,
            },
            "daily_target": {
                "score_col": "v7_8_final_score",
                "overlay_col": "v7_8_dynamic_position_multiplier",
                "top_k": 3,
                "min_score": 0.05,
                "max_gross": 1.0,
            },
            "features": {
                "rvol_lookback_days": 5,
                "rvol_min_days": 2,
                "opening_range_bars": 3,
                "volatility_window_bars": 6,
                "volatility_min_periods": 3,
                "volatility_z_window_bars": 20,
                "volatility_z_min_periods": 6,
                "atr_window_bars": 6,
                "atr_min_periods": 3,
            },
            "confirmation": {
                "weights": {
                    "vwap": 0.25,
                    "momentum_5": 0.15,
                    "momentum_15": 0.20,
                    "rvol": 0.15,
                    "breakout": 0.15,
                    "volatility": 0.10,
                }
            },
            "backtest": {
                "initial_capital": 1_000_000,
                "buy_cost_rate": 0.0006,
                "sell_cost_rate": 0.0011,
                "enter_threshold": 0.1,
                "reduce_threshold": -0.05,
                "exit_threshold": -0.3,
                "rebalance_band": 0.005,
                "earliest_entry_bar": 0,
                "latest_entry_bar": 44,
            },
        },
    }


def _daily_signals(dates: pd.DatetimeIndex) -> pd.DataFrame:
    symbols = ["688981.SH", "600001.SH", "600002.SH", "600003.SH"]
    rows = []
    for i, dt in enumerate(dates):
        for j, symbol in enumerate(symbols):
            score = 0.30 - 0.04 * j + 0.01 * np.sin(i / 3)
            rows.append(
                {
                    "datetime": dt,
                    "symbol": symbol,
                    "v7_8_final_score": score,
                    "v7_alpha": score,
                    "v7_8_dynamic_position_multiplier": 0.8,
                    "v7_7_dynamic_position_multiplier": 0.8,
                }
            )
    return pd.DataFrame(rows)


def _bars(dates: pd.DatetimeIndex, bars_per_day: int = 48) -> pd.DataFrame:
    rows = []
    for di, day in enumerate(dates):
        times = list(pd.date_range(day + pd.Timedelta(hours=9, minutes=35), periods=24, freq="5min"))
        times += list(pd.date_range(day + pd.Timedelta(hours=13, minutes=5), periods=bars_per_day - 24, freq="5min"))
        prev = 50.0 + di * 0.2
        for i, dt in enumerate(times):
            ret = 0.0006 * np.sin((i + di) / 4) + 0.00015
            close = prev * (1 + ret)
            high = max(prev, close) * 1.0005
            low = min(prev, close) * 0.9995
            volume = 100_000 * (1.0 + abs(i - bars_per_day / 2) / bars_per_day)
            rows.append(
                {
                    "datetime": dt,
                    "symbol": "688981.SH",
                    "open": prev,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "amount": volume * (prev + close) / 2,
                }
            )
            prev = close
    return pd.DataFrame(rows)


def test_features_are_bounded_and_opening_range_is_causal() -> None:
    dates = pd.bdate_range("2024-01-02", periods=8)
    features = build_intraday_features(_bars(dates), _config()["v7_9"]["features"])
    for col in [
        "vwap_strength_score",
        "momentum_5_score",
        "momentum_15_score",
        "rvol_score",
        "breakout_score",
        "volatility_control_score",
    ]:
        assert features[col].fillna(0).between(-1, 1).all()
    early = features.loc[features["bar_index"] < 3, "breakout_score"]
    assert early.eq(0.0).all()


def test_rvol_does_not_use_future_days() -> None:
    dates = pd.bdate_range("2024-01-02", periods=8)
    bars = _bars(dates)
    cfg = _config()["v7_9"]["features"]
    first = build_intraday_features(bars, cfg)
    changed = bars.copy()
    last_day = changed["datetime"].dt.normalize().max()
    changed.loc[changed["datetime"].dt.normalize().eq(last_day), "volume"] *= 100.0
    second = build_intraday_features(changed, cfg)
    prior = first["trade_date"] < last_day
    np.testing.assert_allclose(
        first.loc[prior, "expected_slot_volume"].fillna(-1).to_numpy(),
        second.loc[prior, "expected_slot_volume"].fillna(-1).to_numpy(),
    )


def test_daily_target_uses_next_session_not_same_day() -> None:
    dates = pd.bdate_range("2024-01-02", periods=6)
    signals = _daily_signals(dates[:-1])
    targets = build_focus_daily_targets(signals, pd.Index(dates), "688981.SH", _config()["v7_9"]["daily_target"])
    first_nonzero = targets.loc[targets["target_weight"] > 0].iloc[0]
    assert pd.Timestamp(first_nonzero["trade_date"]) > pd.Timestamp(first_nonzero["signal_date"])


def test_confirmation_executes_on_next_bar_open() -> None:
    dates = pd.bdate_range("2024-01-02", periods=5)
    bars = build_intraday_features(_bars(dates), _config()["v7_9"]["features"])
    scored = build_confirmation_score(bars, _config()["v7_9"]["confirmation"])
    # Force a strong positive score so an entry is generated immediately.
    scored["confirmation_score"] = 1.0
    targets = pd.DataFrame(
        {
            "trade_date": dates,
            "signal_date": dates - pd.offsets.BDay(1),
            "target_weight": 0.2,
        }
    )
    result = simulate_focus_strategy(scored, targets, _config()["v7_9"]["backtest"], mode="confirmation")
    buys = result.trades.loc[result.trades["side"].eq("BUY")]
    assert not buys.empty
    assert (pd.to_datetime(buys["datetime"]) > pd.to_datetime(buys["signal_datetime"])).all()


def test_t_plus_one_blocks_same_day_sale_of_new_purchase() -> None:
    day = pd.Timestamp("2024-01-02")
    bars = _bars(pd.DatetimeIndex([day]))
    # Create a dramatic price increase after entry so position weight exceeds target; negative
    # confirmation would normally ask for a reduction, but today's purchased shares are unsellable.
    bars.loc[bars.index[5]:, ["open", "high", "low", "close"]] *= 2.0
    features = build_intraday_features(bars, _config()["v7_9"]["features"])
    features["confirmation_score"] = 1.0
    features.loc[features["bar_index"] >= 6, "confirmation_score"] = -1.0
    targets = pd.DataFrame({"trade_date": [day], "signal_date": [day - pd.offsets.BDay(1)], "target_weight": [0.2]})
    result = simulate_focus_strategy(features, targets, _config()["v7_9"]["backtest"], mode="confirmation")
    assert (result.trades["side"] == "BUY").any()
    assert not (result.trades["side"] == "SELL").any()


def test_demo_quality_fails_research_gate(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "out"
    manager = V79IntradayDataManager(data_dir)
    dates = pd.bdate_range("2024-01-02", periods=8)
    bars = _bars(dates)
    manager.save(
        bars,
        IntradayManifest(
            symbol="688981.SH",
            source="demo",
            quality=DEMO_5MIN_QUALITY,
            rows=len(bars),
            start_datetime=str(bars["datetime"].min()),
            end_datetime=str(bars["datetime"].max()),
            point_in_time=False,
        ),
    )
    _, summary = V79ResearchPipeline(_config()).run(_daily_signals(dates), data_dir, output_dir)
    assert summary["intraday_data_gate_passed"] is False
    assert not (output_dir / "veighna_order_intents_v7_9.csv").exists()


def test_real_quality_can_pass_gate(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "out"
    manager = V79IntradayDataManager(data_dir)
    dates = pd.bdate_range("2024-01-02", periods=10)
    bars = _bars(dates)
    manager.save(
        bars,
        IntradayManifest(
            symbol="688981.SH",
            source="real-test",
            quality=REAL_5MIN_QUALITY,
            rows=len(bars),
            start_datetime=str(bars["datetime"].min()),
            end_datetime=str(bars["datetime"].max()),
            point_in_time=True,
        ),
    )
    _, summary = V79ResearchPipeline(_config()).run(_daily_signals(dates), data_dir, output_dir)
    assert summary["intraday_data_gate_passed"] is True
    assert (output_dir / "ablation_v7_9.csv").is_file()
    assert (output_dir / "V7_9_REPORT.md").is_file()


def test_baostock_symbol_mapping_and_normalization() -> None:
    assert BaoStockV79MinuteDownloader._to_baostock_code("688981.SH") == "sh.688981"
    assert BaoStockV79MinuteDownloader._to_baostock_code("000001.SZ") == "sz.000001"

    raw = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02"],
            "time": ["20240102093500000", "20240102094000000"],
            "code": ["sh.688981", "sh.688981"],
            "open": ["53.0200", "52.7900"],
            "high": ["53.0600", "53.0800"],
            "low": ["52.7000", "52.7600"],
            "close": ["52.7700", "53.0300"],
            "volume": ["866909", "445708"],
            "amount": ["45837172.0000", "23595588.0000"],
            "adjustflag": ["3", "3"],
        }
    )

    result = BaoStockV79MinuteDownloader._normalize_raw(raw, "688981.SH")

    assert list(result["symbol"].unique()) == ["688981.SH"]
    assert result.iloc[0]["datetime"] == pd.Timestamp("2024-01-02 09:35:00")
    assert result.iloc[1]["datetime"] == pd.Timestamp("2024-01-02 09:40:00")
    assert float(result.iloc[0]["close"]) == 52.77
    assert float(result.iloc[0]["volume"]) == 866909.0


def test_entry_confirmation_only_daily_exit_executes_at_first_bar_open() -> None:
    """Daily V7.8 target reduction must not wait for a negative 5min signal."""

    dates = pd.bdate_range("2024-01-02", periods=2)
    bars = build_intraday_features(
        _bars(dates),
        _config()["v7_9"]["features"],
    )

    # Day 1: strong confirmation -> enter.
    # Day 2: confirmation remains strongly positive, but daily target goes to zero.
    # Entry-only mode must still exit at the first bar open because V7.8 owns exits.
    bars["confirmation_score"] = 1.0

    targets = pd.DataFrame(
        {
            "trade_date": dates,
            "signal_date": dates - pd.offsets.BDay(1),
            "target_weight": [0.2, 0.0],
        }
    )

    result = simulate_focus_strategy(
        bars,
        targets,
        _config()["v7_9"]["backtest"],
        mode="entry_confirmation_only",
    )

    assert (result.trades["side"] == "BUY").any()

    sells = result.trades.loc[result.trades["side"].eq("SELL")].copy()
    assert not sells.empty
    assert sells.iloc[-1]["action"] == "DAILY_EXIT"

    second_day_first_bar = (
        bars.loc[bars["trade_date"].eq(dates[1]), "datetime"]
        .sort_values()
        .iloc[0]
    )

    assert pd.Timestamp(sells.iloc[-1]["datetime"]) == pd.Timestamp(second_day_first_bar)


def test_entry_confirmation_only_never_uses_intraday_reduce_or_exit() -> None:
    """Negative 5min scores cannot override a still-positive frozen daily target."""

    dates = pd.bdate_range("2024-01-02", periods=3)
    bars = build_intraday_features(
        _bars(dates),
        _config()["v7_9"]["features"],
    )

    # Enter on day 1, then make all subsequent intraday confirmation extremely weak.
    bars["confirmation_score"] = -1.0
    first_day = bars["trade_date"].min()
    bars.loc[bars["trade_date"].eq(first_day), "confirmation_score"] = 1.0

    targets = pd.DataFrame(
        {
            "trade_date": dates,
            "signal_date": dates - pd.offsets.BDay(1),
            "target_weight": [0.2, 0.2, 0.2],
        }
    )

    result = simulate_focus_strategy(
        bars,
        targets,
        _config()["v7_9"]["backtest"],
        mode="entry_confirmation_only",
    )

    assert (result.trades["side"] == "BUY").any()

    if "action" in result.trades.columns:
        assert not result.trades["action"].isin(["REDUCE", "EXIT"]).any()

    assert int(result.metrics.get("intraday_reduce_count", 0)) == 0
    assert int(result.metrics.get("intraday_exit_count", 0)) == 0


def test_entry_confirmation_only_still_uses_next_bar_for_buy() -> None:
    """The asymmetric structure must preserve next-bar execution for entries."""

    dates = pd.bdate_range("2024-01-02", periods=2)
    bars = build_intraday_features(
        _bars(dates),
        _config()["v7_9"]["features"],
    )
    bars["confirmation_score"] = 1.0

    targets = pd.DataFrame(
        {
            "trade_date": dates,
            "signal_date": dates - pd.offsets.BDay(1),
            "target_weight": [0.2, 0.2],
        }
    )

    result = simulate_focus_strategy(
        bars,
        targets,
        _config()["v7_9"]["backtest"],
        mode="entry_confirmation_only",
    )

    buys = result.trades.loc[result.trades["side"].eq("BUY")]
    assert not buys.empty

    confirmed_buys = buys.loc[buys["signal_datetime"].notna()]
    assert not confirmed_buys.empty

    assert (
        pd.to_datetime(confirmed_buys["datetime"])
        > pd.to_datetime(confirmed_buys["signal_datetime"])
    ).all()


def test_pipeline_exports_structural_v79_variants(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "out"

    manager = V79IntradayDataManager(data_dir)
    dates = pd.bdate_range("2024-01-02", periods=12)
    bars = _bars(dates)

    manager.save(
        bars,
        IntradayManifest(
            symbol="688981.SH",
            source="real-test",
            quality=REAL_5MIN_QUALITY,
            rows=len(bars),
            start_datetime=str(bars["datetime"].min()),
            end_datetime=str(bars["datetime"].max()),
            point_in_time=True,
        ),
    )

    _, summary = V79ResearchPipeline(_config()).run(
        _daily_signals(dates),
        data_dir,
        output_dir,
    )

    ablation = pd.read_csv(output_dir / "ablation_v7_9.csv")

    assert "v7_9_entry_confirmation_only" in set(ablation["variant"])
    assert "v7_9_entry_confirmation_no_momentum" in set(ablation["variant"])

    assert "metrics_v7_9_entry_confirmation_only" in summary
    assert "metrics_v7_9_entry_confirmation_no_momentum" in summary

    assert (output_dir / "trades_v7_9_entry_confirmation_only.csv").is_file()
    assert (output_dir / "ablation_v7_9_development_only.csv").is_file()
