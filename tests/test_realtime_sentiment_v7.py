from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.realtime_market_snapshot_v7 import (
    market_session,
    summarize_spot_market,
    summarize_tushare_daily,
)
from src.v7.realtime_sentiment_v7 import build_realtime_state


def test_spot_summary_converts_percent_and_amount_units():
    spot = pd.DataFrame({
        "代码": ["600000", "688981", "000001"],
        "最新价": [10.0, 50.0, 12.0],
        "涨跌幅": [1.0, -2.0, 0.0],
        "成交额": [1_000_000.0, 2_000_000.0, 3_000_000.0],
    })
    out = summarize_spot_market(spot, pd.Timestamp("2026-08-19"))
    assert len(out) == 1
    assert out.loc[0, "market_count"] == 3
    assert np.isclose(out.loc[0, "market_equal_return"], (-0.01) / 3.0)
    # Yuan -> thousand yuan to stay compatible with Tushare summary units.
    assert np.isclose(out.loc[0, "total_amount"], 6000.0)


def test_tushare_summary_units_are_preserved():
    raw = pd.DataFrame({
        "ts_code": ["600000.SH", "688981.SH"],
        "pct_chg": [1.0, -2.0],
        "amount": [1000.0, 2000.0],
    })
    out = summarize_tushare_daily(raw, "20260818")
    assert out.loc[0, "market_count"] == 2
    assert np.isclose(out.loc[0, "total_amount"], 3000.0)


def test_session_detection():
    assert market_session(pd.Timestamp("2026-08-19 08:40", tz="Asia/Shanghai")) == "PREOPEN"
    assert market_session(pd.Timestamp("2026-08-19 10:00", tz="Asia/Shanghai")) == "MORNING_SESSION"
    assert market_session(pd.Timestamp("2026-08-19 12:00", tz="Asia/Shanghai")) == "LUNCH_BREAK"
    assert market_session(pd.Timestamp("2026-08-19 14:00", tz="Asia/Shanghai")) == "AFTERNOON_SESSION"


def test_realtime_state_builds_position_multiplier():
    dates = pd.bdate_range("2024-01-02", periods=520)
    rng = np.random.default_rng(7)
    rets = rng.normal(0.0002, 0.01, size=len(dates))
    hist = pd.DataFrame({
        "datetime": dates,
        "market_count": 5000,
        "up_count": 2600,
        "down_count": 2300,
        "flat_count": 100,
        "strong_up_count": 100,
        "strong_down_count": 80,
        "approx_limit_up_count": 20,
        "approx_limit_down_count": 8,
        "limit_up_count": 20,
        "limit_down_count": 8,
        "limit_count_source": "APPROX_FROM_DAILY_RETURN",
        "market_equal_return": rets,
        "market_median_return": rets * 0.8,
        "cross_section_vol": 0.018 + np.abs(rng.normal(0, 0.002, size=len(dates))),
        "total_amount": 1e9 + rng.normal(0, 5e7, size=len(dates)),
        "breadth": np.clip(0.5 + rets * 8, 0.05, 0.95),
        "strong_up_rate": 0.02,
        "strong_down_rate": 0.015,
    })
    sig = pd.DataFrame({
        "datetime": dates.repeat(3),
        "symbol": ["A", "B", "C"] * len(dates),
        "future_return_5": rng.normal(0.001, 0.02, size=len(dates) * 3),
    })
    sector = pd.DataFrame({
        "datetime": dates,
        "pct_chg": rng.normal(0.0003, 0.012, size=len(dates)),
        "volume": 1e8 + rng.normal(0, 1e7, size=len(dates)),
        "amount": np.nan,
        "sector_source": "TEST",
        "sector_quality": "REAL_HISTORICAL_SECTOR_ETF_PROXY",
    })
    focus = pd.DataFrame({
        "datetime": dates,
        "return_1": rng.normal(0.0004, 0.015, size=len(dates)),
    })
    focus["return_5"] = focus["return_1"].rolling(5, min_periods=3).sum()
    config = {
        "v7_7": {"transition": {"base_exposure": 0.78, "min_exposure": 0.55, "max_exposure": 1.0}},
        "v7_8": {
            "market_sentiment": {"z_window": 120, "z_min_periods": 40},
            "sector": {"corr_window": 20, "z_window": 120, "z_min_periods": 40},
            "gates": {"min_market_count": 1500},
        },
    }
    state = build_realtime_state(
        historical_market_summary=hist,
        incremental_market_summary=pd.DataFrame(),
        live_market_summary=pd.DataFrame(),
        historical_signals=sig,
        sector_history=sector,
        focus_history=focus,
        sector_live=pd.DataFrame(),
        focus_live=pd.DataFrame(),
        config=config,
        as_of=pd.Timestamp(dates[-1]).tz_localize("Asia/Shanghai"),
        session="POST_CLOSE",
        live_requested=False,
    )
    mult = state["dynamic_position_management"]["realtime_cycle_position_multiplier"]
    assert state["selected_variant"] == "v7_8_market_return_only"
    assert 0.55 <= mult <= 1.0
    assert state["market_count"] == 5000
    assert state["sector"]["gate"] is True


def test_tushare_query_fallback_accepts_second_call_form():
    from src.data.realtime_market_snapshot_v7 import RealtimeMarketDataClient

    class FakePro:
        def daily(self, **kwargs):
            if "trade_date" in kwargs:
                return pd.DataFrame()
            return pd.DataFrame({
                "ts_code": ["600000.SH", "000001.SZ"],
                "pct_chg": [1.0, -0.5],
                "amount": [1000.0, 2000.0],
            })

        def query(self, *args, **kwargs):
            return pd.DataFrame()

    raw, method, errors = RealtimeMarketDataClient._query_tushare_daily_cross_section(FakePro(), "20260818")
    assert len(raw) == 2
    assert method == "daily(start_date=end_date)"
    assert errors == ["daily(trade_date): EMPTY"]


def test_tushare_query_fallback_reports_all_empty():
    from src.data.realtime_market_snapshot_v7 import RealtimeMarketDataClient

    class FakePro:
        def daily(self, **kwargs):
            return pd.DataFrame()

        def query(self, *args, **kwargs):
            return pd.DataFrame()

    raw, method, errors = RealtimeMarketDataClient._query_tushare_daily_cross_section(FakePro(), "20260818")
    assert raw.empty
    assert method == "NONE"
    assert len(errors) == 3
    assert all("EMPTY" in x for x in errors)


def test_normalize_market_index_history_percent_units():
    from src.data.realtime_market_snapshot_v7 import normalize_market_index_history

    raw = pd.DataFrame({
        "日期": ["2025-12-30", "2025-12-31"],
        "收盘": [100.0, 101.0],
        "涨跌幅": [0.5, 1.0],
    })
    out = normalize_market_index_history(raw, symbol="000985")
    assert len(out) == 2
    assert np.isclose(out.loc[0, "index_return"], 0.005)
    assert np.isclose(out.loc[1, "index_return"], 0.01)
    assert out.loc[0, "proxy_index_quality"] == "REAL_HISTORICAL_BROAD_INDEX"


def test_calibrated_index_proxy_only_extends_after_frozen_history():
    from src.data.realtime_market_snapshot_v7 import calibrate_broad_index_return_proxy

    dates = pd.bdate_range("2024-01-02", periods=420)
    rng = np.random.default_rng(42)
    idx_ret = rng.normal(0.0001, 0.01, len(dates) + 5)
    # Full-A equal return is deliberately a linear but not identical mapping.
    base_ret = 0.0002 + 0.82 * idx_ret[: len(dates)] + rng.normal(0, 0.001, len(dates))
    base = pd.DataFrame({"datetime": dates, "market_equal_return": base_ret})
    future_dates = dates.append(pd.bdate_range(dates[-1] + pd.Timedelta(days=1), periods=5))
    idx = pd.DataFrame({
        "datetime": future_dates,
        "index_return": idx_ret,
        "proxy_index_symbol": "000985",
        "proxy_index_source": "TEST_INDEX",
    })
    proxy, meta = calibrate_broad_index_return_proxy(base, idx, min_overlap=250)
    assert len(proxy) == 5
    assert pd.to_datetime(proxy["datetime"]).min() > dates[-1]
    assert proxy["snapshot_quality"].eq("REAL_EOD_BROAD_INDEX_RETURN_PROXY").all()
    assert int(meta["calibration_dates"]) == len(dates)
    assert float(meta["correlation"]) > 0.9
    assert 0.6 < float(meta["beta"]) < 1.0


def test_proxy_latest_makes_frozen_overlay_usable_but_full_sentiment_diagnostic_stale():
    dates = pd.bdate_range("2024-01-02", periods=520)
    rng = np.random.default_rng(17)
    rets = rng.normal(0.0002, 0.01, size=len(dates))
    hist = pd.DataFrame({
        "datetime": dates,
        "market_count": 5000,
        "up_count": 2600,
        "down_count": 2300,
        "flat_count": 100,
        "strong_up_count": 100,
        "strong_down_count": 80,
        "approx_limit_up_count": 20,
        "approx_limit_down_count": 8,
        "limit_up_count": 20,
        "limit_down_count": 8,
        "limit_count_source": "APPROX_FROM_DAILY_RETURN",
        "market_equal_return": rets,
        "market_median_return": rets * 0.8,
        "cross_section_vol": 0.018,
        "total_amount": 1e9,
        "breadth": 0.52,
        "strong_up_rate": 0.02,
        "strong_down_rate": 0.015,
        "snapshot_quality": "REAL_POINT_IN_TIME",
    })
    proxy_date = dates[-1] + pd.offsets.BDay(1)
    proxy = pd.DataFrame([{
        "datetime": proxy_date,
        "market_equal_return": 0.006,
        "market_count": np.nan,
        "total_amount": np.nan,
        "snapshot_source": "TEST BROAD INDEX",
        "snapshot_quality": "REAL_EOD_BROAD_INDEX_RETURN_PROXY",
        "proxy_index_symbol": "000985",
        "proxy_calibration_corr": 0.95,
        "proxy_calibration_r2": 0.90,
        "proxy_calibration_dates": 500,
        "proxy_calibration_cutoff": dates[-1].strftime("%Y-%m-%d"),
    }])
    sig = pd.DataFrame({
        "datetime": dates.repeat(2),
        "symbol": ["A", "B"] * len(dates),
        "future_return_5": rng.normal(0.001, 0.02, size=len(dates) * 2),
    })
    sector = pd.DataFrame({
        "datetime": dates,
        "pct_chg": rng.normal(0.0003, 0.012, size=len(dates)),
        "volume": 1e8,
        "amount": np.nan,
        "sector_source": "TEST",
        "sector_quality": "REAL_HISTORICAL_SECTOR_ETF_PROXY",
    })
    focus = pd.DataFrame({"datetime": dates, "return_1": rng.normal(0.0004, 0.015, size=len(dates))})
    focus["return_5"] = focus["return_1"].rolling(5, min_periods=3).sum()
    config = {
        "v7_7": {"transition": {"base_exposure": 0.78, "min_exposure": 0.55, "max_exposure": 1.0}},
        "v7_8": {
            "market_sentiment": {"z_window": 120, "z_min_periods": 40},
            "sector": {"corr_window": 20, "z_window": 120, "z_min_periods": 40},
            "gates": {"min_market_count": 1500},
        },
    }
    state = build_realtime_state(
        historical_market_summary=hist,
        incremental_market_summary=proxy,
        live_market_summary=pd.DataFrame(),
        historical_signals=sig,
        sector_history=sector,
        focus_history=focus,
        sector_live=pd.DataFrame(),
        focus_live=pd.DataFrame(),
        config=config,
        as_of=pd.Timestamp(proxy_date).tz_localize("Asia/Shanghai"),
        session="PREOPEN",
        live_requested=False,
    )
    assert state["market_gate"] is True
    assert state["overlay_gate"] is True
    assert state["full_market_diagnostic_gate"] is True  # only one business day stale in this synthetic test
    assert state["runtime_quality"] == "LATEST_EOD_BROAD_INDEX_RETURN_PROXY"
    assert state["frozen_market_return_overlay"]["proxy_applied"] is True
    assert state["frozen_market_return_overlay"]["proxy_index_symbol"] == "000985"
