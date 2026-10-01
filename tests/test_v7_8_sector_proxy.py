from pathlib import Path

import numpy as np
import pandas as pd

from src.data.v78_sector_proxy_baostock import (
    BaoStockSectorProxyDownloader,
    SECTOR_PROXY_QUALITY,
)
from src.v7.sector_linkage_v78 import build_historical_board_linkage


def _raw(n: int = 260) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=n)
    close = 1.0 + np.arange(n) * 0.001
    pre = np.r_[np.nan, close[:-1]]
    pct = (close / pre - 1.0) * 100.0
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "code": "sh.512480",
            "open": close - 0.001,
            "high": close + 0.002,
            "low": close - 0.002,
            "close": close,
            "preclose": pre,
            "volume": 1_000_000 + np.arange(n),
            "amount": 2_000_000 + np.arange(n) * 1000,
            "pctChg": pct,
        }
    )


def test_baostock_sector_proxy_normalization() -> None:
    out = BaoStockSectorProxyDownloader.normalize_raw(_raw())
    assert len(out) == 260
    assert out["proxy_symbol"].eq("512480.SH").all()
    assert out["tracked_index"].eq("H30184.CSI").all()
    assert out["sector_quality"].eq(SECTOR_PROXY_QUALITY).all()
    assert out["sector_survivorship_flag"].eq(False).all()
    assert out["point_in_time"].eq(True).all()
    assert out["pct_chg"].dropna().abs().max() < 0.01


def test_sector_proxy_validation() -> None:
    out = BaoStockSectorProxyDownloader.normalize_raw(_raw())
    summary = BaoStockSectorProxyDownloader.validate(out)
    assert summary.rows == 260
    assert summary.quality == SECTOR_PROXY_QUALITY
    assert summary.duplicate_dates == 0
    assert summary.positive_price_share == 1.0


def test_historical_linkage_preserves_proxy_metadata() -> None:
    board = BaoStockSectorProxyDownloader.normalize_raw(_raw())
    dates = board["datetime"]
    focus_ret = 0.0015 * np.sin(np.arange(len(dates)) / 8.0)
    focus = pd.DataFrame(
        {
            "datetime": dates,
            "return_1": focus_ret,
            "return_5": pd.Series(focus_ret).rolling(5, min_periods=1).sum().to_numpy(),
        }
    )
    out = build_historical_board_linkage(
        board,
        focus,
        z_window=60,
        z_min_periods=20,
    )
    assert not out.empty
    assert out["historical_board_data_quality"].eq(SECTOR_PROXY_QUALITY).all()
    assert out["historical_board_source"].str.contains("512480.SH").all()
    assert out["historical_board_survivorship_flag"].eq(False).all()
    assert out["historical_board_strength_score"].between(-1, 1).all()
    assert out["historical_board_stock_link_score"].between(-1, 1).all()
