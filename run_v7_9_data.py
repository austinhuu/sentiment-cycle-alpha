from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT
from src.data.v79_intraday_downloader import (
    DEMO_5MIN_QUALITY,
    IntradayManifest,
    TushareV79MinuteDownloader,
    BaoStockV79MinuteDownloader,
    V79IntradayDataManager,
    generate_smoke_5min_from_daily,
)


def _load_daily() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "raw" / "all_stocks_daily.csv"
    if not path.is_file():
        raise FileNotFoundError(f"缺少日线文件: {path}")
    d = pd.read_csv(path, low_memory=False)
    d["datetime"] = pd.to_datetime(d["datetime"], errors="coerce")
    return d


def main() -> None:
    parser = argparse.ArgumentParser(description="V7.9 5-minute data preparation")
    parser.add_argument("--source", choices=["tushare", "baostock", "csv", "demo"], required=True)
    parser.add_argument("--symbol", default="688981.SH")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--csv", default="")
    parser.add_argument("--pause", type=float, default=1.2)
    parser.add_argument("--chunk-days", type=int, default=90)
    parser.add_argument("--baostock-adjustflag", choices=["1", "2", "3"], default="3")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / "data" / "v7_9"
    manager = V79IntradayDataManager(data_dir)
    symbol = args.symbol.upper()

    if args.source == "tushare":
        downloader = TushareV79MinuteDownloader(
            manager,
            pause_seconds=args.pause,
            chunk_days=args.chunk_days,
        )
        path = downloader.download_symbol(symbol, args.start, args.end)
        print(f"[V7.9 data] real 5min saved: {path}")

    elif args.source == "baostock":
        downloader = BaoStockV79MinuteDownloader(
            manager,
            pause_seconds=args.pause,
            chunk_days=max(args.chunk_days, 180),
            adjustflag=args.baostock_adjustflag,
        )
        path = downloader.download_symbol(symbol, args.start, args.end)
        print(f"[V7.9 data] BaoStock real 5min saved: {path}")

    elif args.source == "csv":
        if not args.csv:
            raise ValueError("--source csv 时必须提供 --csv 文件路径")
        path = manager.register_csv(Path(args.csv), symbol)
        print(f"[V7.9 data] local 5min registered: {path}")

    else:
        daily = _load_daily()
        bars = generate_smoke_5min_from_daily(daily, symbol, args.start, args.end)
        manifest = IntradayManifest(
            symbol=symbol,
            source="synthetic_from_daily",
            quality=DEMO_5MIN_QUALITY,
            rows=len(bars),
            start_datetime=str(bars["datetime"].min()),
            end_datetime=str(bars["datetime"].max()),
            bar_minutes=5,
            volume_unit="shares_or_daily_native",
            point_in_time=False,
            notes="Engineering smoke test only. Never use for performance claims.",
        )
        path = manager.save(bars, manifest)
        print(f"[V7.9 data] DEMO smoke bars saved: {path}")
        print("[V7.9 data] IMPORTANT: quality=SIMULATED_SMOKE_ONLY; research gate will remain False.")

    manifest = manager.read_manifest()
    print("[V7.9 data] COMPLETE")
    if not manifest.empty:
        print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
