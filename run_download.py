from __future__ import annotations

import logging

from src.config import PROJECT_ROOT, load_config
from src.data.tushare_daily_downloader import DailyDownloadConfig, TushareDailyDownloader


def main() -> None:
    config = load_config()
    data = config["data"]
    if str(data.get("source", "tushare_daily")).lower() != "tushare_daily":
        raise RuntimeError("当前 data.source 不是 tushare_daily；CSV模式不需要运行下载脚本。")

    trading_pool = [str(x).strip().upper() for x in data.get("trading_pool", [])]
    sentiment_pool = [str(x).strip().upper() for x in data.get("sentiment_pool", [])]
    symbols = list(dict.fromkeys(trading_pool + sentiment_pool))
    if not symbols:
        raise ValueError("data.trading_pool 与 data.sentiment_pool 不能同时为空。")

    downloader = TushareDailyDownloader(
        DailyDownloadConfig(
            start_date=str(data["start_date"]),
            end_date=str(data["end_date"]),
            request_pause_seconds=float(data.get("request_pause_seconds", 1.5)),
            max_retries=int(data.get("max_retries", 4)),
            rate_limit_wait_seconds=float(data.get("rate_limit_wait_seconds", 65)),
        ),
        PROJECT_ROOT / "data" / "raw",
    )
    combined = downloader.download_pool(symbols)
    print(f"下载完成：{len(combined):,} 行，股票数：{combined['symbol'].nunique()}")
    print(f"交易池：{len(trading_pool)} 只；情绪参考池：{len(sentiment_pool)} 只")
    print(PROJECT_ROOT / "data" / "raw" / "all_stocks_daily.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
