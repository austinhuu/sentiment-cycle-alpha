from __future__ import annotations

import logging
from pathlib import Path

from src.config import PROJECT_ROOT, load_config
from src.data.akshare_text_downloader import (
    AKShareTextDownloader,
    TextDownloadConfig,
)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    config = load_config()
    data_config = config["data"]
    download_config = config.get("text_download", {})

    downloader = AKShareTextDownloader(
        TextDownloadConfig(
            start_date=str(data_config["start_date"]),
            end_date=str(data_config["end_date"]),
            symbols=list(data_config.get("trading_pool", [])),
            events_output_csv=_resolve(
                str(
                    download_config.get(
                        "events_output_csv",
                        "data/raw/text_events.csv",
                    )
                )
            ),
            daily_output_csv=_resolve(
                str(
                    download_config.get(
                        "daily_output_csv",
                        "data/raw/news_sentiment_daily.csv",
                    )
                )
            ),
            request_pause_seconds=float(
                download_config.get("request_pause_seconds", 1.5)
            ),
            max_retries=int(download_config.get("max_retries", 3)),
            availability_lag_days=int(
                download_config.get("availability_lag_days", 1)
            ),
            include_recent_news=bool(
                download_config.get("include_recent_news", True)
            ),
            recent_news_max_age_days=int(
                download_config.get("recent_news_max_age_days", 180)
            ),
            include_event_content=bool(
                download_config.get("include_event_content", True)
            ),
        )
    )
    events, daily = downloader.download()
    print(
        f"Text events: {len(events):,}\n"
        f"Daily sentiment rows: {len(daily):,}\n"
        f"Events output: {downloader.config.events_output_csv}\n"
        f"Daily output: {downloader.config.daily_output_csv}"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()