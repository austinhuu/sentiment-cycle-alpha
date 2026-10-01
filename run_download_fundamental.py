from __future__ import annotations

import logging
from pathlib import Path

from src.config import PROJECT_ROOT, load_config
from src.data.akshare_fundamental_downloader import (
    AKShareFundamentalDownloader,
    FundamentalDownloadConfig,
)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    config = load_config()
    data_config = config["data"]
    download_config = config.get("fundamental_download", {})
    downloader = AKShareFundamentalDownloader(
        FundamentalDownloadConfig(
            start_date=str(data_config["start_date"]),
            end_date=str(data_config["end_date"]),
            symbols=list(data_config.get("trading_pool", [])),
            output_csv=_resolve(
                str(download_config.get(
                    "output_csv", "data/raw/fundamental_scores.csv"
                ))
            ),
            raw_output_csv=_resolve(
                str(download_config.get(
                    "raw_output_csv", "data/raw/fundamental_raw.csv"
                ))
            ),
            request_pause_seconds=float(
                download_config.get("request_pause_seconds", 1.0)
            ),
            max_retries=int(download_config.get("max_retries", 3)),
            availability_lag_days=int(
                download_config.get("availability_lag_days", 1)
            ),
            explicit_date_grace_days=int(
                download_config.get("explicit_date_grace_days", 60)
            ),
            rolling_history_quarters=int(
                download_config.get("rolling_history_quarters", 8)
            ),
            min_history_quarters=int(
                download_config.get("min_history_quarters", 3)
            ),
        )
    )
    result = downloader.download()
    print(
        f"Fundamental rows: {len(result):,}\n"
        f"Symbols: {result['symbol'].nunique()}\n"
        f"Explicit date share: "
        f"{result['available_date_source'].eq('explicit_announcement_date').mean():.2%}\n"
        f"Output: {downloader.config.output_csv}"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()