from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from src.config import PROJECT_ROOT
from src.data.v78_real_data_downloader import DatasetManifest, V78RealDataDownloader
from src.data.v78_sector_proxy_sina import (
    DEFAULT_PROXY_SYMBOL,
    DEFAULT_TRACKED_INDEX,
    SECTOR_PROXY_QUALITY,
    SinaSectorProxyDownloader,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V7 Final Step A: real historical semiconductor sector proxy"
    )
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--symbol", default=DEFAULT_PROXY_SYMBOL)
    parser.add_argument("--tracked-index", default=DEFAULT_TRACKED_INDEX)
    parser.add_argument("--source", choices=["sina", "csv"], default="sina")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / "data" / "v7_8"
    downloader = SinaSectorProxyDownloader(data_dir)

    if args.source == "csv":
        if not args.csv:
            raise RuntimeError("--source csv requires --csv PATH")
        frame, summary = downloader.load_csv(
            Path(args.csv),
            args.start,
            args.end,
            symbol=args.symbol,
            tracked_index=args.tracked_index,
        )
    else:
        frame, summary = downloader.download(
            args.start,
            args.end,
            symbol=args.symbol,
            tracked_index=args.tracked_index,
        )

    source = str(frame["sector_source"].iloc[-1])
    manifest_writer = V78RealDataDownloader(data_dir)
    manifest_writer.manifests.append(
        DatasetManifest(
            dataset="semiconductor_board_daily",
            source=source,
            rows=int(len(frame)),
            start_date=summary.start_date,
            end_date=summary.end_date,
            quality=SECTOR_PROXY_QUALITY,
            survivorship_flag=False,
            notes=(
                "Historical traded semiconductor ETF proxy. Not reconstructed "
                "from today's constituents. Sina history supplies volume but no "
                "turnover amount; sector linkage uses causal volume ratio as the "
                "liquidity factor. Eligible only when config explicitly accepts "
                "REAL_HISTORICAL_SECTOR_ETF_PROXY and coverage passes."
            ),
        )
    )
    manifest_writer.write_manifest()

    print("[V7 Final Step A] COMPLETE")
    for key, value in asdict(summary).items():
        print(f"{key}={value}")
    print(f"output={data_dir / 'semiconductor_board_daily.csv'}")
    print(f"quality={SECTOR_PROXY_QUALITY}")
    print("survivorship_flag=False point_in_time=True")


if __name__ == "__main__":
    main()
