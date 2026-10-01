from __future__ import annotations

import argparse

import pandas as pd

from src.config import PROJECT_ROOT
from src.data.v78_real_data_downloader import V78RealDataDownloader


def _business_dates(start: str, end: str, recent_days: int = 60) -> list[str]:
    dates = pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end))
    return [x.strftime("%Y-%m-%d") for x in dates[-recent_days:]]


def _attempt(label: str, func):
    try:
        return func()
    except Exception as exc:
        print(f"[V7.8 data] WARNING {label} failed: {exc}", flush=True)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="V7.8 real A-share data acquisition")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--mode", choices=["historical", "recent", "all"], default="all")
    parser.add_argument("--all-a-source", choices=["tushare", "eastmoney", "skip"], default="tushare")
    parser.add_argument("--max-symbols", type=int, default=None, help="Eastmoney smoke-test cap only")
    parser.add_argument("--recent-days", type=int, default=60)
    parser.add_argument(
        "--sector-history-source",
        choices=["eastmoney", "skip"],
        default="skip",
        help=(
            "Historical sector source. Default skip avoids repeatedly hitting unstable Eastmoney endpoints. "
            "If a verified semiconductor_board_daily.csv already exists, run_v7_8.py will use it through the data gate."
        ),
    )
    parser.add_argument(
        "--current-sector-universe",
        choices=["fetch", "skip"],
        default="skip",
        help="Current semiconductor constituents are diagnostic only; skip by default.",
    )
    parser.add_argument(
        "--peer-history-source",
        choices=["tushare", "eastmoney", "skip"],
        default="skip",
        help="Current-member peer history is survivorship-biased diagnostic data; skip by default.",
    )
    parser.add_argument(
        "--tushare-rpm",
        type=int,
        default=45,
        help="Client-side Tushare pacing. Use <=45 for a 120-point account (server limit 50/min observed on this account).",
    )
    parser.add_argument("--recent-end", default=None, help="Recent Eastmoney end date; default today")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "data" / "v7_8"
    dl = V78RealDataDownloader(out_dir, tushare_requests_per_minute=args.tushare_rpm)

    if args.mode in {"historical", "all"}:
        if args.current_sector_universe == "fetch":
            print("[V7.8 data] semiconductor current universe (diagnostic)...", flush=True)
            _attempt("semiconductor current universe", dl.fetch_semiconductor_universe)
        else:
            print("[V7.8 data] semiconductor current universe skipped (diagnostic only)", flush=True)

        if args.sector_history_source == "eastmoney":
            print("[V7.8 data] semiconductor historical board index via Eastmoney...", flush=True)
            _attempt(
                "semiconductor historical board index",
                lambda: dl.fetch_semiconductor_board_history(args.start, args.end),
            )
        else:
            print("[V7.8 data] semiconductor historical board download skipped; frozen V7.7 sector remains fallback", flush=True)

        if args.peer_history_source == "eastmoney":
            print("[V7.8 data] current-membership peer history via Eastmoney (diagnostic)...", flush=True)
            _attempt("semiconductor current-membership history", lambda: dl.fetch_semiconductor_history(args.start, args.end))
        elif args.peer_history_source == "tushare":
            from config.tushare_token import TUSHARE_TOKEN

            universe_path = out_dir / "semiconductor_universe_current.csv"
            if universe_path.is_file():
                universe = pd.read_csv(universe_path, low_memory=False)
                print("[V7.8 data] current-membership peer history via Tushare (diagnostic)...", flush=True)
                peer = _attempt(
                    "semiconductor current-membership Tushare history",
                    lambda: dl.fetch_symbol_history_tushare(
                        TUSHARE_TOKEN, universe["symbol"].tolist(), args.start, args.end
                    ),
                )
                if peer is not None:
                    peer.to_csv(out_dir / "semiconductor_daily.csv", index=False, encoding="utf-8-sig")
            else:
                print("[V7.8 data] WARNING no current semiconductor universe; peer history skipped", flush=True)
        else:
            print("[V7.8 data] semiconductor current-membership peer history skipped (diagnostic only)", flush=True)

        if args.all_a_source == "tushare":
            from config.tushare_token import TUSHARE_TOKEN

            print(f"[V7.8 data] full-A PIT history via Tushare (paced at {args.tushare_rpm}/min)...", flush=True)
            _attempt(
                "full-A Tushare PIT history",
                lambda: dl.fetch_all_a_history_tushare(TUSHARE_TOKEN, args.start, args.end),
            )
        elif args.all_a_source == "eastmoney":
            print("[V7.8 data] full-A current-universe history via Eastmoney (survivorship flagged)...", flush=True)
            _attempt(
                "full-A Eastmoney current-universe history",
                lambda: dl.fetch_all_a_history(args.start, args.end, max_symbols=args.max_symbols),
            )

    if args.mode in {"recent", "all"}:
        today = pd.Timestamp.now().normalize()
        end = min(pd.Timestamp(args.recent_end).normalize(), today) if args.recent_end else today
        start = end - pd.Timedelta(days=max(90, args.recent_days * 2))
        print("[V7.8 data] recent Eastmoney limit pools...", flush=True)
        _attempt(
            "recent Eastmoney limit pools",
            lambda: dl.fetch_recent_limit_pools(
                _business_dates(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), args.recent_days)
            ),
        )
        print("[V7.8 data] recent focus/industry news...", flush=True)
        _attempt(
            "recent focus/industry news",
            lambda: dl.fetch_focus_news(["688981", "中芯国际", "半导体", "芯片"]),
        )

    manifest = dl.write_manifest()
    print("[V7.8 data] COMPLETE", flush=True)
    if not manifest.empty:
        print(manifest[["dataset", "rows", "quality", "survivorship_flag"]].to_string(index=False))
    print(f"output={out_dir}")


if __name__ == "__main__":
    main()
