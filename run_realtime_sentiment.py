from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.data.realtime_market_snapshot_v7 import (
    RealtimeMarketDataClient,
    focus_live_row_from_market,
    live_session_usable,
    market_session,
    normalize_sector_live_row,
    shanghai_now,
)
from src.v7.realtime_sentiment_v7 import build_realtime_state


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.is_file() else pd.DataFrame()


def _latest_closed_date(now: pd.Timestamp) -> pd.Timestamp:
    local = now.tz_convert("Asia/Shanghai") if now.tzinfo is not None else now
    day = local.tz_localize(None).normalize() if getattr(local, "tzinfo", None) else local.normalize()
    # Before 16:30 use the previous calendar day as the upper bound; the index endpoint itself
    # determines which dates are actual trading days.
    if local.time() < pd.Timestamp("16:30").time():
        day -= pd.Timedelta(days=1)
    return day


def main() -> None:
    parser = argparse.ArgumentParser(description="V7 Final Step B realtime sentiment-cycle refresh")
    parser.add_argument("--mode", choices=["auto", "live", "eod"], default="auto")
    parser.add_argument("--skip-backfill", action="store_true")
    parser.add_argument("--focus", default=None)
    parser.add_argument("--sector", default="512480.SH")
    parser.add_argument("--market-index", default="000985", help="preferred broad-market index; default CSI All Share 000985")
    parser.add_argument("--pause", type=float, default=0.15)
    args = parser.parse_args()

    config = load_config()
    focus = str(args.focus or config.get("v7_8", {}).get("focus_symbol", "688981.SH")).upper()
    data_v78 = PROJECT_ROOT / "data" / "v7_8"
    data_rt = PROJECT_ROOT / "data" / "v7_realtime"
    out_rt = PROJECT_ROOT / "outputs" / "v7_realtime"
    data_rt.mkdir(parents=True, exist_ok=True)
    out_rt.mkdir(parents=True, exist_ok=True)

    base_market = _read(data_v78 / "all_a_market_summary.csv")
    if base_market.empty:
        raise RuntimeError("缺少 data/v7_8/all_a_market_summary.csv；请先完成 V7.8 全A PIT 数据层")
    base_market["datetime"] = pd.to_datetime(base_market["datetime"], errors="coerce")
    base_market = base_market.dropna(subset=["datetime"]).sort_values("datetime")

    signals_path = PROJECT_ROOT / "outputs" / "v7_8" / "v7_8_signals.csv"
    if not signals_path.is_file():
        raise RuntimeError("缺少 outputs/v7_8/v7_8_signals.csv；请先运行 run_v7_8.py")
    sig_header = pd.read_csv(signals_path, nrows=0).columns
    needed = [c for c in ["datetime", "symbol", "future_return_5"] if c in sig_header]
    historical_signals = pd.read_csv(signals_path, usecols=needed, low_memory=False)

    client = RealtimeMarketDataClient(data_rt, pause_seconds=args.pause)
    now = shanghai_now()
    session = market_session(now)
    closed_end = _latest_closed_date(now)

    proxy_path = data_rt / "market_return_proxy_incremental.csv"
    incremental = _read(proxy_path)
    if not args.skip_backfill:
        calibration_start = max(
            pd.Timestamp("2020-01-01"),
            pd.Timestamp(base_market["datetime"].min()).normalize(),
        )
        print(
            f"[V7 Final Step B] broad-index EOD bridge {args.market_index} "
            f"{calibration_start.date()}~{closed_end.date()}...",
            flush=True,
        )
        proxy, meta = client.build_market_return_proxy_incremental(
            base_market_summary=base_market,
            start_date=calibration_start.strftime("%Y-%m-%d"),
            end_date=closed_end.strftime("%Y-%m-%d"),
            preferred_symbol=args.market_index,
        )
        incremental = proxy
        print(
            "[V7 Final Step B] index proxy "
            f"symbol={meta.get('proxy_symbol')} provider={meta.get('provider')} "
            f"calibration_dates={meta.get('calibration_dates')} "
            f"corr={float(meta.get('correlation', 0.0)):.3f} r2={float(meta.get('r2', 0.0)):.3f} "
            f"cutoff={meta.get('calibration_cutoff')}",
            flush=True,
        )
        if incremental.empty:
            latest_base = pd.Timestamp(base_market["datetime"].max()).normalize()
            if latest_base < closed_end:
                raise RuntimeError(
                    "Step B broad-index bridge produced no post-history rows although the frozen "
                    f"full-A history ends at {latest_base.date()} and latest closed date is {closed_end.date()}."
                )
        else:
            print(
                f"[V7 Final Step B] EOD proxy rows={len(incremental):,} "
                f"latest={pd.to_datetime(incremental['datetime']).max().date()}",
                flush=True,
            )

    want_live = args.mode == "live" or (args.mode == "auto" and live_session_usable(session))
    live_market = pd.DataFrame()
    sector_live = pd.DataFrame()
    focus_live = pd.DataFrame()
    live_error = None
    if want_live:
        try:
            bundle = client.fetch_live_bundle(sector_code=args.sector.split(".")[0])
            live_market = bundle.market_summary
            sector_live = normalize_sector_live_row(bundle.sector_spot, bundle.data_date)
            focus_live = focus_live_row_from_market(bundle.market_spot, focus, bundle.data_date)
        except Exception as exc:
            live_error = str(exc)
            print(f"[V7 Final Step B] WARNING live snapshot failed; falling back to EOD proxy: {exc}", flush=True)

    print("[V7 Final Step B] sector history refresh via AKShare/Sina...", flush=True)
    try:
        sector_history = client.fetch_sector_history_sina(args.sector)
    except Exception as exc:
        print(f"[V7 Final Step B] WARNING sector history unavailable: {exc}", flush=True)
        sector_history = _read(data_v78 / "semiconductor_board_daily.csv")

    focus_start = "2025-01-01"
    focus_end = now.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d")
    print("[V7 Final Step B] focus daily refresh via BaoStock...", flush=True)
    try:
        focus_history = client.fetch_focus_daily_baostock(focus, focus_start, focus_end)
    except Exception as exc:
        print(f"[V7 Final Step B] WARNING focus daily unavailable: {exc}", flush=True)
        full_sig_header = pd.read_csv(signals_path, nrows=0).columns
        focus_cols = [c for c in ["datetime", "symbol", "return_1", "return_5"] if c in full_sig_header]
        tmp = pd.read_csv(signals_path, usecols=focus_cols, low_memory=False)
        focus_history = tmp.loc[tmp["symbol"].astype(str).str.upper().eq(focus)].copy()

    state = build_realtime_state(
        historical_market_summary=base_market,
        incremental_market_summary=incremental,
        live_market_summary=live_market,
        historical_signals=historical_signals,
        sector_history=sector_history,
        focus_history=focus_history,
        sector_live=sector_live,
        focus_live=focus_live,
        config=config,
        as_of=now,
        session=session,
        live_requested=want_live,
        output_dir=out_rt,
    )
    state["live_error"] = live_error
    (out_rt / "realtime_sentiment_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("[V7 Final Step B] COMPLETE", flush=True)
    print(
        f"as_of={state['as_of']} data_date={state['data_date']} session={state['session']} "
        f"quality={state['runtime_quality']}"
    )
    print(
        f"market_gate={state['market_gate']} overlay_gate={state['overlay_gate']} "
        f"full_market_diagnostic_gate={state['full_market_diagnostic_gate']} "
        f"market_count(last_full_A)={state['market_count']} live_applied={state['live_applied']}"
    )
    print(
        f"full_sentiment(diagnostic_date={state['market']['data_date']})="
        f"{state['market']['full_sentiment_score']:.4f} full_phase={state['market']['full_cycle_phase']}"
    )
    ov = state["frozen_market_return_overlay"]
    print(
        f"selected_variant={state['selected_variant']} component_score={ov['component_score']:.4f} "
        f"phase={ov['cycle_phase']} transition={ov['transition']}"
    )
    print(
        f"position_multiplier={ov['position_multiplier']:.4f} "
        f"transition_confidence={ov['transition_confidence']:.3f} "
        f"proxy_applied={ov['proxy_applied']} proxy_index={ov['proxy_index_symbol']}"
    )
    sec = state["sector"]
    print(
        f"sector_gate={sec['gate']} sector_strength={sec['strength_score']} "
        f"sector_stock_link={sec['stock_link_score']}"
    )
    print(f"output={out_rt}")


if __name__ == "__main__":
    main()
