from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .market_sentiment_v78 import build_real_market_sentiment
from .sector_linkage_v78 import build_historical_board_linkage
from .transition_overlay import build_causal_transition_overlay


def _phase_from_score(score: pd.Series) -> pd.Series:
    s = pd.to_numeric(score, errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    momentum = s.diff(3) / 3.0
    return pd.Series(
        np.select(
            [
                (s <= -0.35) & (momentum <= 0),
                (s <= -0.15) & (momentum > 0),
                (s > -0.15) & (s < 0.45) & (momentum > 0),
                (s >= 0.45) & (momentum >= -0.01),
                (s >= 0.15) & (momentum < -0.01),
            ],
            ["ICE_POINT", "REPAIR", "FERMENTATION", "CLIMAX", "DIVERGENCE"],
            default="EBB",
        ),
        index=score.index,
        dtype="object",
    )


def _market_return_component(market: pd.DataFrame) -> pd.DataFrame:
    out = market[["datetime"]].copy()
    z = pd.to_numeric(market.get("market_equal_return_z"), errors="coerce").fillna(0.0)
    out["component_score"] = np.tanh(z / 1.5).clip(-1.0, 1.0)
    out["component_phase"] = _phase_from_score(out["component_score"])
    return out


def _combine_market_history(*frames: pd.DataFrame) -> pd.DataFrame:
    parts = [x.copy() for x in frames if x is not None and not x.empty]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True, sort=False)
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    return out.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates("datetime", keep="last")


def build_realtime_state(
    *,
    historical_market_summary: pd.DataFrame,
    incremental_market_summary: pd.DataFrame,
    live_market_summary: pd.DataFrame,
    historical_signals: pd.DataFrame,
    sector_history: pd.DataFrame,
    focus_history: pd.DataFrame,
    sector_live: pd.DataFrame,
    focus_live: pd.DataFrame,
    config: dict[str, Any],
    as_of: pd.Timestamp,
    session: str,
    live_requested: bool,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    v78 = dict(config.get("v7_8", {}))
    market_cfg = dict(v78.get("market_sentiment", {}))
    sector_cfg = dict(v78.get("sector", {}))
    gate_cfg = dict(v78.get("gates", {}))

    combined_summary = _combine_market_history(
        historical_market_summary,
        incremental_market_summary,
        live_market_summary,
    )
    if combined_summary.empty:
        raise RuntimeError("Step B has no market summary history")

    market = build_real_market_sentiment(
        combined_summary,
        exact_limit_pool=None,
        z_window=int(market_cfg.get("z_window", 120)),
        min_periods=int(market_cfg.get("z_min_periods", 40)),
    )
    if market.empty:
        raise RuntimeError("Step B market sentiment builder returned empty data")

    component = _market_return_component(market)
    sig = historical_signals.copy()
    sig["datetime"] = pd.to_datetime(sig["datetime"], errors="coerce")
    target = (
        sig.groupby("datetime", as_index=False)["future_return_5"].mean()
        if "future_return_5" in sig.columns
        else pd.DataFrame(columns=["datetime", "future_return_5"])
    )
    daily = component.merge(target, on="datetime", how="left")
    daily["cycle_score"] = daily["component_score"]
    daily["cycle_phase"] = daily["component_phase"]
    trans = build_causal_transition_overlay(
        daily,
        target_col="future_return_5",
        cfg=dict(config.get("v7_7", {}).get("transition", {})),
    )
    latest_market = market.iloc[-1]
    latest_component = component.iloc[-1]
    latest_trans = trans.iloc[-1] if not trans.empty else pd.Series(dtype=object)

    # The selected frozen V7.8 variant only needs the market-return component.
    # When Step B bridges missing EOD full-A history with a calibrated broad-index return,
    # keep full cross-sectional sentiment explicitly diagnostic rather than pretending the
    # proxy contains breadth/dispersion/limit-board information.
    latest_quality = str(latest_market.get("snapshot_quality", ""))
    proxy_applied = latest_quality == "REAL_EOD_BROAD_INDEX_RETURN_PROXY"
    full_mask = ~market.get(
        "snapshot_quality", pd.Series("", index=market.index, dtype="object")
    ).astype(str).eq("REAL_EOD_BROAD_INDEX_RETURN_PROXY")
    full_rows = market.loc[full_mask].copy()
    latest_full_market = full_rows.iloc[-1] if not full_rows.empty else latest_market

    # Live/updated sector context. Historical ETF proxy + current live ETF row, and focus history
    # + current focus row, are kept source-separated and causal.
    sector_all = _combine_market_history(sector_history, sector_live)
    focus_all = _combine_market_history(focus_history, focus_live)
    sector_link = build_historical_board_linkage(
        sector_all,
        focus_all,
        corr_window=int(sector_cfg.get("corr_window", 20)),
        z_window=int(sector_cfg.get("z_window", 120)),
        z_min_periods=int(sector_cfg.get("z_min_periods", 40)),
    ) if not sector_all.empty and not focus_all.empty else pd.DataFrame()
    latest_sector = sector_link.iloc[-1] if not sector_link.empty else pd.Series(dtype=object)

    data_date = pd.Timestamp(latest_market["datetime"]).normalize()
    full_market_date = pd.Timestamp(latest_full_market["datetime"]).normalize()
    full_count_num = pd.to_numeric(latest_full_market.get("market_count"), errors="coerce")
    market_count = int(full_count_num) if pd.notna(full_count_num) else 0
    min_count = int(gate_cfg.get("min_market_count", 1500))
    position_multiplier = float(pd.to_numeric(latest_trans.get("transition_cycle_multiplier"), errors="coerce"))
    if not np.isfinite(position_multiplier):
        position_multiplier = float(config.get("v7_7", {}).get("transition", {}).get("base_exposure", 0.78))
    position_multiplier = float(np.clip(position_multiplier, 0.0, 1.0))

    as_of_ts = pd.Timestamp(as_of)
    if as_of_ts.tzinfo is not None:
        local_date = as_of_ts.tz_convert("Asia/Shanghai").tz_localize(None).normalize()
    else:
        local_date = as_of_ts.normalize()
    freshness_days = int((local_date - data_date).days)
    full_freshness_days = int((local_date - full_market_date).days)
    overlay_gate = bool(
        pd.notna(latest_market.get("market_equal_return_z"))
        and pd.notna(latest_component.get("component_score"))
        and freshness_days <= 5
    )
    full_market_diagnostic_gate = bool(
        market_count >= min_count
        and pd.notna(latest_full_market.get("market_equal_return_z"))
        and full_freshness_days <= 5
    )
    # For the frozen selected variant, market_gate means the market-return overlay is usable.
    market_gate = overlay_gate
    live_applied = bool(not live_market_summary.empty and data_date == local_date)
    runtime_quality = (
        "REALTIME_INTRADAY_POINT_IN_TIME"
        if live_applied
        else ("LATEST_EOD_BROAD_INDEX_RETURN_PROXY" if proxy_applied else "LATEST_EOD_POINT_IN_TIME")
    )

    state = {
        "version": "V7 Final Step B",
        "selected_variant": "v7_8_market_return_only",
        "as_of": as_of_ts.isoformat(),
        "data_date": data_date.strftime("%Y-%m-%d"),
        "session": str(session),
        "live_requested": bool(live_requested),
        "live_applied": live_applied,
        "runtime_quality": runtime_quality,
        "point_in_time": True,
        "market_gate": market_gate,
        "overlay_gate": overlay_gate,
        "full_market_diagnostic_gate": full_market_diagnostic_gate,
        "market_count": market_count,
        "history_freshness_calendar_days": freshness_days,
        "full_market_history_freshness_calendar_days": full_freshness_days,
        "market": {
            "data_date": full_market_date.strftime("%Y-%m-%d"),
            "breadth": float(pd.to_numeric(latest_full_market.get("breadth"), errors="coerce")),
            "equal_return": float(pd.to_numeric(latest_full_market.get("market_equal_return"), errors="coerce")),
            "cross_section_vol": float(pd.to_numeric(latest_full_market.get("cross_section_vol"), errors="coerce")),
            "limit_up_count": int(pd.to_numeric(latest_full_market.get("limit_up_count"), errors="coerce") or 0),
            "limit_down_count": int(pd.to_numeric(latest_full_market.get("limit_down_count"), errors="coerce") or 0),
            "full_sentiment_score": float(pd.to_numeric(latest_full_market.get("real_sentiment_score"), errors="coerce")),
            "full_cycle_phase": str(latest_full_market.get("real_cycle_phase", "UNKNOWN")),
            "diagnostic_only": bool(proxy_applied),
        },
        "frozen_market_return_overlay": {
            "component_score": float(pd.to_numeric(latest_component.get("component_score"), errors="coerce")),
            "cycle_phase": str(latest_component.get("component_phase", "UNKNOWN")),
            "transition": str(latest_trans.get("transition", "NA")),
            "transition_edge_score": float(pd.to_numeric(latest_trans.get("transition_edge_score"), errors="coerce") or 0.0),
            "transition_confidence": float(pd.to_numeric(latest_trans.get("transition_confidence"), errors="coerce") or 0.0),
            "position_multiplier": position_multiplier,
            "data_source": str(latest_market.get("snapshot_source", "FULL_A_MARKET_SUMMARY")),
            "data_quality": str(latest_market.get("snapshot_quality", "REAL_POINT_IN_TIME")),
            "proxy_applied": proxy_applied,
            "proxy_index_symbol": str(latest_market.get("proxy_index_symbol", "")) if proxy_applied else None,
            "proxy_calibration_corr": float(pd.to_numeric(latest_market.get("proxy_calibration_corr"), errors="coerce")) if proxy_applied else None,
            "proxy_calibration_r2": float(pd.to_numeric(latest_market.get("proxy_calibration_r2"), errors="coerce")) if proxy_applied else None,
            "proxy_calibration_dates": int(pd.to_numeric(latest_market.get("proxy_calibration_dates"), errors="coerce")) if proxy_applied else None,
            "proxy_calibration_cutoff": str(latest_market.get("proxy_calibration_cutoff", "")) if proxy_applied else None,
        },
        "dynamic_position_management": {
            "rule": "target_gross = base_strategy_gross * realtime_cycle_position_multiplier",
            "realtime_cycle_position_multiplier": position_multiplier,
            "usable": overlay_gate,
        },
        "sector": {
            "gate": bool(not sector_link.empty),
            "source": str(latest_sector.get("historical_board_source", "UNAVAILABLE")),
            "quality": str(latest_sector.get("historical_board_data_quality", "UNAVAILABLE")),
            "liquidity_basis": str(latest_sector.get("historical_board_liquidity_basis", "UNAVAILABLE")),
            "strength_score": float(pd.to_numeric(latest_sector.get("historical_board_strength_score"), errors="coerce")) if not latest_sector.empty else None,
            "stock_link_score": float(pd.to_numeric(latest_sector.get("historical_board_stock_link_score"), errors="coerce")) if not latest_sector.empty else None,
        },
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        market.to_csv(output_dir / "realtime_market_sentiment_history.csv", index=False, encoding="utf-8-sig")
        component.to_csv(output_dir / "realtime_market_return_component.csv", index=False, encoding="utf-8-sig")
        trans.to_csv(output_dir / "realtime_transition_overlay.csv", index=False, encoding="utf-8-sig")
        sector_link.to_csv(output_dir / "realtime_sector_stock_linkage.csv", index=False, encoding="utf-8-sig")
        (output_dir / "realtime_sentiment_state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        instruction = pd.DataFrame([{
            "as_of": state["as_of"],
            "data_date": state["data_date"],
            "session": state["session"],
            "market_gate": state["market_gate"],
            "cycle_phase": state["frozen_market_return_overlay"]["cycle_phase"],
            "transition": state["frozen_market_return_overlay"]["transition"],
            "realtime_cycle_position_multiplier": state["dynamic_position_management"]["realtime_cycle_position_multiplier"],
            "rule": state["dynamic_position_management"]["rule"],
        }])
        instruction.to_csv(output_dir / "realtime_position_instruction.csv", index=False, encoding="utf-8-sig")

    return state
