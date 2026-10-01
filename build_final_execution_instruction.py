#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V7 FINAL execution-instruction builder.

This builder bridges the frozen research system to the VeighNa adapter.

Important:
- It does not invent a base target weight.
- It reads the current Step B realtime state.
- Step B sector gate is read from state["sector"]["gate"] first.
- It prints resolved gate values before building, so schema mismatches are visible.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

BUILDER_SCHEMA = "V7_FINAL_BUILDER_2026_08_23_SECTOR_GATE_FIX"

TARGET_WEIGHT_COLUMNS = (
    "target_weight",
    "daily_target_weight",
    "target_position_weight",
    "target_exposure",
    "position_weight",
    "weight",
)
SYMBOL_COLUMNS = ("symbol", "ts_code", "code", "vt_symbol")
DATE_COLUMNS = ("datetime", "date", "trade_date", "signal_time", "timestamp", "as_of")
ACTION_COLUMNS = ("action", "intraday_action", "decision", "signal", "trade_action")


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off", "", "none", "null", "nan"}:
        return False
    return bool(value)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _nested_get(data: dict[str, Any], paths: Iterable[tuple[str, ...]]) -> Any:
    for path in paths:
        cur: Any = data
        ok = True
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if ok:
            return cur
    return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _row_symbol(row: dict[str, Any]) -> str:
    for col in SYMBOL_COLUMNS:
        value = str(row.get(col, "") or "").strip()
        if value:
            return value
    return ""


def _symbol_matches(value: str, symbol: str, vt_symbol: str) -> bool:
    def norm(s: str) -> str:
        return s.upper().replace(".", "")

    return norm(value) in {
        norm(symbol),
        norm(vt_symbol),
        norm(symbol.split(".")[0]),
        norm(vt_symbol.split(".")[0]),
    }


def _latest_row(
    rows: list[dict[str, str]],
    symbol: str,
    vt_symbol: str,
) -> dict[str, str] | None:
    candidates: list[tuple[str, int, dict[str, str]]] = []
    for i, row in enumerate(rows):
        row_symbol = _row_symbol(row)
        if row_symbol and not _symbol_matches(row_symbol, symbol, vt_symbol):
            continue

        dt = ""
        for col in DATE_COLUMNS:
            if row.get(col):
                dt = str(row[col])
                break
        candidates.append((dt, i, row))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[-1][2]


def _extract_float(row: dict[str, Any] | None, columns: Iterable[str]) -> float | None:
    if not row:
        return None
    for col in columns:
        value = _as_float(row.get(col))
        if value is not None:
            return value
    return None


def _extract_text(row: dict[str, Any] | None, columns: Iterable[str]) -> str | None:
    if not row:
        return None
    for col in columns:
        value = str(row.get(col, "") or "").strip()
        if value:
            return value
    return None


def load_base_target_weight(
    root: Path,
    symbol: str,
    vt_symbol: str,
    explicit: float | None,
) -> tuple[float, str]:
    if explicit is not None:
        if not 0 <= explicit <= 1:
            raise ValueError("--base-target-weight must be in [0, 1]")
        return explicit, "CLI_OVERRIDE"

    candidates = [
        root / "outputs" / "v7_9" / "daily_targets_v7_9.csv",
        root / "outputs" / "v7_8" / "v7_8_signals.csv",
        root / "outputs" / "v7_7" / "v7_7_signals.csv",
    ]

    for path in candidates:
        row = _latest_row(_read_csv_rows(path), symbol, vt_symbol)
        value = _extract_float(row, TARGET_WEIGHT_COLUMNS)
        if value is None:
            continue
        if 1 < value <= 100:
            value /= 100.0
        if not 0 <= value <= 1:
            raise ValueError(f"Invalid target weight in {path}: {value}")
        return value, str(path)

    raise RuntimeError(
        "No base target weight found. Pass --base-target-weight explicitly "
        "or expose a target_weight-like column in frozen outputs."
    )


def load_cycle_state(root: Path) -> tuple[float, dict[str, Any], str]:
    path = root / "outputs" / "v7_realtime" / "realtime_sentiment_state.json"
    data = _read_json(path)
    if not data:
        raise RuntimeError(f"Missing Step B state: {path}")

    multiplier = _as_float(
        _nested_get(
            data,
            (
                ("frozen_market_return_overlay", "position_multiplier"),
                ("dynamic_position_management", "realtime_cycle_position_multiplier"),
                ("dynamic_position_management", "position_multiplier"),
                ("position_multiplier",),
            ),
        )
    )
    if multiplier is None:
        raise RuntimeError(f"No position multiplier in {path}")

    market_gate = _to_bool(
        _nested_get(data, (("market_gate",), ("gates", "market_gate")))
    )
    overlay_raw = _nested_get(data, (("overlay_gate",), ("gates", "overlay_gate")))
    overlay_gate = True if overlay_raw is None else _to_bool(overlay_raw)

    if not (market_gate and overlay_gate):
        raise RuntimeError(
            f"Step B market/overlay gate is closed: "
            f"market_gate={market_gate} overlay_gate={overlay_gate}"
        )

    return _clamp(multiplier, 0.0, 1.5), data, str(path)


def load_event_state(
    root: Path,
    symbol: str,
    vt_symbol: str,
) -> tuple[float, dict[str, Any], str]:
    csv_path = root / "outputs" / "v7_event_llm" / "event_risk_instruction.csv"
    json_path = root / "outputs" / "v7_event_llm" / "event_state.json"

    row = _latest_row(_read_csv_rows(csv_path), symbol, vt_symbol)
    value = _extract_float(
        row,
        ("event_risk_multiplier", "risk_multiplier", "multiplier"),
    )
    if value is not None:
        return _clamp(value, 0.0, 1.0), row or {}, str(csv_path)

    data = _read_json(json_path)
    value = (
        _as_float(
            _nested_get(
                data,
                (
                    ("event_risk_multiplier",),
                    ("risk_instruction", "event_risk_multiplier"),
                    ("event_state", "event_risk_multiplier"),
                ),
            )
        )
        if data
        else None
    )
    if value is not None:
        return _clamp(value, 0.0, 1.0), data, str(json_path)

    raise RuntimeError(
        "No Step C event_risk_multiplier found. "
        "Run run_step_c_event_scoring.py first."
    )


def load_intraday_action(
    root: Path,
    symbol: str,
    vt_symbol: str,
) -> tuple[str, dict[str, Any], str]:
    candidates = [
        root / "outputs" / "v7_9" / "intraday_decisions_v7_9.csv",
        root / "outputs" / "v7_9" / "intraday_decisions.csv",
    ]

    for path in candidates:
        row = _latest_row(_read_csv_rows(path), symbol, vt_symbol)
        action = _extract_text(row, ACTION_COLUMNS)
        if not action:
            continue
        action = action.upper()
        if action not in {"ENTER", "WAIT", "REDUCE", "EXIT", "HOLD"}:
            raise RuntimeError(f"Unknown action {action!r} in {path}")
        return action, row or {}, str(path)

    raise RuntimeError("No V7.9 intraday decision found.")


def resolve_sector_gate(
    root: Path,
    cycle_state: dict[str, Any],
) -> tuple[bool, str, Any]:
    """
    Step B current schema:
        state["sector"]["gate"]

    Backward-compatible fallbacks are retained.

    If the gate field is absent (not false), a non-empty Step B sector-linkage
    output is accepted as evidence because Step B itself defines sector.gate
    as bool(not sector_link.empty).
    """
    raw = _nested_get(
        cycle_state,
        (
            ("sector", "gate"),
            ("sector_gate",),
            ("gates", "sector_gate"),
        ),
    )
    if raw is not None:
        return _to_bool(raw), "realtime_sentiment_state.json", raw

    linkage_path = (
        root
        / "outputs"
        / "v7_realtime"
        / "realtime_sector_stock_linkage.csv"
    )
    rows = _read_csv_rows(linkage_path)
    if rows:
        return True, str(linkage_path), "NON_EMPTY_LINKAGE_FALLBACK"

    return False, "MISSING", None


def build_instruction(
    *,
    symbol: str,
    vt_symbol: str,
    base_target_weight: float,
    cycle_multiplier: float,
    event_risk_multiplier: float,
    intraday_action: str,
    market_gate: bool,
    sector_gate: bool,
    event_gate: bool,
    intraday_gate: bool,
    max_target_weight: float,
    as_of: str | None = None,
    provenance: dict[str, str] | None = None,
) -> dict[str, Any]:
    closed = [
        name
        for name, gate in (
            ("market_gate", market_gate),
            ("sector_gate", sector_gate),
            ("event_gate", event_gate),
            ("intraday_gate", intraday_gate),
        )
        if not gate
    ]
    if closed:
        raise RuntimeError("Execution gate closed: " + ", ".join(closed))

    calculated = (
        0.0
        if intraday_action == "EXIT"
        else base_target_weight * cycle_multiplier * event_risk_multiplier
    )
    final_weight = _clamp(calculated, 0.0, max_target_weight)

    content = {
        "version": "V7_FINAL",
        "builder_schema": BUILDER_SCHEMA,
        "symbol": symbol,
        "vt_symbol": vt_symbol,
        "as_of": as_of or datetime.now().astimezone().isoformat(),
        "base_target_weight": round(base_target_weight, 8),
        "cycle_multiplier": round(cycle_multiplier, 8),
        "event_risk_multiplier": round(event_risk_multiplier, 8),
        "intraday_action": intraday_action,
        "final_target_weight": round(final_weight, 8),
        "max_target_weight": round(max_target_weight, 8),
        "market_gate": bool(market_gate),
        "sector_gate": bool(sector_gate),
        "event_gate": bool(event_gate),
        "intraday_gate": bool(intraday_gate),
        "quality": "V7_FINAL_EXECUTION_READY",
        "provenance": provenance or {},
    }
    content["signal_id"] = hashlib.sha256(
        json.dumps(content, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    return content


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--symbol", default="688981.SH")
    parser.add_argument("--vt-symbol", default="688981.SSE")
    parser.add_argument("--base-target-weight", type=float, default=None)
    parser.add_argument("--max-target-weight", type=float, default=0.50)
    parser.add_argument(
        "--output",
        default="outputs/v7_final/final_execution_instruction.json",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()

    base_weight, base_source = load_base_target_weight(
        root, args.symbol, args.vt_symbol, args.base_target_weight
    )
    cycle_multiplier, cycle_state, cycle_source = load_cycle_state(root)
    event_multiplier, event_state, event_source = load_event_state(
        root, args.symbol, args.vt_symbol
    )
    action, action_row, action_source = load_intraday_action(
        root, args.symbol, args.vt_symbol
    )

    market_gate = _to_bool(
        _nested_get(
            cycle_state,
            (("market_gate",), ("gates", "market_gate")),
        )
    )
    overlay_raw = _nested_get(
        cycle_state,
        (("overlay_gate",), ("gates", "overlay_gate")),
    )
    overlay_gate = True if overlay_raw is None else _to_bool(overlay_raw)
    market_gate = market_gate and overlay_gate

    sector_gate, sector_gate_source, sector_gate_raw = resolve_sector_gate(
        root, cycle_state
    )

    event_gate_raw = _nested_get(
        event_state if isinstance(event_state, dict) else {},
        (
            ("event_gate",),
            ("gates", "event_gate"),
        ),
    )
    event_gate = True if event_gate_raw is None else _to_bool(event_gate_raw)

    intraday_gate_raw = (
        action_row.get("intraday_gate")
        if isinstance(action_row, dict)
        else None
    )
    intraday_gate = (
        True
        if intraday_gate_raw is None
        else _to_bool(intraday_gate_raw)
    )

    print(f"[V7 FINAL] builder_schema={BUILDER_SCHEMA}")
    print(
        "[V7 FINAL] gate debug "
        f"market_gate={market_gate} "
        f"sector_gate={sector_gate} "
        f"event_gate={event_gate} "
        f"intraday_gate={intraday_gate}"
    )
    print(
        "[V7 FINAL] sector gate source="
        f"{sector_gate_source} raw={sector_gate_raw!r}"
    )

    instruction = build_instruction(
        symbol=args.symbol,
        vt_symbol=args.vt_symbol,
        base_target_weight=base_weight,
        cycle_multiplier=cycle_multiplier,
        event_risk_multiplier=event_multiplier,
        intraday_action=action,
        market_gate=market_gate,
        sector_gate=sector_gate,
        event_gate=event_gate,
        intraday_gate=intraday_gate,
        max_target_weight=args.max_target_weight,
        provenance={
            "base_target": base_source,
            "cycle_state": cycle_source,
            "event_state": event_source,
            "intraday_action": action_source,
            "sector_gate": sector_gate_source,
        },
    )

    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)

    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(
        json.dumps(instruction, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(output)

    print("[V7 FINAL] execution instruction built")
    print(f"symbol={instruction['symbol']} vt_symbol={instruction['vt_symbol']}")
    print(
        f"base={instruction['base_target_weight']:.2%} "
        f"cycle={instruction['cycle_multiplier']:.4f} "
        f"event={instruction['event_risk_multiplier']:.4f} "
        f"action={instruction['intraday_action']}"
    )
    print(f"final_target_weight={instruction['final_target_weight']:.2%}")
    print(f"signal_id={instruction['signal_id']}")
    print(f"output={output}")


if __name__ == "__main__":
    main()