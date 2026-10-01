from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class OrderPlanSummary:
    signal_date: str
    capital: float
    buy_orders: int
    sell_orders: int
    blocked_orders: int
    hold_rows: int
    estimated_buy_notional: float
    estimated_sell_notional: float
    estimated_transaction_cost: float
    target_gross_exposure: float
    estimated_cash_after_orders: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OrderPlanner:
    """Translate latest target weights into A-share paper orders.

    This planner does not send orders. It creates an auditable, lot-rounded order
    list and blocks trades that are not tradable or appear limit constrained.
    """

    def __init__(self, config: dict[str, Any], project_root: Path) -> None:
        self.config = config
        self.project_root = Path(project_root)

    def _load_holdings(self) -> pd.DataFrame:
        path = self.project_root / self.config.get(
            "holdings_csv", "data/live/current_holdings.csv"
        )
        if not path.exists():
            return pd.DataFrame(columns=["symbol", "shares"])
        frame = pd.read_csv(path)
        if "symbol" not in frame or "shares" not in frame:
            raise ValueError("Holdings CSV must contain symbol and shares columns")
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
        frame["shares"] = pd.to_numeric(frame["shares"], errors="coerce").fillna(0).astype(int)
        return frame[["symbol", "shares"]].groupby("symbol", as_index=False)["shares"].sum()

    @staticmethod
    def _price_column(frame: pd.DataFrame, requested: str) -> str:
        for column in [requested, "signal_open", "open", "close"]:
            if column in frame.columns:
                return column
        raise KeyError("No usable execution price column found")

    def build(
        self,
        decisions: pd.DataFrame,
        allocations: pd.DataFrame,
    ) -> tuple[pd.DataFrame, OrderPlanSummary]:
        decisions_local = decisions.copy()
        allocations_local = allocations.copy()
        decisions_local["datetime"] = pd.to_datetime(decisions_local["datetime"], errors="coerce")
        allocations_local["datetime"] = pd.to_datetime(allocations_local["datetime"], errors="coerce")
        decisions_local["symbol"] = decisions_local["symbol"].astype(str).str.upper()
        allocations_local["symbol"] = allocations_local["symbol"].astype(str).str.upper()
        signal_date = pd.Timestamp(allocations_local["datetime"].max())
        current = allocations_local.loc[allocations_local["datetime"].eq(signal_date)].copy()
        latest_decisions = decisions_local.loc[decisions_local["datetime"].eq(signal_date)].copy()
        price_column = self._price_column(latest_decisions, str(self.config.get("price_column", "signal_open")))
        latest_decisions["execution_price"] = pd.to_numeric(latest_decisions[price_column], errors="coerce")
        columns = ["symbol", "execution_price"]
        for optional in ["is_tradable_sample", "is_limit_up", "is_limit_down", "limit_up_price", "limit_down_price", "model_health_state"]:
            if optional in latest_decisions:
                columns.append(optional)
        market = latest_decisions[columns].drop_duplicates("symbol")
        current = current.merge(market, on="symbol", how="left")
        current["target_weight"] = pd.to_numeric(current["target_weight"], errors="coerce").fillna(0.0)
        current["execution_price"] = pd.to_numeric(current["execution_price"], errors="coerce")

        capital = float(self.config.get("capital", 1_000_000.0))
        lot_size = max(int(self.config.get("lot_size", 100)), 1)
        minimum_trade_value = float(self.config.get("minimum_trade_value", 5_000.0))
        holdings = self._load_holdings()
        if holdings.empty and bool(self.config.get("derive_holdings_from_previous_target", True)):
            previous_dates = sorted(allocations_local["datetime"].dropna().unique())
            previous_date = previous_dates[-2] if len(previous_dates) >= 2 else None
            if previous_date is not None:
                previous = allocations_local.loc[allocations_local["datetime"].eq(previous_date), ["symbol", "target_weight"]].copy()
                previous = previous.merge(market[["symbol", "execution_price"]], on="symbol", how="left")
                previous["shares"] = np.floor(
                    previous["target_weight"] * capital / previous["execution_price"] / lot_size
                ).fillna(0).astype(int) * lot_size
                holdings = previous[["symbol", "shares"]]
        current = current.merge(holdings, on="symbol", how="outer")
        current["target_weight"] = pd.to_numeric(current["target_weight"], errors="coerce").fillna(0.0)
        current["shares"] = pd.to_numeric(current["shares"], errors="coerce").fillna(0).astype(int)
        current["execution_price"] = pd.to_numeric(current["execution_price"], errors="coerce")
        target_value = current["target_weight"] * capital
        current["target_shares"] = (
            np.floor(target_value / current["execution_price"] / lot_size).fillna(0).astype(int) * lot_size
        )
        current["share_change"] = current["target_shares"] - current["shares"]
        current["estimated_notional"] = current["share_change"].abs() * current["execution_price"].fillna(0.0)
        current["side"] = np.select(
            [current["share_change"].gt(0), current["share_change"].lt(0)],
            ["BUY", "SELL"],
            default="HOLD",
        )
        current["blocked_reason"] = ""
        missing_price = current["execution_price"].isna() | current["execution_price"].le(0)
        current.loc[missing_price & current["side"].ne("HOLD"), "blocked_reason"] = "missing_price"
        if "is_tradable_sample" in current:
            not_tradable = pd.to_numeric(current["is_tradable_sample"], errors="coerce").fillna(0).le(0)
            current.loc[not_tradable & current["side"].ne("HOLD"), "blocked_reason"] = "not_tradable"
        if bool(self.config.get("block_limit_up_buys", True)) and "is_limit_up" in current:
            blocked = pd.to_numeric(current["is_limit_up"], errors="coerce").fillna(0).gt(0) & current["side"].eq("BUY")
            current.loc[blocked, "blocked_reason"] = "limit_up_buy_blocked"
        if bool(self.config.get("block_limit_down_sells", True)) and "is_limit_down" in current:
            blocked = pd.to_numeric(current["is_limit_down"], errors="coerce").fillna(0).gt(0) & current["side"].eq("SELL")
            current.loc[blocked, "blocked_reason"] = "limit_down_sell_blocked"
        too_small = current["estimated_notional"].lt(minimum_trade_value) & current["side"].ne("HOLD")
        current.loc[too_small, "blocked_reason"] = "below_minimum_trade_value"
        current["order_status"] = np.where(
            current["blocked_reason"].ne(""), "BLOCKED", np.where(current["side"].eq("HOLD"), "NO_ORDER", "READY")
        )
        current["order_shares"] = np.where(current["order_status"].eq("READY"), current["share_change"].abs(), 0).astype(int)
        buy_cost = float(self.config.get("buy_commission_rate", 0.0001)) + float(self.config.get("slippage_rate", 0.0005))
        sell_cost = float(self.config.get("sell_commission_rate", 0.0001)) + float(self.config.get("stamp_duty_rate", 0.0005)) + float(self.config.get("slippage_rate", 0.0005))
        current["estimated_cost"] = np.where(
            current["order_status"].eq("READY") & current["side"].eq("BUY"),
            current["estimated_notional"] * buy_cost,
            np.where(
                current["order_status"].eq("READY") & current["side"].eq("SELL"),
                current["estimated_notional"] * sell_cost,
                0.0,
            ),
        )
        ready = current["order_status"].eq("READY")
        buy_notional = float(current.loc[ready & current["side"].eq("BUY"), "estimated_notional"].sum())
        sell_notional = float(current.loc[ready & current["side"].eq("SELL"), "estimated_notional"].sum())
        costs = float(current["estimated_cost"].sum())
        current_value = float((current["shares"] * current["execution_price"].fillna(0.0)).sum())
        estimated_cash = capital - current_value - buy_notional + sell_notional - costs
        summary = OrderPlanSummary(
            signal_date=signal_date.date().isoformat(),
            capital=capital,
            buy_orders=int((ready & current["side"].eq("BUY")).sum()),
            sell_orders=int((ready & current["side"].eq("SELL")).sum()),
            blocked_orders=int(current["order_status"].eq("BLOCKED").sum()),
            hold_rows=int(current["order_status"].eq("NO_ORDER").sum()),
            estimated_buy_notional=buy_notional,
            estimated_sell_notional=sell_notional,
            estimated_transaction_cost=costs,
            target_gross_exposure=float(current["target_weight"].sum()),
            estimated_cash_after_orders=estimated_cash,
        )
        output_columns = [
            "symbol", "side", "order_status", "order_shares", "execution_price",
            "estimated_notional", "estimated_cost", "shares", "target_shares",
            "target_weight", "blocked_reason",
        ]
        return current[output_columns].sort_values(["order_status", "side", "symbol"]), summary


def save_order_plan(
    orders: pd.DataFrame,
    summary: OrderPlanSummary,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    orders.to_csv(output_dir / "latest_order_plan.csv", index=False, encoding="utf-8-sig")
    (output_dir / "latest_order_plan_summary.json").write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )