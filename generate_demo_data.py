from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config


def generate_symbol(
    symbol: str,
    dates: pd.DatetimeIndex,
    seed: int,
    start_price: float,
    inject_split: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    market_cycle = np.sin(np.linspace(0, 14 * np.pi, len(dates))) * 0.0010
    regime = np.repeat([0.00035, -0.00015, 0.00010], repeats=[len(dates)//3, len(dates)//3, len(dates)-2*(len(dates)//3)])
    returns = market_cycle + regime + rng.normal(0.0, 0.014, len(dates))
    economic_close = start_price * np.exp(np.cumsum(returns))
    overnight = rng.normal(0.0, 0.003, len(dates))
    economic_open = np.r_[economic_close[0], economic_close[:-1]] * (1.0 + overnight)
    spread = np.abs(rng.normal(0.011, 0.004, len(dates)))
    economic_high = np.maximum(economic_open, economic_close) * (1.0 + spread)
    economic_low = np.minimum(economic_open, economic_close) * (1.0 - spread)

    raw_factor = np.ones(len(dates))
    if inject_split and len(dates) > 700:
        raw_factor[len(dates) // 2 :] = 0.5
    close = economic_close * raw_factor
    open_price = economic_open * raw_factor
    high = economic_high * raw_factor
    low = economic_low * raw_factor
    pre_close = close / (1.0 + returns)
    change = close - pre_close
    pct_chg = returns * 100.0
    volume = rng.lognormal(mean=13.0, sigma=0.45, size=len(dates)) * (1.0 + 5.0 * np.abs(returns))

    return pd.DataFrame(
        {
            "datetime": dates,
            "symbol": symbol,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "pre_close": pre_close,
            "change": change,
            "pct_chg": pct_chg,
            "volume": volume,
            "amount": volume * close,
        }
    )


def main() -> None:
    config = load_config()
    data_cfg = config["data"]
    symbols = list(dict.fromkeys(data_cfg.get("trading_pool", []) + data_cfg.get("sentiment_pool", [])))
    dates = pd.bdate_range("2020-01-02", "2025-12-31")
    frames = [
        generate_symbol(symbol, dates, 42 + i, 12.0 + i * 3.0, inject_split=(i in {5, 12}))
        for i, symbol in enumerate(symbols)
    ]
    data = pd.concat(frames, ignore_index=True).sort_values(["datetime", "symbol"])
    path = PROJECT_ROOT / "data" / "raw" / "demo_all_stocks_daily.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"演示日线数据已生成：{path}\nRows: {len(data):,}; Symbols: {data['symbol'].nunique()}")


if __name__ == "__main__":
    main()
