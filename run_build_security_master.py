from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, load_config


def main() -> None:
    config = load_config()
    fundamental_path = PROJECT_ROOT / config.get("alternative_data", {}).get(
        "fundamental_csv", "data/raw/fundamental_scores.csv"
    )
    output_path = PROJECT_ROOT / config.get("portfolio", {}).get(
        "security_master_csv", "data/raw/security_master.csv"
    )
    if not fundamental_path.exists():
        raise FileNotFoundError(
            "未找到fundamental_scores.csv，请先运行python run_download_fundamental.py"
        )
    frame = pd.read_csv(fundamental_path, encoding="utf-8-sig")
    if "symbol" not in frame.columns:
        raise ValueError("fundamental_scores.csv缺少symbol字段")
    if "industry" not in frame.columns:
        frame["industry"] = "UNKNOWN"
    if "available_date" in frame.columns:
        frame["available_date"] = pd.to_datetime(
            frame["available_date"], errors="coerce"
        )
        frame = frame.sort_values(["symbol", "available_date"])
    latest = frame.drop_duplicates("symbol", keep="last").copy()
    latest["sector"] = latest["industry"].fillna("UNKNOWN").astype(str)
    universe = pd.DataFrame(
        {"symbol": config.get("data", {}).get("trading_pool", [])}
    )
    result = universe.merge(
        latest[["symbol", "sector"]],
        on="symbol",
        how="left",
    )
    result["sector"] = result["sector"].fillna("UNKNOWN")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(
        f"Security master rows: {len(result):,}\n"
        f"Sectors: {result['sector'].nunique()}\n"
        f"Output: {output_path}"
    )


if __name__ == "__main__":
    main()