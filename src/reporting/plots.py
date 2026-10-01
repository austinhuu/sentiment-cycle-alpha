from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(portfolio: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dates = pd.to_datetime(portfolio["datetime"])
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(dates, portfolio["equity"], label="Multi-Agent Strategy")
    if "equal_weight_equity" in portfolio.columns:
        ax.plot(dates, portfolio["equal_weight_equity"], label="Equal-Weight Benchmark")
    if "exposure_matched_equity" in portfolio.columns:
        ax.plot(dates, portfolio["exposure_matched_equity"], label="Exposure-Matched Benchmark")
    ax.set_title("Equity Curve")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Portfolio Equity")
    ax.grid(True, alpha=0.25)
    if len(ax.lines) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
