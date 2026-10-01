from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CovarianceEstimate:
    matrix: np.ndarray
    observations: int
    condition_number: float
    average_correlation: float


def open_to_open_returns(decisions: pd.DataFrame) -> pd.DataFrame:
    """Build returns known at each signal date from consecutive signal opens."""
    frame = decisions.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    price = pd.to_numeric(frame.get("signal_open", frame.get("open")), errors="coerce")
    frame["_price"] = price
    frame = frame.dropna(subset=["datetime", "symbol", "_price"])
    frame = frame.sort_values(["symbol", "datetime"])
    frame["_known_return"] = frame.groupby("symbol", sort=False)["_price"].pct_change()
    return frame.pivot(index="datetime", columns="symbol", values="_known_return").sort_index()


def shrink_covariance(
    returns: pd.DataFrame,
    shrinkage: float = 0.45,
    variance_floor: float = 1.0e-6,
) -> CovarianceEstimate:
    """Estimate a positive-stable covariance matrix with diagonal shrinkage."""
    if returns.empty:
        matrix = np.eye(1, dtype=float) * variance_floor
        return CovarianceEstimate(matrix, 0, 1.0, 0.0)
    cleaned = returns.replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.dropna(how="all")
    observations = int(len(cleaned))
    if observations < 2:
        variances = cleaned.var(ddof=0).fillna(variance_floor).clip(lower=variance_floor)
        matrix = np.diag(variances.to_numpy(dtype=float))
        return CovarianceEstimate(matrix, observations, 1.0, 0.0)

    sample = cleaned.cov(min_periods=max(2, min(observations, 10))).fillna(0.0)
    sample = sample.reindex(index=cleaned.columns, columns=cleaned.columns, fill_value=0.0)
    matrix = sample.to_numpy(dtype=float)
    diagonal = np.clip(np.diag(matrix), variance_floor, None)
    np.fill_diagonal(matrix, diagonal)
    shrinkage = float(np.clip(shrinkage, 0.0, 1.0))
    target = np.diag(diagonal)
    matrix = (1.0 - shrinkage) * matrix + shrinkage * target
    matrix = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, variance_floor, None)
    matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    std = np.sqrt(np.clip(np.diag(matrix), variance_floor, None))
    corr = matrix / np.outer(std, std)
    off_diagonal = corr[~np.eye(len(corr), dtype=bool)]
    average_correlation = float(np.nanmean(off_diagonal)) if off_diagonal.size else 0.0
    condition = float(np.linalg.cond(matrix)) if len(matrix) else 1.0
    return CovarianceEstimate(matrix, observations, condition, average_correlation)