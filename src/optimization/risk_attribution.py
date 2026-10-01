from __future__ import annotations

import numpy as np
import pandas as pd


def portfolio_risk_attribution(
    symbols: list[str],
    weights: np.ndarray,
    covariance: np.ndarray,
    annualization: int = 252,
) -> pd.DataFrame:
    weights = np.asarray(weights, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    variance = float(weights @ covariance @ weights)
    daily_volatility = float(np.sqrt(max(variance, 0.0)))
    annualized_volatility = daily_volatility * np.sqrt(max(int(annualization), 1))
    marginal = covariance @ weights
    component_variance = weights * marginal
    if variance > 1.0e-16:
        contribution_share = component_variance / variance
    else:
        contribution_share = np.zeros_like(weights)
    return pd.DataFrame(
        {
            "symbol": symbols,
            "target_weight": weights,
            "marginal_variance": marginal,
            "component_variance": component_variance,
            "risk_contribution_share": contribution_share,
            "portfolio_annualized_volatility": annualized_volatility,
        }
    )