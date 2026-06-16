"""Monte Carlo simulation of portfolio value over a holding horizon."""

import numpy as np
import pandas as pd

from src.config import HOLDING_PERIOD_DAYS, MC_ZERO_DRIFT, MONTE_CARLO_SIMULATIONS
from src.prices import daily_returns, ewma_volatility


def _correlated_shocks(corr_matrix: np.ndarray, n_sims: int, n_days: int) -> np.ndarray:
    L = np.linalg.cholesky(corr_matrix + 1e-10 * np.eye(len(corr_matrix)))
    z = np.random.standard_normal((n_sims, n_days, len(corr_matrix)))
    return z @ L.T


def simulate(
    exposures: dict[str, float],
    prices: dict[str, pd.Series],
    horizon: int = HOLDING_PERIOD_DAYS,
    n_sims: int = MONTE_CARLO_SIMULATIONS,
    seed: int = 42,
    zero_drift: bool = MC_ZERO_DRIFT,
) -> dict:

    np.random.seed(seed)
    metals = list(exposures.keys())
    exp_arr = np.array([exposures[m] for m in metals], dtype=float)
    initial_value = float(exp_arr.sum()) if len(exp_arr) else 0.0
    pct_levels = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    if not metals or initial_value <= 0:
        paths = np.zeros((n_sims, horizon + 1))
        final_values = paths[:, -1]
        return {
            "paths": paths,
            "final_values": final_values,
            "initial_value": initial_value,
            "percentiles": {p: 0.0 for p in pct_levels},
            "metrics": {
                "mean": 0.0,
                "std": 0.0,
                "prob_loss": 0.0,
                "expected_loss_given_loss": 0.0,
                "var_95": 0.0,
                "cvar_95": 0.0,
            },
        }

    returns_map = {m: daily_returns(prices[m]) for m in metals}
    aligned = pd.DataFrame(returns_map).dropna()
    if aligned.empty:
        paths = np.repeat(initial_value, n_sims * (horizon + 1)).reshape(n_sims, horizon + 1)
        final_values = paths[:, -1]
        return {
            "paths": paths,
            "final_values": final_values,
            "initial_value": initial_value,
            "percentiles": {p: initial_value for p in pct_levels},
            "metrics": {
                "mean": initial_value,
                "std": 0.0,
                "prob_loss": 0.0,
                "expected_loss_given_loss": 0.0,
                "var_95": 0.0,
                "cvar_95": 0.0,
            },
        }

    mu = np.zeros(len(metals)) if zero_drift else np.asarray(aligned.mean(), dtype=float)
    sigma = np.array([ewma_volatility(aligned[m]) for m in metals])
    if np.any(~np.isfinite(sigma)) or np.allclose(sigma, 0.0):
        paths = np.repeat(initial_value, n_sims * (horizon + 1)).reshape(n_sims, horizon + 1)
        final_values = paths[:, -1]
        return {
            "paths": paths,
            "final_values": final_values,
            "initial_value": initial_value,
            "percentiles": {p: initial_value for p in pct_levels},
            "metrics": {
                "mean": initial_value,
                "std": 0.0,
                "prob_loss": 0.0,
                "expected_loss_given_loss": 0.0,
                "var_95": 0.0,
                "cvar_95": 0.0,
            },
        }
    cov = np.asarray(aligned.cov(), dtype=float)

    D_inv = np.diag(1 / sigma)
    corr = D_inv @ cov @ D_inv
    corr = np.clip(corr, -1, 1)
    np.fill_diagonal(corr, 1.0)

    shocks = _correlated_shocks(corr, n_sims, horizon)

    # zero-drift VaR convention: martingale in price (drift = -0.5 sigma^2)
    drift = mu - 0.5 * sigma ** 2
    diffusion = sigma

    log_increments = drift + diffusion * shocks
    cum_log_returns = np.cumsum(log_increments, axis=1)

    price_ratios = np.exp(cum_log_returns)
    price_ratios = np.concatenate(
        [np.ones((n_sims, 1, len(metals))), price_ratios], axis=1
    )

    portfolio_paths = (price_ratios * exp_arr).sum(axis=2)

    final_values = portfolio_paths[:, -1]
    percentiles = {p: float(np.percentile(final_values, p)) for p in pct_levels}

    return {
        "paths": portfolio_paths,
        "final_values": final_values,
        "initial_value": initial_value,
        "percentiles": percentiles,
        "metrics": {
            "mean": float(np.mean(final_values)),
            "std": float(np.std(final_values)),
            "prob_loss": float(np.mean(final_values < initial_value)),
            "expected_loss_given_loss": float(
                np.mean(initial_value - final_values[final_values < initial_value])
                if np.any(final_values < initial_value) else 0.0
            ),
            "var_95": float(initial_value - np.percentile(final_values, 5)),
            "cvar_95": float(
                initial_value - np.mean(final_values[final_values <= np.percentile(final_values, 5)])
            ),
        },
    }
