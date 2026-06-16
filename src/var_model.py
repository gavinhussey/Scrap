"""Value at Risk and Expected Shortfall (CVaR) calculations."""

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.config import EWMA_LAMBDA, HOLDING_PERIOD_DAYS, VAR_CONFIDENCE_LEVELS
from src.prices import ewma_volatility


def _scale_to_horizon(one_day_var: float, horizon: int) -> float:
    return one_day_var * np.sqrt(horizon)


def parametric_var(
    exposure: float,
    returns: pd.Series,
    confidence: float,
    horizon: int = HOLDING_PERIOD_DAYS,
) -> tuple[float, float]:
    sigma_daily = float(returns.std())
    if exposure <= 0 or not np.isfinite(sigma_daily):
        return 0.0, 0.0
    z = norm.ppf(confidence)
    var_1d = exposure * sigma_daily * z
    var_nh = _scale_to_horizon(var_1d, horizon)
    cvar_nh = exposure * sigma_daily * norm.pdf(z) / (1 - confidence) * np.sqrt(horizon)
    return var_nh, cvar_nh


def ewma_var(
    exposure: float,
    returns: pd.Series,
    confidence: float,
    horizon: int = HOLDING_PERIOD_DAYS,
    lam: float = EWMA_LAMBDA,
) -> tuple[float, float]:
    if exposure <= 0 or returns.dropna().empty:
        return 0.0, 0.0
    sigma_daily = ewma_volatility(returns, lam)
    if not np.isfinite(sigma_daily):
        return 0.0, 0.0
    z = norm.ppf(confidence)
    var_1d = exposure * sigma_daily * z
    var_nh = _scale_to_horizon(var_1d, horizon)
    cvar_nh = exposure * sigma_daily * norm.pdf(z) / (1 - confidence) * np.sqrt(horizon)
    return var_nh, cvar_nh


def historical_var(
    exposure: float,
    returns: pd.Series,
    confidence: float,
    horizon: int = HOLDING_PERIOD_DAYS,
) -> tuple[float, float]:
    r = returns.dropna().values
    if exposure <= 0 or len(r) == 0:
        return 0.0, 0.0

    if len(r) > horizon:
        window_returns = np.array([
            np.prod(1 + r[i: i + horizon]) - 1
            for i in range(len(r) - horizon + 1)
        ])
    else:
        window_returns = r

    losses = -exposure * window_returns
    var = float(np.percentile(losses, confidence * 100))
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if len(tail) > 0 else var
    return var, cvar


def portfolio_var(
    exposures: dict[str, float],
    returns_map: dict[str, pd.Series],
    confidence: float = 0.95,
    horizon: int = HOLDING_PERIOD_DAYS,
    method: str = "historical",
) -> dict:
    metals = list(exposures.keys())
    total_exposure = float(sum(exposures.values()))
    if not metals or total_exposure <= 0:
        return {"per_metal": {}, "portfolio_var": 0.0, "portfolio_cvar": 0.0, "diversification_benefit": 0.0}

    aligned = pd.DataFrame({m: returns_map[m] for m in metals}).dropna()
    if aligned.empty:
        return {
            "per_metal": {m: {"var": 0.0, "cvar": 0.0} for m in metals},
            "portfolio_var": 0.0,
            "portfolio_cvar": 0.0,
            "diversification_benefit": 0.0,
        }

    per_metal: dict[str, dict] = {}
    for metal in metals:
        exp = exposures[metal]
        r = aligned[metal]
        if method == "parametric":
            v, cv = parametric_var(exp, r, confidence, horizon)
        elif method == "ewma":
            v, cv = ewma_var(exp, r, confidence, horizon)
        else:
            v, cv = historical_var(exp, r, confidence, horizon)
        per_metal[metal] = {"var": v, "cvar": cv}

    weights = np.array([exposures[m] for m in metals])
    if method == "ewma":
        ewma_vols = np.array([ewma_volatility(aligned[m], EWMA_LAMBDA) for m in metals])
        D = np.diag(ewma_vols)
        corr_hist = aligned.corr().values
        cov = D @ corr_hist @ D * horizon
    else:
        cov = aligned.cov().values * horizon
    port_variance = float(weights @ cov @ weights)
    port_sigma = np.sqrt(max(port_variance, 0))

    if method in ("parametric", "ewma"):
        z = norm.ppf(confidence)
        port_var = z * port_sigma
        port_cvar = norm.pdf(z) / (1 - confidence) * port_sigma
    else:
        portfolio_returns = aligned @ (weights / total_exposure)
        pr = portfolio_returns.values
        if len(pr) > horizon:
            window_losses = np.array([
                    -total_exposure * (np.prod(1 + pr[i: i + horizon]) - 1)
                for i in range(len(pr) - horizon + 1)
            ])
        else:
            window_losses = -(pr * total_exposure)
        port_var = float(np.percentile(window_losses, confidence * 100))
        tail = window_losses[window_losses >= port_var]
        port_cvar = float(tail.mean()) if len(tail) > 0 else port_var

    return {
        "per_metal": per_metal,
        "portfolio_var": port_var,
        "portfolio_cvar": port_cvar,
        "diversification_benefit": sum(v["var"] for v in per_metal.values()) - port_var,
    }


def var_table(
    exposures: dict[str, float],
    returns_map: dict[str, pd.Series],
    horizon: int = HOLDING_PERIOD_DAYS,
) -> pd.DataFrame:
    rows = []
    for conf in VAR_CONFIDENCE_LEVELS:
        for method in ("parametric", "ewma", "historical"):
            result = portfolio_var(exposures, returns_map, conf, horizon, method)
            rows.append({
                "confidence": f"{int(conf * 100)}%",
                "method": method.capitalize(),
                "portfolio_var": result["portfolio_var"],
                "portfolio_cvar": result["portfolio_cvar"],
                "diversification_benefit": result["diversification_benefit"],
            })
    return pd.DataFrame(rows)
