import pandas as pd

from src.monte_carlo import simulate
from src.var_model import portfolio_var


def test_portfolio_var_returns_zero_for_zero_exposure():
    returns = pd.Series([0.01, -0.02, 0.01])
    result = portfolio_var({"copper": 0.0}, {"copper": returns})

    assert result["portfolio_var"] == 0.0
    assert result["portfolio_cvar"] == 0.0


def test_monte_carlo_handles_zero_volatility_series():
    prices = {"copper": pd.Series([100.0] * 40)}
    result = simulate({"copper": 1_000.0}, prices, horizon=30, n_sims=100)

    assert result["initial_value"] == 1_000.0
    assert result["metrics"]["var_95"] == 0.0
    assert result["metrics"]["cvar_95"] == 0.0
    assert result["percentiles"][5] == 1_000.0
