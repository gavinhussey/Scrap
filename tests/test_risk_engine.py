import pandas as pd
import pytest

from src import risk_engine


# VaR and monte carlo should see the vol adjusted series, not the raw spot prices
def test_build_market_risk_uses_adjusted_risk_series_for_var_and_mc(monkeypatch):
    dates = pd.bdate_range("2026-01-01", periods=4)
    spot = pd.Series([100.0, 101.0, 102.0, 103.0], index=dates, name="copper")
    adjusted = pd.Series([100.0, 110.0, 121.0, 133.1], index=dates, name="copper")
    captured = {}

    inventory = pd.DataFrame([{
        "purchase_date": pd.Timestamp("2026-01-01"),
        "metal": "copper",
        "grade": "COPPER TIER 1",
        "quantity_tonnes": 1.0,
        "purchase_price_per_tonne": 50.0,
        "market_price_per_tonne": 80.0,
    }])

    def fake_var_table(exposures, returns_map):
        captured["var_returns"] = returns_map["copper"].copy()
        return pd.DataFrame([{
            "confidence": "95%",
            "method": "Parametric",
            "portfolio_var": 1.0,
            "portfolio_cvar": 2.0,
            "diversification_benefit": 0.0,
        }])

    def fake_simulate(exposures, price_map, horizon, seed):
        captured.setdefault("mc_prices", price_map["copper"].copy())
        return {
            "paths": [],
            "final_values": [],
            "initial_value": 1.0,
            "percentiles": {1: 1.0, 5: 1.0, 10: 1.0, 25: 1.0, 50: 1.0, 75: 1.0, 90: 1.0, 95: 1.0, 99: 1.0},
            "metrics": {"mean": 1.0, "std": 0.0, "prob_loss": 0.0, "expected_loss_given_loss": 0.0, "var_95": 0.0, "cvar_95": 0.0},
        }

    monkeypatch.setattr(risk_engine, "fetch_all_prices", lambda force_refresh=False: {"copper": spot})
    monkeypatch.setattr(risk_engine, "risk_series", lambda metal, series: adjusted)
    monkeypatch.setattr(risk_engine, "load_inventory", lambda: inventory)
    monkeypatch.setattr(risk_engine, "var_table", fake_var_table)
    monkeypatch.setattr(risk_engine, "simulate", fake_simulate)
    monkeypatch.setattr(risk_engine, "price_source_summary", lambda: {"copper": "test"})

    result = risk_engine.build_market_risk(horizons=[30])

    assert result.prices["copper"].equals(spot)
    assert result.risk_prices["copper"].equals(adjusted)
    assert captured["var_returns"].equals(adjusted.pct_change().dropna())
    assert captured["mc_prices"].equals(adjusted)


# with synthetic prices banned the build should refuse to run
def test_build_market_risk_can_fail_on_synthetic_prices(monkeypatch):
    monkeypatch.setattr(risk_engine, "fetch_all_prices", lambda force_refresh=False: {"copper": pd.Series([1.0, 2.0])})
    monkeypatch.setattr(risk_engine, "price_source_summary", lambda: {"copper": "synthetic"})
    monkeypatch.setattr(risk_engine, "risk_series", lambda metal, series: series)

    inventory = pd.DataFrame([{
        "purchase_date": pd.Timestamp("2026-01-01"),
        "metal": "copper",
        "grade": "COPPER TIER 1",
        "quantity_tonnes": 1.0,
        "purchase_price_per_tonne": 50.0,
        "market_price_per_tonne": 80.0,
    }])
    monkeypatch.setattr(risk_engine, "load_inventory", lambda: inventory)
    monkeypatch.setattr(risk_engine, "snapshot_validation_warnings", lambda: [])
    monkeypatch.setattr(risk_engine, "snapshot_provenance", lambda: {})
    monkeypatch.setattr(risk_engine, "var_table", lambda exposures, returns: pd.DataFrame())
    monkeypatch.setattr(
        risk_engine,
        "simulate",
        lambda exposures, prices, horizon, seed: {"initial_value": 0, "percentiles": {}, "metrics": {}},
    )
    monkeypatch.setattr(risk_engine, "run_scenarios", lambda exposures: pd.DataFrame([{"scenario": "x", "total_pnl": 0, "total_pnl_pct": 0}]))
    monkeypatch.setattr(risk_engine, "worst_case_summary", lambda df: {})

    with pytest.raises(RuntimeError, match="Synthetic price series not allowed"):
        risk_engine.build_market_risk(horizons=[30], allow_synthetic=False)
