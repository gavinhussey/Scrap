"""Shared assembly for market-risk calculations."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.greenspark_inventory import snapshot_provenance, snapshot_validation_warnings
from src.inventory import (
    exposure_by_metal,
    load_inventory,
    mark_to_market,
    portfolio_summary,
    total_exposure,
)
from src.monte_carlo import simulate
from src.prices import (
    daily_returns,
    fetch_all_prices,
    merge_exposures_by_driver,
    price_source_summary,
    risk_series,
)
from src.scenarios import run_scenarios, worst_case_summary
from src.var_model import var_table


@dataclass
class RiskResults:
    prices: dict[str, pd.Series]
    risk_prices: dict[str, pd.Series]
    spot_prices: dict[str, float]
    inventory: pd.DataFrame
    mtm: pd.DataFrame
    summary: pd.DataFrame
    total_mtm: float
    total_pnl: float
    exposures: dict[str, float]
    returns_map: dict[str, pd.Series]
    mc_exposures: dict[str, float]
    mc_returns: dict[str, pd.Series]
    var_df: pd.DataFrame
    horizons: list[int]
    mc_results: list[dict]
    mc: dict
    scenario_df: pd.DataFrame
    worst_case: dict
    price_sources: dict[str, str]
    validation_warnings: list[str]
    input_provenance: dict[str, str | int | None]
    data_quality_metrics: dict[str, float]
    model_status: str
    model_warnings: list[str]


def _model_warnings(
    price_sources: dict[str, str],
    mtm: pd.DataFrame,
    validation_warnings: list[str],
) -> list[str]:
    warnings: list[str] = []
    synthetic = sorted(m for m, source in price_sources.items() if "synthetic" in source)
    proxied = sorted(m for m, source in price_sources.items() if source.startswith("proxy:"))
    unpriced = mtm[mtm["valuation_confidence"] == "Unpriced"]

    if synthetic:
        warnings.append(f"Synthetic price series in use: {', '.join(synthetic)}")
    if proxied:
        warnings.append(f"Proxy price series in use: {', '.join(proxied)}")
    if not unpriced.empty:
        value = float(unpriced["risk_exposure_value"].sum())
        warnings.append(f"Unpriced inventory included in risk via conservative proxy: ${value:,.0f}")
    warnings.extend(validation_warnings)

    return warnings


def _data_quality_metrics(mtm: pd.DataFrame, exposures: dict[str, float]) -> dict[str, float]:
    total_tonnes = float(mtm["quantity_tonnes"].sum())
    priced = mtm[mtm["valuation_confidence"] != "Unpriced"]
    unpriced = mtm[mtm["valuation_confidence"] == "Unpriced"]
    total_exposure = float(sum(exposures.values()))
    max_metal_exposure = max(exposures.values()) if exposures else 0.0

    return {
        "priced_tonnes_pct": float(priced["quantity_tonnes"].sum() / total_tonnes) if total_tonnes else 0.0,
        "unpriced_tonnes": float(unpriced["quantity_tonnes"].sum()),
        "unpriced_risk_exposure": float(unpriced["risk_exposure_value"].sum()),
        "max_metal_exposure_pct": float(max_metal_exposure / total_exposure) if total_exposure else 0.0,
    }


def _status(
    price_sources: dict[str, str],
    validation_warnings: list[str],
    metrics: dict[str, float],
) -> str:
    has_synthetic = any("synthetic" in source for source in price_sources.values())
    has_stale_cache = any("stale-cache" in source for source in price_sources.values())
    if has_synthetic:
        return "do_not_use"
    if has_stale_cache or validation_warnings or metrics["priced_tonnes_pct"] < 0.90:
        return "degraded"
    if metrics["unpriced_tonnes"] > 0 or metrics["max_metal_exposure_pct"] > 0.50:
        return "review"
    return "ok"


def build_market_risk(
    refresh: bool = False,
    horizons: list[int] | None = None,
    *,
    allow_synthetic: bool = True,
) -> RiskResults:
    if horizons is None:
        horizons = [30, 60, 90, 180]

    prices = fetch_all_prices(force_refresh=refresh)
    risk_prices = {metal: risk_series(metal, series) for metal, series in prices.items()}
    spot_prices = {metal: float(series.iloc[-1]) for metal, series in prices.items()}

    inventory = load_inventory()
    mtm = mark_to_market(inventory, spot_prices)
    summary = portfolio_summary(mtm)
    total_mtm = total_exposure(mtm)
    total_pnl = float(mtm["unrealised_pnl"].sum())
    exposures = exposure_by_metal(mtm)

    returns_map = {metal: daily_returns(risk_prices[metal]) for metal in exposures}
    mc_exposures, mc_returns = merge_exposures_by_driver(exposures, returns_map)
    var_df = var_table(mc_exposures, mc_returns)
    mc_results = [simulate(mc_exposures, risk_prices, horizon=h, seed=42) for h in horizons]
    scenario_df = run_scenarios(exposures)

    price_sources = price_source_summary()
    validation_warnings = snapshot_validation_warnings()
    synthetic = sorted(m for m, source in price_sources.items() if "synthetic" in source)
    if synthetic and not allow_synthetic:
        raise RuntimeError(f"Synthetic price series not allowed: {', '.join(synthetic)}")
    warnings = _model_warnings(price_sources, mtm, validation_warnings)
    metrics = _data_quality_metrics(mtm, exposures)
    if metrics["max_metal_exposure_pct"] > 0.50:
        warnings.append(f"Concentration warning: largest metal is {metrics['max_metal_exposure_pct']:.1%} of risk exposure")

    return RiskResults(
        prices=prices,
        risk_prices=risk_prices,
        spot_prices=spot_prices,
        inventory=inventory,
        mtm=mtm,
        summary=summary,
        total_mtm=total_mtm,
        total_pnl=total_pnl,
        exposures=exposures,
        returns_map=returns_map,
        mc_exposures=mc_exposures,
        mc_returns=mc_returns,
        var_df=var_df,
        horizons=horizons,
        mc_results=mc_results,
        mc=mc_results[0],
        scenario_df=scenario_df,
        worst_case=worst_case_summary(scenario_df),
        price_sources=price_sources,
        validation_warnings=validation_warnings,
        input_provenance=snapshot_provenance(),
        data_quality_metrics=metrics,
        model_status=_status(price_sources, validation_warnings, metrics),
        model_warnings=warnings,
    )
