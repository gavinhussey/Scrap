"""Chart rendering for risk model outputs."""

from __future__ import annotations

from pathlib import Path

from src.monte_carlo import simulate
from src.prices import daily_returns
from src.report import (
    chart_aging,
    chart_commodity_breakdown,
    chart_monte_carlo,
    chart_price_history,
    chart_return_distributions,
    chart_scenarios,
    chart_var_summary,
)
from src.risk_engine import RiskResults
from src.scenarios import run_scenarios
from src.var_model import var_table


def render_risk_charts(results: RiskResults, per_commodity: bool = True) -> list[Path]:
    """Render every chart referenced by the HTML risk report."""
    rendered: list[Path] = []
    merged_prices = {metal: results.prices[metal] for metal in results.mc_exposures}

    rendered.extend([
        chart_commodity_breakdown(results.summary),
        chart_price_history(merged_prices),
        chart_return_distributions(results.mc_returns),
        chart_monte_carlo(results.mc_results),
        chart_scenarios(results.scenario_df),
        chart_var_summary(results.var_df),
        chart_aging(results.summary),
    ])

    if per_commodity:
        for metal in sorted(results.exposures):
            exp_single = {metal: results.exposures[metal]}
            prices_single = {metal: results.risk_prices[metal]}
            returns_single = {metal: daily_returns(prices_single[metal])}

            rendered.extend([
                chart_var_summary(var_table(exp_single, returns_single), label=metal),
                chart_monte_carlo(
                    [simulate(exp_single, prices_single, horizon=h, seed=42) for h in results.horizons],
                    label=metal,
                ),
                chart_scenarios(run_scenarios(exp_single), label=metal),
            ])

    return rendered
