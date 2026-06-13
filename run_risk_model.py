"""
Entry point for the Scrapyard Risk Model.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src import config
from src.inventory import (
    exposure_by_metal,
    load_inventory,
    mark_to_market,
    portfolio_summary,
    total_exposure,
)
from src.monte_carlo import simulate
from src.prices import daily_returns, fetch_all_prices, merge_exposures_by_driver
from src.report import (
    chart_aging,
    chart_commodity_breakdown,
    chart_monte_carlo,
    chart_price_history,
    chart_return_distributions,
    chart_scenarios,
    chart_var_summary,
    print_inventory,
    print_mc_horizons,
    print_monte_carlo,
    print_scenarios,
    print_var,
)
from src.config import METALS
from src.scenarios import run_scenarios, worst_case_summary
from src.var_model import var_table


def main(refresh: bool = False, charts: bool = True) -> None:
    print("\nFetching metal prices...")
    prices = fetch_all_prices(force_refresh=refresh)
    spot_prices = {m: float(s.iloc[-1]) for m, s in prices.items()}

    print("Building inventory...")
    inventory = load_inventory()
    mtm = mark_to_market(inventory, spot_prices)
    summary = portfolio_summary(mtm)
    total_mtm = total_exposure(mtm)
    total_pnl = float(mtm["unrealised_pnl"].sum())
    exposures = exposure_by_metal(mtm)

    # Build per-metal returns; proxy metals (brass) share their driver's series
    returns_map = {m: daily_returns(prices[m]) for m in exposures}

    # For VaR and Monte Carlo, collapse metals that share the same price driver
    # (e.g. brass + copper both driven by HG=F) to avoid a singular covariance matrix.
    mc_exposures, mc_returns = merge_exposures_by_driver(exposures, returns_map)

    print("Running VaR calculations...")
    var_df = var_table(mc_exposures, mc_returns)

    print("Running Monte Carlo simulation (4 horizons)...")
    mc_horizons = [30, 60, 90, 180]
    mc_results = [simulate(mc_exposures, prices, horizon=h, seed=42) for h in mc_horizons]
    mc = mc_results[0]

    print("Running stress scenarios...")
    scenario_df = run_scenarios(exposures)
    ws = worst_case_summary(scenario_df)

    # ── Print report ──────────────────────────────────────────────────────────
    print_inventory(summary, total_mtm, total_pnl)
    print_var(var_df)
    print_monte_carlo(mc)
    print_mc_horizons(mc_results, mc_horizons)
    print_scenarios(scenario_df, ws)

    # ── Charts ────────────────────────────────────────────────────────────────
    if charts:
        print("Generating charts...")
        merged_prices = {m: prices[m] for m in mc_exposures}
        p0 = chart_commodity_breakdown(summary)
        p1 = chart_price_history(merged_prices)
        p2 = chart_return_distributions(mc_returns)
        p3 = chart_monte_carlo(mc_results)
        p4 = chart_scenarios(scenario_df)
        p5 = chart_var_summary(var_df)
        p6 = chart_aging(summary)
        print(f"\n  Charts saved to: {config.CHARTS_DIR}")
        for p in [p0, p1, p2, p3, p4, p5, p6]:
            print(f"    {p.name}")

    # ── Per-commodity risk charts ─────────────────────────────────────────────
    if charts:
        print("Generating per-commodity charts...")
        for metal in sorted(exposures.keys()):
            exp_single = {metal: exposures[metal]}
            driver = METALS.get(metal, {}).get("price_proxy", metal)
            prices_single = {metal: prices[driver].rename(metal)}
            returns_single = {metal: daily_returns(prices_single[metal])}

            var_single   = var_table(exp_single, returns_single)
            mc_single    = [simulate(exp_single, prices_single, horizon=h, seed=42) for h in mc_horizons]
            scen_single  = run_scenarios(exp_single)

            chart_var_summary(var_single, label=metal)
            chart_monte_carlo(mc_single, label=metal)
            chart_scenarios(scen_single, label=metal)
            print(f"    {metal} ✓")

    print("\nDone.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrapyard Risk Model")
    parser.add_argument("--refresh", action="store_true", help="Force re-download of price data")
    parser.add_argument("--no-charts", dest="charts", action="store_false", help="Skip chart generation")
    args = parser.parse_args()
    main(refresh=args.refresh, charts=args.charts)
