"""Entry point for the Scrapyard Risk Model."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src import config
from src.charting import render_risk_charts
from src.report import (
    print_inventory,
    print_mc_horizons,
    print_monte_carlo,
    print_scenarios,
    print_var,
)
from src.risk_engine import build_market_risk


def main(refresh: bool = False, charts: bool = True, allow_synthetic: bool = True) -> None:
    print("\nFetching metal prices...")
    print("Building inventory...")
    results = build_market_risk(refresh=refresh, allow_synthetic=allow_synthetic)
    for warning in results.model_warnings:
        print(f"  [!] {warning}")

    print("Running VaR calculations...")
    var_df = results.var_df

    print("Running Monte Carlo simulation (4 horizons)...")
    mc_horizons = results.horizons
    mc_results = results.mc_results
    mc = results.mc

    print("Running stress scenarios...")
    scenario_df = results.scenario_df
    ws = results.worst_case

    # Print report
    print_inventory(results.summary, results.total_mtm, results.total_pnl)
    print_var(var_df)
    print_monte_carlo(mc)
    print_mc_horizons(mc_results, mc_horizons)
    print_scenarios(scenario_df, ws)

    # Charts
    if charts:
        print("Generating charts...")
        rendered = render_risk_charts(results)
        print(f"\n  Charts saved to: {config.CHARTS_DIR}")
        for p in rendered:
            print(f"    {p.name}")

    print("\nDone.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrapyard Risk Model")
    parser.add_argument("--refresh", action="store_true", help="Force re-download of price data")
    parser.add_argument("--no-charts", dest="charts", action="store_false", help="Skip chart generation")
    parser.add_argument(
        "--fail-on-synthetic",
        action="store_true",
        help="Fail instead of producing a report if synthetic price data is required.",
    )
    args = parser.parse_args()
    main(refresh=args.refresh, charts=args.charts, allow_synthetic=not args.fail_on_synthetic)
