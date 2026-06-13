"""Applies named price-shock stress scenarios to the portfolio."""

import pandas as pd

from src.config import STRESS_SCENARIOS


def run_scenarios(
    exposures: dict[str, float],
    scenarios: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:

    if scenarios is None:
        scenarios = STRESS_SCENARIOS

    total_exposure = sum(exposures.values())
    rows = []

    for name, shocks in scenarios.items():
        metal_pnl = {}
        total_pnl = 0.0
        for metal, exposure in exposures.items():
            shock = shocks.get(metal, 0.0)
            pnl = exposure * shock
            metal_pnl[metal] = pnl
            total_pnl += pnl

        rows.append({
            "scenario": name,
            **{f"{m}_pnl": metal_pnl.get(m, 0.0) for m in exposures},
            "total_pnl": total_pnl,
            "total_pnl_pct": (total_pnl / total_exposure * 100) if total_exposure else 0.0,
            "portfolio_after": total_exposure + total_pnl,
        })

    return pd.DataFrame(rows)


def worst_case_summary(scenario_df: pd.DataFrame) -> dict:
    worst = scenario_df.loc[scenario_df["total_pnl"].idxmin()]
    best = scenario_df.loc[scenario_df["total_pnl"].idxmax()]
    return {
        "worst_scenario": worst["scenario"],
        "worst_pnl": worst["total_pnl"],
        "worst_pnl_pct": worst["total_pnl_pct"],
        "best_scenario": best["scenario"],
        "best_pnl": best["total_pnl"],
        "best_pnl_pct": best["total_pnl_pct"],
    }
