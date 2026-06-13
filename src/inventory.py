"""Values the current inventory at market and computes unrealised P&L."""

import pandas as pd

from src.config import METALS
from src.positions import current_inventory
from src.prices import latest_price

last_reconciliation: dict[str, pd.DataFrame] = {}


def load_inventory(inbound_paths=None, outbound_paths=None) -> pd.DataFrame:
    inv, recon_grade, recon_metal, margin = current_inventory(inbound_paths, outbound_paths)
    last_reconciliation.update(
        recon_grade=recon_grade, recon_metal=recon_metal, margin=margin
    )

    bought = recon_metal["bought_t"].sum()
    sold = recon_metal["sold_t"].sum()
    on_hand = recon_metal["on_hand_t"].sum()
    print(
        f"  [+] Netted inventory: {bought:,.0f}t bought - {sold:,.0f}t sold "
        f"= {on_hand:,.0f}t on hand ({len(inv)} open lots)"
    )
    shortfalls = recon_grade[recon_grade["shortfall_t"] > 1e-6]
    if not shortfalls.empty:
        print(f"  [!] {len(shortfalls)} grade(s) sold more than purchased YTD "
              f"(floored to zero — likely opening inventory or reclassification):")
        for _, r in shortfalls.iterrows():
            print(f"        {r['grade']:<26} shortfall {r['shortfall_t']:>8,.1f}t")
    return inv


def _grade_basis(metal: str, grade: str) -> float:
    basis_map = METALS[metal]["grade_basis"]
    return basis_map.get(grade, basis_map["default"])


def mark_to_market(df: pd.DataFrame, spot_prices: dict[str, float] | None = None) -> pd.DataFrame:
    if spot_prices is None:
        spot_prices = {m: latest_price(m) for m in df["metal"].unique()}

    result = df.copy()
    result["spot_price_per_tonne"] = result["metal"].map(spot_prices)
    result["basis_factor"] = result.apply(
        lambda r: _grade_basis(r["metal"], r.get("grade", "default")), axis=1
    )
    result["scrap_price_per_tonne"] = result["spot_price_per_tonne"] * result["basis_factor"]
    result["book_value"] = result["purchase_price_per_tonne"] * result["quantity_tonnes"]
    result["mtm_value"] = result["scrap_price_per_tonne"] * result["quantity_tonnes"]
    result["unrealised_pnl"] = result["mtm_value"] - result["book_value"]

    today = pd.Timestamp.today().normalize()
    result["days_held"] = (today - result["purchase_date"]).dt.days

    result["sensitivity_per_dollar"] = result["quantity_tonnes"] * result["basis_factor"]

    return result


def portfolio_summary(mtm_df: pd.DataFrame) -> pd.DataFrame:
    df = mtm_df.copy()
    df["_basis_x_tonnes"] = df["basis_factor"] * df["quantity_tonnes"]

    summary = (
        df.groupby("metal")
        .agg(
            total_tonnes=("quantity_tonnes", "sum"),
            book_value=("book_value", "sum"),
            mtm_value=("mtm_value", "sum"),
            unrealised_pnl=("unrealised_pnl", "sum"),
            spot_price=("spot_price_per_tonne", "first"),
            avg_days_held=("days_held", "mean"),
            max_days_held=("days_held", "max"),
            sensitivity_per_dollar=("sensitivity_per_dollar", "sum"),
            _basis_x_tonnes=("_basis_x_tonnes", "sum"),
        )
        .reset_index()
    )

    summary["avg_cost_per_tonne"] = summary["book_value"] / summary["total_tonnes"]
    summary["avg_basis"] = summary["_basis_x_tonnes"] / summary["total_tonnes"]
    summary["breakeven_price"] = summary["avg_cost_per_tonne"] / summary["avg_basis"]
    summary = summary.drop(columns=["_basis_x_tonnes"])
    return summary


def total_exposure(mtm_df: pd.DataFrame) -> float:
    return float(mtm_df["mtm_value"].sum())


def exposure_by_metal(mtm_df: pd.DataFrame) -> dict[str, float]:
    return mtm_df.groupby("metal")["mtm_value"].sum().to_dict()
