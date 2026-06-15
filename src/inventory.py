"""Values the current inventory at market and computes unrealised P&L."""

import pandas as pd

from src.config import LIQUIDATION_HAIRCUTS, METALS, NET_REALIZABLE_HAIRCUTS
from src.greenspark_inventory import LBS_PER_TONNE, load_greenspark_lots
from src.positions import current_inventory
from src.prices import latest_price

last_reconciliation: dict[str, pd.DataFrame] = {}


def load_inventory(source: str = "greenspark", inbound_paths=None, outbound_paths=None) -> pd.DataFrame:
    """Build the current on-hand lot table.

    source="greenspark" (default): physical truth from the GreenSpark snapshot.
    source="netting": legacy buys-minus-sales FIFO (understates the book — the
    outbound feed double-counts inter-yard transfers; kept for comparison only).
    """
    if source == "netting":
        inv, recon_grade, recon_metal, margin = current_inventory(inbound_paths, outbound_paths)
        last_reconciliation.update(recon_grade=recon_grade, recon_metal=recon_metal, margin=margin)
        on_hand = recon_metal["on_hand_t"].sum()
        print(f"  [+] Netted inventory: {recon_metal['bought_t'].sum():,.0f}t bought - "
              f"{recon_metal['sold_t'].sum():,.0f}t sold = {on_hand:,.0f}t on hand ({len(inv)} lots)")
        return inv

    lots, dropped = load_greenspark_lots()
    on_hand_t = lots["quantity_tonnes"].sum()
    has_mkt = lots.get("market_price_per_tonne")
    priced = int(has_mkt.notna().sum()) if has_mkt is not None else 0
    print(f"  [+] GreenSpark inventory: {on_hand_t * LBS_PER_TONNE:,.0f} lbs "
          f"({on_hand_t:,.0f}t) across {len(lots)} grades; "
          f"market price matched on {priced}/{len(lots)} "
          f"(unmatched lines are held at book unless fallback is explicit).")
    if not dropped.empty:
        d_lbs = dropped["quantity_tonnes"].sum() * LBS_PER_TONNE
        print(f"  [!] Excluded {len(dropped)} non-modelled lines ({d_lbs:,.0f} lbs, "
              f"${dropped['cost'].sum():,.0f} cost): "
              f"{', '.join(sorted(dropped['grade'].unique()))}")
    return lots


def _grade_basis(metal: str, grade: str) -> float:
    basis_map = METALS[metal]["grade_basis"]
    return basis_map.get(grade, basis_map["default"])


def mark_to_market(
    df: pd.DataFrame,
    spot_prices: dict[str, float] | None = None,
    *,
    allow_futures_fallback: bool = False,
) -> pd.DataFrame:
    if spot_prices is None:
        spot_prices = {m: latest_price(m) for m in df["metal"].unique()}

    result = df.copy()
    result["spot_price_per_tonne"] = result["metal"].map(spot_prices)
    modelled_scrap = result.apply(
        lambda r: r["spot_price_per_tonne"] * _grade_basis(r["metal"], r.get("grade", "default")),
        axis=1,
    )
    # Prefer real transacted sale price per grade. Missing prices are held at
    # book by default; futures-basis fallback must be explicitly requested.
    real = result["market_price_per_tonne"] if "market_price_per_tonne" in result else pd.Series(pd.NA, index=result.index)
    result["book_value"] = result["purchase_price_per_tonne"] * result["quantity_tonnes"]
    real_price = pd.to_numeric(real, errors="coerce")
    has_real = real_price.notna() & (real_price > 0)

    if allow_futures_fallback:
        result["scrap_price_per_tonne"] = real_price.fillna(modelled_scrap)
        result["price_source"] = has_real.map({True: "real sale", False: "futures x basis"})
        result["valuation_confidence"] = has_real.map({True: "A: real sale", False: "E: futures fallback"})
        result["basis_factor"] = result["scrap_price_per_tonne"] / result["spot_price_per_tonne"]
    else:
        result["scrap_price_per_tonne"] = real_price.where(has_real, result["purchase_price_per_tonne"])
        result["price_source"] = has_real.map({True: "real sale", False: "unpriced - held at book"})
        result["valuation_confidence"] = has_real.map({True: "A: real sale", False: "Unpriced"})
        result["basis_factor"] = (result["scrap_price_per_tonne"] / result["spot_price_per_tonne"]).where(has_real, 0.0)

    result["mtm_value"] = result["scrap_price_per_tonne"] * result["quantity_tonnes"]
    result["unrealised_pnl"] = result["mtm_value"] - result["book_value"]
    result["gross_mtm_value"] = result["mtm_value"]
    result["zero_cost_flag"] = result["book_value"].abs() < 0.01
    result["zero_cost_policy"] = result["zero_cost_flag"].map({
        True: "zero cost basis - review",
        False: "normal cost basis",
    })

    result["net_realizable_haircut"] = result["metal"].map(NET_REALIZABLE_HAIRCUTS).fillna(0.10)
    result["liquidation_haircut"] = result["metal"].map(LIQUIDATION_HAIRCUTS).fillna(0.20)
    result["net_realizable_value"] = result["mtm_value"] * (1 - result["net_realizable_haircut"])
    result["liquidation_value"] = result["mtm_value"] * (1 - result["liquidation_haircut"])
    result["net_unrealised_pnl"] = result["net_realizable_value"] - result["book_value"]
    result["liquidation_pnl"] = result["liquidation_value"] - result["book_value"]
    result["risk_exposure_value"] = result["net_realizable_value"].where(has_real, 0.0)

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
            net_realizable_value=("net_realizable_value", "sum"),
            liquidation_value=("liquidation_value", "sum"),
            risk_exposure_value=("risk_exposure_value", "sum"),
            unrealised_pnl=("unrealised_pnl", "sum"),
            net_unrealised_pnl=("net_unrealised_pnl", "sum"),
            liquidation_pnl=("liquidation_pnl", "sum"),
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


def exposure_by_metal(mtm_df: pd.DataFrame, value_col: str = "risk_exposure_value") -> dict[str, float]:
    return mtm_df.groupby("metal")[value_col].sum().to_dict()
