"""Nets inbound against outbound (FIFO) to derive current on-hand inventory."""

from __future__ import annotations

import pandas as pd

from src.data_loader import load_inbound
from src.outbound_loader import load_outbound

_LOT_COLS = [
    "purchase_date", "metal", "grade", "quantity_tonnes",
    "purchase_price_per_tonne", "yard", "ticket", "customer",
    "material_name", "material_code",
]


def _fifo_remaining(lots: pd.DataFrame, sold_tonnes: float) -> tuple[pd.DataFrame, float]:
    lots = lots.sort_values("purchase_date")
    to_consume = sold_tonnes
    survivors: list[pd.Series] = []

    for _, lot in lots.iterrows():
        q = lot["quantity_tonnes"]
        if to_consume >= q:
            to_consume -= q
            continue
        if to_consume > 0:
            lot = lot.copy()
            lot["quantity_tonnes"] = q - to_consume
            to_consume = 0.0
        survivors.append(lot)

    survivors_df = pd.DataFrame(survivors, columns=lots.columns) if survivors \
        else lots.iloc[0:0].copy()
    return survivors_df, max(to_consume, 0.0)


def build_positions(
    inbound: pd.DataFrame,
    outbound: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sold_by_grade = outbound.groupby(["metal", "grade"])["quantity_tonnes"].sum()

    inventory_rows: list[pd.DataFrame] = []
    recon: list[dict] = []

    all_keys = sorted(
        set(map(tuple, inbound[["metal", "grade"]].drop_duplicates().values))
        | set(sold_by_grade.index)
    )

    for metal, grade in all_keys:
        lots = inbound[(inbound["metal"] == metal) & (inbound["grade"] == grade)]
        bought = float(lots["quantity_tonnes"].sum())
        sold = float(sold_by_grade.get((metal, grade), 0.0))

        survivors, shortfall = _fifo_remaining(lots, sold)
        on_hand = float(survivors["quantity_tonnes"].sum())
        if not survivors.empty:
            inventory_rows.append(survivors)

        recon.append({
            "metal": metal,
            "grade": grade,
            "bought_t": bought,
            "sold_t": sold,
            "on_hand_t": on_hand,
            "shortfall_t": shortfall,
        })

    inventory_df = (
        pd.concat(inventory_rows, ignore_index=True)
        if inventory_rows else pd.DataFrame(columns=_LOT_COLS)
    )
    inventory_df = inventory_df.reset_index(drop=True)

    recon_grade = pd.DataFrame(recon).sort_values(["metal", "grade"]).reset_index(drop=True)
    recon_metal = (
        recon_grade.groupby("metal")[["bought_t", "sold_t", "on_hand_t", "shortfall_t"]]
        .sum()
        .reset_index()
    )

    margin_df = realized_margin(outbound)
    return inventory_df, recon_grade, recon_metal, margin_df


def realized_margin(outbound: pd.DataFrame) -> pd.DataFrame:
    g = outbound.groupby(["metal", "grade"]).agg(
        sold_t=("quantity_tonnes", "sum"),
        revenue=("revenue", "sum"),
        cogs=("cogs", "sum"),
    ).reset_index()
    g["gross_margin"] = g["revenue"] - g["cogs"]
    g["margin_pct"] = (g["gross_margin"] / g["revenue"].where(g["revenue"] != 0)) * 100
    return g.sort_values(["metal", "grade"]).reset_index(drop=True)


def current_inventory(
    inbound_paths=None,
    outbound_paths=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inbound = load_inbound(inbound_paths)
    outbound = load_outbound(outbound_paths)
    return build_positions(inbound, outbound)
