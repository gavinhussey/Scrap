"""Validation summaries comparing GreenSpark inventory to YTD flow files."""

from __future__ import annotations

import pandas as pd

from src.data_loader import load_inbound
from src.outbound_loader import load_outbound
from src.positions import current_inventory


def build_flow_validation(snapshot_inventory: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return metal-level flow validation and realized-margin summaries.
    The GreenSpark snapshot remains the physical inventory source. This summary
    exists to expose whether YTD inbound/outbound flow files reconcile cleanly or
    contain opening-balance/transfer effects that make netting unreliable.
    """
    inbound = load_inbound()
    outbound = load_outbound()
    _, _, recon_metal, margin = current_inventory()

    snapshot = (
        snapshot_inventory.groupby("metal", as_index=False)["quantity_tonnes"]
        .sum()
        .rename(columns={"quantity_tonnes": "snapshot_t"})
    )

    inb = (
        inbound.groupby("metal", as_index=False)["quantity_tonnes"]
        .sum()
        .rename(columns={"quantity_tonnes": "inbound_t"})
    )
    out = (
        outbound.groupby("metal", as_index=False)["quantity_tonnes"]
        .sum()
        .rename(columns={"quantity_tonnes": "outbound_t"})
    )

    validation = (
        snapshot.merge(inb, on="metal", how="outer")
        .merge(out, on="metal", how="outer")
        .merge(recon_metal, on="metal", how="outer")
        .fillna(0.0)
    )
    validation["flow_net_t"] = validation["inbound_t"] - validation["outbound_t"]
    validation["snapshot_vs_flow_net_t"] = validation["snapshot_t"] - validation["flow_net_t"]
    validation["shortfall_t"] = validation.get("shortfall_t", 0.0)
    validation["flow_reliable_for_inventory"] = validation["shortfall_t"].abs() < 1e-9

    cols = [
        "metal",
        "snapshot_t",
        "inbound_t",
        "outbound_t",
        "flow_net_t",
        "on_hand_t",
        "shortfall_t",
        "snapshot_vs_flow_net_t",
        "flow_reliable_for_inventory",
    ]
    validation = validation[cols].sort_values("snapshot_t", ascending=False).reset_index(drop=True)

    margin_by_metal = (
        margin.groupby("metal", as_index=False)
        .agg(
            sold_t=("sold_t", "sum"),
            revenue=("revenue", "sum"),
            cogs=("cogs", "sum"),
            gross_margin=("gross_margin", "sum"),
        )
        .sort_values("gross_margin", ascending=False)
    )
    margin_by_metal["margin_pct"] = (
        margin_by_metal["gross_margin"] / margin_by_metal["revenue"].where(margin_by_metal["revenue"] != 0) * 100
    )

    return validation, margin_by_metal.reset_index(drop=True)


def economic_reconciliation(mtm_df: pd.DataFrame, margin_by_metal: pd.DataFrame) -> pd.DataFrame:
    unreal = (
        mtm_df.groupby("metal", as_index=False)
        .agg(
            gross_unrealised_pnl=("unrealised_pnl", "sum"),
            net_unrealised_pnl=("net_unrealised_pnl", "sum"),
            liquidation_pnl=("liquidation_pnl", "sum"),
        )
    )
    realized = margin_by_metal[["metal", "gross_margin"]].rename(columns={"gross_margin": "realized_margin"})
    out = unreal.merge(realized, on="metal", how="outer").fillna(0.0)
    out["economic_pnl_signal"] = out["realized_margin"] + out["net_unrealised_pnl"]
    return out.sort_values("economic_pnl_signal", ascending=False).reset_index(drop=True)
