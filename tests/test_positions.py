"""Tests for the FIFO inbound/outbound netting engine."""

import pandas as pd
import pytest

from src.positions import build_positions, _fifo_remaining


def _lot(date, qty, price=100.0, grade="STEEL", metal="steel"):
    return {
        "purchase_date": pd.Timestamp(date), "metal": metal, "grade": grade,
        "quantity_tonnes": qty, "purchase_price_per_tonne": price, "yard": "Y",
        "ticket": "t", "customer": "c", "material_name": "m", "material_code": "mc",
    }


def _sale(date, qty, grade="STEEL", metal="steel", rev=0.0, cogs=0.0):
    return {
        "sale_date": pd.Timestamp(date), "metal": metal, "grade": grade,
        "quantity_tonnes": qty, "sale_price_per_tonne": 0.0, "cogs_per_tonne": 0.0,
        "revenue": rev, "cogs": cogs, "yard": "Y", "ticket": "t",
        "customer": "c", "status": "PAID",
    }


def test_simple_net():
    inbound = pd.DataFrame([_lot("2026-01-01", 100)])
    outbound = pd.DataFrame([_sale("2026-02-01", 70)])
    inv, recon_g, recon_m, _ = build_positions(inbound, outbound)
    assert inv["quantity_tonnes"].sum() == pytest.approx(30.0)
    assert recon_g.loc[0, "on_hand_t"] == pytest.approx(30.0)
    assert recon_g.loc[0, "shortfall_t"] == pytest.approx(0.0)


def test_fifo_consumes_oldest_first():
    inbound = pd.DataFrame([
        _lot("2026-01-01", 10, price=50),
        _lot("2026-03-01", 10, price=80),
    ])
    outbound = pd.DataFrame([_sale("2026-04-01", 15)])
    inv, _, _, _ = build_positions(inbound, outbound)
    assert inv["quantity_tonnes"].sum() == pytest.approx(5.0)
    assert (inv["purchase_price_per_tonne"] == 80).all()
    assert (inv["purchase_date"] == pd.Timestamp("2026-03-01")).all()


def test_oversold_floors_to_zero_and_flags_shortfall():
    inbound = pd.DataFrame([_lot("2026-01-01", 100)])
    outbound = pd.DataFrame([_sale("2026-02-01", 120)])
    inv, recon_g, _, _ = build_positions(inbound, outbound)
    assert inv["quantity_tonnes"].sum() == pytest.approx(0.0)
    assert recon_g.loc[0, "on_hand_t"] == pytest.approx(0.0)
    assert recon_g.loc[0, "shortfall_t"] == pytest.approx(20.0)


def test_sell_only_grade_reported():
    inbound = pd.DataFrame([_lot("2026-01-01", 10, grade="STEEL", metal="steel")])
    outbound = pd.DataFrame([_sale("2026-02-01", 5, grade="COPPER TIER 1", metal="copper")])
    _, recon_g, _, _ = build_positions(inbound, outbound)
    cu = recon_g[recon_g["grade"] == "COPPER TIER 1"].iloc[0]
    assert cu["bought_t"] == pytest.approx(0.0)
    assert cu["shortfall_t"] == pytest.approx(5.0)


def test_metal_rollup_sums_grades():
    inbound = pd.DataFrame([
        _lot("2026-01-01", 10, grade="COPPER TIER 1", metal="copper"),
        _lot("2026-01-02", 20, grade="COPPER TIER 2", metal="copper"),
    ])
    outbound = pd.DataFrame([_sale("2026-02-01", 5, grade="COPPER TIER 1", metal="copper")])
    _, _, recon_m, _ = build_positions(inbound, outbound)
    cu = recon_m[recon_m["metal"] == "copper"].iloc[0]
    assert cu["bought_t"] == pytest.approx(30.0)
    assert cu["on_hand_t"] == pytest.approx(25.0)


def test_partial_lot_preserves_cost_basis():
    survivors, shortfall = _fifo_remaining(
        pd.DataFrame([_lot("2026-01-01", 10, price=42)]), sold_tonnes=4
    )
    assert shortfall == 0.0
    assert survivors["quantity_tonnes"].sum() == pytest.approx(6.0)
    assert (survivors["purchase_price_per_tonne"] == 42).all()


def test_realized_margin():
    inbound = pd.DataFrame([_lot("2026-01-01", 10)])
    outbound = pd.DataFrame([_sale("2026-02-01", 5, rev=1000.0, cogs=600.0)])
    _, _, _, margin = build_positions(inbound, outbound)
    row = margin.iloc[0]
    assert row["gross_margin"] == pytest.approx(400.0)
    assert row["margin_pct"] == pytest.approx(40.0)
