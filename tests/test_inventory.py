import pandas as pd
import pytest

from src.inventory import mark_to_market


# one fake lead lot to feed mark_to_market, market price optional
def _inventory_row(market_price_per_tonne=pd.NA):
    return pd.DataFrame([{
        "purchase_date": pd.Timestamp("2026-06-12"),
        "metal": "lead",
        "grade": "LEAD",
        "quantity_tonnes": 10.0,
        "purchase_price_per_tonne": 100.0,
        "market_price_per_tonne": market_price_per_tonne,
    }])


# no market price means we hold at book but still carry conservative proxy risk
def test_missing_market_price_is_held_at_book_by_default():
    mtm = mark_to_market(_inventory_row(), {"lead": 2_000.0})
    row = mtm.iloc[0]

    assert row["price_source"] == "unpriced - held at book"
    assert row["valuation_confidence"] == "Unpriced"
    assert row["scrap_price_per_tonne"] == pytest.approx(100.0)
    assert row["mtm_value"] == pytest.approx(1_000.0)
    assert row["unrealised_pnl"] == pytest.approx(0.0)
    assert row["sensitivity_per_dollar"] == pytest.approx(0.0)
    assert row["risk_exposure_source"] == "conservative proxy - unpriced"
    assert row["risk_exposure_value"] == pytest.approx(7_920.0)


# opting into the futures fallback values it at spot times basis instead
def test_missing_market_price_can_use_explicit_futures_fallback():
    mtm = mark_to_market(_inventory_row(), {"lead": 2_000.0}, allow_futures_fallback=True)
    row = mtm.iloc[0]

    assert row["price_source"] == "futures x basis"
    assert row["valuation_confidence"] == "E: futures fallback"
    assert row["scrap_price_per_tonne"] == pytest.approx(900.0)
    assert row["mtm_value"] == pytest.approx(9_000.0)
    assert row["unrealised_pnl"] == pytest.approx(8_000.0)


# a real sale price flows straight through to mtm and the haircuts
def test_real_market_price_is_used_for_mtm():
    mtm = mark_to_market(_inventory_row(250.0), {"lead": 2_000.0})
    row = mtm.iloc[0]

    assert row["price_source"] == "real sale"
    assert row["valuation_confidence"] == "A: real sale"
    assert row["scrap_price_per_tonne"] == pytest.approx(250.0)
    assert row["mtm_value"] == pytest.approx(2_500.0)
    assert row["unrealised_pnl"] == pytest.approx(1_500.0)
    assert row["net_realizable_value"] == pytest.approx(2_200.0)
    assert row["liquidation_value"] == pytest.approx(1_750.0)
    assert row["risk_exposure_source"] == "priced nrv"
    assert row["risk_exposure_value"] == pytest.approx(2_200.0)


# a zero cost lot gets flagged so someone takes a look
def test_zero_cost_inventory_is_flagged_for_review():
    row = _inventory_row(250.0)
    row["purchase_price_per_tonne"] = 0.0
    mtm = mark_to_market(row, {"lead": 2_000.0})

    assert mtm.iloc[0]["zero_cost_flag"]
    assert mtm.iloc[0]["zero_cost_policy"] == "zero cost basis - review"
