"""Loads current on-hand inventory straight from the GreenSpark snapshot.

This replaces the flow-netting in ``positions.py`` as the inventory source for the
risk model. Netting (buys - sales YTD) collapsed the book to ~48t because outbound
tickets include inter-yard transfers that were never booked as purchases, so the
backward identity breaks. The GreenSpark "combined inventory" export is the physical
truth: weight and Total Cost per grade per yard, with zero inventory variance.

Each grade line becomes one lot in the shape ``mark_to_market`` expects. Real per-grade
market prices (from our own transacted sales, via ``inventory_valuation.py``) are merged
in from ``output/valuation_today_by_grade.csv`` when present, so mark-to-market uses
actual sale prices instead of the hardcoded futures-basis haircuts.
"""

from __future__ import annotations

import re

import pandas as pd

from src.config import COMMODITY_TO_METAL, DATA_DIR, OUTPUT_DIR

LBS_PER_TONNE = 2204.62
_SNAPSHOT_GLOB = "combined inventory *.csv"
_VALUATION_CSV = "valuation_today_by_grade.csv"


def _latest_snapshot():
    files = sorted(DATA_DIR.glob(_SNAPSHOT_GLOB))
    if not files:
        raise FileNotFoundError(
            f"No GreenSpark snapshot ({_SNAPSHOT_GLOB}) in {DATA_DIR}. "
            "Run data/merge_inventory.py first."
        )
    return files[-1]


def _snapshot_date(path) -> pd.Timestamp:
    m = re.search(r"(\d{8})", path.name)
    return pd.to_datetime(m.group(1), format="%Y%m%d") if m else pd.Timestamp.today().normalize()


def _market_price_per_tonne_by_code() -> pd.Series | None:
    """Real $/tonne sale price per material code from inventory_valuation.py output."""
    path = OUTPUT_DIR / _VALUATION_CSV
    if not path.exists():
        return None
    v = pd.read_csv(path)
    v["code"] = v["code"].astype(str)
    price = pd.to_numeric(v["price"], errors="coerce")          # $/lb, identical per code across yards
    s = (price * LBS_PER_TONNE).groupby(v["code"]).first()
    return s[s > 0]


def load_greenspark_lots() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (modelled lots, dropped lines) from the latest GreenSpark snapshot.

    Dropped lines are grades whose commodity has no metal in the risk model
    (LEAD / ZINC / OTHER / ZWASTE) — returned so the caller can report what is
    excluded rather than silently losing it.
    """
    path = _latest_snapshot()
    raw = pd.read_csv(path, thousands=",")
    grade = raw["Commodity Name"].astype(str).str.strip().str.upper()

    df = pd.DataFrame({
        "purchase_date": _snapshot_date(path),
        "metal": grade.map(COMMODITY_TO_METAL),
        "grade": grade,
        "quantity_tonnes": pd.to_numeric(raw["Total Net Weight"], errors="coerce").fillna(0.0) / LBS_PER_TONNE,
        "cost": pd.to_numeric(raw["Total Cost"], errors="coerce").fillna(0.0),
        "yard": raw["Location"],
        "ticket": "",
        "customer": "GREENSPARK SNAPSHOT",
        "material_name": raw["Material Name"],
        "material_code": raw["Material Code"].astype(str),
    })
    df = df[df["quantity_tonnes"] > 0].copy()
    df["purchase_price_per_tonne"] = df["cost"] / df["quantity_tonnes"]

    mp = _market_price_per_tonne_by_code()
    if mp is not None:
        df["market_price_per_tonne"] = df["material_code"].map(mp)

    dropped = df[df["metal"].isna()].copy()
    lots = df[df["metal"].notna()].drop(columns=["cost"]).reset_index(drop=True)
    return lots, dropped
