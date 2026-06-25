# sell_signal/build_scrap_prices.py
"""Build a REAL internal scrap-price panel from outbound (sale) tickets.

Output is a wide CSV (date index, one column per material) of $/lb prices,
suitable for `Config.material_price_csv` in sell_signal/pipeline.py.
That turns the sell-signal analysis from "proxy vs the same proxy" (correlation 1.0
by construction) into "real scrap price vs market proxy" (a genuine relationship).

METHODOLOGY — fixed-weight index (why not a plain weekly average):
    Each material sells as several grade *tiers* of very different value
    (e.g. COPPER TIER 1/2/3). A naive weight-weighted weekly price swings with the
    grade *mix* sold that week, not with the underlying market. So we instead:
      1. price each (material, grade) per week  = sum(revenue) / sum(lbs),
      2. forward-fill each grade's price (carry the last known print),
      3. combine grades into one material index with FIXED weights = each grade's
         share of total period volume, renormalized over the grades priced that week.
    This isolates price movement from mix movement (the CPI fixed-basket idea).

Run (from the project root):
    python sell_signal/build_scrap_prices.py
"""

from __future__ import annotations

import os
import sys

# make the project root importable so `import src` works no matter where this is run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.data_loader import LBS_PER_TONNE
from src.outbound_loader import load_outbound

# scrap-yard metal (lowercase, as emitted by the loader) -> sell-signal material name
METAL_TO_MATERIAL: dict[str, str] = {
    "copper": "COPPER",
    "aluminium": "ALUMINUM",
    "steel": "STEEL",
    "stainless": "STAINLESS",
    "brass": "BRASS",
    "zinc": "ZINC",
}

DEFAULT_OUT = "output/sell_signal/scrap_material_prices.csv"
DEFAULT_FREQ_RULE = "W-FRI"  # weekly; ~6 months of tickets -> ~25 weekly points


def _material_index(sub: pd.DataFrame, freq_rule: str) -> pd.Series:
    """Fixed-weight $/lb index for one material from its grade-tier sale tickets."""
    # weekly price per grade = total revenue / total pounds (weight-weighted within grade)
    grp = sub.groupby([pd.Grouper(freq=freq_rule), "grade"]).agg(
        rev=("revenue", "sum"), tonnes=("quantity_tonnes", "sum"))
    grp["price_lb"] = grp["rev"] / (grp["tonnes"] * LBS_PER_TONNE)
    price = grp["price_lb"].unstack("grade").ffill()         # carry last known grade price
    if price.empty:
        return pd.Series(dtype=float)

    # fixed weights = each grade's share of total period volume
    weights = sub.groupby("grade")["quantity_tonnes"].sum()
    weights = weights.reindex(price.columns).fillna(0.0)
    if weights.sum() == 0:
        return pd.Series(dtype=float)

    weighted = price.mul(weights, axis=1)                    # NaN where a grade has no price yet
    present_w = price.notna().mul(weights, axis=1).sum(axis=1)   # renormalize over priced grades
    index = weighted.sum(axis=1, skipna=True) / present_w.replace(0, pd.NA)
    return index.dropna()


def build_scrap_prices(freq_rule: str = DEFAULT_FREQ_RULE, out_path: str = DEFAULT_OUT) -> pd.DataFrame:
    df = load_outbound()
    df = df[df["metal"].isin(METAL_TO_MATERIAL) & df["revenue"].notna() & (df["revenue"] > 0)].copy()
    df["material"] = df["metal"].map(METAL_TO_MATERIAL)
    df = df.set_index("sale_date").sort_index()

    cols: dict[str, pd.Series] = {}
    for material in sorted(df["material"].unique()):
        idx = _material_index(df[df["material"] == material], freq_rule)
        if not idx.empty:
            cols[material] = idx
    if not cols:
        raise RuntimeError("No material price series could be built from outbound tickets.")

    panel = pd.concat(cols, axis=1, sort=True).sort_index()
    panel.index.name = "date"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    panel.to_csv(out_path)

    span = f"{panel.index.min().date()} -> {panel.index.max().date()}"
    print(f"wrote {out_path}  ({len(panel)} periods x {panel.shape[1]} materials, {span})")
    print("non-null periods per material:")
    print(panel.notna().sum().to_string())
    print("\nmedian $/lb per material:")
    print(panel.median().round(3).to_string())
    return panel


if __name__ == "__main__":
    build_scrap_prices()
