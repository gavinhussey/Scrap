"""Loads and normalises outbound (sales) tickets from the CSV exports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import COMMODITY_TO_METAL, DAILY_INPUT_DIR, DATA_DIR
from src.data_loader import LBS_PER_TONNE, _parse_float, parse_dates

_OUTBOUND_GLOB = "*outbound*.csv"

DEPLETING_STATUSES = {"PAID", "INVOICED", "SHIPPED"}


def find_outbound_csvs(directory: Path | None = None) -> list[Path]:
    directory = directory or DAILY_INPUT_DIR
    combined = sorted(directory.glob("*combined*outbound*.csv"))
    if combined:
        return combined
    return sorted(p for p in directory.glob(_OUTBOUND_GLOB) if "combined" not in p.name.lower())


def _read_outbound_one(path: Path | str) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    raw.columns = [c.strip() for c in raw.columns]

    grade = raw["Commodity Name"].str.strip().str.upper()
    metal = grade.map(COMMODITY_TO_METAL)

    weight_lbs = raw["Net Weight"].map(_parse_float)
    qty_tonnes = weight_lbs / LBS_PER_TONNE
    revenue = raw.get("Expected Value", pd.Series([""] * len(raw))).map(_parse_float)
    cogs = raw.get("Cost", pd.Series([""] * len(raw))).map(_parse_float)
    qty_safe = qty_tonnes.where(qty_tonnes > 0)

    df = pd.DataFrame(
        {
            "sale_date": parse_dates(raw["Date In"]),
            "metal": metal,
            "grade": grade,
            "quantity_tonnes": qty_tonnes,
            "sale_price_per_tonne": revenue / qty_safe,
            "cogs_per_tonne": cogs / qty_safe,
            "revenue": revenue,
            "cogs": cogs,
            "yard": raw.get("Yard Name", pd.Series([""] * len(raw))).str.strip(),
            "ticket": raw.get("Outbound Ticket #", pd.Series([""] * len(raw))),
            "customer": raw.get("Customer Name", pd.Series([""] * len(raw))),
            "status": raw.get("Status", pd.Series([""] * len(raw))).str.strip().str.upper(),
        }
    )

    df = df[df["metal"].notna() & (df["quantity_tonnes"] > 0)]
    df = df.dropna(subset=["sale_date"])
    return df


def load_outbound(
    paths: list[Path | str] | None = None,
    statuses: set[str] | None = DEPLETING_STATUSES,
) -> pd.DataFrame:
    if paths is None:
        paths = find_outbound_csvs()
    if not paths:
        raise FileNotFoundError(f"No outbound CSV ({_OUTBOUND_GLOB}) found in {DAILY_INPUT_DIR}")

    frames = [_read_outbound_one(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    if statuses is not None:
        df = df[df["status"].isin(statuses)]
    df = df.sort_values("sale_date").reset_index(drop=True)
    return df
