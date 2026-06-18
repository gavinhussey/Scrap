"""Loads and normalises inbound (purchase) tickets from the CSV exports."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.config import COMMODITY_TO_METAL, DAILY_INPUT_DIR, DATA_DIR
from src.input_validation import require_columns

LBS_PER_TONNE = 2204.62

_INBOUND_GLOB = "*inbound*.csv"
_INBOUND_REQUIRED = {
    "Commodity Name",
    "Cost",
    "Net Weight",
    "Effective Date",
}


# strip the currency junk off a value and turn it into a number, never blow up
def _parse_float(s: object) -> float:
    if s is None:
        return 0.0
    cleaned = re.sub(r"[$,()]", "", str(s)).strip()
    if not cleaned or cleaned == "-":
        return 0.0
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


# the date column is messy, try iso first then fall back to the old us format
def parse_dates(raw: pd.Series) -> pd.Series:
    s = raw.astype(str).str.replace(r"\[[^\]]*\]\s*$", "", regex=True).str.strip()
    s = s.str.replace(r"\s+t$", "", regex=True)

    dt = pd.to_datetime(s, format="ISO8601", utc=True, errors="coerce")

    missing = dt.isna()
    if missing.any():
        legacy = pd.to_datetime(s[missing], format="%m/%d/%y", errors="coerce")
        dt.loc[missing] = legacy.dt.tz_localize("UTC")

    return dt.dt.tz_localize(None).dt.normalize()


# prefer a prebuilt combined file, otherwise grab the individual inbound exports
def find_inbound_csvs(directory: Path | None = None) -> list[Path]:
    search = directory if directory is not None else DAILY_INPUT_DIR
    combined = sorted(search.glob("*combined*inbound*.csv"))
    if not combined and directory is None:
        combined = sorted(DATA_DIR.glob("*combined*inbound*.csv"))
    if combined:
        return combined
    return sorted(p for p in search.glob(_INBOUND_GLOB) if "combined" not in p.name.lower())


# read one inbound csv into our standard lot shape and drop the unusable rows
def _read_inbound_one(path: Path | str) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    raw.columns = [c.strip() for c in raw.columns]
    require_columns(raw, _INBOUND_REQUIRED, str(path))

    grade = raw["Commodity Name"].str.strip().str.upper()
    metal = grade.map(COMMODITY_TO_METAL)

    cost = raw["Cost"].map(_parse_float)
    weight_lbs = raw["Net Weight"].map(_parse_float)
    qty_tonnes = weight_lbs / LBS_PER_TONNE

    df = pd.DataFrame(
        {
            "purchase_date": parse_dates(raw["Effective Date"]),
            "metal": metal,
            "grade": grade,
            "quantity_tonnes": qty_tonnes,
            "purchase_price_per_tonne": cost / qty_tonnes.where(qty_tonnes > 0),
            "yard": raw.get("Yard Name", pd.Series([""] * len(raw))).str.strip(),
            "ticket": raw.get("Ticket #", pd.Series([""] * len(raw))),
            "customer": raw.get("Customer Name", pd.Series([""] * len(raw))),
            "material_name": raw.get("Material Name", pd.Series([""] * len(raw))),
            "material_code": raw.get("Material Code", pd.Series([""] * len(raw))),
        }
    )

    df = df[df["metal"].notna() & (df["quantity_tonnes"] > 0) & (cost.values > 0)]
    df = df.dropna(subset=["purchase_date"])
    return df


# read every inbound file and stack them into one sorted purchase history
def load_inbound(paths: list[Path | str] | None = None) -> pd.DataFrame:
    if paths is None:
        paths = find_inbound_csvs()
    if not paths:
        raise FileNotFoundError(f"No inbound CSV ({_INBOUND_GLOB}) found in {DAILY_INPUT_DIR}")

    frames = [_read_inbound_one(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("purchase_date").reset_index(drop=True)
    return df
