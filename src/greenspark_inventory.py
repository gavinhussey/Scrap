"""Loads current on-hand inventory straight from the GreenSpark snapshot."""

from __future__ import annotations

import re

import pandas as pd

from src.config import COMMODITY_TO_METAL, DAILY_INPUT_DIR, DATA_DIR, OUTPUT_DIR
from src.input_validation import file_provenance, require_columns, require_nonnegative, validate_greenspark_snapshot

LBS_PER_TONNE = 2204.62
_SNAPSHOT_GLOB = "combined inventory *.csv"
_VALUATION_CSV = "valuation_today_by_grade.csv"
_BRACKET = re.compile(r"\[[^\]]*\]$")
_SNAPSHOT_REQUIRED = {
    "Location",
    "Material Code",
    "Material Name",
    "Commodity Name",
    "Total Net Weight",
    "Total Cost",
}
LAST_SNAPSHOT_VALIDATION_WARNINGS: list[str] = []
LAST_SNAPSHOT_PROVENANCE: dict[str, str | int | None] = {}


def _inbound_avg_date_by_code() -> pd.Series | None:
    dirs = [DAILY_INPUT_DIR, DATA_DIR]
    paths: list = []
    for d in dirs:
        paths = [p for p in d.glob("2026 ytd *inbound.csv") if "combined" not in p.name.lower()]
        if paths:
            break
    if not paths:
        return None
    frames = []
    for p in paths:
        raw = pd.read_csv(p, thousands=",")
        dt = pd.to_datetime(
            raw["Effective Date"].astype("string").str.replace(_BRACKET, "", regex=True),
            utc=True, errors="coerce", format="mixed",
        ).dt.tz_localize(None)
        frames.append(pd.DataFrame({
            "code": raw["Material Code"].astype(str),
            "wt": pd.to_numeric(raw["Net Weight"], errors="coerce"),
            "dt": dt,
        }))
    f = pd.concat(frames, ignore_index=True).dropna(subset=["code", "wt", "dt"])
    f = f[f["wt"] > 0]
    ref = pd.Timestamp("2025-01-01")
    f["days"] = (f["dt"] - ref).dt.total_seconds() / 86400.0
    f["wd"] = f["days"] * f["wt"]
    agg = f.groupby("code").agg(wd=("wd", "sum"), w=("wt", "sum"))
    return ref + pd.to_timedelta(agg["wd"] / agg["w"], unit="D")


def _latest_snapshot():
    files = sorted(DAILY_INPUT_DIR.glob(_SNAPSHOT_GLOB))
    if not files:
        files = sorted(DATA_DIR.glob(_SNAPSHOT_GLOB))
    if not files:
        raise FileNotFoundError(
            f"No GreenSpark snapshot ({_SNAPSHOT_GLOB}) in {DAILY_INPUT_DIR}. "
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
    price = pd.to_numeric(v["price"], errors="coerce")
    s = (price * LBS_PER_TONNE).groupby(v["code"]).first()
    return s[s > 0]


def load_greenspark_lots() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = _latest_snapshot()
    raw = pd.read_csv(path, thousands=",")
    require_columns(raw, _SNAPSHOT_REQUIRED, str(path))
    require_nonnegative(raw["Total Net Weight"], f"{path} Total Net Weight")
    require_nonnegative(raw["Total Cost"], f"{path} Total Cost")
    LAST_SNAPSHOT_VALIDATION_WARNINGS.clear()
    LAST_SNAPSHOT_VALIDATION_WARNINGS.extend(
        validate_greenspark_snapshot(raw, path, COMMODITY_TO_METAL)
    )
    LAST_SNAPSHOT_PROVENANCE.clear()
    LAST_SNAPSHOT_PROVENANCE.update(file_provenance(path))
    grade = raw["Commodity Name"].astype(str).str.strip().str.upper()

    df = pd.DataFrame({
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

    acq = _inbound_avg_date_by_code()
    snap = _snapshot_date(path)
    df["purchase_date"] = (df["material_code"].map(acq).fillna(snap)
                           if acq is not None else snap)

    mp = _market_price_per_tonne_by_code()
    if mp is not None:
        df["market_price_per_tonne"] = df["material_code"].map(mp)

    dropped = df[df["metal"].isna()].copy()
    lots = df[df["metal"].notna()].drop(columns=["cost"]).reset_index(drop=True)
    return lots, dropped


def snapshot_validation_warnings() -> list[str]:
    return list(LAST_SNAPSHOT_VALIDATION_WARNINGS)


def snapshot_provenance() -> dict[str, str | int | None]:
    return dict(LAST_SNAPSHOT_PROVENANCE)
