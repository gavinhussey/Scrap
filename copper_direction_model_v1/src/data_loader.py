"""Load and normalize a raw COMEX copper CSV into a clean OHLCV frame.

Output columns are always: date, open, high, low, close, volume (volume optional).
No future information is used anywhere here.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from .utils import ensure_dir, resolve_path, setup_logging

logger = setup_logging()

# Accepted source column names, mapped to canonical names.
_DATE_CANDIDATES = ["date", "Date", "timestamp", "Timestamp", "DATE"]
_OPEN_CANDIDATES = ["open", "Open", "OPEN"]
_HIGH_CANDIDATES = ["high", "High", "HIGH"]
_LOW_CANDIDATES = ["low", "Low", "LOW"]
_CLOSE_CANDIDATES = ["close", "Close", "CLOSE"]
_SETTLE_CANDIDATES = ["settle", "Settle", "SETTLE", "Settlement", "settlement"]
_VOLUME_CANDIDATES = ["volume", "Volume", "VOLUME", "vol", "Vol"]


def _first_present(columns: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in columns:
            return c
    return None


def load_and_clean(config: Dict) -> pd.DataFrame:
    """Read the configured CSV, standardize columns, sort, and persist a clean copy.

    Returns a DataFrame indexed 0..N-1 with columns
    [date, open, high, low, close, volume]; ``volume`` is omitted if absent.
    """
    data_cfg = config["data"]
    csv_path = resolve_path(data_cfg["input_csv"])
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found at {csv_path}. Place your COMEX copper CSV there "
            f"or update data.input_csv in the config."
        )

    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    logger.info("Loaded %s (%d rows). Columns: %s", csv_path.name, len(df), cols)

    # --- date ---
    date_col = data_cfg.get("date_column") or _first_present(cols, _DATE_CANDIDATES)
    if date_col is None:
        raise ValueError(f"No date column found. Looked for {_DATE_CANDIDATES}.")

    # --- price (close vs settle) ---
    explicit_price = data_cfg.get("price_column")
    close_col = _first_present(cols, _CLOSE_CANDIDATES)
    settle_col = _first_present(cols, _SETTLE_CANDIDATES)
    if explicit_price:
        price_col = explicit_price
    elif close_col and settle_col:
        price_col = settle_col if data_cfg.get("prefer_settle", True) else close_col
    else:
        price_col = close_col or settle_col
    if price_col is None or price_col not in cols:
        raise ValueError(
            f"No usable close/settle column. Looked for "
            f"{_CLOSE_CANDIDATES + _SETTLE_CANDIDATES}."
        )
    logger.info("Using '%s' as the close/settlement price.", price_col)

    # --- assemble canonical frame ---
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce")
    out["close"] = pd.to_numeric(df[price_col], errors="coerce")

    for canon, cands in (
        ("open", _OPEN_CANDIDATES),
        ("high", _HIGH_CANDIDATES),
        ("low", _LOW_CANDIDATES),
    ):
        src = _first_present(cols, cands)
        out[canon] = pd.to_numeric(df[src], errors="coerce") if src else pd.NA

    vol_col = _first_present(cols, _VOLUME_CANDIDATES)
    has_volume = vol_col is not None
    if has_volume:
        out["volume"] = pd.to_numeric(df[vol_col], errors="coerce")
    else:
        logger.warning("No volume column found; continuing without volume features.")

    # --- clean: sort ascending, drop undated / missing-close rows ---
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    before = len(out)
    out = out.dropna(subset=["close"]).reset_index(drop=True)
    if len(out) < before:
        logger.info("Dropped %d rows with missing close.", before - len(out))

    # Forward-fill OHL using PAST values only (never future). Close already clean.
    for col in ["open", "high", "low"]:
        if col in out.columns:
            out[col] = out[col].ffill()

    # Reorder canonical columns.
    ordered = ["date", "open", "high", "low", "close"]
    if has_volume:
        ordered.append("volume")
    out = out[ordered]

    # Persist a processed copy.
    proc_dir = ensure_dir("data/processed")
    proc_path = proc_dir / "clean_prices.csv"
    out.to_csv(proc_path, index=False)
    logger.info("Saved cleaned prices -> %s (%d rows).", proc_path, len(out))

    return out
