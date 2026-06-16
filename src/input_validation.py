"""Input validation helpers for model CSV files."""

from __future__ import annotations

import re
import hashlib
from pathlib import Path

import pandas as pd

STALE_SNAPSHOT_BUSINESS_DAYS = 3
EXTREME_ROW_WEIGHT_LBS = 500_000
EXTREME_COST_PER_LB = 10.0


def require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{label} missing required column(s): {', '.join(missing)}")


def require_nonnegative(series: pd.Series, label: str) -> None:
    numeric = pd.to_numeric(series, errors="coerce")
    bad = numeric < 0
    if bad.any():
        examples = ", ".join(str(i) for i in list(series[bad].index[:5]))
        raise ValueError(f"{label} has negative value(s) at row index: {examples}")


def _snapshot_date(path: Path) -> pd.Timestamp | None:
    match = re.search(r"(\d{8})", path.name)
    if not match:
        return None
    return pd.to_datetime(match.group(1), format="%Y%m%d", errors="coerce")


def _business_days_old(snapshot_date: pd.Timestamp, today: pd.Timestamp) -> int:
    start = snapshot_date.normalize() + pd.Timedelta(days=1)
    end = today.normalize()
    if start > end:
        return 0
    return len(pd.bdate_range(start, end))


def validate_greenspark_snapshot(
    raw: pd.DataFrame,
    path: Path,
    commodity_map: dict[str, str | None],
    *,
    today: pd.Timestamp | None = None,
    stale_business_days: int = STALE_SNAPSHOT_BUSINESS_DAYS,
) -> list[str]:
    """Return non-fatal warnings for a GreenSpark inventory snapshot."""
    warnings: list[str] = []
    today = today or pd.Timestamp.today().normalize()

    snap_date = _snapshot_date(path)
    if snap_date is None or pd.isna(snap_date):
        warnings.append(f"Snapshot filename has no YYYYMMDD date: {path.name}")
    else:
        age = _business_days_old(snap_date, today)
        if age > stale_business_days:
            warnings.append(
                f"Inventory snapshot is {age} business days old "
                f"({snap_date.date()} from {path.name})"
            )

    dup_cols = ["Location", "Material Code"]
    if set(dup_cols) <= set(raw.columns):
        dup = raw.duplicated(dup_cols, keep=False)
        if dup.any():
            examples = (
                raw.loc[dup, dup_cols]
                .astype(str)
                .drop_duplicates()
                .head(5)
                .agg(" / ".join, axis=1)
                .tolist()
            )
            warnings.append(
                f"Duplicate Location + Material Code rows in snapshot: {int(dup.sum())} row(s)"
                f" ({'; '.join(examples)})"
            )

    if "Commodity Name" in raw.columns:
        commodities = raw["Commodity Name"].astype(str).str.strip().str.upper()
        unmapped = sorted(c for c in commodities.unique() if c not in commodity_map)
        if unmapped:
            preview = ", ".join(unmapped[:8])
            suffix = "..." if len(unmapped) > 8 else ""
            warnings.append(f"Unmapped commodity name(s): {preview}{suffix}")

    if {"Total Net Weight", "Total Cost"} <= set(raw.columns):
        weight = pd.to_numeric(raw["Total Net Weight"], errors="coerce")
        cost = pd.to_numeric(raw["Total Cost"], errors="coerce")
        positive_weight = weight > 0

        zero_cost = positive_weight & (cost.fillna(0.0) == 0.0)
        if zero_cost.any():
            examples = _row_examples(raw, zero_cost)
            warnings.append(
                f"Positive-weight rows with zero total cost: {int(zero_cost.sum())}"
                f" ({'; '.join(examples)})"
            )

        extreme_weight = weight > EXTREME_ROW_WEIGHT_LBS
        if extreme_weight.any():
            examples = _row_examples(raw, extreme_weight)
            warnings.append(
                f"Rows above {EXTREME_ROW_WEIGHT_LBS:,.0f} lbs net weight: {int(extreme_weight.sum())}"
                f" ({'; '.join(examples)})"
            )

        cost_per_lb = cost / weight.where(weight > 0)
        extreme_cost = cost_per_lb > EXTREME_COST_PER_LB
        if extreme_cost.any():
            examples = _row_examples(raw, extreme_cost)
            warnings.append(
                f"Rows above ${EXTREME_COST_PER_LB:,.2f}/lb cost: {int(extreme_cost.sum())}"
                f" ({'; '.join(examples)})"
            )

    return warnings


def _row_examples(raw: pd.DataFrame, mask: pd.Series, limit: int = 5) -> list[str]:
    cols = [c for c in ["Location", "Material Code", "Material Name"] if c in raw.columns]
    if not cols:
        return [f"row {i}" for i in list(raw.index[mask][:limit])]
    examples = []
    for idx, row in raw.loc[mask, cols].head(limit).iterrows():
        label = " / ".join(str(row[c]) for c in cols)
        examples.append(f"row {idx}: {label}")
    return examples


def file_provenance(path: Path, latest_date: pd.Timestamp | None = None) -> dict[str, str | int | None]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    snap_date = _snapshot_date(path)
    return {
        "filename": path.name,
        "path": str(path),
        "sha256": digest,
        "rows": int(len(pd.read_csv(path, usecols=[0]))),
        "snapshot_date": str(snap_date.date()) if snap_date is not None else None,
        "latest_data_date": str(latest_date.date()) if latest_date is not None and not pd.isna(latest_date) else None,
    }
