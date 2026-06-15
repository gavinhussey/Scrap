"""Merge inbound and outbound orders from both locations (Milton/Benton & Merrillville)."""

import re
import pandas as pd

DATA = "daily_inputs"

INBOUND = [
    (f"{DATA}/2026 ytd milton inbound.csv", "Milton"),
    (f"{DATA}/2026 ytd merriville inbound.csv", "Merrillville"),
]
OUTBOUND = [
    (f"{DATA}/2026 ytd milton outbound.csv", "Milton"),
    (f"{DATA}/2026 ytd merriville outbound.csv", "Merrillville"),
]

_BRACKET = re.compile(r"\[[^\]]*\]$")


def parse_dt(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.replace(_BRACKET, "", regex=True)
    return pd.to_datetime(cleaned, utc=True, errors="coerce", format="mixed")


def merge(files, date_col, out_path, row_id_col):
    frames = []
    for path, location in files:
        df = pd.read_csv(path, dtype=str)
        df.insert(0, "Location", location)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["_sort"] = parse_dt(combined[date_col])

    n_bad = int(combined["_sort"].isna().sum())
    if n_bad:
        print(f"  WARNING: {n_bad} row(s) had an unparseable {date_col!r}; placed last")

    combined = combined.sort_values(
        "_sort", ascending=True, na_position="last", kind="stable"
    ).drop(columns="_sort")

    if row_id_col in combined.columns:
        combined = combined.drop(columns=row_id_col)
    combined.insert(0, row_id_col, range(1, len(combined) + 1))

    combined.to_csv(out_path, index=False)
    print(f"  wrote {len(combined)} rows -> {out_path}")


print("Inbound:")
merge(INBOUND, "Effective Date", f"{DATA}/2026 ytd combined inbound.csv", "Inbound Row Id")
print("Outbound:")
merge(OUTBOUND, "Date In", f"{DATA}/2026 ytd combined outbound.csv", "Outbound Row Id")
