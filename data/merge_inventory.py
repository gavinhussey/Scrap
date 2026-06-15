"""Merge the current physical inventory reports from both yards into one file."""

import os
import re
import glob

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
INPUTS = os.path.join(PROJECT, "daily_inputs")

COLS = ["Location", "Material Code", "Material Name", "Commodity Name", "Commodity Type",
        "Total Wip Weight", "Total FG Weight", "Total Net Weight",
        "Total Wip Cost", "Total FG Cost", "Total Cost", "Average Cost"]


def _location(filename: str) -> str:
    name = filename.lower()
    if "milton" in name:
        return "Milton"
    if "merriville" in name or "merrillville" in name:
        return "Merrillville"
    return "Unknown"


def _snapshot_date(filenames) -> str:
    for f in filenames:
        m = re.search(r"(\d{8})", f)
        if m:
            return m.group(1)
    return "latest"


def main() -> None:
    os.makedirs(INPUTS, exist_ok=True)
    paths = [p for p in glob.glob(os.path.join(INPUTS, "*Inventory Report*.csv"))
             if "combined" not in os.path.basename(p).lower()]
    if not paths:
        raise SystemExit("No '*Inventory Report*.csv' files found in daily_inputs/.")

    frames = []
    for path in sorted(paths):
        df = pd.read_csv(path, dtype=str)
        df.insert(0, "Location", _location(os.path.basename(path)))
        frames.append(df)
        print(f"  read {len(df):>4} rows  {_location(os.path.basename(path)):<12} <- {os.path.basename(path)}")

    combined = pd.concat(frames, ignore_index=True)
    ordered = [c for c in COLS if c in combined.columns]
    extra = [c for c in combined.columns if c not in COLS]
    combined = combined[ordered + extra]

    combined = combined.sort_values(
        ["Commodity Type", "Commodity Name", "Material Name", "Location"],
        kind="stable", na_position="last").reset_index(drop=True)

    out_path = os.path.join(INPUTS, f"combined inventory {_snapshot_date([os.path.basename(p) for p in paths])}.csv")
    combined.to_csv(out_path, index=False)

    net = pd.to_numeric(combined["Total Net Weight"], errors="coerce")
    by_loc = net.groupby(combined["Location"]).sum()
    print(f"\n  wrote {len(combined)} rows -> {out_path}")
    print("  net weight on hand (lbs):")
    for loc, w in by_loc.items():
        print(f"    {loc:<12} {w:>14,.0f}")
    print(f"    {'TOTAL':<12} {net.sum():>14,.0f}")


if __name__ == "__main__":
    main()
