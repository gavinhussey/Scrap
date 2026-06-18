"""Calculate current inventory from EOY-2025 closing balances + 2026 YTD flows."""

import glob
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")

INBOUND = os.path.join(DATA, "2026 ytd combined inbound.csv")
OUTBOUND = os.path.join(DATA, "2026 ytd combined outbound.csv")


def _latest(pattern: str) -> str:
    hits = glob.glob(os.path.join(DATA, pattern))
    if not hits:
        raise SystemExit(f"No file matching {pattern!r} in {DATA}/")
    return max(hits, key=os.path.getmtime)


def load_eoy_2025() -> pd.DataFrame:
    """EOY-2025 closing weight per yard/material code, from the GreenSpark closing-balance exports."""
    frames = []
    for path in glob.glob(os.path.join(DATA, "*2025-12-31*.csv")):
        d = pd.read_csv(path)
        d["Location"] = d["Yard Name"].str.title()  # "MILTON" -> "Milton", matches flow/inventory file casing
        frames.append(d[["Location", "Material Code", "Commodity", "Commodity Type", "Closing Weight"]])
    if not frames:
        raise SystemExit(f"No '*2025-12-31*.csv' EOY closing-balance files found in {DATA}/")
    eoy = pd.concat(frames, ignore_index=True)
    eoy = eoy.rename(columns={"Material Code": "code", "Closing Weight": "eoy2025_wt"})
    eoy["eoy2025_wt"] = pd.to_numeric(eoy["eoy2025_wt"], errors="coerce").fillna(0.0)
    return eoy.groupby(["Location", "code"], as_index=False).agg(
        eoy2025_wt=("eoy2025_wt", "sum"),
        commodity=("Commodity", "first"),
        ctype=("Commodity Type", "first"),
    )


def _load_flow(path: str) -> pd.DataFrame:
    d = pd.read_csv(path, thousands=",")
    d["wt"] = pd.to_numeric(d["Net Weight"], errors="coerce").fillna(0.0)
    g = d.groupby(["Location", "Material Code"], as_index=False)["wt"].sum()
    return g.rename(columns={"Material Code": "code"})


def _commodity_tags(path: str) -> pd.DataFrame:
    """Location/code -> commodity/type from a flow file, for codes the EOY file has no balance for."""
    d = pd.read_csv(path, thousands=",", usecols=["Location", "Material Code", "Commodity Name", "Commodity Type"])
    d = d.rename(columns={"Material Code": "code", "Commodity Name": "commodity", "Commodity Type": "ctype"})
    return d.dropna(subset=["code"]).drop_duplicates(["Location", "code"])


def _fill_missing_commodity(df: pd.DataFrame, *extra_paths: str) -> pd.DataFrame:
    """EOY-2025 file omits codes with no 2025 closing balance, so they join with no commodity tag.
    Backfill those from the flow files (and any extra source, e.g. the actual inventory file),
    which carry the tag on every row."""
    if not df["commodity"].isna().any():
        return df
    fallback = (pd.concat([_commodity_tags(p) for p in (INBOUND, OUTBOUND, *extra_paths)], ignore_index=True)
                .drop_duplicates(["Location", "code"], keep="first"))
    df = df.merge(fallback, on=["Location", "code"], how="left", suffixes=("", "_fb"))
    df["commodity"] = df["commodity"].fillna(df["commodity_fb"])
    df["ctype"] = df["ctype"].fillna(df["ctype_fb"])
    return df.drop(columns=["commodity_fb", "ctype_fb"])


def load_actual_inventory(path: str | None = None) -> pd.DataFrame:
    """Latest physical inventory snapshot (data/combined inventory *.csv) unless a path is given."""
    path = path or _latest("combined inventory*.csv")
    inv = pd.read_csv(path)
    inv = inv.rename(columns={"Material Code": "code", "Total Net Weight": "actual_wt"})
    inv["actual_wt"] = pd.to_numeric(inv["actual_wt"], errors="coerce").fillna(0.0)
    return inv.groupby(["Location", "code"], as_index=False)["actual_wt"].sum()


def calc_inventory() -> pd.DataFrame:
    """EOY-2025 closing weight rolled forward by 2026 YTD inbound/outbound, per yard/material code."""
    eoy = load_eoy_2025()
    inb = _load_flow(INBOUND).rename(columns={"wt": "inbound_wt"})
    out = _load_flow(OUTBOUND).rename(columns={"wt": "outbound_wt"})

    # outer merge: keep codes with no EOY balance (new in 2026) and codes with no 2026 flow
    df = eoy.merge(inb, on=["Location", "code"], how="outer").merge(out, on=["Location", "code"], how="outer")
    for c in ["eoy2025_wt", "inbound_wt", "outbound_wt"]:
        df[c] = df[c].fillna(0.0)
    df = _fill_missing_commodity(df)
    df["calc_wt"] = df["eoy2025_wt"] + df["inbound_wt"] - df["outbound_wt"]
    return df


def reconcile(actual_path: str | None = None) -> pd.DataFrame:
    """calc_inventory() joined against the actual inventory, with a diff column."""
    actual_path = actual_path or _latest("combined inventory*.csv")
    calc = calc_inventory()
    actual = load_actual_inventory(actual_path)
    df = calc.merge(actual, on=["Location", "code"], how="outer")
    for c in ["eoy2025_wt", "inbound_wt", "outbound_wt", "calc_wt", "actual_wt"]:
        df[c] = df[c].fillna(0.0)
    # a few (yard, code) pairs only ever show up in the actual-inventory snapshot, never in
    # EOY/inbound/outbound at that yard, so calc_inventory()'s fallback can't tag them either
    df = _fill_missing_commodity(df, actual_path)
    df["diff"] = df["calc_wt"] - df["actual_wt"]
    return df.sort_values("diff", key=abs, ascending=False).reset_index(drop=True)  # biggest mismatch first, either sign


def _with_total_row(df: pd.DataFrame, label_col: str, label: str = "TOTAL") -> pd.DataFrame:
    """Append a row summing every numeric column, for printing/CSVs that should show their own total."""
    total = {c: (label if c == label_col else "") for c in df.columns}
    for c in df.select_dtypes("number").columns:
        total[c] = df[c].sum()
    return pd.concat([df, pd.DataFrame([total])], ignore_index=True)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    df = reconcile()
    df.round(2).to_csv(os.path.join(OUT, "inventory_reconciliation.csv"), index=False)

    # collapse yard-level detail per code: regrade noise within a code mostly cancels out here
    by_code = df.groupby("code", as_index=False).agg(
        commodity=("commodity", "first"), ctype=("ctype", "first"),
        eoy2025_wt=("eoy2025_wt", "sum"), inbound_wt=("inbound_wt", "sum"),
        outbound_wt=("outbound_wt", "sum"), calc_wt=("calc_wt", "sum"),
        actual_wt=("actual_wt", "sum"), diff=("diff", "sum"))
    by_code = by_code.sort_values("diff", key=abs, ascending=False)
    _with_total_row(by_code, "code").round(2).to_csv(os.path.join(OUT, "inventory_reconciliation_by_code.csv"), index=False)

    # collapse codes into their commodity (material type): regrade noise between codes of the
    # same commodity cancels out here too, leaving the more genuine cross-commodity gaps
    by_commodity = df.groupby("commodity", as_index=False).agg(
        ctype=("ctype", "first"),
        eoy2025_wt=("eoy2025_wt", "sum"), inbound_wt=("inbound_wt", "sum"),
        outbound_wt=("outbound_wt", "sum"), calc_wt=("calc_wt", "sum"),
        actual_wt=("actual_wt", "sum"), diff=("diff", "sum"))
    by_commodity = by_commodity.sort_values("diff", key=abs, ascending=False)
    _with_total_row(by_commodity, "commodity").round(2).to_csv(
        os.path.join(OUT, "inventory_reconciliation_by_commodity.csv"), index=False)

    tot = df[["eoy2025_wt", "inbound_wt", "outbound_wt", "calc_wt", "actual_wt", "diff"]].sum()
    print("TOTALS (all yards, lbs)")
    print(f"  EOY-2025 closing : {tot['eoy2025_wt']:>14,.0f}")
    print(f"  + inbound YTD    : {tot['inbound_wt']:>14,.0f}")
    print(f"  - outbound YTD   : {tot['outbound_wt']:>14,.0f}")
    print(f"  = calculated     : {tot['calc_wt']:>14,.0f}")
    print(f"  actual           : {tot['actual_wt']:>14,.0f}")
    pct = tot["diff"] / tot["actual_wt"] * 100 if tot["actual_wt"] else float("nan")
    print(f"  diff (calc-actual): {tot['diff']:>14,.0f}  ({pct:.1f}% of actual)")
    print()
    print("BY YARD")
    for loc, sub in df.groupby("Location"):
        t = sub[["eoy2025_wt", "inbound_wt", "outbound_wt", "calc_wt", "actual_wt", "diff"]].sum()
        print(f"  {loc:<13} eoy {t['eoy2025_wt']:>13,.0f}  +in {t['inbound_wt']:>13,.0f}"
              f"  -out {t['outbound_wt']:>13,.0f}  = calc {t['calc_wt']:>13,.0f}"
              f"  actual {t['actual_wt']:>13,.0f}  diff {t['diff']:>13,.0f}")
    print(f"  {'TOTAL':<13} eoy {tot['eoy2025_wt']:>13,.0f}  +in {tot['inbound_wt']:>13,.0f}"
          f"  -out {tot['outbound_wt']:>13,.0f}  = calc {tot['calc_wt']:>13,.0f}"
          f"  actual {tot['actual_wt']:>13,.0f}  diff {tot['diff']:>13,.0f}")
    print()
    print("TOP 20 LARGEST DISCREPANCIES BY MATERIAL CODE (combined across yards)")
    cols = ["code", "commodity", "ctype", "eoy2025_wt", "inbound_wt", "outbound_wt", "calc_wt", "actual_wt", "diff"]
    top20_with_total = pd.concat([by_code[cols].head(20),
                                   _with_total_row(by_code[cols], "code", "TOTAL (all codes)").tail(1)])
    print(top20_with_total.to_string(index=False))
    print()
    print("BY COMMODITY (material type, combined across codes and yards)")
    com_cols = ["commodity", "ctype", "eoy2025_wt", "inbound_wt", "outbound_wt", "calc_wt", "actual_wt", "diff"]
    print(_with_total_row(by_commodity[com_cols], "commodity").to_string(index=False))
    print(f"\nWrote:\n  {os.path.join(OUT, 'inventory_reconciliation.csv')}"
          f"\n  {os.path.join(OUT, 'inventory_reconciliation_by_code.csv')}"
          f"\n  {os.path.join(OUT, 'inventory_reconciliation_by_commodity.csv')}")


if __name__ == "__main__":
    main()
