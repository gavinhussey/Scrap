"""Reconstruct end-of-year-2025 (opening) inventory by working backwards."""

import os
import re
import glob

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
DAILY = os.path.join(HERE, "daily_inputs")
OUT = os.path.join(HERE, "output")

_TIER = re.compile(r"\s*TIER\s*\d+\s*$", re.I)
_BRACKET = re.compile(r"\[[^\]]*\]$")

MARKET_WINDOW_DAYS = 60


def _glob(pattern):
    hits = glob.glob(os.path.join(DAILY, pattern))
    return hits or glob.glob(os.path.join(DATA, pattern))


def _location(path: str) -> str:
    n = os.path.basename(path).lower()
    if "milton" in n:
        return "Milton"
    if "merriville" in n or "merrillville" in n:
        return "Merrillville"
    return "Unknown"


def _metal(commodity_name: object) -> str:
    if pd.isna(commodity_name):
        return "UNKNOWN"
    return _TIER.sub("", str(commodity_name).strip().upper())


def _latest_inventory() -> str:
    files = sorted(_glob("combined inventory*.csv"))
    if not files:
        raise SystemExit("Run data/merge_inventory.py first (no combined inventory file).")
    return files[-1]


def load_inventory() -> pd.DataFrame:
    inv = pd.read_csv(_latest_inventory())
    inv = inv.rename(columns={"Material Code": "code", "Material Name": "name",
                              "Commodity Name": "commodity", "Commodity Type": "ctype"})
    inv["current_wt"] = pd.to_numeric(inv["Total Net Weight"], errors="coerce").fillna(0.0)
    inv["avg_cost"] = pd.to_numeric(inv["Average Cost"], errors="coerce")
    return inv[["Location", "code", "name", "commodity", "ctype", "current_wt", "avg_cost"]]


def _sum_flow(pattern: str, code_col: str, name_col: str) -> pd.DataFrame:
    frames = []
    for p in _glob(pattern):
        if "combined" in os.path.basename(p):
            continue
        d = pd.read_csv(p, thousands=",")
        d["Location"] = _location(p)
        d["wt"] = pd.to_numeric(d["Net Weight"], errors="coerce")
        frames.append(d.rename(columns={code_col: "code", name_col: "name",
                                         "Commodity Name": "commodity",
                                         "Commodity Type": "ctype"})
                      [["Location", "code", "name", "commodity", "ctype", "wt"]])
    flow = pd.concat(frames, ignore_index=True)
    grouped = flow.groupby(["Location", "code"], as_index=False)["wt"].sum()
    meta = flow.dropna(subset=["code"]).drop_duplicates(["Location", "code"])[
        ["Location", "code", "name", "commodity", "ctype"]]
    return grouped, meta


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    inv = load_inventory()
    inb, inb_meta = _sum_flow("2026 ytd *inbound.csv", "Material Code", "Material Name")
    out, out_meta = _sum_flow("2026 ytd *outbound.csv", "Material Code", "Material")

    inb = inb.rename(columns={"wt": "inbound_wt"})
    out = out.rename(columns={"wt": "outbound_wt"})

    meta = (pd.concat([inv[["Location", "code", "name", "commodity", "ctype"]],
                       inb_meta, out_meta], ignore_index=True)
            .dropna(subset=["code"])
            .drop_duplicates(["Location", "code"]))

    df = (meta
          .merge(inv[["Location", "code", "current_wt", "avg_cost"]], on=["Location", "code"], how="left")
          .merge(inb, on=["Location", "code"], how="left")
          .merge(out, on=["Location", "code"], how="left"))
    for c in ["current_wt", "inbound_wt", "outbound_wt"]:
        df[c] = df[c].fillna(0.0)

    df["opening_wt"] = df["current_wt"] - df["inbound_wt"] + df["outbound_wt"]
    df["metal"] = df["commodity"].map(_metal)
    df["flag"] = ""
    df.loc[df["opening_wt"] < -0.5, "flag"] = "NEGATIVE (reclassified or short)"
    df["opening_value_est"] = (df["opening_wt"].clip(lower=0) * df["avg_cost"]).round(2)

    by_grade = df.sort_values(["Location", "ctype", "commodity", "name"]).reset_index(drop=True)
    by_grade = by_grade[["Location", "code", "name", "commodity", "metal", "ctype",
                         "current_wt", "inbound_wt", "outbound_wt", "opening_wt",
                         "avg_cost", "opening_value_est", "flag"]]

    by_metal = (df.groupby(["Location", "metal"], as_index=False)[
        ["current_wt", "inbound_wt", "outbound_wt", "opening_wt"]].sum()
        .sort_values(["Location", "metal"]).reset_index(drop=True))
    by_metal["flag"] = ""
    by_metal.loc[by_metal["opening_wt"] < -0.5, "flag"] = "NEGATIVE at metal level"

    return by_grade, by_metal


def _flow_lines(pattern: str, value_col: str) -> pd.DataFrame:
    """Line-level (code, $ value, lbs, date) for a flow, for $/lb pricing."""
    frames = []
    for p in _glob(pattern):
        if "combined" in os.path.basename(p):
            continue
        d = pd.read_csv(p, thousands=",")
        date = d["Date In"] if "Date In" in d.columns else d["Effective Date"]
        frames.append(pd.DataFrame({
            "code": d["Material Code"],
            "value": pd.to_numeric(d[value_col], errors="coerce"),
            "wt": pd.to_numeric(d["Net Weight"], errors="coerce"),
            "date": pd.to_datetime(date.astype("string").str.replace(_BRACKET, "", regex=True),
                                   utc=True, errors="coerce", format="mixed").dt.tz_localize(None),
        }))
    f = pd.concat(frames, ignore_index=True).dropna(subset=["code", "value", "wt"])
    return f[f["wt"] > 0]


def _vwap(df: pd.DataFrame) -> pd.Series:
    g = df.groupby("code")
    return g["value"].sum() / g["wt"].sum()


def market_prices(avg_cost_by_code: pd.Series) -> pd.DataFrame:
    """Best market $/lb per grade, with provenance. Ladder, most-trusted first:
    recent sale -> YTD sale -> YTD purchase -> current average cost."""
    ob = _flow_lines("2026 ytd *outbound.csv", "Expected Value")
    inb = _flow_lines("2026 ytd *inbound.csv", "Cost")
    latest = max(ob["date"].max(), inb["date"].max())
    cutoff = latest - pd.Timedelta(days=MARKET_WINDOW_DAYS)

    ladder = [("sale (recent)", _vwap(ob[ob["date"] >= cutoff])),
              ("sale (YTD)",    _vwap(ob)),
              ("purchase (YTD)", _vwap(inb)),
              ("avg cost",      avg_cost_by_code)]

    codes = set().union(*[s.index for _, s in ladder])
    rows = []
    for c in codes:
        for source, series in ladder:
            if c in series.index and pd.notna(series[c]) and series[c] > 0:
                rows.append({"code": c, "market_price": round(float(series[c]), 4),
                             "price_source": source})
                break
    return pd.DataFrame(rows)


def portfolio_weights(by_grade: pd.DataFrame, prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    df = by_grade.copy()
    df["held_wt"] = df["opening_wt"].clip(lower=0)

    grade = (df.groupby(["code", "name", "commodity", "metal", "ctype"], as_index=False)
             ["held_wt"].sum())
    grade = grade[grade["held_wt"] > 0].copy()
    grade = grade.merge(prices, on="code", how="left")

    grade["price_source"] = grade["price_source"].fillna("unpriced")
    grade["market_price"] = grade["market_price"].fillna(0.0)
    grade["value_usd"] = (grade["held_wt"] * grade["market_price"]).round(2)

    tot_v = grade["value_usd"].sum()
    tot_w = grade["held_wt"].sum()
    grade["weight_by_value_pct"] = (100 * grade["value_usd"] / tot_v).round(2)
    grade["weight_by_qty_pct"] = (100 * grade["held_wt"] / tot_w).round(2)
    grade = grade.sort_values("weight_by_value_pct", ascending=False).reset_index(drop=True)
    grade = grade.rename(columns={"held_wt": "qty_lbs", "market_price": "unit_value"})
    grade = grade[["code", "name", "commodity", "metal", "ctype", "qty_lbs",
                   "unit_value", "price_source", "value_usd",
                   "weight_by_value_pct", "weight_by_qty_pct"]]

    metal = (grade.groupby("metal", as_index=False)[["qty_lbs", "value_usd"]].sum())
    metal["weight_by_value_pct"] = (100 * metal["value_usd"] / tot_v).round(2)
    metal["weight_by_qty_pct"] = (100 * metal["qty_lbs"] / tot_w).round(2)
    metal = metal.sort_values("weight_by_value_pct", ascending=False).reset_index(drop=True)
    return grade, metal


def _fmt(w):
    return f"{w:>14,.0f}"


def write(by_grade: pd.DataFrame, by_metal: pd.DataFrame,
          pf_grade: pd.DataFrame, pf_metal: pd.DataFrame) -> None:
    os.makedirs(OUT, exist_ok=True)
    g_path = os.path.join(OUT, "opening_inventory_eoy2025_by_grade.csv")
    m_path = os.path.join(OUT, "opening_inventory_eoy2025_by_metal.csv")
    pg_path = os.path.join(OUT, "opening_portfolio_weights_by_grade.csv")
    pm_path = os.path.join(OUT, "opening_portfolio_weights_by_metal.csv")
    r_path = os.path.join(OUT, "opening_inventory_reconciliation.txt")
    by_grade.round(2).to_csv(g_path, index=False)
    by_metal.round(2).to_csv(m_path, index=False)
    pf_grade.round(2).to_csv(pg_path, index=False)
    pf_metal.round(2).to_csv(pm_path, index=False)

    lines, A = [], lambda s: lines.append(s)
    A("END-OF-YEAR 2025 (OPENING) INVENTORY — BACKWARD RECONSTRUCTION")
    A("=" * 72)
    A("  opening = current physical count - inbound YTD + outbound YTD   (lbs)")
    A("")
    tot = by_grade[["current_wt", "inbound_wt", "outbound_wt", "opening_wt"]].sum()
    A("TOTALS (all yards, all grades)")
    A(f"  current on hand : {_fmt(tot['current_wt'])}")
    A(f"  - inbound YTD   : {_fmt(tot['inbound_wt'])}")
    A(f"  + outbound YTD  : {_fmt(tot['outbound_wt'])}")
    A(f"  = opening 2025  : {_fmt(tot['opening_wt'])}   ({tot['opening_wt']/2000:,.0f} tons)")
    A("")
    A("BY YARD")
    for loc, sub in by_grade.groupby("Location"):
        t = sub[["current_wt", "inbound_wt", "outbound_wt", "opening_wt"]].sum()
        A(f"  {loc:<12} current {_fmt(t['current_wt'])}  - in {_fmt(t['inbound_wt'])}"
          f"  + out {_fmt(t['outbound_wt'])}  = opening {_fmt(t['opening_wt'])}")
    A("")
    A("OPENING BY METAL (grade negatives netted within metal)")
    for _, r in by_metal.iterrows():
        A(f"  {r['Location']:<12} {r['metal']:<22} {_fmt(r['opening_wt'])}   {r['flag']}")
    A("")
    neg = by_grade[by_grade["flag"] != ""]
    A(f"NEGATIVE GRADES (sold/processed more than bought + ending): {len(neg)}")
    A("  -> these indicate material reclassified between buy and sell; they net")
    A("     out at the metal level above. Largest negatives:")
    for _, r in neg.sort_values("opening_wt").head(12).iterrows():
        A(f"     {r['Location']:<12} {r['name'][:34]:<34} {r['metal']:<16} {_fmt(r['opening_wt'])}")
    A("")
    est = pf_metal["value_usd"].sum()
    A(f"ESTIMATED OPENING VALUE (held grades @ MARKET spot price): ${est:,.0f}")
    A("  Market $/lb per grade = volume-weighted transacted price, ladder:")
    A(f"  recent sale (<= {MARKET_WINDOW_DAYS}d) -> YTD sale -> YTD purchase -> avg cost.")
    src = (pf_grade.groupby("price_source")["value_usd"].sum() / est * 100).round(1)
    A("  share of value by price source: "
      + ", ".join(f"{s}={src[s]:.0f}%" for s in src.sort_values(ascending=False).index))
    A("")
    A("PORTFOLIO WEIGHTS BY METAL  (share of the EOY-2025 book, MARKET-priced)")
    A(f"  {'metal':<18}{'value $':>14}{'  wt%(value)':>12}{'  lbs':>14}{'  wt%(lbs)':>11}")
    for _, r in pf_metal.iterrows():
        A(f"  {r['metal']:<18}{r['value_usd']:>14,.0f}{r['weight_by_value_pct']:>11.1f}%"
          f"{r['qty_lbs']:>14,.0f}{r['weight_by_qty_pct']:>10.1f}%")
    A("  -> by VALUE is the risk-relevant split; note how steel dominates by weight")
    A("     but not by value, while copper/brass punch far above their tonnage.")
    A("")
    A("TOP 12 ITEMS BY VALUE WEIGHT")
    for _, r in pf_grade.head(12).iterrows():
        A(f"  {r['name'][:30]:<30} {r['metal']:<14} {r['weight_by_value_pct']:>5.1f}%"
          f"  ({r['qty_lbs']:>10,.0f} lbs @ ${r['unit_value']:.2f} {r['price_source']})")
    A("")
    A("ASSUMPTIONS / CAVEATS")
    A("  - Join key: Material Code + yard (Milton=Benton Metals, Merrillville=")
    A("    Merrillville Metal Recycling). Flow yard taken from source filename.")
    A("  - Weight = inventory 'Total Net Weight' (WIP + finished goods), all lbs.")
    A("  - Conservation assumed: no scrap/shrinkage/moisture adjustment applied.")
    A("  - Inter-yard transfers only cancel if booked as one yard's outbound and")
    A("    the other's inbound; otherwise per-yard openings carry that error.")
    A("  - Snapshot date of the count must align with the YTD flow coverage.")
    with open(r_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote:\n  {g_path}\n  {m_path}\n  {pg_path}\n  {pm_path}\n  {r_path}")


def main() -> None:
    by_grade, by_metal = build()
    avg_cost = (by_grade.dropna(subset=["avg_cost"])
                .groupby("code")["avg_cost"].mean())
    prices = market_prices(avg_cost)
    pf_grade, pf_metal = portfolio_weights(by_grade, prices)
    write(by_grade, by_metal, pf_grade, pf_metal)


if __name__ == "__main__":
    main()
