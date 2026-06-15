"""Calibrate a per-grade price basis from real transactions."""

import os
import re
import glob

import numpy as np
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")

_BRACKET = re.compile(r"\[[^\]]*\]$")
_TIER = re.compile(r"\s*TIER\s*\d+\s*$", re.I)
_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%")

BENCH = {"COPPER": ("HG=F", 1.0), "BRASS": ("HG=F", 1.0),
         "STEEL": ("HRC=F", 2000.0), "ALUMINUM": ("ALI=F", 2204.62)}
BASIS_HIGH_FLAG = 1.10     # scrap rarely exceeds benchmark; flag for review above this
MIN_TRADE_LBS = 200        # below this total weight, mark basis low-confidence


def _metal(commodity: object) -> str:
    return _TIER.sub("", str(commodity).strip().upper()) if pd.notna(commodity) else "UNKNOWN"


def _flow(pattern: str, value_col: str) -> pd.DataFrame:
    rows = []
    for p in glob.glob(os.path.join(DATA, pattern)):
        if "combined" in os.path.basename(p):
            continue
        d = pd.read_csv(p, thousands=",")
        name = d["Material Name"] if "Material Name" in d.columns else d["Material"]
        date = d["Date In"] if "Date In" in d.columns else d["Effective Date"]
        rows.append(pd.DataFrame({
            "code": d["Material Code"], "name": name,
            "commodity": d["Commodity Name"], "ctype": d["Commodity Type"],
            "value": pd.to_numeric(d[value_col], errors="coerce"),
            "wt": pd.to_numeric(d["Net Weight"], errors="coerce"),
            "date": pd.to_datetime(date.astype("string").str.replace(_BRACKET, "", regex=True),
                                   utc=True, errors="coerce", format="mixed").dt.tz_localize(None),
        }))
    f = pd.concat(rows, ignore_index=True).dropna(subset=["code", "value", "wt", "date"])
    f = f[f["wt"] > 0].copy()
    f["metal"] = f["commodity"].map(_metal)
    return f


def benchmark_daily(start, end) -> dict:
    """date-indexed USD/lb series per benchmarked metal (ffilled over non-trading days)."""
    idx = pd.date_range(start.normalize(), end.normalize(), freq="D")
    raw = {}
    for ticker in {t for t, _ in BENCH.values()}:
        d = yf.download(ticker, start=start - pd.Timedelta(days=7), end=end + pd.Timedelta(days=2),
                        progress=False, auto_adjust=True)
        raw[ticker] = d["Close"].squeeze().dropna()
    out = {}
    for metal, (ticker, div) in BENCH.items():
        s = raw[ticker] / div
        out[metal] = s.reindex(idx.union(s.index)).sort_index().ffill().reindex(idx)
    return out


def benchmark_now() -> dict:
    out = {}
    for metal, (ticker, div) in BENCH.items():
        d = yf.download(ticker, period="10d", progress=False, auto_adjust=True)
        out[metal] = float(d["Close"].squeeze().dropna().iloc[-1]) / div
    return out


def _attach_bench(df: pd.DataFrame, bench: dict) -> pd.DataFrame:
    df = df.copy()
    df["bench"] = np.nan
    for metal, s in bench.items():
        m = df["metal"] == metal
        df.loc[m, "bench"] = df.loc[m, "date"].dt.normalize().map(s).to_numpy()
    return df


def _calib(df: pd.DataFrame) -> pd.DataFrame:
    """per-code volume-weighted basis (value / benchmark-value) and $/lb."""
    df = df.assign(benchval=df["bench"] * df["wt"])
    g = df.groupby("code")
    out = g.agg(value=("value", "sum"), wt=("wt", "sum"),
                benchval=("benchval", "sum"), n=("value", "size"))
    out["basis"] = out["value"] / out["benchval"]      # NaN where no benchmark
    out["price"] = out["value"] / out["wt"]
    return out


def calibrate() -> tuple[pd.DataFrame, dict, dict]:
    sale = _flow("2026 ytd *outbound.csv", "Expected Value")
    buy = _flow("2026 ytd *inbound.csv", "Cost")
    lo = min(sale["date"].min(), buy["date"].min())
    hi = max(sale["date"].max(), buy["date"].max())
    bench = benchmark_daily(lo, hi)
    now = benchmark_now()

    s = _calib(_attach_bench(sale, bench))
    b = _calib(_attach_bench(buy, bench))

    # metadata per code: inventory first, then flows
    inv = pd.read_csv(glob.glob(os.path.join(DATA, "combined inventory*.csv"))[-1])
    inv_meta = inv.rename(columns={"Material Code": "code", "Material Name": "name",
                                   "Commodity Name": "commodity", "Commodity Type": "ctype"})
    meta = (pd.concat([inv_meta[["code", "name", "commodity", "ctype"]],
                       sale[["code", "name", "commodity", "ctype"]],
                       buy[["code", "name", "commodity", "ctype"]]], ignore_index=True)
            .dropna(subset=["code"]).drop_duplicates("code").set_index("code"))
    meta["metal"] = meta["commodity"].map(_metal)

    # per-metal buy->sale margin uplift (from grades that have both, weighted by sale lbs)
    both = s[["basis", "wt"]].join(b[["basis"]], rsuffix="_buy", how="inner").dropna()
    both = both.join(meta["metal"])
    both = both[both["metal"].isin(BENCH)]               # only benchmarked metals have a basis
    uplift = {}
    for metal, gdf in both.groupby("metal"):
        sb = np.average(gdf["basis"], weights=gdf["wt"])
        bb = np.average(gdf["basis_buy"], weights=gdf["wt"])
        if bb > 0:
            uplift[metal] = sb / bb
    denom = np.average(both["basis_buy"], weights=both["wt"]) if len(both) else 0.0
    overall = (np.average(both["basis"], weights=both["wt"]) / denom) if denom > 0 else 1.0

    # commodity- and metal-average sale basis (weighted) for thin grades
    sj = s.join(meta[["commodity", "metal"]])
    comm_avg = sj.dropna(subset=["basis"]).groupby("commodity").apply(
        lambda g: np.average(g["basis"], weights=g["wt"]), include_groups=False)
    metal_avg = sj.dropna(subset=["basis"]).groupby("metal").apply(
        lambda g: np.average(g["basis"], weights=g["wt"]), include_groups=False)
    return _assemble(meta, s, b, uplift, overall, comm_avg, metal_avg, now), now, uplift


def _assemble(meta, s, b, uplift, overall, comm_avg, metal_avg, now) -> pd.DataFrame:
    rows = []
    for code, m in meta.iterrows():
        metal = m["metal"]
        benched = metal in BENCH
        basis = np.nan
        ref_price = np.nan
        source = "none"
        n = lbs = 0

        if benched:
            if code in s.index and s.at[code, "benchval"] > 0:
                basis, source = s.at[code, "basis"], "sale"
                n, lbs = int(s.at[code, "n"]), s.at[code, "wt"]
            elif code in b.index and b.at[code, "benchval"] > 0:
                up = uplift.get(metal, overall)
                basis, source = b.at[code, "basis"] * up, "buy+margin"
                n, lbs = int(b.at[code, "n"]), b.at[code, "wt"]
            elif (pm := _PCT.search(str(m["name"]))):
                basis, source = float(pm.group(1)) / 100.0, "name-%"
            elif m["commodity"] in comm_avg.index:
                basis, source = float(comm_avg[m["commodity"]]), "tier-avg"
            elif metal in metal_avg.index:
                basis, source = float(metal_avg[metal]), "metal-avg"
            price_now = basis * now[metal] if pd.notna(basis) else np.nan
        else:                                            # no benchmark -> absolute $/lb
            if code in s.index:
                ref_price, source, n, lbs = s.at[code, "price"], "absolute (sale)", int(s.at[code, "n"]), s.at[code, "wt"]
            elif code in b.index:
                ref_price, source, n, lbs = b.at[code, "price"], "absolute (buy)", int(b.at[code, "n"]), b.at[code, "wt"]
            price_now = ref_price

        flag = ""
        if pd.notna(basis) and basis > BASIS_HIGH_FLAG:
            flag = "basis>benchmark (review)"
        elif source in ("tier-avg", "metal-avg", "name-%", "none"):
            flag = "no own trades (modeled)"
        elif lbs and lbs < MIN_TRADE_LBS:
            flag = "thin volume"

        rows.append({
            "code": code, "name": m["name"], "commodity": m["commodity"], "metal": metal,
            "ctype": m["ctype"], "benchmark": BENCH[metal][0] if benched else "none",
            "basis": round(basis, 4) if pd.notna(basis) else "",
            "ref_price_per_lb": round(ref_price, 4) if pd.notna(ref_price) else "",
            "price_now_per_lb": round(price_now, 4) if pd.notna(price_now) else "",
            "basis_source": source, "n_trades": n, "trade_lbs": round(lbs, 0), "flag": flag,
        })
    df = pd.DataFrame(rows).sort_values(["metal", "commodity", "code"]).reset_index(drop=True)
    return df


def main() -> None:
    table, now, uplift = calibrate()
    path = os.path.join(OUT, "grade_basis_calibration.csv")
    table.to_csv(path, index=False)

    L = []
    L.append("GRADE BASIS CALIBRATION  (grade $/lb = basis x benchmark $/lb)")
    L.append("=" * 70)
    L.append("Benchmarks now (USD/lb): " +
             ", ".join(f"{m}={now[m]:.4f}" for m in ["COPPER", "STEEL", "ALUMINUM"]))
    L.append("Buy->sale margin uplift used for buy-only grades: " +
             ", ".join(f"{k}={v:.2f}x" for k, v in uplift.items()))
    L.append("")
    L.append(f"grades calibrated: {len(table)}")
    vc = table["basis_source"].value_counts()
    L.append("  by source: " + ", ".join(f"{k}={v}" for k, v in vc.items()))
    flagged = table[table["flag"] != ""]
    L.append(f"  flagged for review: {len(flagged)}")
    L.append("")
    L.append("SAMPLE (benchmarked grades, basis vs metal):")
    show = table[(table["basis"] != "") & (table["basis_source"] == "sale")].copy()
    show["basis_f"] = pd.to_numeric(show["basis"])
    for _, r in show.sort_values("trade_lbs", ascending=False).head(14).iterrows():
        L.append(f"  {str(r['name'])[:30]:<30} {r['metal']:<10} basis={r['basis_f']:.3f}"
                 f"  -> ${r['price_now_per_lb']}/lb  ({int(r['n_trades'])} trades)")
    out = "\n".join(L)
    with open(os.path.join(OUT, "grade_basis_calibration.txt"), "w") as f:
        f.write(out + "\n")
    print(out)
    print(f"\nWrote: {path}")


if __name__ == "__main__":
    main()
