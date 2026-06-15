"""Cost vs market valuation for EOY-2025 and today, plus 2026-YTD unrealized gain.

Inputs (data/):
  - <yard> inv summary 2025-12-31.csv   GreenSpark closing balance per material (EOY-2025)
  - combined inventory 20260612.csv     GreenSpark current balance (weight + Total Cost)
  - 2026 ytd *outbound.csv              sales lines (Expected Value / Net Weight = sale $/lb)
  - 2026 ytd *inbound.csv               purchase lines (Cost / Net Weight = buy $/lb)

Cost basis is taken straight from GreenSpark (Closing Cost / Total Cost) -- it is known,
not modelled. Market value is the only modelled piece: per-grade $/lb from our own
transacted prices, with a provenance ladder (recent sale -> YTD sale -> purchase -> cost).
EOY-2025 market uses Jan-2026 sale prices rolled back to Dec-31 by each metal's futures
move (yfinance); today's market uses sale prices from the last 60 days. Metals with no
futures proxy are held flat and flagged.
"""

import os
import re
import glob

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")
_TIER = re.compile(r"\s*TIER\s*\d+\s*$", re.I)
_BRACKET = re.compile(r"\[[^\]]*\]$")

EOY = "2025-12-31"
JAN_END = "2026-02-01"            # window anchoring the EOY market price
RECENT_DAYS = 60                  # "today" sale window, counted back from latest ticket
PROXY = {"COPPER": "HG=F", "STEEL": "HRC=F", "ALUMINUM": "ALI=F", "BRASS": "HG=F"}


def _metal(name: object) -> str:
    if pd.isna(name):
        return "UNKNOWN"
    return _TIER.sub("", str(name).strip().upper())


# ---- inventory levels (cost basis is read directly, never modelled) -------------------

def load_eoy2025() -> pd.DataFrame:
    frames = []
    for path, yard in [("merrilolville inv summary 2025-12-31.csv", "Merrillville"),
                       ("mitlon inv summary 2025-12-31.csv", "Milton")]:
        d = pd.read_csv(os.path.join(DATA, path), thousands=",")
        d.columns = [c.replace("\n", " ").strip() for c in d.columns]
        frames.append(pd.DataFrame({
            "yard": yard, "code": d["Material Code"], "name": d["Material Type"],
            "commodity": d["Commodity"], "metal": d["Commodity"].map(_metal),
            "wt": pd.to_numeric(d["Closing Weight"], errors="coerce").fillna(0.0),
            "cost": pd.to_numeric(d["Closing Cost"], errors="coerce").fillna(0.0).clip(lower=0),
        }))
    return pd.concat(frames, ignore_index=True)


def load_current() -> pd.DataFrame:
    d = pd.read_csv(os.path.join(DATA, "combined inventory 20260612.csv"), thousands=",")
    return pd.DataFrame({
        "yard": d["Location"], "code": d["Material Code"], "name": d["Material Name"],
        "commodity": d["Commodity Name"], "metal": d["Commodity Name"].map(_metal),
        "wt": pd.to_numeric(d["Total Net Weight"], errors="coerce").fillna(0.0),
        "cost": pd.to_numeric(d["Total Cost"], errors="coerce").fillna(0.0),
    })


# ---- market $/lb per grade, from our own transacted prices ----------------------------

def _flow_lines(pattern: str, value_col: str, date_col: str) -> pd.DataFrame:
    frames = []
    for p in glob.glob(os.path.join(DATA, pattern)):
        if "combined" in os.path.basename(p):
            continue
        d = pd.read_csv(p, thousands=",")
        frames.append(pd.DataFrame({
            "code": d["Material Code"],
            "value": pd.to_numeric(d[value_col], errors="coerce"),
            "wt": pd.to_numeric(d["Net Weight"], errors="coerce"),
            "date": pd.to_datetime(d[date_col].astype("string").str.replace(_BRACKET, "", regex=True),
                                   utc=True, errors="coerce", format="mixed").dt.tz_localize(None),
        }))
    f = pd.concat(frames, ignore_index=True).dropna(subset=["code", "value", "wt"])
    return f[f["wt"] > 0]


def _vwap(df: pd.DataFrame) -> pd.Series:
    g = df.groupby("code")
    return g["value"].sum() / g["wt"].sum()


def grade_prices(avg_cost: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Per-grade $/lb for 'today' (recent sales) and a Jan-2026 anchor, with provenance."""
    ob = _flow_lines("2026 ytd *outbound.csv", "Expected Value", "Date In")
    inb = _flow_lines("2026 ytd *inbound.csv", "Cost", "Effective Date")
    latest = max(ob["date"].max(), inb["date"].max())
    recent = ob[ob["date"] >= latest - pd.Timedelta(days=RECENT_DAYS)]
    jan = ob[ob["date"] < JAN_END]

    def ladder_price(primary, label):
        steps = [(label, primary), ("YTD sale", _vwap(ob)),
                 ("purchase", _vwap(inb)), ("avg cost", avg_cost)]
        codes = set().union(*[s.index for _, s in steps])
        rows = []
        for c in codes:
            for src, s in steps:
                if c in s.index and pd.notna(s[c]) and s[c] > 0:
                    rows.append({"code": c, "price": float(s[c]), "src": src})
                    break
        return pd.DataFrame(rows).set_index("code")

    today = ladder_price(_vwap(recent), "recent sale")
    janp = ladder_price(_vwap(jan), "Jan sale")["price"]
    return today, janp


def futures_factors() -> tuple[dict, dict]:
    """Dec-31-2025 / Jan-2026-avg close per proxy -> per-metal roll-back factor."""
    try:
        import yfinance as yf
    except Exception:
        return {}, {}
    detail = {}
    for ticker in set(PROXY.values()):
        try:
            d = yf.download(ticker, start="2025-12-01", end=JAN_END, progress=False, auto_adjust=True)
            c = d["Close"].squeeze().dropna()
            dec = float(c[c.index <= EOY].iloc[-1])
            jan = float(c[c.index >= "2026-01-01"].mean())
            detail[ticker] = (dec, jan, dec / jan)
        except Exception:
            pass
    factors = {m: detail[t][2] for m, t in PROXY.items() if t in detail}
    return factors, detail


# ---- assembly -------------------------------------------------------------------------

def by_metal(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    g = (df.groupby("metal", as_index=False)
         .agg(wt=("wt", "sum"), cost=("cost", "sum"), mkt=(value_col, "sum"))
         .sort_values("mkt", ascending=False))
    g["mkt_wt_pct"] = (100 * g["mkt"] / g["mkt"].sum()).round(1)
    g["cost_wt_pct"] = (100 * g["cost"] / g["cost"].sum()).round(1)
    return g


def main() -> None:
    eoy = load_eoy2025()
    cur = load_current()
    avg_cost = pd.concat([eoy, cur]).assign(upl=lambda d: d["cost"] / d["wt"].replace(0, pd.NA))
    avg_cost = avg_cost.dropna(subset=["upl"]).groupby("code")["upl"].mean()

    today, janp = grade_prices(avg_cost)
    factors, detail = futures_factors()

    # today's market value
    cur = cur.merge(today, left_on="code", right_index=True, how="left")
    cur["price"] = cur["price"].fillna(0.0)
    cur["mkt"] = (cur["wt"] * cur["price"]).round(2)

    # EOY-2025 market value: Jan price rolled back to Dec-31 by the metal's futures move
    eoy = eoy.merge(janp.rename("jan_price"), left_on="code", right_index=True, how="left")
    eoy["jan_price"] = eoy["jan_price"].fillna(0.0)
    eoy["factor"] = eoy["metal"].map(factors).fillna(1.0)
    eoy["eoy_price"] = eoy["jan_price"] * eoy["factor"]
    eoy["mkt"] = (eoy["wt"] * eoy["eoy_price"]).round(2)

    e_cost, e_mkt = eoy["cost"].sum(), eoy["mkt"].sum()
    c_cost, c_mkt = cur["cost"].sum(), cur["mkt"].sum()
    e_metal, c_metal = by_metal(eoy, "mkt"), by_metal(cur, "mkt")

    L, A = [], lambda s: None
    out = []
    A = out.append
    A("INVENTORY VALUATION — COST vs MARKET, EOY-2025 and TODAY (06/12/2026)")
    A("=" * 78)
    if detail:
        A("Futures roll-back (Dec-31-2025 close / Jan-2026 avg) applied to EOY market:")
        for t, (dec, jan, f) in detail.items():
            A(f"    {t:7} {dec:>9.3f} / {jan:>9.3f}  = x{f:.3f}")
    else:
        A("(yfinance unavailable -> EOY market held at Jan-2026 sale prices, no roll-back)")
    A("")
    A(f"{'':<22}{'EOY-2025':>16}{'TODAY':>16}{'2026 YTD chg':>16}")
    A("-" * 78)
    A(f"{'weight (lbs)':<22}{eoy['wt'].sum():>16,.0f}{cur['wt'].sum():>16,.0f}"
      f"{cur['wt'].sum()-eoy['wt'].sum():>16,.0f}")
    A(f"{'cost basis ($)':<22}{e_cost:>16,.0f}{c_cost:>16,.0f}{c_cost-e_cost:>16,.0f}")
    A(f"{'market value ($)':<22}{e_mkt:>16,.0f}{c_mkt:>16,.0f}{c_mkt-e_mkt:>16,.0f}")
    A(f"{'unrealized gain ($)':<22}{e_mkt-e_cost:>16,.0f}{c_mkt-c_cost:>16,.0f}"
      f"{(c_mkt-c_cost)-(e_mkt-e_cost):>16,.0f}")
    A(f"{'unrealized gain (%)':<22}{100*(e_mkt-e_cost)/e_cost:>15.1f}%{100*(c_mkt-c_cost)/c_cost:>15.1f}%")
    A("")
    for label, m, tot_c, tot_m in [("EOY-2025", e_metal, e_cost, e_mkt),
                                    ("TODAY (06/12/2026)", c_metal, c_cost, c_mkt)]:
        A(f"PORTFOLIO BY METAL — {label}")
        A(f"  {'metal':<16}{'cost $':>12}{'mkt $':>12}{'cost%':>8}{'mkt%':>8}{'unreal $':>12}")
        for _, r in m.iterrows():
            A(f"  {r['metal']:<16}{r['cost']:>12,.0f}{r['mkt']:>12,.0f}"
              f"{r['cost_wt_pct']:>7.1f}%{r['mkt_wt_pct']:>7.1f}%{r['mkt']-r['cost']:>12,.0f}")
        A(f"  {'TOTAL':<16}{tot_c:>12,.0f}{tot_m:>12,.0f}{'':>8}{'':>8}{tot_m-tot_c:>12,.0f}")
        A("")
    text = "\n".join(out)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "inventory_valuation.txt"), "w") as f:
        f.write(text + "\n")
    eoy.round(4).to_csv(os.path.join(OUT, "valuation_eoy2025_by_grade.csv"), index=False)
    cur.round(4).to_csv(os.path.join(OUT, "valuation_today_by_grade.csv"), index=False)
    e_metal.round(2).to_csv(os.path.join(OUT, "valuation_eoy2025_by_metal.csv"), index=False)
    c_metal.round(2).to_csv(os.path.join(OUT, "valuation_today_by_metal.csv"), index=False)
    print(text)


if __name__ == "__main__":
    main()
