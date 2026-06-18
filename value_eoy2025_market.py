"""Value the reconstructed EOY-2025 portfolio at end-of-2025 MARKET prices."""

import os
import re
import glob

import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
DAILY = os.path.join(HERE, "daily_inputs")
OUT = os.path.join(HERE, "output")
_BRACKET = re.compile(r"\[[^\]]*\]$")


def _glob(pattern):
    hits = glob.glob(os.path.join(DAILY, pattern))
    return hits or glob.glob(os.path.join(DATA, pattern))

EOY = "2025-12-31"
JAN_END = "2026-02-01"
PROXY = {"COPPER": "HG=F", "STEEL": "HRC=F", "ALUMINUM": "ALI=F", "BRASS": "HG=F"}


def _flow_lines(pattern: str, value_col: str) -> pd.DataFrame:
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


def _vwap(df):
    g = df.groupby("code")
    return g["value"].sum() / g["wt"].sum()


def jan_scrap_prices() -> tuple[pd.DataFrame, dict]:
    """Per-grade $/lb anchored to Jan-2026, with provenance ladder."""
    ob = _flow_lines("2026 ytd *outbound.csv", "Expected Value")
    inb = _flow_lines("2026 ytd *inbound.csv", "Cost")
    jan_ob = ob[ob["date"] < JAN_END]
    avg_cost = (pd.read_csv(os.path.join(OUT, "opening_inventory_eoy2025_by_grade.csv"))
                .groupby("code")["avg_cost"].mean())
    ladder = [("Jan sale", _vwap(jan_ob)), ("YTD sale", _vwap(ob)),
              ("YTD purchase", _vwap(inb)), ("avg cost", avg_cost)]
    codes = set().union(*[s.index for _, s in ladder])
    rows = []
    for c in codes:
        for src, s in ladder:
            if c in s.index and pd.notna(s[c]) and s[c] > 0:
                rows.append({"code": c, "jan_price": float(s[c]), "price_source": src})
                break
    return pd.DataFrame(rows), {}


def futures_factors() -> dict:
    """Dec-31-2025 / Jan-2026-avg close for each proxy ticker -> per-metal factor."""
    factors, detail = {}, {}
    for ticker in set(PROXY.values()):
        d = yf.download(ticker, start="2025-12-01", end=JAN_END, progress=False, auto_adjust=True)
        c = d["Close"].squeeze().dropna()
        dec = float(c[c.index <= EOY].iloc[-1])
        jan = float(c[c.index >= "2026-01-01"].mean())
        detail[ticker] = (dec, jan, dec / jan)
    for metal, ticker in PROXY.items():
        factors[metal] = detail[ticker][2]
    return factors, detail


def main() -> None:
    holdings = pd.read_csv(os.path.join(OUT, "opening_portfolio_weights_by_grade.csv"))[
        ["code", "name", "commodity", "metal", "qty_lbs"]]
    prices, _ = jan_scrap_prices()
    factors, detail = futures_factors()

    df = holdings.merge(prices, on="code", how="left")
    df["jan_price"] = df["jan_price"].fillna(0.0)
    df["factor"] = df["metal"].map(factors).fillna(1.0)
    df["adjusted"] = df["metal"].isin(factors)
    df["eoy_price"] = (df["jan_price"] * df["factor"]).round(4)
    df["jan_value"] = (df["qty_lbs"] * df["jan_price"]).round(2)
    df["eoy_value"] = (df["qty_lbs"] * df["eoy_price"]).round(2)

    by_metal = (df.groupby("metal", as_index=False)
                .agg(qty_lbs=("qty_lbs", "sum"), jan_value=("jan_value", "sum"),
                     eoy_value=("eoy_value", "sum"), factor=("factor", "first"),
                     adjusted=("adjusted", "first"))
                .sort_values("eoy_value", ascending=False))
    tot_eoy = by_metal["eoy_value"].sum()
    by_metal["weight_pct"] = (100 * by_metal["eoy_value"] / tot_eoy).round(2)

    df.sort_values("eoy_value", ascending=False).round(4).to_csv(
        os.path.join(OUT, "eoy2025_market_value_by_grade.csv"), index=False)
    by_metal.round(2).to_csv(os.path.join(OUT, "eoy2025_market_value_by_metal.csv"), index=False)

    L = []
    A = L.append
    A("EOY-2025 PORTFOLIO MARKET VALUE  (Jan-2026 scrap prices rolled back via yfinance)")
    A("=" * 76)
    A("Futures move applied (Dec-31-2025 close / Jan-2026 avg close):")
    for t, (dec, jan, f) in detail.items():
        A(f"  {t:7} Dec-31={dec:>10.3f}  Jan-avg={jan:>10.3f}  factor={f:.3f}")
    A("")
    A(f"{'metal':<18}{'qty lbs':>14}{'EOY-2025 $':>16}{'  wt%':>8}{'  basis':>16}")
    for _, r in by_metal.iterrows():
        basis = f"x{r['factor']:.3f}" if r["adjusted"] else "Jan price (no proxy)"
        A(f"{r['metal']:<18}{r['qty_lbs']:>14,.0f}{r['eoy_value']:>16,.0f}"
          f"{r['weight_pct']:>7.1f}%   {basis}")
    A("-" * 76)
    A(f"{'TOTAL EOY-2025 MARKET VALUE':<32}{tot_eoy:>16,.0f}")
    A(f"{'(for reference) at Jan-2026 prices':<32}{by_metal['jan_value'].sum():>16,.0f}")
    A("")
    A("CAVEATS")
    A("  - Quantities are the reconstructed EOY-2025 opening (held, floored >=0).")
    A("  - Per-grade scrap price = real Jan-2026 sale vwap (true basis), rolled to")
    A("    Dec-31 by the metal's futures move. ~weeks apart, so a small correction.")
    A("  - Stainless/lead/zinc/other have NO Dec-2025 futures proxy (NI=F delisted)")
    A("    -> held at Jan price; their EOY value is a Jan proxy, not Dec-31.")
    A("  - This is MARKET (sale/liquidation) value, not cost/book value.")
    out = "\n".join(L)
    with open(os.path.join(OUT, "eoy2025_market_value.txt"), "w") as f:
        f.write(out + "\n")
    print(out)
    print(f"\nWrote: output/eoy2025_market_value{{.txt,_by_metal.csv,_by_grade.csv}}")


if __name__ == "__main__":
    main()
