"""Fetch LME Copper 3-month official settlement prices via Bloomberg Terminal.

Requires Bloomberg Terminal running and logged in (port 8194).
Ticker: LMCADS03 Comdty — LME Copper 3-month official settlement, USD/tonne.

Output: data/external/lme_copper.csv with columns [date, lme_close]
This file is automatically picked up by copper_close_morning_model.py.

Install one of:
    pip install pdblp
    pip install xbbg
"""
import sys
import pandas as pd
from pathlib import Path

OUT = Path("data/external/lme_copper.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

START = "20140601"
END   = pd.Timestamp.today().strftime("%Y%m%d")
TICKER = "LMCADS03 Comdty"


def fetch_pdblp() -> pd.DataFrame:
    import pdblp
    con = pdblp.BCon(debug=False, port=8194, timeout=5000)
    con.start()
    try:
        df = con.bdh(tickers=[TICKER], flds=["PX_LAST"],
                     start_date=START, end_date=END)
        df = df.droplevel(0, axis=1).reset_index()
        df.columns = ["date", "lme_close"]
        return df
    finally:
        con.stop()


def fetch_xbbg() -> pd.DataFrame:
    from xbbg import blp
    result = blp.bdh(tickers=TICKER, flds="PX_LAST", start_date="2014-06-01")
    # xbbg may return a narwhals DataFrame in newer versions — convert to pandas
    if hasattr(result, "to_native"):
        df = result.to_native()
    elif hasattr(result, "to_pandas"):
        df = result.to_pandas()
    else:
        df = pd.DataFrame(result)
    if df is None or len(df) == 0:
        raise RuntimeError("Bloomberg returned empty data — is Terminal running and logged in?")
    # Handle long format: [ticker, date, field, value]
    if "value" in df.columns and "date" in df.columns:
         
        df = df[df["field"] == "PX_LAST"][["date", "value"]].copy()
        df.columns = ["date", "lme_close"]
        return df
    # Handle wide format with MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(0, axis=1).reset_index()
    else:
        df = df.reset_index()
    df.columns = ["date", "lme_close"]
    return df


def main():
    df = None
    for name, fn in [("pdblp", fetch_pdblp), ("xbbg", fetch_xbbg)]:
        try:
            print(f"Trying {name}...")
            df = fn()
            print(f"  {name} succeeded")
            break
        except ImportError:
            print(f"  {name} not installed — skipping")
        except Exception as e:
            print(f"  {name} failed: {e}")

    if df is None:
        print("\nBoth methods failed. Make sure Bloomberg Terminal is running and logged in.")
        print("Install a Bloomberg Python library:  pip install pdblp  OR  pip install xbbg")
        sys.exit(1)

    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["lme_close"]).sort_values("date").drop_duplicates("date")
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} rows ({df['date'].min().date()} - {df['date'].max().date()}) to {OUT}")
    print("Re-run copper_close_morning_model.py to incorporate LME features.")


if __name__ == "__main__":
    main()
