"""Fetch metals settlement prices via Bloomberg Terminal.

Pulls four instruments, all with Ring official settlement available by ~7:15 AM CT
(1:15 PM London), safely before the 8:30 AM CT prediction deadline.

  LMCADS03 Comdty  — LME Copper 3m official settlement      -> data/external/lme_copper.csv
  LMAHDS03 Comdty  — LME Aluminum 3m official settlement    -> data/external/lme_aluminum.csv
  LMZSDS03 Comdty  — LME Zinc 3m official settlement        -> data/external/lme_zinc.csv
  CU1 Comdty       — SHFE Copper front month                 -> data/external/shfe_copper.csv
                     (SHFE day session closes ~3 PM Beijing = ~1 AM CT)

Requires Bloomberg Terminal running and logged in (port 8194).
Install:  pip install xbbg   OR   pip install pdblp

Run:  python bloombergCopper.py
"""
import sys
import pandas as pd
from pathlib import Path

START = "20140601"
END   = pd.Timestamp.today().strftime("%Y%m%d")

TICKERS = [
    ("LMCADS03 Comdty", "lme_close",    "data/external/lme_copper.csv"),
    ("LMAHDS03 Comdty", "lme_al_close", "data/external/lme_aluminum.csv"),
    ("LMZSDS03 Comdty", "lme_zn_close", "data/external/lme_zinc.csv"),
    ("CU1 Comdty",      "shfe_cu_close","data/external/shfe_copper.csv"),
]


def fetch_xbbg(ticker: str) -> pd.DataFrame:
    from xbbg import blp
    result = blp.bdh(tickers=ticker, flds="PX_LAST", start_date="2014-06-01")
    if hasattr(result, "to_native"):
        df = result.to_native()
    elif hasattr(result, "to_pandas"):
        df = result.to_pandas()
    else:
        df = pd.DataFrame(result)
    if df is None or len(df) == 0:
        raise RuntimeError(f"{ticker}: Bloomberg returned empty data — is Terminal running?")
    # Long format: [ticker, date, field, value]
    if "value" in df.columns and "date" in df.columns:
        return df[df["field"] == "PX_LAST"][["date", "value"]].copy()
    # Wide / MultiIndex format
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(0, axis=1).reset_index()
    else:
        df = df.reset_index()
    df = df.iloc[:, :2].copy()
    df.columns = ["date", "value"]
    return df


def fetch_pdblp(ticker: str) -> pd.DataFrame:
    import pdblp
    con = pdblp.BCon(debug=False, port=8194, timeout=5000)
    con.start()
    try:
        df = con.bdh(tickers=[ticker], flds=["PX_LAST"],
                     start_date=START, end_date=END)
        df = df.droplevel(0, axis=1).reset_index()
        df.columns = ["date", "value"]
        return df
    finally:
        con.stop()


def fetch(ticker: str) -> pd.DataFrame:
    for name, fn in [("xbbg", fetch_xbbg), ("pdblp", fetch_pdblp)]:
        try:
            return fn(ticker)
        except ImportError:
            pass
        except Exception as e:
            print(f"  {name} failed for {ticker}: {e}")
    raise RuntimeError(f"All methods failed for {ticker}. Is Bloomberg Terminal running?")


def main():
    any_failed = False
    for ticker, col_name, out_path in TICKERS:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"Fetching {ticker} -> {out_path} ...")
        try:
            raw = fetch(ticker)
            raw["date"] = pd.to_datetime(raw["date"])
            raw = raw.dropna().sort_values("date").drop_duplicates("date")
            raw.columns = ["date", col_name]
            raw.to_csv(out, index=False)
            print(f"  Saved {len(raw)} rows "
                  f"({raw['date'].min().date()} - {raw['date'].max().date()})")
        except Exception as e:
            print(f"  FAILED: {e}")
            any_failed = True

    if any_failed:
        print("\nSome tickers failed. Re-run after checking Bloomberg connection.")
        sys.exit(1)
    print("\nAll done. Re-run copper_close_morning_model.py to use new features.")


if __name__ == "__main__":
    main()
