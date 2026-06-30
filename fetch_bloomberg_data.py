"""Single-entry Bloomberg fetch script — run this on the Bloomberg PC.

Fetches all Bloomberg data needed for the three direction models and writes CSVs
to the correct data/external/ directories. Then commit and push from this PC.

SETUP (Bloomberg PC only — do this once):
    pip install xbbg pandas

USAGE:
    python fetch_bloomberg_data.py

    Optional: skip a metal with  --skip-steel  --skip-aluminum
    Optional: dry run (list tickers without fetching)  --dry-run

AFTER IT FINISHES:
    git add -A
    git commit -m "Add Bloomberg data"
    git push

Then on your Mac:  git pull  and re-run the models.

⚠️  VERIFY TICKERS BEFORE RUNNING — see TICKER_NOTES below.
    Type each ticker in Bloomberg and run  DES <GO>  to confirm it is correct.
    Edit the STEEL_TICKERS or ALUMINUM_TICKERS lists below if any are wrong.
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent

# ──────────────────────────────────────────────────────────────────────────────
# STEEL — China ferrous complex
# Outputs -> steel_direction_model_v1/data/external/
# ⚠️  Verify each ticker: open Bloomberg, type ticker <GO>, then DES <GO>
# ──────────────────────────────────────────────────────────────────────────────
STEEL_OUT = BASE / "steel_direction_model_v1" / "data" / "external"

STEEL_TICKERS = [
    # ticker               column           filename                  description
    ("RBTA Comdty",  "shfe_rb_close",  "shfe_rebar.csv",       "SHFE rebar 1st generic (CNY/t)"),
    ("HCA Comdty",   "shfe_hrc_close", "shfe_hrc.csv",         "SHFE hot-rolled coil 1st generic (CNY/t)"),
    ("IOE1 Comdty",  "dce_io_close",   "dce_iron_ore.csv",     "DCE iron ore 1st generic (CNY/t)"),
    ("JMA Comdty",   "dce_cc_close",   "dce_coking_coal.csv",  "DCE coking coal 1st generic (CNY/t)"),
    ("SCO1 Comdty",  "sgx_io_close",   "sgx_iron_ore.csv",     "SGX iron ore 62% Fe (USD/t)"),
]

# ──────────────────────────────────────────────────────────────────────────────
# ALUMINUM — SHFE aluminum, LME inventory, European gas
# Outputs -> aluminum_direction_model_v1/data/external/
# ⚠️  Verify each ticker before running
# ──────────────────────────────────────────────────────────────────────────────
ALUMINUM_OUT = BASE / "aluminum_direction_model_v1" / "data" / "external"

ALUMINUM_TICKERS = [
    # ticker               column            filename                       description
    ("ALUA Comdty",   "shfe_al_close",  "shfe_aluminum.csv",           "SHFE aluminum 1st generic (CNY/t)"),
    ("LMAHSTKS Index","lme_al_stocks",  "lme_al_inventory.csv",        "LME aluminum on-warrant stocks (t)"),
    ("LMAHCWRTS Index","lme_al_cw",    "lme_al_cancelled_warrants.csv","LME aluminum cancelled warrants (%)"),
    ("TTFMMA0 Comdty","ttf_close",     "ttf_gas.csv",                  "TTF natural gas front-month (EUR/MWh)"),
]

# ──────────────────────────────────────────────────────────────────────────────
# TICKER NOTES — common alternatives if a ticker fails DES <GO> check
# ──────────────────────────────────────────────────────────────────────────────
TICKER_NOTES = """
Ticker verification guide (type each in Bloomberg, then DES <GO>):

  RBTA Comdty   → SHFE rebar. Alternatives: RBA Comdty, RBT1 Comdty
  HCA Comdty    → SHFE hot-rolled coil. Alternatives: HRC1 Comdty, SHFRHRCA Index
  IOE1 Comdty   → DCE iron ore. Alternatives: I01 Comdty, IOA Comdty
  JMA Comdty    → DCE coking coal. Alternatives: JM01 Comdty, JMCA Comdty
  SCO1 Comdty   → SGX iron ore. Alternatives: TIO1 Comdty, TIO=F (yfinance only)

  ALUA Comdty   → SHFE aluminum. Alternatives: ALU1 Comdty, ALUA1 Comdty
  LMAHSTKS Index → LME aluminum stocks. Search: LME ALUMINIUM WAREHOUSE STOCKS
  LMAHCWRTS Index → LME aluminum cancelled warrants. Search: LME ALUMINIUM CANCELLED
  TTFMMA0 Comdty → TTF gas. Alternatives: TTFGAS Index, GASNLH Comdty
"""

START = "20140601"
END = pd.Timestamp.today().strftime("%Y%m%d")


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
        raise RuntimeError("Bloomberg returned empty data")
    if "value" in df.columns and "date" in df.columns:
        return df[df["field"] == "PX_LAST"][["date", "value"]].copy()
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


def fetch_one(ticker: str) -> pd.DataFrame:
    for name, fn in [("xbbg", fetch_xbbg), ("pdblp", fetch_pdblp)]:
        try:
            return fn(ticker)
        except ImportError:
            pass
        except Exception as e:
            print(f"      {name}: {e}")
    raise RuntimeError(f"All Bloomberg methods failed for {ticker}")


def run_group(label: str, tickers: list, out_dir: Path, dry_run: bool) -> tuple[int, int]:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ok, fail = 0, 0
    for ticker, col, filename, desc in tickers:
        out = out_dir / filename
        print(f"\n  {ticker}  —  {desc}")
        if dry_run:
            print(f"    [dry-run] would write -> {out.relative_to(BASE)}")
            continue
        print(f"    fetching...", end="", flush=True)
        try:
            raw = fetch_one(ticker)
            raw["date"] = pd.to_datetime(raw["date"])
            raw = raw.dropna().sort_values("date").drop_duplicates("date")
            raw.columns = ["date", col]
            raw.to_csv(out, index=False)
            print(f"  {len(raw)} rows  "
                  f"({raw['date'].min().date()} → {raw['date'].max().date()})")
            print(f"    -> {out.relative_to(BASE)}")
            ok += 1
        except Exception as e:
            print(f"\n    FAILED: {e}")
            print(f"    *** Verify '{ticker}' on the Terminal: DES <GO> ***")
            fail += 1
    return ok, fail


def main():
    parser = argparse.ArgumentParser(description="Fetch Bloomberg data for metal direction models")
    parser.add_argument("--skip-steel",    action="store_true", help="Skip steel/iron ore tickers")
    parser.add_argument("--skip-aluminum", action="store_true", help="Skip aluminum tickers")
    parser.add_argument("--dry-run",       action="store_true", help="List tickers without fetching")
    args = parser.parse_args()

    print(TICKER_NOTES)

    if args.dry_run:
        print("DRY RUN — no data will be fetched\n")

    total_ok, total_fail = 0, 0

    if not args.skip_steel:
        ok, fail = run_group("STEEL — China ferrous complex", STEEL_TICKERS, STEEL_OUT, args.dry_run)
        total_ok += ok; total_fail += fail

    if not args.skip_aluminum:
        ok, fail = run_group("ALUMINUM — fundamentals", ALUMINUM_TICKERS, ALUMINUM_OUT, args.dry_run)
        total_ok += ok; total_fail += fail

    print(f"\n{'='*60}")
    if args.dry_run:
        print("  Dry run complete — run without --dry-run to fetch data")
    else:
        print(f"  Done: {total_ok} succeeded, {total_fail} failed")
        if total_fail:
            print("  Fix the failed tickers (see *** lines above) and re-run.")
            print("  Successful CSVs are already saved — use --skip-* flags to skip them.")
        if total_ok:
            print("""
  Next steps on this PC:
    git add steel_direction_model_v1/data/external/
    git add aluminum_direction_model_v1/data/external/
    git commit -m "Add Bloomberg data: China ferrous + aluminum fundamentals"
    git push

  Then on your Mac:
    git pull
    cd steel_direction_model_v1  &&  python iron_ore_direction_model.py
    cd steel_direction_model_v1  &&  python iron_ore_morning_model.py
    cd aluminum_direction_model_v1  &&  python lme_aluminum_morning_model.py
""")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
