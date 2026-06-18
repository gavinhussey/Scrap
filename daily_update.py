"""Run the full daily data refresh after uploading current exports."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DAILY = ROOT / "daily_inputs"
DATA = ROOT / "data"


def _latest_inventory() -> Path | None:
    files = sorted(DATA.glob("combined inventory *.csv")) or sorted(DAILY.glob("combined inventory *.csv"))
    return files[-1] if files else None


def _require(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")


# the combined files live in data and are generated, the raw uploads live in daily_inputs
def _require_inputs(skip_dtc: bool) -> None:
    inv = _latest_inventory()
    if inv is None:
        raise SystemExit("Missing inventory snapshot: data/combined inventory YYYYMMDD.csv (run --merge-inventory)")

    _require(DATA / "2026 ytd combined inbound.csv", "combined inbound export")
    _require(DATA / "2026 ytd combined outbound.csv", "combined outbound export")

    if not skip_dtc and not sorted(DAILY.glob("*DTC*raw*data*.csv")):
        raise SystemExit("Missing DTC raw export: daily_inputs/*DTC*raw*data*.csv")

    print("Inputs:", flush=True)
    print(f"  inventory : {inv.relative_to(ROOT)}", flush=True)
    print("  inbound   : data/2026 ytd combined inbound.csv", flush=True)
    print("  outbound  : data/2026 ytd combined outbound.csv", flush=True)
    if not skip_dtc:
        print(f"  dtc       : {sorted(DAILY.glob('*DTC*raw*data*.csv'))[-1].relative_to(ROOT)}", flush=True)


# run one step as a subprocess and blow up loudly if it fails
def _run(label: str, args: list[str]) -> None:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/private/tmp")
    print(f"\n==> {label}", flush=True)
    print("    " + " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run daily Scrapyard refresh workflow")
    parser.add_argument(
        "--no-refresh-prices",
        action="store_true",
        help="Use cached market prices for the risk model instead of refreshing them.",
    )
    parser.add_argument(
        "--skip-dtc",
        action="store_true",
        help="Skip DTC ticket matching.",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Skip rebuilding the combined files and reuse the existing ones in data/.",
    )
    args = parser.parse_args()

    # merge fresh by default since the combined files are derived from the daily uploads
    if not args.no_merge:
        _run("Merge yard inbound/outbound exports", [sys.executable, "data/merge_orders.py"])
        _run("Merge yard inventory exports", [sys.executable, "data/merge_inventory.py"])

    _require_inputs(skip_dtc=args.skip_dtc)

    # the actual pipeline, valuation then risk then html then dtc then tests
    _run("Update inventory valuation outputs", [sys.executable, "inventory_valuation.py"])

    risk_cmd = [sys.executable, "run_risk_model.py"]
    if not args.no_refresh_prices:
        risk_cmd.append("--refresh")
    _run("Rebuild risk model charts", risk_cmd)

    _run("Rebuild HTML risk report", [sys.executable, "generate_risk_report_html.py"])

    if not args.skip_dtc:
        _run("Rebuild DTC ticket matching outputs", [sys.executable, "dtc/dtc_weight_match.py"])

    _run("Run tests", [sys.executable, "-m", "pytest"])
    print("\nDaily refresh complete.", flush=True)


if __name__ == "__main__":
    main()
