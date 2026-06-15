"""Run the full daily data refresh after uploading current exports."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DTC = ROOT / "dtc"


def _latest_inventory() -> Path | None:
    files = sorted(DATA.glob("combined inventory *.csv"))
    return files[-1] if files else None


def _require(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")


def _require_inputs(skip_dtc: bool) -> None:
    inv = _latest_inventory()
    if inv is None:
        raise SystemExit("Missing inventory snapshot: data/combined inventory YYYYMMDD.csv")

    _require(DATA / "2026 ytd combined inbound.csv", "combined inbound export")
    _require(DATA / "2026 ytd combined outbound.csv", "combined outbound export")

    if not skip_dtc and not sorted(DTC.glob("*DTC_raw_data*.csv")):
        raise SystemExit("Missing DTC raw export: dtc/*DTC_raw_data*.csv")

    print("Inputs:", flush=True)
    print(f"  inventory : {inv.relative_to(ROOT)}", flush=True)
    print("  inbound   : data/2026 ytd combined inbound.csv", flush=True)
    print("  outbound  : data/2026 ytd combined outbound.csv", flush=True)
    if not skip_dtc:
        print(f"  dtc       : {sorted(DTC.glob('*DTC_raw_data*.csv'))[-1].relative_to(ROOT)}", flush=True)


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
        help="Skip DTC model training and ticket matching.",
    )
    parser.add_argument(
        "--skip-dtc-training",
        action="store_true",
        help="Run DTC matching but skip retraining inbound/outbound rankers.",
    )
    parser.add_argument(
        "--merge-orders",
        action="store_true",
        help="Rebuild combined inbound/outbound files from yard-level exports first.",
    )
    parser.add_argument(
        "--merge-inventory",
        action="store_true",
        help="Rebuild combined inventory snapshot from yard inventory reports first.",
    )
    args = parser.parse_args()

    if args.merge_orders:
        _run("Merge yard inbound/outbound exports", [sys.executable, "data/merge_orders.py"])

    if args.merge_inventory:
        _run("Merge yard inventory exports", [sys.executable, "data/merge_inventory.py"])

    _require_inputs(skip_dtc=args.skip_dtc)

    _run("Update inventory valuation outputs", [sys.executable, "inventory_valuation.py"])

    risk_cmd = [sys.executable, "run_risk_model.py"]
    if not args.no_refresh_prices:
        risk_cmd.append("--refresh")
    _run("Rebuild risk model charts", risk_cmd)

    _run("Rebuild HTML risk report", [sys.executable, "generate_risk_report_html.py"])

    if not args.skip_dtc:
        if not args.skip_dtc_training:
            _run("Train outbound DTC ranker", [sys.executable, "dtc/train_outbound_model.py"])
            _run("Train inbound DTC ranker", [sys.executable, "dtc/train_inbound_model.py"])
        _run("Rebuild DTC ticket matching outputs", [sys.executable, "dtc/dtc_ticket_match.py"])

    _run("Run tests", [sys.executable, "-m", "pytest"])
    print("\nDaily refresh complete.", flush=True)


if __name__ == "__main__":
    main()
