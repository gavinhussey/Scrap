# Scrapyard Project

Analysis and matching tools for scrapyard inventory, commodity exposure, and DTC order reconciliation.

## Main Workflows

### Daily Full Refresh

After uploading the latest daily files:

- `daily_inputs/Milton - Inventory Report - Weights and Costs - YYYYMMDD.csv`
- `daily_inputs/Merriville - Inventory Report - Weights and Costs - YYYYMMDD.csv`
- `daily_inputs/2026 ytd milton inbound.csv`
- `daily_inputs/2026 ytd merriville inbound.csv`
- `daily_inputs/2026 ytd milton outbound.csv`
- `daily_inputs/2026 ytd merriville outbound.csv`
- `daily_inputs/*DTC_raw_data*.csv`

run:

```sh
source .venv/bin/activate
MPLCONFIGDIR=/private/tmp python daily_update.py
```

Useful options:

```sh
python daily_update.py --no-refresh-prices
python daily_update.py --skip-dtc
python daily_update.py --skip-merge-orders
python daily_update.py --skip-merge-inventory
```

`daily_inputs/` is the only folder that needs daily upload/replacement. The
two yard inventory reports are merged automatically into a combined inventory
snapshot, and the four yard-level order exports are merged automatically into
combined inbound and outbound files. The `data/` folder is for historical/static
inputs and merge scripts.

### DTC Ticket Matching

The DTC workflow links DTC order rows to inbound and outbound ticket rows using material, weight, date, customer, price, and row-level identifiers.

Key files:

- `dtc/dtc_ticket_match.py` - main DTC matching pipeline.
- `dtc/train_outbound_model.py` - outbound model training using deterministic `Outbound Ticket Id` labels.
- `dtc/train_inbound_model.py` - inbound weak-supervision model training from strict pseudo-labels.
- `data/merge_orders.py` - merges inbound/outbound source exports and creates stable row IDs.

Run:

```sh
python3 data/merge_orders.py
env MPLCONFIGDIR=/private/tmp python3 dtc/train_outbound_model.py
env MPLCONFIGDIR=/private/tmp python3 dtc/train_inbound_model.py
env MPLCONFIGDIR=/private/tmp python3 dtc/dtc_ticket_match.py
```

Primary outputs:

- `dtc/matched_dtc_orders.csv`
- `dtc/exceptions_unmatched_or_ambiguous.csv`
- `dtc/candidate_matches_inbound.csv`
- `dtc/candidate_matches_outbound.csv`
- `dtc/match_summary.txt`
- `dtc/dtc_ticket_match.png`

### Inventory and Market Risk

The inventory/risk scripts estimate market exposure and portfolio risk for inventory positions.

Key files:

- `run_risk_model.py`
- `reconstruct_opening_inventory.py`
- `grade_basis_calibration.py`
- `value_eoy2025_market.py`
- `src/`

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Privacy

The CSV files include business data such as customer/vendor names, addresses, weights, values, and costs. Keep this repository private unless the data is removed or redacted.

## Tests

```sh
python3 -m pytest
```
