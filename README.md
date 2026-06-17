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

The DTC workflow links each DTC order to its partner inbound and outbound tickets by weight, treating DTC as a same-day pass-through (one ticket in = one ticket out, exact net). Outbound links authoritatively via the DTC `Outbound Ticket Id`; the inbound leg is matched same-day (within 2 business days). Matches are corroborated by supplier vs consumer and filtered/annotated against the BMR Transport roster.

Inputs are read from `daily_inputs/` (combined inbound/outbound plus the DTC raw export).

Key files:

- `dtc/dtc_weight_match.py` - the DTC matching pipeline (weight-based same-day pass-through).
- `dtc/BMR_Transport_Data.csv` - roster of valid `(material, supplier company)` DTC combinations. A material or company absent from it cannot be a DTC order; for genuine missing inbound legs it names the supplier companies to chase in GreenSpark.
- `data/merge_orders.py` - merges yard inbound/outbound exports into the combined files in `daily_inputs/` and creates stable row IDs.

Run:

```sh
python3 data/merge_orders.py          # optional: rebuild combined inbound/outbound first
env MPLCONFIGDIR=/private/tmp python3 dtc/dtc_weight_match.py
```

Primary outputs:

- `dtc/dtc_weight_matches.csv` - every DTC order with its matched inbound/outbound tickets and CSV row references.
- `dtc/dtc_match_exceptions.csv` - rows needing review (missing inbound legs, outside-same-day, not-a-DTC-material).
- `dtc/dtc_ticket_match.png` - color-coded summary chart.
- `dtc/dtc_ticket_match.csv` - CSV twin of the chart (identical rows and columns).

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
