# Inventory & Grades

## Inventory Source
The risk model uses the latest GreenSpark combined inventory snapshot, normally:

`daily_inputs/combined inventory YYYYMMDD.csv`

The snapshot is treated as the physical inventory source of truth. YTD inbound/outbound files are used for realized-margin signals, valuation support, and flow validation, not for primary inventory quantity.

## Required Snapshot Columns
- `Location`
- `Material Code`
- `Material Name`
- `Commodity Name`
- `Total Net Weight`
- `Total Cost`

Negative `Total Net Weight` or `Total Cost` values are rejected.

## Commodity Mapping
`Commodity Name` is mapped to modeled metal using `COMMODITY_TO_METAL` in `src/config.py`. `OTHER` and `ZWASTE` are excluded from modeled market risk and reported as dropped lines.

## Valuation Policy
- If a grade has a usable realized market price, MTM uses that price.
- If a grade is unpriced, MTM is held at book value.
- Unpriced inventory still contributes to risk exposure using the higher of net-realizable book value and a futures-basis proxy value.

## Aging
Inventory aging is approximate. The model uses the weight-weighted average inbound date by material code when available. If no inbound date exists, it falls back to the snapshot date in the filename.

## Assumptions
Grade basis, haircuts, proxy metals, and unpriced-risk treatment are documented in `ASSUMPTIONS` in `src/config.py` and shown in the HTML report.
