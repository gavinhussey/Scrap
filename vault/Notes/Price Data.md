# Price Data

## Source
All price data is pulled from **Yahoo Finance** via the `yfinance` Python library.

## Tickers
| Metal | Ticker | Exchange | Raw Unit | Converted To |
|-------|--------|----------|----------|--------------|
| Copper | `HG=F` | COMEX (CME Group) | USD/lb | USD/tonne (× 2204.62) |
| Aluminium | `ALI=F` | CME Group | USD/tonne | USD/tonne (no change) |

## Caching
- On first run: downloads 5 years of daily closing prices, saves to `cache/copper_prices.csv` and `cache/aluminium_prices.csv`
- On subsequent runs: uses cache if last date is yesterday or today
- If cache is stale (older than yesterday): re-downloads automatically
- Force refresh at any time: `python run_risk_model.py --refresh`
- If download fails: falls back to synthetic price series so the model doesn't crash

## Current Price Snapshot (as of 2026-06-09)
| Metal | Price (USD/tonne) |
|-------|-------------------|
| Copper | ~$13,960 |
| Aluminium | ~$3,576 |

## Data Range
- Start: June 2021
- End: Current (updates on each run)
- ~1,257 trading days for copper, ~1,255 for aluminium

## Limitations
- Yahoo Finance is a free, unofficial source — not suitable for regulated reporting
- No scrap-specific prices (scrap data requires paid feeds e.g. AMM, Fastmarkets, Metal Bulletin)
- `ALI=F` (aluminium) is less liquid than LME contracts — may have gaps or stale closes
- Daily closes only — not suitable for intraday analysis
- Prices available after ~4pm EST market close; runs before close use previous day's price

## Upgrading the Data Source
Only `_download()` in `src/prices.py` needs to change to swap in a different provider. The rest of the model consumes a standard pandas Series of daily USD/tonne prices.

## Related Notes
- [[Risk Model Overview]]
- [[Inventory & Grades]]
