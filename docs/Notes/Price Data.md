# Price Data

## Source
Price history is loaded through `src/prices.py`. Exchange prices are fetched with `yfinance` and cached in `cache/`.

## Run Source Labels
Each run tracks price source quality:
- `yahoo`: downloaded during the run
- `cache`: loaded from local cache
- `proxy:<metal>`: using another metal's price series
- `synthetic`: generated fallback series because live/cache data was unavailable

Synthetic data degrades model quality and should not be used for final business decisions without review.

## Proxy Metals
Some scrap categories use proxy series:
- brass -> copper
- stainless -> copper, with additional volatility multiplier
- lead -> copper
- zinc -> copper

These proxies are pragmatic approximations and should be replaced with better market data when available.

## Risk Series
The model separates display/spot prices from risk prices. Risk prices apply the scrap basis volatility overlay before VaR and Monte Carlo calculations.

## Limitations
- Yahoo Finance is not a regulated data source.
- Some tickers can be stale, unavailable, or revised.
- Scrap-specific price data is not directly modeled unless realized transaction prices exist.
- Daily closes are not suitable for intraday risk.
