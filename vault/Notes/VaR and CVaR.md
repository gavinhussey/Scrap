# VaR and CVaR

## What They Measure
VaR estimates a loss threshold over the configured horizon. CVaR estimates the average loss beyond that threshold.

The current model uses net-realizable risk exposure, including unpriced inventory through a conservative proxy.

## Methods
- Parametric VaR: normal distribution, historical daily volatility, square-root horizon scaling.
- EWMA VaR: normal distribution with RiskMetrics-style recent volatility weighting.
- Historical VaR: overlapping horizon windows from empirical returns.

## Interpretation
VaR is a market-risk estimate, not a maximum loss. It does not cover operational failures, bad data, customer credit, environmental events, theft, or liquidity gaps beyond the configured haircuts.

## Data Quality
VaR should be interpreted as degraded if:
- synthetic price series are used
- large inventory value is unpriced
- key metals are proxied to another metal
- the GreenSpark snapshot is stale or fails validation

## Where To Review Assumptions
See `ASSUMPTIONS` in `src/config.py` and the HTML report methodology section.
