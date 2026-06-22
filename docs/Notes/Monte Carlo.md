# Monte Carlo

## What It Does
Monte Carlo simulates portfolio value paths over 30, 60, 90, and 180 trading-day horizons using correlated geometric Brownian motion.

## Inputs
- Risk exposure by metal
- Basis-vol-adjusted risk price series
- EWMA volatility
- Historical correlations after proxy metals are collapsed to their price driver

## Convention
The model uses a zero-drift risk convention by default. This avoids embedding historical bull-market drift into loss estimates.

## Output
The report shows median, 5th percentile, 95th percentile, probability of loss, MC VaR, and MC CVaR.

## Limitations
- GBM assumes continuous lognormal price moves.
- Tail risk can be understated if historical volatility misses crisis behavior.
- Proxy metals inherit another metal's return behavior.
- Synthetic price series make the simulation degraded.
