# Monte Carlo Simulation

## What It Does
Runs 10,000 simulated price paths for copper and aluminium simultaneously over a 30-day horizon, using their historical volatility and the correlation between the two metals. Each path is one possible future for the portfolio.

## How to Read the Chart (monte_carlo.png)
- **X-axis**: Time (trading days 0–30)
- **Y-axis**: Total portfolio value in USD
- **Dotted black line**: Today's portfolio value (starting point)
- **Dark band**: 25th–75th percentile — the middle 50% of outcomes
- **Light band**: 5th–95th percentile — the outer 90% of outcomes
- **Blue line**: Median path (50th percentile)

The cone shape widens over time — uncertainty compounds the further out you go.

## Current Results (~$964K portfolio, 30-day horizon)
| Percentile | Portfolio Value | P&L vs. Today |
|------------|----------------|----------------|
| 1st | ~$770K | -$194K |
| 5th | ~$825K | -$139K |
| 25th | ~$909K | -$55K |
| 50th (median) | ~$971K | +$6K |
| 75th | ~$1.038M | +$74K |
| 95th | ~$1.141M | +$177K |
| 99th | ~$1.226M | +$261K |

- **Probability of any loss**: ~47%
- **95% VaR (Monte Carlo)**: ~$139K
- **95% CVaR (Monte Carlo)**: ~$173K

## Method
Uses correlated Geometric Brownian Motion (GBM):
- Daily drift and EWMA volatility calibrated from 5 years of historical prices
- Cholesky decomposition to preserve the historical correlation between copper and aluminium
- 10,000 independent simulations, each stepping one trading day at a time

## Business Interpretation
The wide spread by day 30 (~$315K from 5th to 95th percentile) shows that holding inventory is not a passive act — every day held is a day of price risk absorbed. If sitting on significant unrealised profit, the Monte Carlo quantifies the chance of that profit reversing before sale.

## Related Notes
- [[Risk Model Overview]]
- [[VaR and CVaR]]
- [[How to Use This for Business Decisions]]
