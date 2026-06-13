# Risk Model Overview

## Purpose
A quantitative risk model built to help the scrapyard gauge price exposure on physical metal inventory (copper and aluminium) and support expansion decisions.

## What It Does
- Marks the physical inventory to market daily using live exchange prices adjusted for scrap grade discounts
- Calculates Value at Risk (VaR) and Expected Shortfall (CVaR) across three methods
- Runs 10,000 Monte Carlo simulations of portfolio value over a 30-day horizon
- Stress tests the portfolio against named historical crash scenarios
- Generates five charts saved to `output/charts/`

## How to Run
```bash
python run_risk_model.py              # standard run
python run_risk_model.py --refresh    # force re-download of price data
python run_risk_model.py --no-charts  # skip chart generation
```

## Project Structure
```
scrapyard_project/
├── run_risk_model.py         # entry point
├── requirements.txt
├── data/
│   └── sample_inventory.csv  # physical inventory — edit this with real data
├── src/
│   ├── config.py             # all tunable parameters
│   ├── prices.py             # fetch & cache metal prices from Yahoo Finance
│   ├── inventory.py          # mark-to-market with grade basis discounts
│   ├── var_model.py          # VaR and CVaR calculations
│   ├── monte_carlo.py        # correlated GBM simulation
│   ├── scenarios.py          # stress scenario engine
│   └── report.py             # console output and charts
├── cache/                    # auto-generated price cache (CSV)
└── output/charts/            # generated chart images
```

## Key Parameters (src/config.py)
| Parameter | Value | Description |
|-----------|-------|-------------|
| LOOKBACK_YEARS | 5 | Years of price history used |
| HOLDING_PERIOD_DAYS | 30 | VaR and Monte Carlo horizon |
| MONTE_CARLO_SIMULATIONS | 10,000 | Number of simulated paths |
| EWMA_LAMBDA | 0.94 | RiskMetrics decay factor for volatility |

## Related Notes
- [[Price Data]]
- [[Inventory & Grades]]
- [[VaR and CVaR]]
- [[Monte Carlo]]
- [[Stress Scenarios]]
- [[How to Use This for Business Decisions]]
