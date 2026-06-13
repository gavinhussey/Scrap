# How to Use This for Business Decisions

## Cash Reserve
Use the **95% Parametric VaR (~$129K)** as the minimum liquid cash buffer to hold against the current inventory. If they cannot absorb this loss, they are over-exposed.

Scale linearly with inventory size:
- 2× inventory → hold ~$258K in reserve
- 3× inventory → hold ~$387K in reserve

## Sell Triggers
Use unrealised P&L from the MTM report to set rules:
- When unrealised P&L reaches a target (e.g. +20%) → sell
- When unrealised P&L falls below a floor (e.g. -5%) → stop buying that metal until conditions improve

The Monte Carlo shows there is a ~47% chance of the portfolio losing value over 30 days — sitting on profit is not risk-free.

## Expansion Decisions
The key question: would a worst-case stress scenario wipe out more than the company's equity?

1. Take the worst-case stress loss (2008 = -$594K at current size)
2. Compare to total equity
3. If loss > equity: cannot expand without hedging or raising capital first
4. If loss < equity: expansion is feasible — recalculate at the new inventory size

## Position Limits by Metal
The model shows copper (~$634K MTM) carries far more risk than aluminium (~$330K MTM) because:
- Copper spot price is higher
- Copper has higher historical volatility
- Copper represents ~66% of total portfolio value

Consider setting a maximum % allocation to copper to avoid over-concentration.

## Holding Period Decisions
The Monte Carlo cone widens significantly by day 30. Practical rule: the longer inventory sits unsold, the more VaR accumulates. If the business regularly holds metal for 60–90 days, re-run the model with `HOLDING_PERIOD_DAYS = 60` or `90` in `src/config.py` to see the true risk exposure.

## Hedging
If stress scenario losses are unacceptable, the business can hedge using LME/COMEX futures:
- Sell futures contracts equivalent to the inventory weight
- This locks in a price and removes directional price risk
- The remaining risk is **basis risk** (the gap between scrap prices and futures prices)
- The model already tracks basis by grade — this data can be used to size hedges correctly

## Grade Mix Management
Higher grades (No.1 Bare Bright copper at 92% basis) carry less basis risk than lower grades (Mixed/Cast aluminium at 60% basis). When acquiring new inventory, prioritise higher-grade material to improve the portfolio's effective sell price and reduce basis drag.

## Monitoring
Run `python run_risk_model.py` each morning to get:
- Updated MTM values with overnight price moves
- Fresh VaR calculations
- Current unrealised P&L per line item

## Related Notes
- [[Risk Model Overview]]
- [[VaR and CVaR]]
- [[Monte Carlo]]
- [[Stress Scenarios]]
- [[Inventory & Grades]]
