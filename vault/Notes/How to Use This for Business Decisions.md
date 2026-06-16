# How to Use This for Business Decisions

## First Check Model Quality
Before using VaR or Monte Carlo, check the model-quality warnings:
- synthetic price series
- proxied metals
- unpriced inventory
- stale or invalid input files

If warnings are material, treat VaR as directional rather than decision-grade.

## Cash Reserve
Use the 95% VaR as a starting reserve estimate, then add management judgment for operational risks not modeled here.

## Position Limits
Use risk exposure and stress losses to set limits by metal and by valuation quality. Unpriced inventory should have tighter review limits because its MTM is less reliable.

## Sell Triggers
Use net unrealized P&L, aging, concentration, and model quality together. A high MTM value with low price coverage should not be treated the same as a high MTM value supported by recent realized sales.

## Expansion Decisions
Scale VaR and stress losses with projected inventory, but do not assume perfect linearity if grade mix, liquidity, or price coverage changes.

## Hedging
For hedge sizing, use risk exposure and sensitivity by metal. Proxy metals and basis risk mean a futures hedge may reduce directional exposure without eliminating scrap spread risk.

## Governance
Review assumptions in `src/config.py` before major decisions. Low-confidence assumptions should be challenged, calibrated, or stress-tested.
