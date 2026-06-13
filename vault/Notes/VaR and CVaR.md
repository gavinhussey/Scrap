# VaR and CVaR (Value at Risk & Expected Shortfall)

## What They Answer
- **VaR**: "What is the most we should expect to lose in a bad month?" (the threshold)
- **CVaR (Expected Shortfall)**: "If it does go badly, how bad does it get on average?" (the average of the worst outcomes)

CVaR is always higher than VaR. It answers what happens once you've crossed the VaR threshold.

## Current Results (95% confidence, 30-day horizon, ~$964K portfolio)
| Method | VaR | CVaR |
|--------|-----|------|
| Parametric | ~$129K | ~$162K |
| EWMA | ~$129K | ~$162K |
| Historical | ~$90K | ~$129K |

## Three Methods Explained

### Parametric
Assumes returns follow a perfect bell curve (normal distribution). Clean and fast but underestimates tail risk — real metal prices have fatter tails than a normal distribution suggests.

### EWMA (Exponentially Weighted Moving Average)
Same bell curve assumption but weights recent volatility more heavily than older data using a decay factor (λ = 0.94, the RiskMetrics standard). When EWMA matches Parametric, current volatility is in line with the 5-year average.

### Historical Simulation
Uses the actual distribution of past returns with no assumptions about shape. Lower VaR than Parametric here means real historical data shows fewer extreme monthly losses than the normal distribution predicts.

## Which to Use for Business Decisions
Use the **Parametric figure (~$129K VaR)** for setting cash reserves and position limits. It is more conservative and builds in a buffer for market conditions worse than the last 5 years.

## Practical Application
- **Cash reserve**: Hold at least the 95% VaR amount (~$129K) in liquid cash to absorb a bad month without missing obligations
- **Position limits**: Scale VaR proportionally to inventory size — if doubling inventory, expect to need ~$258K in reserve
- **Expansion**: If a worst-case scenario would exceed equity, the business cannot afford to scale without hedging or raising capital

## Diversification Benefit
The model calculates how much VaR is reduced by holding both copper and aluminium instead of just one metal. At 95%:
- Sum of individual VaRs: ~$148K
- Portfolio VaR: ~$129K
- **Diversification benefit: ~$19K** — the two metals are correlated but not perfectly, so holding both reduces total risk slightly

## Related Notes
- [[Risk Model Overview]]
- [[Monte Carlo]]
- [[How to Use This for Business Decisions]]
