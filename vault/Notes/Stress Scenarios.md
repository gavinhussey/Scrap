# Stress Scenarios

## What They Do
Apply an instantaneous % price shock to the current portfolio to show what a historical or hypothetical crisis would cost. Unlike VaR and Monte Carlo (which are probabilistic), stress scenarios are deterministic — they answer "what if X happened today?"

## Current Results (~$964K portfolio)
| Scenario | Copper Shock | Aluminium Shock | Total P&L | % Change |
|----------|-------------|-----------------|-----------|----------|
| 2008 Financial Crisis | -65% | -55% | -$594K | -61.6% |
| Demand Collapse (-40%) | -40% | -35% | -$369K | -38.3% |
| 2015 China Slowdown | -35% | -25% | -$304K | -31.6% |
| 2022 Rate Shock | -30% | -20% | -$256K | -26.6% |
| Supply Spike (+40%) | +40% | +35% | +$369K | +38.3% |

## Key Takeaway
A 2008-style crash would wipe out **61.6% of the portfolio value**. If this exceeds the company's equity base, they cannot afford to hold current inventory levels without hedging or additional capital.

## Adding or Changing Scenarios
Edit the `STRESS_SCENARIOS` dictionary in `src/config.py`:
```python
STRESS_SCENARIOS = {
    "My Scenario": {"copper": -0.25, "aluminium": -0.20},
}
```

## How to Use for Business Decisions
- **Position limits**: If the worst-case scenario loss would exceed equity, inventory is too high
- **Expansion planning**: Scale the worst-case loss proportionally to evaluate how much equity is needed at 2× or 3× current inventory
- **Hedging trigger**: If a 2008-style scenario is a real concern, use LME futures to hedge a portion of inventory

## Related Notes
- [[Risk Model Overview]]
- [[VaR and CVaR]]
- [[How to Use This for Business Decisions]]
