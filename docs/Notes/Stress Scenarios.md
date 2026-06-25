# Stress Scenarios

## What They Do
Stress scenarios apply deterministic price shocks to current risk exposure. They answer: "What would this portfolio lose if this shock happened now?"

## Current Scenario Types
- Historical-style commodity crashes
- Demand collapse
- Supercycle rally
- Basis compression stress

## Important Detail
Stress scenarios use current risk exposure, not gross book value. Unpriced inventory is included through the conservative risk proxy.

## Limitations
- Scenario shocks are hard-coded assumptions in `src/config.py`.
- They are not full macroeconomic simulations.
- They do not include credit, operational, environmental, or liquidity failures except through configured realization haircuts.

## Maintenance
Review scenario shocks periodically and document the source, rationale, confidence, and review date in `ASSUMPTIONS`.
