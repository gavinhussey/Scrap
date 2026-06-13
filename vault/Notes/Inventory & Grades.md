# Inventory & Grades

## Inventory File
Physical inventory is stored in `data/sample_inventory.csv`. Edit this file directly to add or update positions. The model re-reads it on every run — no restart needed.

## CSV Format
```
metal,quantity_tonnes,purchase_price_per_tonne,purchase_date,location,grade
copper,15.5,8500,2026-05-15,Bay A,No.1 Bare Bright
aluminium,45.0,2400,2026-05-10,Bay B,Clean Sheet
```

| Column | Required | Notes |
|--------|----------|-------|
| metal | Yes | `copper` or `aluminium` (lowercase) |
| quantity_tonnes | Yes | Weight in metric tonnes |
| purchase_price_per_tonne | Yes | Cost basis — used for unrealised P&L |
| purchase_date | Yes | Used to calculate days held |
| location | No | Informational only |
| grade | Yes | Must match grade names exactly (see below) |

## Grade Basis Discounts
Grade determines what fraction of the exchange spot price the scrap actually sells for.

### Copper
| Grade | Basis | Sells At |
|-------|-------|----------|
| No.1 Bare Bright | 0.92 | 92% of COMEX spot |
| No.2 Copper | 0.82 | 82% of COMEX spot |
| default (fallback) | 0.85 | 85% of COMEX spot |

### Aluminium
| Grade | Basis | Sells At |
|-------|-------|----------|
| Clean Sheet | 0.82 | 82% of LME spot |
| UBC | 0.70 | 70% of LME spot |
| Mixed/Cast | 0.60 | 60% of LME spot |
| default (fallback) | 0.75 | 75% of LME spot |

## Mark-to-Market Formula
```
Scrap Price = Spot Price × Basis Factor
MTM Value   = Scrap Price × Quantity (tonnes)
Unrealised P&L = MTM Value - (Purchase Price × Quantity)
```

## Important: Grade Name Matching
Grade names must match exactly (case and spacing). A typo silently falls back to the `default` basis. For example:
- `No.1 Bare Bright` ✓
- `No. 1 Bare Bright` ✗ → falls back to 0.85 instead of 0.92

To add new grades or change discounts, edit the `grade_basis` dictionaries in `src/config.py`.

## Limitation
The basis percentages are currently hardcoded estimates. For accuracy, log actual achieved sell prices against spot on each transaction and calculate real basis from trade history.

## Related Notes
- [[Risk Model Overview]]
- [[Price Data]]
- [[How to Use This for Business Decisions]]
