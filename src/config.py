"""Configuration: file paths, metal/grade mappings, and risk parameters."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"
OUTPUT_DIR = BASE_DIR / "output"
CHARTS_DIR = OUTPUT_DIR / "charts"

METALS: dict = {
    "copper": {
        "ticker": "HG=F",
        "usd_per_tonne_multiplier": 2204.62,
        "grade_basis": {
            "COPPER TIER 1":    0.625,
            "COPPER TIER 2":    0.410,
            "COPPER TIER 3":    0.035,
            "No.1 Bare Bright": 0.92,
            "No.2 Copper":      0.82,
            "default":          0.40,
        },
    },
    "aluminium": {
        "ticker": "ALI=F",
        "usd_per_tonne_multiplier": 1.0,
        "grade_basis": {
            "ALUMINUM":    0.34,
            "Clean Sheet": 0.82,
            "UBC":         0.70,
            "Mixed/Cast":  0.60,
            "default":     0.34,
        },
    },
    "steel": {
        "ticker": "HRC=F",
        "usd_per_tonne_multiplier": 1.10231,
        "grade_basis": {
            "STEEL":   0.17,
            "default": 0.17,
        },
    },
    "stainless": {
        # NI=F (nickel) is delisted on Yahoo, so the old config fell back to a
        # SYNTHETIC, independent random walk -> stainless showed ~0 correlation to
        # every real metal and injected fake diversification. Proxy to copper for
        # real base-metals co-movement; vol_multiplier lifts it to nickel-like vol
        # (nickel ~1.4x copper historically).
        "price_proxy": "copper",
        "vol_multiplier": 1.15,
        "grade_basis": {
            "STAINLESS STEEL TIER 1": 0.035,
            "STAINLESS STEEL TIER 2": 0.096,
            "default": 0.055,
        },
    },
    "brass": {
        "price_proxy": "copper",
        "grade_basis": {
            "BRASS TIER 1": 0.44,
            "BRASS TIER 2": 0.29,
            "default":      0.36,
        },
    },
    # Lead/zinc have no free live future; proxy to copper so they carry base-metals
    # risk instead of being dropped (was ~1.5% of the book at zero modelled risk).
    "lead": {
        "price_proxy": "copper",
        "grade_basis": {"default": 0.45},
    },
    "zinc": {
        "price_proxy": "copper",
        "grade_basis": {"default": 0.50},
    },
}

# Basis-risk overlay: scrap prices are more volatile than the underlying futures
# (the scrap/futures basis widens in selloffs). Six months of mix-varying tickets
# can't estimate this cleanly, so apply a transparent, conservative multiplier to
# every metal's flat-price vol used in VaR/MC. Documented assumption, not measured.
SCRAP_BASIS_VOL_MULT = 1.25

# VaR convention: simulate with zero drift rather than embedding the 5y realized
# (bull-market) mean, which optimistically lifted the loss distribution.
MC_ZERO_DRIFT = True

COMMODITY_TO_METAL: dict[str, str | None] = {
    "STEEL":                  "steel",
    "ALUMINUM":               "aluminium",
    "COPPER TIER 1":          "copper",
    "COPPER TIER 2":          "copper",
    "COPPER TIER 3":          "copper",
    "BRASS TIER 1":           "brass",
    "BRASS TIER 2":           "brass",
    "STAINLESS STEEL TIER 1": "stainless",
    "STAINLESS STEEL TIER 2": "stainless",
    "LEAD":                   "lead",
    "ZINC":                   "zinc",
    "OTHER":                  None,
    "ZWASTE":                 None,
}

LOOKBACK_YEARS = 5
HOLDING_PERIOD_DAYS = 30
VAR_CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]
MONTE_CARLO_SIMULATIONS = 10_000
EWMA_LAMBDA = 0.94

STRESS_SCENARIOS: dict[str, dict[str, float]] = {
    "2008 Financial Crisis": {
        "copper":    -0.65,
        "aluminium": -0.55,
        "steel":     -0.50,
        "stainless": -0.55,
        "brass":     -0.58,
        "lead":      -0.60,
        "zinc":      -0.55,
    },
    "2015 China Slowdown": {
        "copper":    -0.35,
        "aluminium": -0.25,
        "steel":     -0.40,
        "stainless": -0.30,
        "brass":     -0.32,
        "lead":      -0.30,
        "zinc":      -0.28,
    },
    "2022 Rate Shock": {
        "copper":    -0.30,
        "aluminium": -0.20,
        "steel":     -0.25,
        "stainless": -0.30,
        "brass":     -0.28,
        "lead":      -0.25,
        "zinc":      -0.25,
    },
    "Demand Collapse (-40%)": {
        "copper":    -0.40,
        "aluminium": -0.35,
        "steel":     -0.40,
        "stainless": -0.40,
        "brass":     -0.40,
        "lead":      -0.40,
        "zinc":      -0.40,
    },
    "2021 Supercycle Rally": {
        "copper":    +0.55,
        "aluminium": +0.40,
        "steel":     +0.70,
        "stainless": +0.35,
        "brass":     +0.50,
        "lead":      +0.45,
        "zinc":      +0.50,
    },
}
