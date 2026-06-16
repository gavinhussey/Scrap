"""Configuration: file paths, metal/grade mappings, and risk parameters."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DAILY_INPUT_DIR = BASE_DIR / "daily_inputs"
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
    "lead": {
        "price_proxy": "copper",
        "grade_basis": {"default": 0.45},
    },
    "zinc": {
        "price_proxy": "copper",
        "grade_basis": {"default": 0.50},
    },
}

SCRAP_BASIS_VOL_MULT = 1.25

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

NET_REALIZABLE_HAIRCUTS: dict[str, float] = {
    "copper": 0.04,
    "aluminium": 0.06,
    "steel": 0.08,
    "stainless": 0.07,
    "brass": 0.05,
    "lead": 0.12,
    "zinc": 0.10,
}

LIQUIDATION_HAIRCUTS: dict[str, float] = {
    "copper": 0.10,
    "aluminium": 0.15,
    "steel": 0.20,
    "stainless": 0.18,
    "brass": 0.12,
    "lead": 0.30,
    "zinc": 0.25,
}

BASIS_STRESS_POINTS = [0.02, 0.05, 0.10]

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

ASSUMPTIONS: dict[str, dict[str, str]] = {
    "scrap_basis_vol_multiplier": {
        "value": str(SCRAP_BASIS_VOL_MULT),
        "source": "management/model overlay",
        "confidence": "low",
        "last_reviewed": "2026-06-16",
        "rationale": "Scrap prices can move more sharply than exchange futures when basis widens in selloffs.",
    },
    "net_realizable_haircuts": {
        "value": ", ".join(f"{k}={v:.0%}" for k, v in NET_REALIZABLE_HAIRCUTS.items()),
        "source": "management estimate",
        "confidence": "medium-low",
        "last_reviewed": "2026-06-16",
        "rationale": "Approximates freight, handling, shrink, bid/ask, and normal sale friction.",
    },
    "liquidation_haircuts": {
        "value": ", ".join(f"{k}={v:.0%}" for k, v in LIQUIDATION_HAIRCUTS.items()),
        "source": "management estimate",
        "confidence": "medium-low",
        "last_reviewed": "2026-06-16",
        "rationale": "Approximates forced-sale discount by metal.",
    },
    "proxy_metals": {
        "value": ", ".join(
            f"{metal}->{cfg['price_proxy']}" for metal, cfg in METALS.items() if "price_proxy" in cfg
        ),
        "source": "free price-data availability",
        "confidence": "low",
        "last_reviewed": "2026-06-16",
        "rationale": "Used where no reliable free exchange ticker is available for the scrap category.",
    },
    "stress_scenarios": {
        "value": ", ".join(STRESS_SCENARIOS.keys()),
        "source": "model-defined historical/hypothetical shocks",
        "confidence": "medium-low",
        "last_reviewed": "2026-06-16",
        "rationale": "Applies deterministic commodity shocks to current risk exposure.",
    },
    "unpriced_inventory_risk_proxy": {
        "value": "max(net realizable book value, futures-basis proxy value)",
        "source": "conservative model policy",
        "confidence": "medium",
        "last_reviewed": "2026-06-16",
        "rationale": "Unpriced rows are held at book for MTM but still included in risk exposure using the higher conservative risk value.",
    },
}
