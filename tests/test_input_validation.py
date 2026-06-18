from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import load_inbound
from src.input_validation import file_provenance, require_nonnegative, validate_greenspark_snapshot
from src.outbound_loader import load_outbound


# both loaders should complain loudly if a required column is missing
def test_inbound_loader_reports_missing_required_columns(tmp_path):
    path = tmp_path / "bad inbound.csv"
    path.write_text("Commodity Name,Cost\nSTEEL,10\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required column"):
        load_inbound([path])


def test_outbound_loader_reports_missing_required_columns(tmp_path):
    path = tmp_path / "bad outbound.csv"
    path.write_text("Commodity Name,Net Weight\nSTEEL,10\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required column"):
        load_outbound([path])


# a negative value where one shouldn't exist is a hard error
def test_require_nonnegative_rejects_negative_values():
    with pytest.raises(ValueError, match="negative value"):
        require_nonnegative(pd.Series([1, -2, 3]), "test series")


# one messy snapshot should trip every soft warning at once
def test_greenspark_snapshot_validation_warns_on_stale_duplicates_unmapped_and_outliers():
    raw = pd.DataFrame([
        {
            "Location": "Milton",
            "Material Code": "A1",
            "Material Name": "Known",
            "Commodity Name": "STEEL",
            "Total Net Weight": 600_000,
            "Total Cost": 10_000,
        },
        {
            "Location": "Milton",
            "Material Code": "A1",
            "Material Name": "Duplicate",
            "Commodity Name": "MYSTERY",
            "Total Net Weight": 100,
            "Total Cost": 0,
        },
        {
            "Location": "Merrillville",
            "Material Code": "B2",
            "Material Name": "High Cost",
            "Commodity Name": "COPPER TIER 1",
            "Total Net Weight": 10,
            "Total Cost": 500,
        },
    ])

    warnings = validate_greenspark_snapshot(
        raw,
        Path("combined inventory 20250101.csv"),
        {"STEEL": "steel", "COPPER TIER 1": "copper"},
        today=pd.Timestamp("2025-01-10"),
        stale_business_days=3,
    )

    assert any("business days old" in w for w in warnings)
    assert any("Duplicate Location + Material Code" in w for w in warnings)
    assert any("Unmapped commodity" in w and "MYSTERY" in w for w in warnings)
    assert any("zero total cost" in w for w in warnings)
    assert any("net weight" in w for w in warnings)
    assert any("cost" in w and "/lb" in w for w in warnings)


# provenance should capture the name, row count, date and a full hash
def test_file_provenance_includes_rows_snapshot_date_and_hash(tmp_path):
    path = tmp_path / "combined inventory 20260612.csv"
    path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    provenance = file_provenance(path)

    assert provenance["filename"] == "combined inventory 20260612.csv"
    assert provenance["rows"] == 2
    assert provenance["snapshot_date"] == "2026-06-12"
    assert len(str(provenance["sha256"])) == 64
