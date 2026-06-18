from src.data_loader import find_inbound_csvs
from src.outbound_loader import find_outbound_csvs


# when a combined file exists the loaders should pick it over the per yard ones
def test_inbound_loader_prefers_combined_export(tmp_path):
    combined = tmp_path / "2026 ytd combined inbound.csv"
    yard = tmp_path / "2026 ytd milton inbound.csv"
    combined.write_text("", encoding="utf-8")
    yard.write_text("", encoding="utf-8")

    assert find_inbound_csvs(tmp_path) == [combined]


def test_outbound_loader_prefers_combined_export(tmp_path):
    combined = tmp_path / "2026 ytd combined outbound.csv"
    yard = tmp_path / "2026 ytd milton outbound.csv"
    combined.write_text("", encoding="utf-8")
    yard.write_text("", encoding="utf-8")

    assert find_outbound_csvs(tmp_path) == [combined]


# with no combined file they fall back to the individual yard exports
def test_loaders_fall_back_to_yard_exports_without_combined(tmp_path):
    inbound = tmp_path / "2026 ytd milton inbound.csv"
    outbound = tmp_path / "2026 ytd milton outbound.csv"
    inbound.write_text("", encoding="utf-8")
    outbound.write_text("", encoding="utf-8")

    assert find_inbound_csvs(tmp_path) == [inbound]
    assert find_outbound_csvs(tmp_path) == [outbound]
