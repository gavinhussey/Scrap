import pandas as pd

from src import prices


def test_price_source_summary_marks_synthetic_and_proxy(monkeypatch, tmp_path):
    dates = pd.bdate_range("2026-01-01", periods=120)

    monkeypatch.setattr(prices, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(prices, "_download", lambda metal, years: None)
    monkeypatch.setattr(
        prices,
        "_synthetic",
        lambda metal, years: pd.Series(range(100, 220), index=dates, name=metal),
    )

    prices.fetch_all_prices()
    sources = prices.price_source_summary()

    assert sources["copper"] == "synthetic"
    assert sources["steel"] == "synthetic"
    assert sources["brass"] == "proxy:copper (synthetic)"
    assert sources["lead"] == "proxy:copper (synthetic)"


def test_synthetic_fallback_does_not_write_real_cache(monkeypatch, tmp_path):
    dates = pd.bdate_range("2026-01-01", periods=120)

    monkeypatch.setattr(prices, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(prices, "_download", lambda metal, years: None)
    monkeypatch.setattr(
        prices,
        "_synthetic",
        lambda metal, years: pd.Series(range(100, 220), index=dates, name=metal),
    )

    result = prices.fetch_prices("copper", force_refresh=True)

    assert len(result) == 120
    assert not (tmp_path / "copper_prices.csv").exists()
    assert not (tmp_path / "copper_prices.meta.json").exists()
    assert (tmp_path / "copper_synthetic_prices.csv").exists()
    assert prices.price_source_summary()["copper"] == "synthetic"


def test_stale_real_cache_is_used_before_synthetic(monkeypatch, tmp_path):
    old_dates = pd.bdate_range("2025-01-01", periods=120)
    real_cache = pd.Series(range(200, 320), index=old_dates, name="copper")
    real_cache.to_csv(tmp_path / "copper_prices.csv")

    monkeypatch.setattr(prices, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(prices, "_download", lambda metal, years: None)
    monkeypatch.setattr(
        prices,
        "_synthetic",
        lambda metal, years: pd.Series(range(100, 220), index=old_dates, name=metal),
    )

    result = prices.fetch_prices("copper", force_refresh=True)

    assert result.equals(real_cache)
    assert not (tmp_path / "copper_synthetic_prices.csv").exists()
    assert prices.price_source_summary()["copper"] == "stale-cache"
