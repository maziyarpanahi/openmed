from __future__ import annotations

import json

import pytest

from openmed.eval.cost import cost_vs_cloud_report, load_cloud_prices


def _perf_report():
    return {
        "model_name": "synthetic-model",
        "device": "cpu",
        "docs_per_second": 10.0,
        "metadata": {"chars_per_document": 1000},
    }


def _hardware_cost_model():
    return {
        "purchase_price_usd": 1000.0,
        "useful_life_hours": 10000.0,
        "power_watts": 50.0,
        "electricity_usd_per_kwh": 0.2,
    }


def _prices():
    return {
        "schema_version": 1,
        "currency": "USD",
        "prices": [
            {
                "provider": "Example Cloud",
                "service": "Clinical text",
                "region": "test",
                "tier": "paid",
                "minimum_monthly_characters": 0,
                "maximum_monthly_characters": None,
                "price_per_1000_characters_usd": 0.1,
                "source_url": "https://example.com/pricing",
                "captured_at": "2026-08-21",
                "source_effective_at": "2026-01-01",
                "verify": True,
            }
        ],
    }


def test_cost_report_computes_local_cloud_and_breakeven_math():
    report = cost_vs_cloud_report(
        _perf_report(),
        _prices(),
        _hardware_cost_model(),
    )

    assert report.chars_per_second == 10000.0
    assert report.amortized_local_cost_per_hour_usd == pytest.approx(0.11)
    assert report.local_cost_per_million_characters_usd == pytest.approx(0.003055555556)
    row = report.comparisons[0]
    assert row.cloud_cost_per_million_characters_usd == 100.0
    assert row.savings_per_million_characters_usd == pytest.approx(99.996944444444)
    assert row.breakeven_characters == 10000028


def test_breakeven_beyond_hardware_useful_life_is_never():
    hardware = _hardware_cost_model()
    hardware["useful_life_hours"] = 1.0
    prices = _prices()
    prices["prices"][0]["price_per_1000_characters_usd"] = 0.000001

    report = cost_vs_cloud_report(_perf_report(), prices, hardware)

    assert report.comparisons[0].breakeven_characters is None
    assert "| never |" in report.to_markdown()


def test_cost_report_is_deterministic_and_aggregate_only():
    first = cost_vs_cloud_report(_perf_report(), _prices(), _hardware_cost_model())
    second = cost_vs_cloud_report(_perf_report(), _prices(), _hardware_cost_model())

    assert first.to_json() == second.to_json()
    assert first.input_fingerprint.startswith("sha256:")
    assert "synthetic-model" in first.to_markdown()
    assert "[captured 2026-08-21](https://example.com/pricing)" in first.to_markdown()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("source_url", None, "source_url"),
        ("captured_at", None, "captured_at"),
        ("verify", False, "verify"),
    ],
)
def test_price_rows_require_citation_date_and_verify_marker(field, value, match):
    prices = _prices()
    prices["prices"][0][field] = value

    with pytest.raises((TypeError, ValueError), match=match):
        cost_vs_cloud_report(_perf_report(), prices, _hardware_cost_model())


def test_bundled_cloud_price_table_is_cited_and_loadable():
    prices = load_cloud_prices()
    report = cost_vs_cloud_report(
        _perf_report(),
        prices,
        _hardware_cost_model(),
    )

    assert len(report.comparisons) == 7
    assert {row.provider for row in report.comparisons} == {"AWS", "Azure"}
    assert all(row.source_url.startswith("https://") for row in report.comparisons)


def test_report_writes_json_and_markdown(tmp_path):
    report = cost_vs_cloud_report(_perf_report(), _prices(), _hardware_cost_model())

    json_path = report.write_json(tmp_path / "cost.json")
    markdown_path = report.write_markdown(tmp_path / "cost.md")

    assert json.loads(json_path.read_text(encoding="utf-8"))["schema_version"] == 1
    assert markdown_path.read_text(encoding="utf-8").startswith(
        "# Cost vs cloud benchmark"
    )


def test_missing_chars_per_document_fails_closed():
    perf = _perf_report()
    perf["metadata"] = {}

    with pytest.raises(TypeError, match="chars_per_document"):
        cost_vs_cloud_report(perf, _prices(), _hardware_cost_model())
