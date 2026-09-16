from __future__ import annotations

import json

import pytest

from openmed.eval import cost as cost_module
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
    assert "[captured 2026-08-21](<https://example.com/pricing>)" in (
        first.to_markdown()
    )


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
    assert prices["captured_at"] == "2026-08-21"
    aws = [row for row in report.comparisons if row.provider == "AWS"]
    assert [
        (
            row.minimum_monthly_characters,
            row.maximum_monthly_characters,
            row.price_per_1000_characters_usd,
        )
        for row in aws
    ] == [
        (0, 100_000_000, 0.1),
        (100_000_000, 200_000_000, 0.05),
        (200_000_000, None, 0.01),
    ]
    azure = [row for row in report.comparisons if row.provider == "Azure"]
    assert [
        (
            row.minimum_monthly_characters,
            row.maximum_monthly_characters,
            row.price_per_1000_characters_usd,
        )
        for row in azure
    ] == [
        (5_000_000, 500_000_000, 0.02),
        (500_000_000, 2_500_000_000, 0.015),
        (2_500_000_000, 10_000_000_000, 0.006),
        (10_000_000_000, None, 0.005),
    ]


def test_report_writes_json_and_markdown_without_fchmod(tmp_path, monkeypatch):
    monkeypatch.delattr(cost_module.os, "fchmod", raising=False)
    report = cost_vs_cloud_report(_perf_report(), _prices(), _hardware_cost_model())

    json_path = report.write_json(tmp_path / "cost.json")
    markdown_path = report.write_markdown(tmp_path / "cost.md")

    assert json.loads(json_path.read_text(encoding="utf-8"))["schema_version"] == 1
    assert markdown_path.read_text(encoding="utf-8").startswith(
        "# Cost vs cloud benchmark"
    )
    assert sorted(path.name for path in tmp_path.iterdir()) == ["cost.json", "cost.md"]


def test_missing_chars_per_document_fails_closed():
    perf = _perf_report()
    perf["metadata"] = {}

    with pytest.raises(TypeError, match="chars_per_document"):
        cost_vs_cloud_report(perf, _prices(), _hardware_cost_model())


def test_cost_report_ignores_raw_non_semantic_fields_and_hides_local_paths():
    perf = _perf_report()
    perf["model_name"] = "/Users/Ada/private-model"
    perf["raw_text"] = "Patient Ada, MRN 12345"
    hardware = _hardware_cost_model()
    hardware["owner"] = "Ada"
    prices = _prices()
    prices["notes"] = ["Ada"]

    first = cost_vs_cloud_report(perf, prices, hardware)
    perf["raw_text"] = "Patient Bob, MRN 99999"
    hardware["owner"] = "Bob"
    prices["notes"] = ["Bob"]
    second = cost_vs_cloud_report(perf, prices, hardware)

    assert first.input_fingerprint == second.input_fingerprint
    assert first.model_name.startswith("local-sha256-")
    assert "Users/Ada" not in first.to_json()
    assert "Patient Ada" not in first.to_json()


def test_cost_report_escapes_markdown_cells_and_link_destinations():
    prices = _prices()
    row = prices["prices"][0]
    row["provider"] = "Cloud|<script>"
    row["source_url"] = "https://example.com/pricing_(east)"

    markdown = cost_vs_cloud_report(
        _perf_report(), prices, _hardware_cost_model()
    ).to_markdown()

    assert "Cloud\\|&lt;script&gt;" in markdown
    assert "pricing_%28east%29" in markdown
    assert "<script>" not in markdown


def test_cost_report_rejects_unsafe_source_urls_and_future_snapshots():
    prices = _prices()
    prices["prices"][0]["source_url"] = "https://user:pass@example.com/pricing"
    with pytest.raises(ValueError, match="safe HTTPS URL"):
        cost_vs_cloud_report(_perf_report(), prices, _hardware_cost_model())

    prices = _prices()
    prices["prices"][0]["captured_at"] = "2999-01-01"
    with pytest.raises(ValueError, match="future"):
        cost_vs_cloud_report(_perf_report(), prices, _hardware_cost_model())


def test_cost_report_rejects_overlapping_or_extended_price_rows():
    prices = _prices()
    prices["prices"][0]["maximum_monthly_characters"] = 1_000
    overlapping = dict(prices["prices"][0])
    overlapping["tier"] = "overlap"
    overlapping["minimum_monthly_characters"] = 500
    overlapping["maximum_monthly_characters"] = None
    prices["prices"].append(overlapping)
    with pytest.raises(ValueError, match="overlapping"):
        cost_vs_cloud_report(_perf_report(), prices, _hardware_cost_model())

    prices = _prices()
    prices["prices"][0]["raw_text"] = "not part of schema"
    with pytest.raises(ValueError, match="invalid schema"):
        cost_vs_cloud_report(_perf_report(), prices, _hardware_cost_model())


def test_cost_input_loader_rejects_oversized_json_before_parsing(
    tmp_path,
    monkeypatch,
):
    input_path = tmp_path / "input.json"
    input_path.write_text('{"value": 123}', encoding="utf-8")
    monkeypatch.setattr(cost_module, "_MAX_INPUT_BYTES", 8)

    with pytest.raises(ValueError, match="size limit"):
        cost_module.load_cost_input(input_path, name="cost input")


def test_cost_report_rejects_unbounded_or_non_finite_numbers():
    perf = _perf_report()
    perf["docs_per_second"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        cost_vs_cloud_report(perf, _prices(), _hardware_cost_model())

    hardware = _hardware_cost_model()
    hardware["purchase_price_usd"] = 1_000_000_001
    with pytest.raises(ValueError, match="within the limit"):
        cost_vs_cloud_report(_perf_report(), _prices(), hardware)

    perf = _perf_report()
    perf["docs_per_second"] = 5e-324
    with pytest.raises(ValueError, match="derived"):
        cost_vs_cloud_report(perf, _prices(), _hardware_cost_model())


def test_cost_report_validates_indent_and_documents_marginal_tiers():
    report = cost_vs_cloud_report(_perf_report(), _prices(), _hardware_cost_model())

    with pytest.raises(ValueError, match="indent"):
        report.to_json(indent=100)
    markdown = report.to_markdown()
    assert "Marginal USD / 1M chars" in markdown
    assert "marginal monthly bands" in markdown
