"""Focused tests for the cost-vs-cloud benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.cli import main_module
from openmed.eval.cost import (
    PriceCitationError,
    cost_vs_cloud_report,
    load_cloud_prices,
)


def _price_table() -> dict[str, object]:
    return {
        "hardware_cost_model": {
            "purchase_price_usd": 1_000.0,
            "amortization_hours": 1_000.0,
            "operating_cost_usd_per_hour": 0.0,
        },
        "prices": [
            {
                "id": "synthetic-cloud",
                "provider": "Synthetic Cloud",
                "service": "Synthetic medical text",
                "price_usd_per_1k_chars": 0.01,
                "source_url": "https://example.invalid/synthetic-pricing",
                "capture_date": "2026-08-08",
                "verify": True,
            }
        ],
    }


def _perf_report() -> dict[str, object]:
    return {
        "model_name": "synthetic-local-model",
        "device": "cpu",
        "docs_per_second": 10.0,
        "chars_per_doc": 1_000.0,
    }


def test_cost_report_computes_cloud_local_and_breakeven_math() -> None:
    report = cost_vs_cloud_report(
        _perf_report(),
        _price_table(),
        _price_table()["hardware_cost_model"],
    )

    comparison = report.comparisons[0]
    assert report.chars_per_second == pytest.approx(10_000.0)
    assert comparison.cloud_cost_per_million_chars_usd == pytest.approx(10.0)
    assert comparison.local_cost_per_million_chars_usd == pytest.approx(1 / 36)
    assert comparison.breakeven_volume_chars == pytest.approx(100_000_000.0)
    assert report["comparisons"][0]["price"]["source_url"].startswith("https://")


def test_cost_report_rejects_price_rows_without_citation() -> None:
    prices = _price_table()
    prices["prices"] = [
        {
            "id": "missing-citation",
            "provider": "Synthetic Cloud",
            "price_usd_per_1k_chars": 0.01,
            "verify": True,
        }
    ]

    with pytest.raises(PriceCitationError, match="source_url.*capture_date"):
        cost_vs_cloud_report(_perf_report(), prices, prices["hardware_cost_model"])


def test_committed_price_table_has_dated_cited_verified_rows() -> None:
    table_path = (
        Path(__file__).parents[3] / "openmed" / "eval" / "data" / "cloud_prices.json"
    )
    table = load_cloud_prices(table_path)

    assert isinstance(table, dict)
    rows = table["prices"]
    assert isinstance(rows, list)
    assert {row["provider"] for row in rows} == {"AWS", "Azure"}
    assert all(row["source_url"].startswith("https://") for row in rows)
    assert all(row["capture_date"] for row in rows)
    assert all(row["verify"] is True for row in rows)


def test_cost_cli_writes_json_and_markdown_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    perf_path = tmp_path / "perf.json"
    prices_path = tmp_path / "prices.json"
    output_dir = tmp_path / "reports"
    perf_path.write_text(json.dumps(_perf_report()), encoding="utf-8")
    prices_path.write_text(json.dumps(_price_table()), encoding="utf-8")

    result = main_module.main(
        [
            "benchmark",
            "cost",
            "--perf",
            str(perf_path),
            "--prices",
            str(prices_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result == 0
    assert "Cost comparison reports written" in capsys.readouterr().out
    json_report = output_dir / "cost.json"
    markdown_report = output_dir / "cost.md"
    assert json_report.is_file()
    assert markdown_report.is_file()
    payload = json.loads(json_report.read_text(encoding="utf-8"))
    assert payload["benchmark"] == "cost_vs_cloud"
    assert payload["comparisons"][0][
        "cloud_cost_per_million_chars_usd"
    ] == pytest.approx(10.0)
    assert "## Cloud comparison" in markdown_report.read_text(encoding="utf-8")
