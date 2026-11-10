"""Tests for per-model evaluation scorecards."""

from __future__ import annotations

import json

from openmed.eval import (
    ModelScorecard,
    render_model_scorecard,
    write_model_scorecard,
    write_model_scorecard_json,
)
from openmed.eval.report import BenchmarkReport

MIB = 1024 * 1024


def _manifest_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "repo_id": "OpenMed/pii-tiny",
        "family": "PII",
        "tier": "Tiny",
        "formats": ["cpu", "mlx-fp"],
    }
    row.update(overrides)
    return row


def _report(
    *,
    suite: str,
    device: str,
    fixture_count: int,
    recall: tuple[int, int],
    leakage: tuple[int, int],
    p50_ms: float,
    p95_ms: float,
    peak_rss_mib: float,
    model_size_mib: float,
) -> BenchmarkReport:
    recall_numerator, recall_denominator = recall
    leakage_numerator, leakage_denominator = leakage
    return BenchmarkReport(
        suite=suite,
        model_name="OpenMed/pii-tiny",
        device=device,
        fixture_count=fixture_count,
        generated_at=f"2026-06-2{fixture_count}T00:00:00Z",
        metadata={
            "raw_fixture": "Patient John Doe has SSN 123-45-6789.",
        },
        metrics={
            "character_recall": {
                "rate": recall_numerator / recall_denominator,
                "numerator": recall_numerator,
                "denominator": recall_denominator,
            },
            "leakage": {
                "overall": leakage_numerator / leakage_denominator,
                "leaked_chars": leakage_numerator,
                "total_chars": leakage_denominator,
            },
            "latency": {
                "p50_ms": p50_ms,
                "p95_ms": p95_ms,
                "count": fixture_count,
            },
            "resources": {
                "peak_rss_bytes": int(peak_rss_mib * MIB),
                "peak_rss_mib": peak_rss_mib,
                "model_size_bytes": int(model_size_mib * MIB),
                "model_size_mib": model_size_mib,
            },
            "unsafe_examples": ["Patient John Doe", "123-45-6789"],
        },
    )


def test_model_scorecard_renders_multiple_suite_reports_by_device_tier() -> None:
    reports = [
        _report(
            suite="golden",
            device="cpu",
            fixture_count=2,
            recall=(98, 100),
            leakage=(0, 100),
            p50_ms=11,
            p95_ms=20,
            peak_rss_mib=128,
            model_size_mib=64,
        ),
        _report(
            suite="multilingual",
            device="mlx-fp",
            fixture_count=3,
            recall=(270, 300),
            leakage=(6, 300),
            p50_ms=8,
            p95_ms=16,
            peak_rss_mib=96,
            model_size_mib=61.5,
        ),
    ]

    markdown = render_model_scorecard(reports, manifest_rows=[_manifest_row()])

    assert markdown == render_model_scorecard(reports, manifest_rows=[_manifest_row()])
    assert "| Family | `PII` |" in markdown
    assert "| Model Tier | `Tiny` |" in markdown
    assert "| Tier RAM Limit MB | 350 |" in markdown
    assert "| Tier Latency Budget p50/p95 ms | `60 / 150` |" in markdown
    assert (
        "| `cpu` | yes | 1 | 2 | 98.00% | n/a | 0.00% | n/a | n/a | n/a | "
        "n/a | 11 / 20 | 128 | 64 |"
    ) in markdown
    assert (
        "| `mlx-fp` | yes | 1 | 3 | 90.00% | n/a | 2.00% | n/a | n/a | n/a | "
        "n/a | 8 / 16 | 96 | 61.5 |"
    ) in markdown
    assert (
        "| `mlx-8bit` | no | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
    ) in markdown
    assert (
        "| `coreml` | no | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
    ) in markdown


def test_model_scorecard_json_is_byte_stable_and_machine_readable() -> None:
    report = _report(
        suite="golden",
        device="cpu",
        fixture_count=2,
        recall=(98, 100),
        leakage=(0, 100),
        p50_ms=11,
        p95_ms=20,
        peak_rss_mib=128,
        model_size_mib=64,
    )
    scorecard = ModelScorecard.from_reports([report], manifest_rows=[_manifest_row()])

    first = scorecard.to_json()
    second = scorecard.to_json()

    assert first == second
    payload = json.loads(first)
    assert payload["family"] == "PII"
    assert payload["model_tier"] == "Tiny"
    assert payload["device_tiers"][0]["device_tier"] == "cpu"
    assert payload["device_tiers"][0]["recall"] == 0.98
    assert payload["device_tiers"][0]["critical_finding_recall"] is None
    assert payload["device_tiers"][0]["leakage_rate"] == 0.0


def test_model_scorecard_missing_tier_and_metrics_use_placeholders() -> None:
    report = BenchmarkReport(
        suite="empty",
        model_name="OpenMed/pii-tiny",
        device="cpu",
        fixture_count=1,
        metrics={},
    )

    markdown = render_model_scorecard(
        [report],
        manifest_rows=[_manifest_row(tier=None, formats=[])],
    )
    payload = ModelScorecard.from_reports(
        [report],
        manifest_rows=[_manifest_row(tier=None, formats=[])],
    ).to_dict()

    assert "| Model Tier | `n/a` |" in markdown
    assert "| Tier RAM Limit MB | n/a |" in markdown
    assert (
        "| `cpu` | yes | 1 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
    ) in markdown
    assert payload["model_tier"] is None
    assert payload["device_tiers"][0]["recall"] is None


def test_model_scorecard_omits_fixture_text_from_markdown_and_json() -> None:
    report = _report(
        suite="golden",
        device="cpu",
        fixture_count=2,
        recall=(98, 100),
        leakage=(0, 100),
        p50_ms=11,
        p95_ms=20,
        peak_rss_mib=128,
        model_size_mib=64,
    )
    scorecard = ModelScorecard.from_reports([report], manifest_rows=[_manifest_row()])

    combined_output = scorecard.to_markdown() + scorecard.to_json()

    assert "John Doe" not in combined_output
    assert "123-45-6789" not in combined_output


def test_model_scorecard_write_helpers(tmp_path) -> None:
    report = _report(
        suite="golden",
        device="cpu",
        fixture_count=2,
        recall=(98, 100),
        leakage=(0, 100),
        p50_ms=11,
        p95_ms=20,
        peak_rss_mib=128,
        model_size_mib=64,
    )

    markdown_path = write_model_scorecard(
        [report],
        tmp_path,
        manifest_rows=[_manifest_row()],
    )
    json_path = write_model_scorecard_json(
        [report],
        tmp_path,
        manifest_rows=[_manifest_row()],
    )

    assert markdown_path.name == "openmed-pii-tiny.md"
    assert json_path.name == "openmed-pii-tiny.json"
    assert markdown_path.read_text(encoding="utf-8").startswith("# Model Scorecard")
    assert json.loads(json_path.read_text(encoding="utf-8"))["model_name"] == (
        "OpenMed/pii-tiny"
    )
