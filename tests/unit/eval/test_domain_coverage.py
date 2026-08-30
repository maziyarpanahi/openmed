"""Focused tests for the synthetic clinical-domain coverage gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.cli import main as cli_main
from openmed.eval.domain_coverage import (
    CLINICAL_DOMAIN_COVERAGE,
    CLINICAL_DOMAIN_FIXTURE_NAMES,
    assert_domain_coverage_gate,
    domain_coverage_metadata,
    run_domain_coverage,
)
from openmed.eval.suites import (
    load_suite_fixtures,
    suite_metadata,
    validate_suite_name,
)


def _write_fixture(path: Path, *, label: str = "Problem") -> None:
    path.write_text(
        json.dumps(
            {
                "text": "Synthetic token.",
                "entities": [{"label": label, "start": 10, "end": 15, "text": "token"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_default_coverage_gate_passes_with_non_empty_per_label_spans() -> None:
    report = assert_domain_coverage_gate()

    assert report.suite == CLINICAL_DOMAIN_COVERAGE
    assert report.passed is True
    assert tuple(domain.domain for domain in report.per_domain) == tuple(
        sorted(CLINICAL_DOMAIN_FIXTURE_NAMES)
    )
    assert report.missing_fixtures == ()
    assert report.orphan_labels == ()
    assert report.missing_labels == ()
    assert all(
        coverage.span_count > 0
        for domain in report.per_domain
        for coverage in domain.per_label
    )


def test_missing_fixture_fails_with_domain_only_evidence(tmp_path: Path) -> None:
    report = run_domain_coverage(
        label_map={"missing_domain": ["Problem"]},
        fixture_dir=tmp_path,
        domains=("missing_domain",),
    )

    assert report.passed is False
    assert report.missing_fixtures == ("missing_domain",)
    assert "Synthetic" not in report.to_json()
    with pytest.raises(AssertionError, match="missing fixtures: missing_domain"):
        assert_domain_coverage_gate(
            label_map={"missing_domain": ["Problem"]},
            fixture_dir=tmp_path,
            domains=("missing_domain",),
        )


def test_orphan_label_fails_and_report_never_contains_fixture_text(
    tmp_path: Path,
) -> None:
    _write_fixture(tmp_path / "orphan_domain.jsonl", label="NotInCatalog")

    report = run_domain_coverage(
        label_map={"orphan_domain": ["Problem"]},
        fixture_dir=tmp_path,
        domains=("orphan_domain",),
    )

    assert report.passed is False
    assert [(issue.label, issue.reason) for issue in report.orphan_labels] == [
        ("NotInCatalog", "label_not_in_catalog")
    ]
    serialized = report.to_json()
    assert "Synthetic token" not in serialized
    assert '"start": 10' in serialized
    assert "NotInCatalog" in serialized


def test_suite_registry_and_metadata_are_discoverable() -> None:
    assert validate_suite_name(CLINICAL_DOMAIN_COVERAGE) == CLINICAL_DOMAIN_COVERAGE
    assert suite_metadata(CLINICAL_DOMAIN_COVERAGE) == domain_coverage_metadata()
    with pytest.raises(ValueError, match="aggregate gate"):
        load_suite_fixtures(CLINICAL_DOMAIN_COVERAGE)


def test_benchmark_cli_writes_machine_readable_coverage_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "coverage.json"

    result = cli_main(
        [
            "benchmark",
            "domain-coverage",
            "--domain",
            "anesthesia",
            "--json",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["suite"] == CLINICAL_DOMAIN_COVERAGE
    assert payload["passed"] is True
    assert payload["summary"]["domain_count"] == 1
    captured = capsys.readouterr()
    assert "General anesthesia" not in captured.out
    assert '"passed": true' in captured.out
