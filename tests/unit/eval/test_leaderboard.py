"""Tests for the archived-report public leaderboard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from openmed.__about__ import __version__
from openmed.core.model_registry import load_manifest_rows
from openmed.eval.leaderboard import (
    LeaderboardError,
    load_leaderboard_rows,
    write_leaderboard,
)
from openmed.eval.report import BenchmarkReport

ROOT = Path(__file__).resolve().parents[3]
PUBLIC_LEADERBOARD = ROOT / "docs" / "eval" / "benchmark-leaderboard"
PAGES_WORKFLOW = ROOT / ".github" / "workflows" / "pages.yml"


def _write_report(
    path: Path,
    *,
    model_name: str,
    suite: str = "privacy",
    family: str = "PII",
    leakage: float = 0.0,
    recall: float = 1.0,
    f1: float = 1.0,
    release_tag: str = "v2.0.0",
    run_date: str = "2026-07-17T12:00:00Z",
    repro_character: str = "a",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    BenchmarkReport(
        suite=suite,
        model_name=model_name,
        device="cpu",
        fixture_count=3,
        generated_at=run_date,
        metrics={
            "exact_span_f1": {"f1": f1, "recall": recall},
            "leakage": {"overall": leakage},
        },
        metadata={
            "family": family,
            "release_tag": release_tag,
            "reproducibility_hash": "sha256:" + repro_character * 64,
            "synthetic": True,
        },
    ).write_json(path)


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(item for item in directory.rglob("*") if item.is_file())
    }


def test_renderer_sorts_leakage_first_then_recall(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    _write_report(
        reports / "high-f1.json",
        model_name="high-f1",
        leakage=0.2,
        recall=0.99,
        f1=0.99,
    )
    _write_report(
        reports / "low-recall.json",
        model_name="low-recall",
        leakage=0.1,
        recall=0.8,
        f1=0.95,
        repro_character="b",
    )
    _write_report(
        reports / "high-recall.json",
        model_name="high-recall",
        leakage=0.1,
        recall=0.9,
        f1=0.4,
        repro_character="c",
    )

    rows = load_leaderboard_rows(reports)

    assert [row.model_name for row in rows] == [
        "high-recall",
        "low-recall",
        "high-f1",
    ]
    assert [(row.leakage, row.recall) for row in rows] == [
        (0.1, 0.9),
        (0.1, 0.8),
        (0.2, 0.99),
    ]


def test_renderer_ignores_non_report_json_in_mixed_archive(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    _write_report(reports / "valid.report.json", model_name="valid")
    (reports / "baseline.json").write_text(
        '{"schema_version": 1, "throughput": 12.5}\n',
        encoding="utf-8",
    )

    rows = load_leaderboard_rows(reports)

    assert [row.model_name for row in rows] == ["valid"]


def test_renderer_emits_grouped_html_json_and_downloads(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    _write_report(
        reports / "nested" / "privacy.json",
        model_name="privacy-model",
        suite="privacy",
        family="PII",
    )
    _write_report(
        reports / "utility.json",
        model_name="utility-model",
        suite="utility",
        family="NER",
        leakage=0.02,
        recall=0.91,
        f1=0.9,
        release_tag="v2.0.1",
        run_date="2026-07-18T08:30:00Z",
        repro_character="d",
    )

    output = tmp_path / "output"
    artifacts = write_leaderboard(reports, output)
    payload = json.loads(artifacts.json_path.read_text(encoding="utf-8"))
    rendered = artifacts.html_path.read_text(encoding="utf-8")

    assert payload["schema_version"] == 1
    assert payload["sort"] == ["leakage:asc", "recall:desc"]
    assert [suite["name"] for suite in payload["suites"]] == [
        "privacy",
        "utility",
    ]
    assert payload["rows"][0]["release_tag"] == "v2.0.0"
    assert payload["rows"][0]["run_date"] == "2026-07-17"
    assert payload["rows"][0]["reproducibility_hash"] == "sha256:" + "a" * 64
    assert 'role="tab"' in rendered
    assert ">PII</h2>" in rendered
    assert ">NER</h2>" in rendered
    assert 'href="reports/nested/privacy.json" download' in rendered
    assert 'href="leaderboard.json" download' in rendered
    assert (output / "reports" / "nested" / "privacy.json").is_file()
    assert (output / "reports" / "utility.json").is_file()


def test_rerendering_same_archive_is_byte_identical(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    _write_report(
        reports / "z.json",
        model_name="z-model",
        leakage=0.03,
        recall=0.9,
        f1=0.88,
    )
    _write_report(
        reports / "a.json",
        model_name="a-model",
        leakage=0.01,
        recall=0.92,
        f1=0.9,
        repro_character="b",
    )
    output = tmp_path / "output"

    write_leaderboard(reports, output)
    first = _snapshot(output)
    write_leaderboard(reports, output)
    second = _snapshot(output)

    assert first == second


def test_manifest_and_release_fallback_stamp_legacy_report(tmp_path: Path) -> None:
    report_path = tmp_path / "reports" / "legacy.json"
    report_path.parent.mkdir()
    BenchmarkReport(
        suite="privacy",
        model_name="OpenMed/legacy",
        device="cpu",
        fixture_count=1,
        generated_at="2026-07-17T01:02:03Z",
        metrics={
            "exact_span_f1": {"f1": 0.9, "recall": 0.92},
            "leakage": {"overall": 0.01},
        },
        metadata={"synthetic": True},
    ).write_json(report_path)
    manifest = [
        {
            "repo_id": "OpenMed/legacy",
            "family": "PII",
            "reproducibility_hash": "sha256:" + "e" * 64,
        }
    ]

    rows = load_leaderboard_rows(
        report_path.parent,
        manifest_rows=manifest,
        release_tag="v2.0.0",
    )

    assert rows[0].model_family == "PII"
    assert rows[0].release_tag == "v2.0.0"
    assert rows[0].run_date == "2026-07-17"
    assert rows[0].reproducibility_hash == "sha256:" + "e" * 64


def test_renderer_fails_closed_when_ranking_evidence_is_missing(
    tmp_path: Path,
) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    BenchmarkReport(
        suite="privacy",
        model_name="incomplete",
        device="cpu",
        fixture_count=1,
        generated_at="2026-07-17T00:00:00Z",
        metrics={"exact_span_f1": {"f1": 0.9, "recall": 0.9}},
        metadata={
            "family": "PII",
            "release_tag": "v2.0.0",
            "reproducibility_hash": "sha256:" + "a" * 64,
            "synthetic": True,
        },
    ).write_json(reports / "incomplete.json")

    with pytest.raises(LeaderboardError, match="missing leakage"):
        load_leaderboard_rows(reports)


def test_renderer_rejects_report_without_explicit_synthetic_provenance(
    tmp_path: Path,
) -> None:
    reports = tmp_path / "reports"
    _write_report(reports / "unmarked.json", model_name="unmarked")
    payload = json.loads((reports / "unmarked.json").read_text(encoding="utf-8"))
    del payload["metadata"]["synthetic"]
    (reports / "unmarked.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(LeaderboardError, match="explicitly marked as synthetic"):
        load_leaderboard_rows(reports)


@pytest.mark.parametrize(
    "run_date",
    ["2026-07-17 trailing text", "2026-13-01", 20260717],
)
def test_renderer_rejects_invalid_run_dates(tmp_path: Path, run_date: object) -> None:
    reports = tmp_path / "reports"
    _write_report(reports / "invalid-date.json", model_name="invalid-date")
    payload = json.loads((reports / "invalid-date.json").read_text(encoding="utf-8"))
    payload["generated_at"] = run_date
    (reports / "invalid-date.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(LeaderboardError, match="generated_at|run date"):
        load_leaderboard_rows(reports)


def test_renderer_escapes_report_metadata_in_html(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    _write_report(
        reports / "unsafe.json",
        model_name="<script>alert(1)</script>",
        family="PII & NER",
    )

    artifacts = write_leaderboard(reports, tmp_path / "output")
    rendered = artifacts.html_path.read_text(encoding="utf-8")

    assert "<script>alert(1)</script>" not in rendered
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered
    assert "PII &amp; NER" in rendered


def test_committed_public_leaderboard_does_not_drift(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    write_leaderboard(
        ROOT / "docs" / "benchmarks",
        generated,
        manifest_rows=load_manifest_rows(ROOT / "models.jsonl"),
        release_tag=f"v{__version__}",
    )

    assert _snapshot(generated) == _snapshot(PUBLIC_LEADERBOARD)


def test_pages_workflow_renders_on_master_and_release_tags() -> None:
    workflow_text = PAGES_WORKFLOW.read_text(encoding="utf-8")
    workflow = yaml.load(
        workflow_text,
        Loader=yaml.BaseLoader,
    )
    push = workflow["on"]["push"]
    steps = workflow["jobs"]["deploy"]["steps"]
    render_step = next(
        step
        for step in steps
        if step.get("name") == "Render public benchmark leaderboard"
    )

    assert push["branches"] == ["master"]
    assert push["tags"] == ["v*"]
    assert workflow_text.count('tags: ["v*"]') == 1
    assert "python -m openmed.eval.leaderboard" in render_step["run"]
    assert "docs/eval/benchmark-leaderboard" in render_step["run"]
    assert 'release_tag="$GITHUB_REF_NAME"' in render_step["run"]
    assert "test -s site/docs/llms.txt" in workflow_text
    assert "test -s site/docs/llms-full.txt" in workflow_text
