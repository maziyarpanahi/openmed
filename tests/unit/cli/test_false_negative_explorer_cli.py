"""Tests for the ``benchmark false-negatives`` explorer CLI command."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from openmed.cli import main_module
from openmed.eval.error_analysis import MISSED, error_report
from openmed.eval.false_negatives import explore_false_negatives
from openmed.eval.harness import BenchmarkFixture

# Synthetic document with planted misses. All identifiers are fabricated.
FIXTURE_TEXT = "Patient John Doe called 555-0100 on 2026-01-02 from Room 5."


def _span(sub: str, label: str) -> dict[str, Any]:
    start = FIXTURE_TEXT.index(sub)
    return {"start": start, "end": start + len(sub), "label": label, "text": sub}


GOLD_SPANS = [
    _span("John Doe", "PERSON"),
    _span("555-0100", "PHONE"),
    _span("2026-01-02", "DATE"),
    _span("Room 5", "LOCATION"),
]

# The model finds only the PERSON span, so PHONE, DATE, and LOCATION are missed.
PREDICTED_SPANS = [_span("John Doe", "PERSON")]


def _metadata_runner(
    fixture: BenchmarkFixture, model: str, device: str
) -> Iterable[dict[str, Any]]:
    return list(fixture.metadata.get("predicted_spans", []))


def _build_report_and_fixtures(tmp_path: Path) -> tuple[Path, Path]:
    """Write a synthetic error-analysis report plus its gold fixture file."""
    fixture = BenchmarkFixture.from_mapping(
        {
            "id": "note-a",
            "text": FIXTURE_TEXT,
            "language": "en",
            "gold_spans": GOLD_SPANS,
            "metadata": {"predicted_spans": PREDICTED_SPANS},
        }
    )
    report = error_report(
        "synthetic-model",
        [fixture],
        suite_name="golden",
        runner=_metadata_runner,
        context_window=10,
    )
    report_path = tmp_path / "report.json"
    report.write_json(report_path)

    fixtures_path = tmp_path / "fixtures.json"
    fixtures_path.write_text(
        json.dumps(
            {
                "fixtures": [
                    {
                        "id": "note-a",
                        "text": FIXTURE_TEXT,
                        "language": "en",
                        "gold_spans": GOLD_SPANS,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return report_path, fixtures_path


def test_explore_false_negatives_lists_only_planted_misses(tmp_path: Path) -> None:
    report_path, fixtures_path = _build_report_and_fixtures(tmp_path)
    from openmed.eval.error_analysis import ErrorAnalysisReport
    from openmed.eval.false_negatives import load_fixture_texts

    report = ErrorAnalysisReport.read_json(report_path)
    exploration = explore_false_negatives(
        report,
        fixture_texts=load_fixture_texts([fixtures_path]),
    )

    assert exploration.total_missed == 3
    assert exploration.shown == 3
    missed = {(r.label, r.span_text) for r in exploration.iter_records()}
    assert missed == {
        ("PHONE", "555-0100"),
        ("DATE", "2026-01-02"),
        ("LOCATION", "Room 5"),
    }
    # The PERSON span was predicted correctly, so it is not a false negative.
    assert all(r.label != "PERSON" for r in exploration.iter_records())


def test_false_negatives_cli_table_reports_correct_labels_and_context(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path, fixtures_path = _build_report_and_fixtures(tmp_path)

    result = main_module.main(
        [
            "benchmark",
            "false-negatives",
            str(report_path),
            "--fixtures",
            str(fixtures_path),
        ]
    )
    assert result == 0

    out = capsys.readouterr().out
    assert "## PHONE (1)" in out
    assert "## DATE (1)" in out
    assert "## LOCATION (1)" in out
    # The correctly predicted PERSON span must not appear as a miss.
    assert "## PERSON" not in out
    # Span text and a surrounding-context window are surfaced from the fixture.
    assert "'555-0100'" in out
    assert "called 555-0100 on" in out
    assert "Missed gold spans: 3" in out


def test_false_negatives_cli_json_emits_valid_records(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path, fixtures_path = _build_report_and_fixtures(tmp_path)

    result = main_module.main(
        [
            "benchmark",
            "false-negatives",
            str(report_path),
            "--fixtures",
            str(fixtures_path),
            "--json",
        ]
    )
    assert result == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["suite"] == "golden"
    assert payload["model_name"] == "synthetic-model"
    assert payload["total_missed"] == 3
    assert payload["shown"] == 3

    records = [rec for group in payload["groups"] for rec in group["records"]]
    assert len(records) == 3
    for record in records:
        assert record["end"] > record["start"]
        assert record["context_start"] <= record["start"]
        assert record["context_end"] >= record["end"]
        assert record["text_hash"].startswith("sha256:")
    phone = next(r for r in records if r["label"] == "PHONE")
    assert phone["span_text"] == "555-0100"
    assert "555-0100" in phone["context"]


def test_false_negatives_cli_honors_label_filter(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path, fixtures_path = _build_report_and_fixtures(tmp_path)

    result = main_module.main(
        [
            "benchmark",
            "false-negatives",
            str(report_path),
            "--fixtures",
            str(fixtures_path),
            "--label",
            "phone",
            "--json",
        ]
    )
    assert result == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["label_filter"] == "PHONE"
    assert payload["total_missed"] == 1
    records = [rec for group in payload["groups"] for rec in group["records"]]
    assert len(records) == 1
    assert records[0]["label"] == "PHONE"
    assert records[0]["span_text"] == "555-0100"


def test_false_negatives_cli_honors_limit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path, fixtures_path = _build_report_and_fixtures(tmp_path)

    result = main_module.main(
        [
            "benchmark",
            "false-negatives",
            str(report_path),
            "--fixtures",
            str(fixtures_path),
            "--limit",
            "1",
            "--json",
        ]
    )
    assert result == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["limit"] == 1
    # Three spans were missed, but only one is shown under the cap.
    assert payload["total_missed"] == 3
    assert payload["shown"] == 1
    records = [rec for group in payload["groups"] for rec in group["records"]]
    assert len(records) == 1


def test_false_negatives_cli_without_fixtures_hides_raw_text(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path, _ = _build_report_and_fixtures(tmp_path)

    result = main_module.main(
        [
            "benchmark",
            "false-negatives",
            str(report_path),
            "--json",
        ]
    )
    assert result == 0

    raw_out = capsys.readouterr().out
    payload = json.loads(raw_out)
    assert payload["has_text"] is False
    records = [rec for group in payload["groups"] for rec in group["records"]]
    assert records
    for record in records:
        # No raw PHI without synthetic fixtures: only offsets and hashes.
        assert "span_text" not in record
        assert "context" not in record
        assert record["text_hash"].startswith("sha256:")
    # No fabricated identifier text should appear anywhere in the output.
    assert "555-0100" not in raw_out


def test_false_negatives_cli_missing_report_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main_module.main(
        [
            "benchmark",
            "false-negatives",
            str(tmp_path / "does-not-exist.json"),
        ]
    )
    assert result == 1
    assert "not found" in capsys.readouterr().err


def test_false_negatives_cli_context_chars_trims_window(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path, fixtures_path = _build_report_and_fixtures(tmp_path)

    result = main_module.main(
        [
            "benchmark",
            "false-negatives",
            str(report_path),
            "--fixtures",
            str(fixtures_path),
            "--label",
            "phone",
            "--context-chars",
            "10",
            "--json",
        ]
    )
    assert result == 0

    payload = json.loads(capsys.readouterr().out)
    record = payload["groups"][0]["records"][0]
    assert len(record["context"]) <= 10
    # The missed span stays visible inside the trimmed window.
    assert "555-0100" in record["context"]


def test_error_analysis_report_round_trips_missed_examples(tmp_path: Path) -> None:
    report_path, _ = _build_report_and_fixtures(tmp_path)
    from openmed.eval.error_analysis import ErrorAnalysisReport

    report = ErrorAnalysisReport.read_json(report_path)
    missed = [
        example
        for examples in report.false_negatives.values()
        for example in examples
        if example.kind == MISSED
    ]
    assert len(missed) == 3
    assert report.to_dict() == json.loads(report_path.read_text(encoding="utf-8"))
