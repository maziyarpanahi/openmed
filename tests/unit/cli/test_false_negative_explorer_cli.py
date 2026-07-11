"""Tests for the ``benchmark false-negatives`` explorer CLI command."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from openmed.cli import main_module
from openmed.eval.error_analysis import MISSED, ErrorAnalysisReport, error_report
from openmed.eval.false_negatives import (
    explore_false_negatives,
    load_fixture_texts,
)
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
            "metadata": {
                "predicted_spans": PREDICTED_SPANS,
                "synthetic": True,
            },
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
                        "metadata": {"synthetic": True},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return report_path, fixtures_path


def _build_capped_report(
    label_counts: dict[str, int], *, example_cap: int
) -> ErrorAnalysisReport:
    fixtures = []
    for label, count in label_counts.items():
        for index in range(count):
            text = f"synthetic-{label}-{index}"
            fixtures.append(
                BenchmarkFixture.from_mapping(
                    {
                        "id": f"{label.lower()}-{index}",
                        "text": text,
                        "gold_spans": [
                            {
                                "start": 0,
                                "end": len(text),
                                "label": label,
                                "text": text,
                            }
                        ],
                        "metadata": {"synthetic": True},
                    }
                )
            )
    return error_report(
        "synthetic-model",
        fixtures,
        runner=lambda fixture, model, device: [],
        example_cap=example_cap,
    )


def test_explore_false_negatives_lists_only_planted_misses(tmp_path: Path) -> None:
    report_path, fixtures_path = _build_report_and_fixtures(tmp_path)
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
    assert payload["available"] == 3
    assert payload["shown"] == 3
    assert payload["examples_truncated"] is False

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


def test_explorer_uses_canonical_counts_beyond_the_example_cap() -> None:
    report = _build_capped_report({"PERSON": 7}, example_cap=5)

    exploration = explore_false_negatives(report)

    assert report.confusion_matrix["PERSON"][MISSED] == 7
    assert exploration.total_missed == 7
    assert exploration.available == 5
    assert exploration.shown == 5
    assert exploration.examples_truncated is True
    assert len(exploration.groups) == 1
    group = exploration.groups[0]
    assert group.label == "PERSON"
    assert group.count == 7
    assert group.available == 5
    assert len(group.records) == 5

    payload = exploration.to_dict()
    assert payload["total_missed"] == 7
    assert payload["available"] == 5
    assert payload["example_cap"] == 5
    assert payload["examples_truncated"] is True
    assert payload["groups"][0]["count"] == 7
    assert payload["groups"][0]["available"] == 5
    assert payload["groups"][0]["shown"] == 5


def test_explorer_orders_labels_by_canonical_miss_frequency() -> None:
    report = _build_capped_report({"DATE": 6, "PHONE": 7}, example_cap=2)

    exploration = explore_false_negatives(report)

    # Both labels have two stored examples, so ordering by the bounded sample
    # would put DATE first. Canonical confusion-matrix counts put PHONE first.
    assert [group.label for group in exploration.groups] == ["PHONE", "DATE"]
    assert [group.count for group in exploration.groups] == [7, 6]
    assert [group.available for group in exploration.groups] == [2, 2]


def test_false_negatives_cli_discloses_capped_examples(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = _build_capped_report({"PERSON": 7}, example_cap=5)
    report_path = tmp_path / "capped-report.json"
    report.write_json(report_path)

    result = main_module.main(["benchmark", "false-negatives", str(report_path)])

    assert result == 0
    output = capsys.readouterr().out
    assert "Missed gold spans: 7  Stored examples: 5  Shown: 5" in output
    assert "Stored examples are capped by the report" in output
    assert "## PERSON (7)" in output
    assert "Stored examples: 5  Shown: 5" in output


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


def test_false_negatives_cli_rejects_negative_context_chars(
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
            "--context-chars",
            "-1",
        ]
    )

    assert result == 1
    assert "context-chars must be non-negative" in capsys.readouterr().err


def test_exploration_only_reports_text_when_fixture_hash_matches(
    tmp_path: Path,
) -> None:
    report_path, _ = _build_report_and_fixtures(tmp_path)
    report = ErrorAnalysisReport.read_json(report_path)

    exploration = explore_false_negatives(
        report,
        fixture_texts={"note-a": "x" * len(FIXTURE_TEXT)},
    )

    assert exploration.has_text is False
    assert all(record.span_text is None for record in exploration.iter_records())


def test_fixture_texts_require_explicit_synthetic_marker() -> None:
    with pytest.raises(ValueError, match="explicitly marked synthetic"):
        load_fixture_texts(
            [
                {
                    "id": "unmarked",
                    "text": "Patient Avery Morgan.",
                    "gold_spans": [],
                }
            ]
        )


@pytest.mark.parametrize(
    ("field", "malformed"),
    [
        ("confusion_matrix", {"PERSON": []}),
        ("false_negatives", {"PERSON": "not-a-list"}),
        ("false_positives", {"PERSON": ["not-an-object"]}),
    ],
)
def test_error_analysis_report_rejects_malformed_nested_payloads(
    tmp_path: Path,
    field: str,
    malformed: object,
) -> None:
    report_path, _ = _build_report_and_fixtures(tmp_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload[field] = malformed

    with pytest.raises(ValueError):
        ErrorAnalysisReport.from_dict(payload)


def test_error_analysis_report_round_trips_missed_examples(tmp_path: Path) -> None:
    report_path, _ = _build_report_and_fixtures(tmp_path)
    report = ErrorAnalysisReport.read_json(report_path)
    missed = [
        example
        for examples in report.false_negatives.values()
        for example in examples
        if example.kind == MISSED
    ]
    assert len(missed) == 3
    assert report.to_dict() == json.loads(report_path.read_text(encoding="utf-8"))
