"""Unit tests for the SHIELD comparison corpus suite."""

from __future__ import annotations

import json
from typing import Mapping

import pytest

from openmed.core.labels import (
    AGE,
    CANONICAL_LABELS,
    DATE,
    ID_NUM,
    LOCATION,
    ORGANIZATION,
    PERSON,
    PHONE,
    URL,
)
from openmed.eval.harness import run_benchmark
from openmed.eval.suites import SHIELD, load_suite_fixtures, suite_metadata
from openmed.eval.suites.shield import (
    IS_HIGH_RECALL_GATE_TARGET,
    PUBLIC_SAMPLE_NOTES_CONFIG,
    PUBLIC_SAMPLE_REPOSITORY,
    PUBLIC_SAMPLE_SPANS_CONFIG,
    SHIELD_LABEL_TO_CANONICAL,
    SUITE_ANNOTATION,
    VERIFIED_LICENSE,
    fixtures_from_rows,
    load_shield_fixtures,
    map_shield_label,
)


def test_shield_label_mapping_is_total_and_canonical() -> None:
    assert SHIELD_LABEL_TO_CANONICAL == {
        "age": AGE,
        "date": DATE,
        "doctor": PERSON,
        "hospital": ORGANIZATION,
        "id": ID_NUM,
        "location": LOCATION,
        "patient": PERSON,
        "phone": PHONE,
        "web": URL,
    }
    assert len(SHIELD_LABEL_TO_CANONICAL) == 9
    assert set(SHIELD_LABEL_TO_CANONICAL.values()) <= CANONICAL_LABELS
    assert map_shield_label(" Doctor ") == PERSON

    with pytest.raises(ValueError, match="unknown SHIELD label"):
        map_shield_label("room")


def test_fixtures_from_rows_joins_notes_and_spans_without_vendored_corpus() -> None:
    note, spans = _synthetic_shield_rows()

    fixtures = fixtures_from_rows([note], spans)

    assert len(fixtures) == 1
    fixture = fixtures[0]
    assert fixture.fixture_id == "note-x"
    assert fixture.metadata["annotation"] == SUITE_ANNOTATION
    assert fixture.metadata["corpus_role"] == "comparison"
    assert fixture.metadata["gate_target"] is IS_HIGH_RECALL_GATE_TARGET
    assert fixture.metadata["license"] == VERIFIED_LICENSE
    assert fixture.metadata["redistribution"] == "not vendored; loaded by reference"
    assert fixture.metadata["repository"] == PUBLIC_SAMPLE_REPOSITORY
    assert {span.label for span in fixture.gold_spans} == set(
        SHIELD_LABEL_TO_CANONICAL.values()
    )
    assert {span.metadata["shield_label"] for span in fixture.gold_spans} == set(
        SHIELD_LABEL_TO_CANONICAL
    )


def test_load_shield_fixtures_uses_public_sample_reference_by_default() -> None:
    note, spans = _synthetic_shield_rows()
    calls: list[tuple[str, str, str]] = []

    def rows_loader(repository: str, config: str, split: str) -> list[Mapping[str, object]]:
        calls.append((repository, config, split))
        if config == PUBLIC_SAMPLE_NOTES_CONFIG:
            return [note]
        if config == PUBLIC_SAMPLE_SPANS_CONFIG:
            return spans
        raise AssertionError(f"unexpected config: {config}")

    fixtures = load_shield_fixtures(rows_loader=rows_loader)

    assert len(fixtures) == 1
    assert calls == [
        (PUBLIC_SAMPLE_REPOSITORY, PUBLIC_SAMPLE_NOTES_CONFIG, "train"),
        (PUBLIC_SAMPLE_REPOSITORY, PUBLIC_SAMPLE_SPANS_CONFIG, "train"),
    ]


def test_suite_registry_loads_shield_and_metadata() -> None:
    note, spans = _synthetic_shield_rows()

    def rows_loader(repository: str, config: str, split: str) -> list[Mapping[str, object]]:
        return [note] if config == PUBLIC_SAMPLE_NOTES_CONFIG else spans

    fixtures = load_suite_fixtures(SHIELD, rows_loader=rows_loader)
    metadata = suite_metadata(SHIELD)

    assert len(fixtures) == 1
    assert metadata["annotation"] == SUITE_ANNOTATION
    assert metadata["label_mapping"]["web"] == URL


def test_shield_report_contains_per_label_leakage_and_recall() -> None:
    note, spans = _synthetic_shield_rows()
    fixture = fixtures_from_rows([note], spans)[0]

    def runner(fixture, model_name, device):
        assert model_name == "fixture-model"
        assert device == "cpu"
        return [
            {"start": span.start, "end": span.end, "label": span.label}
            for span in fixture.gold_spans
            if span.label in {PERSON, AGE}
        ]

    report = run_benchmark(
        [fixture],
        suite=SHIELD,
        model_name="fixture-model",
        runner=runner,
        metadata=suite_metadata(SHIELD),
    )

    data = report.to_dict()
    assert data["suite"] == SHIELD
    assert data["metadata"]["annotation"] == SUITE_ANNOTATION
    assert data["metrics"]["leakage"]["by_label"][PERSON] == 0.0
    assert data["metrics"]["leakage"]["by_label"][PHONE] == 1.0
    assert data["metrics"]["recall_slices"]["by_label"][PERSON] == 1.0
    assert data["metrics"]["recall_slices"]["by_label"][PHONE] == 0.0


def test_cli_benchmark_pii_emits_shield_benchmark_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from openmed.cli import main_module
    from openmed.eval import harness, suites

    note, spans = _synthetic_shield_rows()
    fixtures = fixtures_from_rows([note], spans)

    monkeypatch.setattr(
        suites,
        "load_shield_fixtures",
        lambda **kwargs: fixtures,
    )

    def runner(fixture, model_name, device, **_kwargs):
        return [
            {"start": span.start, "end": span.end, "label": span.label}
            for span in fixture.gold_spans
        ]

    monkeypatch.setattr(harness, "default_model_runner", runner)

    result = main_module.main(
        ["benchmark", "pii", "--suite", "shield", "--models", "fixture-model"]
    )

    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output["suite"] == SHIELD
    assert output["model_name"] == "fixture-model"
    assert output["metadata"]["license"] == VERIFIED_LICENSE
    assert output["metrics"]["leakage"]["by_label"][PERSON] == 0.0
    assert output["metrics"]["recall_slices"]["by_label"][PERSON] == 1.0


def _synthetic_shield_rows() -> tuple[dict[str, object], list[dict[str, object]]]:
    pieces = [
        ("patient", "John Doe"),
        ("age", "45"),
        ("date", "2025-01-15"),
        ("doctor", "Jane Doe"),
        ("hospital", "General Hospital"),
        ("id", "MRN-98765"),
        ("location", "123 Main St"),
        ("phone", "555-0123"),
        ("web", "clinic.example"),
    ]
    text = ""
    spans: list[dict[str, object]] = []
    for index, (label, value) in enumerate(pieces, start=1):
        if text:
            text += " "
        start = len(text)
        text += value
        end = len(text)
        spans.append(
            {
                "span_id": f"span-{index}",
                "note_id": "note-x",
                "span_start": start,
                "span_end": end,
                "span_label": label,
            }
        )

    return (
        {
            "note_id": "note-x",
            "note_text": text,
            "note_type": "synthetic_unit",
        },
        spans,
    )
