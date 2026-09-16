"""Unit tests for the SHIELD comparison corpus suite."""

from __future__ import annotations

import json
import re
from pathlib import Path
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
from openmed.eval.datasets import (
    CLINICAL_PHI_MANIFEST_ID,
    CLINICAL_PHI_MANIFEST_REF,
    CLINICAL_PRIVACY_MODEL_ID,
    clinical_phi_manifest_hash,
)
from openmed.eval.harness import run_benchmark
from openmed.eval.suites import (
    SHIELD,
    load_suite_fixtures,
    run_clinical_phi_shield_benchmark,
    suite_metadata,
)
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

    def rows_loader(
        repository: str, config: str, split: str
    ) -> list[Mapping[str, object]]:
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

    def rows_loader(
        repository: str, config: str, split: str
    ) -> list[Mapping[str, object]]:
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


def test_clinical_phi_flagship_report_binds_comparison_evidence() -> None:
    note, spans = _synthetic_shield_rows()
    fixture = fixtures_from_rows([note], spans)[0]
    checkpoint = _synthetic_checkpoint_manifest()

    def runner(fixture, model_name, device):
        assert model_name == CLINICAL_PRIVACY_MODEL_ID
        assert device == "cpu"
        return [
            {"start": span.start, "end": span.end, "label": span.label}
            for span in fixture.gold_spans
            if span.label in {PERSON, AGE}
        ]

    report = run_clinical_phi_shield_benchmark(
        [fixture],
        checkpoint_manifest=checkpoint,
        checkpoint_manifest_ref=("models.jsonl#OpenMed/OpenMed-ClinicalPrivacy-tier0"),
        runner=runner,
        generated_at="2026-08-01T00:00:00Z",
    )

    data = report.to_dict()
    comparison = data["metrics"]["shield_comparison"]
    assert data["model_name"] == CLINICAL_PRIVACY_MODEL_ID
    assert comparison["evidence_role"] == "comparison"
    assert comparison["high_recall_release_gate"] is False
    assert (
        comparison["aggregate"]["recall"] == data["metrics"]["recall_slices"]["overall"]
    )
    assert comparison["aggregate"]["leakage"] == data["metrics"]["leakage"]["overall"]
    assert comparison["by_label"][PERSON] == {
        "leakage": 0.0,
        "recall": 1.0,
    }
    assert comparison["by_label"][PHONE] == {
        "leakage": 1.0,
        "recall": 0.0,
    }

    metadata = data["metadata"]
    assert metadata["comparison_evidence_only"] is True
    assert metadata["gate_target"] is False
    assert metadata["checkpoint_manifest"]["model_id"] == (CLINICAL_PRIVACY_MODEL_ID)
    assert (
        metadata["checkpoint_manifest"]["reproducibility_hash"]
        == (checkpoint["reproducibility_hash"])
    )
    assert metadata["checkpoint_manifest"]["source_revision"] == "a" * 40
    assert re.fullmatch(
        r"sha256:[0-9a-f]{64}",
        metadata["checkpoint_manifest"]["manifest_content_hash"],
    )
    assert metadata["dataset_manifest"] == {
        "manifest_hash": clinical_phi_manifest_hash(),
        "manifest_id": CLINICAL_PHI_MANIFEST_ID,
        "manifest_ref": CLINICAL_PHI_MANIFEST_REF,
    }
    assert metadata["public_corpus_reference"]["source_url"].endswith(
        PUBLIC_SAMPLE_REPOSITORY
    )
    assert metadata["public_corpus_reference"]["redistribution"] == ("reference-only")
    assert metadata["fixture_ids"] != ["note-x"]
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", metadata["fixture_ids"][0])
    assert re.fullmatch(
        r"sha256:[0-9a-f]{64}",
        metadata["reproducibility"]["fixture_set_hash"],
    )
    assert re.fullmatch(
        r"sha256:[0-9a-f]{64}",
        metadata["reproducibility"]["eval_code_hash"],
    )
    serialized = json.dumps(data, sort_keys=True)
    for raw_value in ("note-x", "John Doe", "MRN-98765", "555-0123", "123 Main St"):
        assert raw_value not in serialized


def test_clinical_phi_flagship_report_rejects_other_checkpoint() -> None:
    note, spans = _synthetic_shield_rows()
    fixture = fixtures_from_rows([note], spans)[0]

    with pytest.raises(ValueError, match="exactly one"):
        run_clinical_phi_shield_benchmark(
            [fixture],
            checkpoint_manifest={
                "repo_id": "OpenMed/another-model",
                "reproducibility_hash": "sha256:" + "0" * 64,
            },
            checkpoint_manifest_ref="models.jsonl#OpenMed/another-model",
            runner=lambda fixture, model_name, device: (),
        )


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

    def runner(fixture, model_name, device):
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


def test_cli_benchmark_pii_emits_manifest_linked_flagship_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    from openmed.cli import main_module
    from openmed.eval import harness, suites

    note, spans = _synthetic_shield_rows()
    fixtures = fixtures_from_rows([note], spans)
    checkpoint_path = tmp_path / "checkpoint.jsonl"
    checkpoint_path.write_text(
        "\n".join(
            (
                json.dumps({"repo_id": "OpenMed/unrelated-model"}),
                json.dumps(_synthetic_checkpoint_manifest()),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        suites,
        "load_shield_fixtures",
        lambda **kwargs: fixtures,
    )
    monkeypatch.setattr(
        harness,
        "default_model_runner",
        lambda fixture, model_name, device: [
            {"start": span.start, "end": span.end, "label": span.label}
            for span in fixture.gold_spans
        ],
    )

    result = main_module.main(
        [
            "benchmark",
            "pii",
            "--suite",
            "shield",
            "--models",
            CLINICAL_PRIVACY_MODEL_ID,
            "--checkpoint-manifest",
            str(checkpoint_path),
            "--checkpoint-manifest-ref",
            "models.jsonl#OpenMed/OpenMed-ClinicalPrivacy-tier0",
        ]
    )

    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output["model_name"] == CLINICAL_PRIVACY_MODEL_ID
    assert output["metadata"]["checkpoint_manifest"]["manifest_ref"].startswith(
        "models.jsonl#"
    )
    assert output["metrics"]["shield_comparison"]["aggregate"]["recall"] == 1.0
    assert output["metrics"]["shield_comparison"]["high_recall_release_gate"] is False


def _synthetic_checkpoint_manifest() -> dict[str, object]:
    return {
        "repo_id": CLINICAL_PRIVACY_MODEL_ID,
        "family": "ClinicalPrivacy",
        "reproducibility_hash": "sha256:" + "1" * 64,
        "provenance": {"source_revision": "a" * 40},
    }


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
