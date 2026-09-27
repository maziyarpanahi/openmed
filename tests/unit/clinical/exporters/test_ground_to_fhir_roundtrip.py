"""Offline conformance and leakage checks for the ground-to-FHIR example."""

from __future__ import annotations

import json
import socket
from collections import Counter
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from examples import ground_then_export_fhir as example
from openmed.clinical.exporters import (
    GROUNDED_CODE_PROVENANCE_EXTENSION_URL,
    check_codeable_concept,
    codeable_concept_from_grounded_concept,
)

_SYSTEM_URI = {
    "rxnorm": "http://www.nlm.nih.gov/research/umls/rxnorm",
    "loinc": "http://loinc.org",
    "icd10cm": "http://hl7.org/fhir/sid/icd-10-cm",
}
_ASSIST_ONLY_EXTENSION_URL = (
    "https://openmed.ai/fhir/StructureDefinition/medical-device-assist"
)
_EXPECTED_BY_CODE = {
    fixture.code: (_SYSTEM_URI[fixture.system], fixture.surface)
    for fixture in example._CONCEPT_FIXTURES
}


def _observation_concepts(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the CodeableConcept values emitted for grounded observations."""
    return [
        entry["resource"]["code"]
        for entry in bundle["entry"]
        if entry["resource"].get("resourceType") == "Observation"
    ]


def _coding_triples(bundle: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Extract ordered coding identity fields for round-trip comparison."""
    return [
        (coding["system"], coding["code"], coding["display"])
        for concept in _observation_concepts(bundle)
        for coding in concept["coding"]
    ]


def test_example_runs_offline_and_emits_thirty_grounded_concepts(
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The runnable example performs de-identification before all downstream work."""
    caplog.set_level("DEBUG")

    bundle = example.main()
    captured = capsys.readouterr()
    concepts = _observation_concepts(bundle)

    assert "=== De-identified text ===" in captured.out
    assert "=== NER spans ===" in captured.out
    assert "=== Grounded concepts ===" in captured.out
    assert "=== FHIR Bundle ===" in captured.out
    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "transaction"
    assert len(concepts) == 30

    systems = Counter(
        coding["system"] for concept in concepts for coding in concept["coding"]
    )
    assert systems == Counter({uri: 10 for uri in _SYSTEM_URI.values()})

    for concept in concepts:
        assert len(concept["coding"]) == 1
        coding = concept["coding"][0]
        assert (coding["system"], coding["display"]) == _EXPECTED_BY_CODE[
            coding["code"]
        ]
        assert (
            check_codeable_concept(
                concept,
                expected_system=coding["system"],
            )
            == []
        )
        assert any(
            extension["url"] == GROUNDED_CODE_PROVENANCE_EXTENSION_URL
            for extension in coding["extension"]
        )
        assert any(
            extension["url"] == _ASSIST_ONLY_EXTENSION_URL
            for extension in concept["extension"]
        )

    for forbidden in example.SYNTHETIC_PHI:
        assert forbidden not in captured.out
        assert forbidden not in captured.err
        assert forbidden not in caplog.text
        assert forbidden not in json.dumps(bundle)
        assert all(forbidden not in record.getMessage() for record in caplog.records)


def test_grounded_coding_provenance_keeps_offsets_and_snapshot_versions() -> None:
    """Every emitted Coding records selection provenance without raw note text."""
    pipeline = example.run_pipeline()
    concepts = _observation_concepts(pipeline["bundle"])

    assert len(pipeline["grounded_spans"]) == len(concepts) == 30
    for span, concept in zip(pipeline["grounded_spans"], concepts):
        coding = concept["coding"][0]
        extension = next(
            item
            for item in coding["extension"]
            if item["url"] == GROUNDED_CODE_PROVENANCE_EXTENSION_URL
        )
        values = {item["url"]: item for item in extension["extension"]}
        assert values["evidence_start"]["valueUnsignedInt"] == span.start
        assert values["evidence_end"]["valueUnsignedInt"] == span.end
        assert values["vocab_version"]["valueString"] == coding["version"]
        assert values["linker"]["valueString"] == "sparse"


def test_bundle_json_round_trip_preserves_code_system_and_display() -> None:
    """JSON serialization and parsing do not mutate any grounded Coding."""
    bundle = example.run_pipeline()["bundle"]
    before = _coding_triples(bundle)
    reparsed = json.loads(json.dumps(bundle, sort_keys=True))

    assert _coding_triples(reparsed) == before
    assert len(before) == 30
    assert {system for system, _, _ in before} == set(_SYSTEM_URI.values())
    assert sorted(before) == sorted(
        (system, code, display) for code, (system, display) in _EXPECTED_BY_CODE.items()
    )


def test_adapter_accepts_one_system_grounded_concept_attributes() -> None:
    """The adapter also consumes the newer one-system concept shape."""
    grounded = SimpleNamespace(
        span=SimpleNamespace(start=4, end=11),
        surface_text="synthetic",
        system="ICD10CM",
        code="E11.9",
        display="synthetic",
        confidence=0.93,
        candidates=(),
        provenance={
            "linker_name": "local-ner",
            "vocabulary_snapshot_version": "snapshot-synthetic-v1",
        },
    )

    concept = codeable_concept_from_grounded_concept(grounded)

    assert concept["text"] == "synthetic"
    assert concept["coding"][0]["system"] == _SYSTEM_URI["icd10cm"]
    assert concept["coding"][0]["code"] == "E11.9"
    assert check_codeable_concept(concept, expected_system="icd10cm") == []


def test_wrong_system_uri_fails_the_opt_in_conformance_check() -> None:
    """A coding copied to another vocabulary is rejected by the checker."""
    concept = deepcopy(_observation_concepts(example.run_pipeline()["bundle"])[0])
    original_system = concept["coding"][0]["system"]
    wrong_system = next(uri for uri in _SYSTEM_URI.values() if uri != original_system)
    concept["coding"][0]["system"] = wrong_system

    findings = check_codeable_concept(concept, expected_system=original_system)

    assert [finding["finding_code"] for finding in findings] == ["system-uri-mismatch"]
    assert findings[0]["severity"] == "error"
    assert findings[0]["expression"] == ["CodeableConcept.coding[0].system"]


def test_pipeline_does_not_open_network_sockets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model and all three terminology snapshots remain local-only."""

    def fail_connection(*_: Any, **__: Any) -> None:
        raise AssertionError("the example must not open a network socket")

    monkeypatch.setattr(socket, "create_connection", fail_connection)

    pipeline = example.run_pipeline()

    assert len(_observation_concepts(pipeline["bundle"])) == 30
