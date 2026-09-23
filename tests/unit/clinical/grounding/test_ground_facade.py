"""Acceptance tests for the typed, offline grounding facade."""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path

import pytest

import openmed
from openmed.clinical.grounding import (
    GroundedConcept,
    GroundingConfigError,
    GroundingResult,
    VocabLoader,
    VocabSource,
    ground,
)
from openmed.core.offline import OfflineModeError


def _loader(tmp_path: Path) -> VocabLoader:
    rows = {
        "rxnorm": {
            "aliases": ["metformin"],
            "canonical_term": "Metformin hydrochloride",
            "concept_id": "860975",
        },
        "loinc": {
            "aliases": ["hemoglobin a1c"],
            "canonical_term": "Hemoglobin A1c",
            "concept_id": "4548-4",
        },
        "icd10cm": {
            "aliases": ["type 2 diabetes", "diabete type 2"],
            "language_aliases": {"fr": ["diabete type 2"]},
            "canonical_term": "Type 2 diabetes mellitus without complications",
            "concept_id": "E11.9",
        },
    }
    registry: dict[str, VocabSource] = {}
    for system, row in rows.items():
        path = tmp_path / f"{system}.jsonl"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        registry[system] = VocabSource(
            system=system,
            path=path,
            version=f"synthetic-{system}-2026",
        )
    return VocabLoader(
        cache_dir=tmp_path / "cache",
        local_only=True,
        registry=registry,
    )


def test_text_and_entity_inputs_return_typed_results(tmp_path: Path) -> None:
    loader = _loader(tmp_path)

    text_result = ground(
        "metformin 500 mg",
        systems=["rxnorm"],
        lang="en",
        top_k=1,
        snapshot=loader,
    )
    assert isinstance(text_result, GroundingResult)
    assert isinstance(text_result.concepts[0], GroundedConcept)
    concept = text_result.concepts[0]
    assert (concept.start, concept.end) == (0, len("metformin"))
    assert concept.surface_text == "metformin"
    assert concept.system == "rxnorm"
    assert concept.code == "860975"
    assert 0.0 <= concept.confidence <= 1.0
    assert concept.provenance["vocabulary_snapshot_version"] == (
        "synthetic-rxnorm-2026"
    )

    entity_result = ground(
        [{"text": "type 2 diabetes", "start": 4, "end": 19}],
        systems=["icd10cm"],
        snapshot=loader,
    )
    assert entity_result[0].start == 4
    assert entity_result.concepts[0].code == "E11.9"


def test_language_routing_and_unknown_system_rejection(tmp_path: Path) -> None:
    result = ground(
        "diabete type 2",
        systems=["icd10cm"],
        lang="fr-FR",
        snapshot=_loader(tmp_path),
    )
    assert result.concepts[0].code == "E11.9"
    assert result.concepts[0].provenance["source_language"] == "fr"

    with pytest.raises(ValueError, match="unsupported grounding system"):
        ground("metformin", systems=["not-a-system"], snapshot=_loader(tmp_path))


def test_missing_snapshot_is_offline_and_does_not_open_a_socket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted = False

    def fail_connect(*_: object, **__: object) -> None:
        nonlocal attempted
        attempted = True
        raise AssertionError("grounding attempted network access")

    monkeypatch.setattr(socket.socket, "connect", fail_connect)
    loader = VocabLoader(
        cache_dir=tmp_path / "empty-cache",
        local_only=True,
        registry={
            "rxnorm": VocabSource(
                system="rxnorm",
                url="https://example.invalid/rxnorm.jsonl",
                sha256="0" * 64,
            )
        },
    )
    with pytest.raises(OfflineModeError, match="vocabulary"):
        ground("metformin", systems=["rxnorm"], snapshot=loader)
    assert attempted is False


def test_restricted_system_requires_a_user_supplied_bridge() -> None:
    with pytest.raises(
        GroundingConfigError, match="user-supplied.*SNOMED|SNOMED.*user-supplied"
    ):
        ground("diabetes", systems=["snomed"])


def test_result_json_round_trip_preserves_spans_and_codes(tmp_path: Path) -> None:
    original = ground(
        "metformin 500 mg",
        systems=["rxnorm"],
        snapshot=_loader(tmp_path),
    )
    payload = json.loads(json.dumps(original.to_dict(), sort_keys=True))
    restored = GroundingResult.from_dict(payload)

    assert restored.concepts[0].span == original.concepts[0].span
    assert restored.concepts[0].code == original.concepts[0].code
    assert restored.concepts[0].surface_text == original.concepts[0].surface_text


def test_three_system_fixture_is_fast_and_offline(tmp_path: Path, monkeypatch) -> None:
    attempted = False

    def fail_socket(*_: object, **__: object) -> None:
        nonlocal attempted
        attempted = True
        raise AssertionError("grounding attempted network access")

    monkeypatch.setattr(socket.socket, "connect", fail_socket)
    monkeypatch.setattr(socket.socket, "connect_ex", fail_socket)
    monkeypatch.setattr(socket, "create_connection", fail_socket)
    text = " ".join(
        f"Sentence {index}: metformin and type 2 diabetes; hemoglobin a1c."
        for index in range(20)
    )
    started = time.perf_counter()
    result = ground(
        text,
        systems=["rxnorm", "loinc", "icd10cm"],
        snapshot=_loader(tmp_path),
    )
    elapsed = time.perf_counter() - started

    assert elapsed < 5.0
    assert attempted is False
    assert any(concept.code == "860975" for concept in result.concepts)
    assert any(concept.code == "E11.9" for concept in result.concepts)
    assert any(concept.code == "4548-4" for concept in result.concepts)
    assert "ground" in openmed.__all__
    assert "GroundingResult" in openmed.__all__


def test_package_contains_no_restricted_terminology_payload() -> None:
    package_root = Path(openmed.__file__).resolve().parent
    restricted_payload_suffixes = {
        ".csv",
        ".json",
        ".jsonl",
        ".obo",
        ".rrf",
        ".tsv",
        ".txt",
        ".xml",
        ".zip",
    }
    restricted_payloads = [
        path.relative_to(package_root)
        for path in package_root.rglob("*")
        if path.is_file()
        and path.suffix.casefold() in restricted_payload_suffixes
        and any(restricted in path.name.casefold() for restricted in ("snomed", "umls"))
    ]
    assert restricted_payloads == []
