"""Contract tests for immutable longitudinal Journey records."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
from jsonschema.validators import validator_for

from openmed.clinical.journey_contracts import (
    JOURNEY_SCHEMA_NAMES,
    ClinicalArtifact,
    ClinicalFact,
    ConflictSet,
    DatasetSnapshot,
    EvidenceLocator,
    JourneyContractError,
    JourneyContractGraphError,
    JourneySchemaVersionError,
    ResolutionEvent,
    canonical_json,
    compute_derivation_hash,
    derived_opaque_id,
    load_all_journey_schemas,
    new_opaque_id,
    sha256_digest,
    validate_contract_graph,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "structured"
    / "journey_contracts.json"
)


def _fixtures() -> dict[str, dict[str, object]]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _records() -> tuple[
    ClinicalArtifact,
    EvidenceLocator,
    ClinicalFact,
    ClinicalFact,
    ConflictSet,
    ResolutionEvent,
    DatasetSnapshot,
]:
    fixture = _fixtures()
    artifact = ClinicalArtifact.from_dict(fixture["clinical_artifact"])
    evidence = EvidenceLocator.from_dict(fixture["evidence_locator"])
    first_fact = ClinicalFact.from_dict(fixture["clinical_fact"])
    second_fact = replace(
        first_fact,
        fact_id="fact_bbbbbbbbbbbbbbbb",
        value={"code": "synthetic-condition", "state": "inactive"},
        status="inactive",
        parent_fact_ids=(first_fact.fact_id,),
    )
    conflict = ConflictSet.from_dict(fixture["conflict_set"])
    resolution = ResolutionEvent.from_dict(fixture["resolution_event"])
    snapshot = ClinicalArtifact.from_dict(fixture["clinical_artifact"])
    dataset = DatasetSnapshot.from_dict(fixture["dataset_snapshot"])
    assert snapshot == artifact
    return artifact, evidence, first_fact, second_fact, conflict, resolution, dataset


def test_all_contracts_round_trip_and_validate_against_bundled_schemas() -> None:
    fixtures = _fixtures()
    classes = {
        "clinical_artifact": ClinicalArtifact,
        "evidence_locator": EvidenceLocator,
        "clinical_fact": ClinicalFact,
        "conflict_set": ConflictSet,
        "resolution_event": ResolutionEvent,
        "dataset_snapshot": DatasetSnapshot,
    }
    schemas = load_all_journey_schemas()

    assert set(schemas) == set(JOURNEY_SCHEMA_NAMES)
    for name, record_type in classes.items():
        schema = schemas[name]
        validator = validator_for(schema)
        validator.check_schema(schema)
        record = record_type.from_dict(fixtures[name])
        restored = record_type.from_json(record.to_json())

        assert restored == record
        assert restored.to_dict() == fixtures[name]
        assert not tuple(validator(schema).iter_errors(record.to_dict()))
        assert record.canonical_hash.startswith("sha256:")


def test_canonical_serialization_is_byte_stable_across_fresh_processes() -> None:
    payload = _fixtures()["clinical_fact"]
    script = (
        "import json; "
        "from openmed.clinical.journey_contracts import ClinicalFact; "
        f"p=json.loads({json.dumps(json.dumps(payload))}); "
        "print(ClinicalFact.from_dict(p).to_json())"
    )

    first = subprocess.check_output([sys.executable, "-c", script], text=True)
    second = subprocess.check_output([sys.executable, "-c", script], text=True)

    assert first == second
    assert first.strip() == ClinicalFact.from_dict(payload).to_json()


def test_supported_additive_fields_survive_round_trip() -> None:
    payload = _fixtures()["clinical_artifact"] | {
        "schema_version": "1.7.3",
        "future_review_state": {"code": "synthetic-state", "rank": 2},
    }

    record = ClinicalArtifact.from_dict(payload)
    restored = ClinicalArtifact.from_json(record.to_json())

    assert restored.schema_version == "1.7.3"
    assert restored.to_dict()["future_review_state"] == {
        "code": "synthetic-state",
        "rank": 2,
    }


def test_unsupported_major_version_fails_without_echoing_payload() -> None:
    canary = "SYNTHETIC-VERSION-CANARY"
    payload = _fixtures()["clinical_artifact"] | {
        "schema_version": "2.0.0",
        "future_value": canary,
    }

    with pytest.raises(JourneySchemaVersionError) as exc_info:
        ClinicalArtifact.from_dict(payload)

    assert canary not in str(exc_info.value)


def test_contracts_are_deeply_immutable() -> None:
    artifact, _, fact, *_ = _records()

    with pytest.raises(FrozenInstanceError):
        artifact.byte_size = 1  # type: ignore[misc]
    with pytest.raises(TypeError):
        artifact.attributes["language"] = "fr"  # type: ignore[index]
    with pytest.raises(TypeError):
        fact.value["code"] = "changed"  # type: ignore[index]


@pytest.mark.parametrize(
    ("location_type", "location"),
    [
        ("text_span", {"start": 4, "end": 18}),
        ("json_pointer", {"pointer": "/entry/0/resource"}),
        ("message_field", {"path": "PID.3.1"}),
        (
            "page_box",
            {
                "page": 1,
                "box": [0.1, 0.2, 0.6, 0.8],
                "coordinate_space": "normalized",
            },
        ),
        (
            "dicom_element",
            {
                "study_uid": "1.2.3",
                "series_uid": "1.2.3.4",
                "instance_uid": "1.2.3.4.5",
                "tag": "0040,A160",
            },
        ),
        ("table_cell", {"sheet": "Labs", "row": 2, "column": 3}),
    ],
)
def test_every_evidence_coordinate_kind_is_supported(
    location_type: str,
    location: dict[str, object],
) -> None:
    locator = EvidenceLocator(
        locator_id=derived_opaque_id("evidence", location_type, location),
        artifact_id="artifact_aaaaaaaaaaaaaaaa",
        location_type=location_type,
        location=location,
    )

    assert locator.location_type == location_type
    assert EvidenceLocator.from_dict(locator.to_dict()) == locator


@pytest.mark.parametrize(
    ("location_type", "location"),
    [
        ("text_span", {"start": 4, "end": 4}),
        ("json_pointer", {"pointer": "/entry/~2"}),
        ("message_field", {"path": "PID"}),
        (
            "page_box",
            {"page": 1, "box": [0.5, 0.2, 0.4, 0.8], "coordinate_space": "normalized"},
        ),
        (
            "page_box",
            {"page": 1, "box": [0.1, 0.2, 1.2, 0.8], "coordinate_space": "normalized"},
        ),
        (
            "dicom_element",
            {
                "study_uid": "invalid",
                "series_uid": "1.2",
                "instance_uid": "1.2.3",
                "tag": "0040,A160",
            },
        ),
        ("table_cell", {"row": 0, "column": 1}),
    ],
)
def test_invalid_evidence_coordinates_are_rejected(
    location_type: str,
    location: dict[str, object],
) -> None:
    with pytest.raises(JourneyContractError):
        EvidenceLocator(
            locator_id="evidence_aaaaaaaaaaaaaaaa",
            artifact_id="artifact_aaaaaaaaaaaaaaaa",
            location_type=location_type,
            location=location,
        )


def test_valid_contract_graph_returns_only_value_free_counts() -> None:
    artifact, evidence, first, second, conflict, resolution, snapshot = _records()

    result = validate_contract_graph(
        artifacts=(artifact,),
        evidence=(evidence,),
        facts=(first, second),
        conflicts=(conflict,),
        resolutions=(resolution,),
        snapshots=(snapshot,),
    )

    assert result.to_dict() == {
        "artifact_count": 1,
        "conflict_count": 1,
        "edge_count": 12,
        "evidence_count": 1,
        "fact_count": 2,
        "node_count": 7,
        "resolution_count": 1,
        "snapshot_count": 1,
        "valid": True,
    }
    assert "synthetic-condition" not in json.dumps(result.to_dict())


def test_contract_graph_rejects_missing_references_and_cycles() -> None:
    artifact, evidence, first, second, conflict, resolution, snapshot = _records()
    missing_parent = replace(
        second,
        parent_fact_ids=("fact_missingmissingmiss",),
    )

    with pytest.raises(JourneyContractGraphError, match="fact parent reference"):
        validate_contract_graph(
            artifacts=(artifact,),
            evidence=(evidence,),
            facts=(first, missing_parent),
            conflicts=(conflict,),
            resolutions=(resolution,),
            snapshots=(snapshot,),
        )

    cyclic_first = replace(first, parent_fact_ids=(second.fact_id,))
    with pytest.raises(JourneyContractGraphError, match="fact derivation graph"):
        validate_contract_graph(
            artifacts=(artifact,),
            evidence=(evidence,),
            facts=(cyclic_first, second),
            conflicts=(conflict,),
            resolutions=(resolution,),
            snapshots=(snapshot,),
        )


def test_contract_graph_rejects_cross_subject_conflicts() -> None:
    artifact, evidence, first, second, conflict, resolution, snapshot = _records()
    other_subject = replace(second, subject_id="subject_bbbbbbbbbbbbbbbb")

    with pytest.raises(JourneyContractGraphError, match="conflict subject"):
        validate_contract_graph(
            artifacts=(artifact,),
            evidence=(evidence,),
            facts=(first, other_subject),
            conflicts=(conflict,),
            resolutions=(resolution,),
            snapshots=(snapshot,),
        )


def test_evidence_schema_rejects_location_type_shape_mismatch() -> None:
    schema = load_all_journey_schemas()["evidence_locator"]
    validator = validator_for(schema)(schema)
    payload = _fixtures()["evidence_locator"] | {
        "location_type": "table_cell",
        "location": {"start": 4, "end": 18},
    }

    assert tuple(validator.iter_errors(payload))


def test_digest_and_identifier_helpers_are_deterministic_and_value_free() -> None:
    first = compute_derivation_hash(
        "journey.extract",
        "1.0.0",
        ("artifact_aaaaaaaaaaaaaaaa", "artifact_bbbbbbbbbbbbbbbb"),
        configuration={"labels": ["condition", "medication"]},
    )
    second = compute_derivation_hash(
        "journey.extract",
        "1.0.0",
        ("artifact_bbbbbbbbbbbbbbbb", "artifact_aaaaaaaaaaaaaaaa"),
        configuration={"labels": ["condition", "medication"]},
    )

    assert first == second
    assert first == sha256_digest(
        canonical_json(
            {
                "component": "journey.extract",
                "component_version": "1.0.0",
                "configuration": {"labels": ["condition", "medication"]},
                "input_ids": [
                    "artifact_aaaaaaaaaaaaaaaa",
                    "artifact_bbbbbbbbbbbbbbbb",
                ],
                "schema_version": "1.0.0",
            }
        )
    )
    assert derived_opaque_id("fact", "synthetic-input") == derived_opaque_id(
        "fact", "synthetic-input"
    )
    assert new_opaque_id("artifact").startswith("artifact_")

    with pytest.raises(JourneyContractError, match="unique"):
        compute_derivation_hash(
            "journey.extract",
            "1.0.0",
            ("artifact_aaaaaaaaaaaaaaaa", "artifact_aaaaaaaaaaaaaaaa"),
        )


def test_dataset_snapshot_rejects_unsafe_paths() -> None:
    payload = _fixtures()["dataset_snapshot"] | {
        "file_hashes": {
            "../escape.jsonl": "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
        }
    }

    with pytest.raises(JourneyContractError, match="safe relative paths"):
        DatasetSnapshot.from_dict(payload)
