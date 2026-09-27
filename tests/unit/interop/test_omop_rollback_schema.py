from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from openmed.interop.omop import (
    OmopMutation,
    OmopMutationBatch,
    VocabularyConcept,
    VocabularySnapshot,
)
from openmed.interop.omop.mutation_batch import MAX_BATCH_MUTATIONS, MutationOperation
from openmed.interop.omop_rollback_manifest import (
    OMOP_ROLLBACK_MANIFEST_SCHEMA,
    OmopRollbackInstruction,
    RollbackStrategy,
    build_omop_rollback_manifest,
)
from openmed.interop.omop_rollback_schema import (
    OMOP_ROLLBACK_MANIFEST_JSON_SCHEMA_ID,
    OMOP_ROLLBACK_MANIFEST_TABLES,
    export_omop_rollback_manifest_schema,
    export_omop_rollback_manifest_schema_json,
)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _manifest() -> dict[str, Any]:
    batch = OmopMutationBatch(
        (
            OmopMutation.insert("person", {"person_id": 101}),
            OmopMutation.update(
                "visit_occurrence",
                {"visit_occurrence_id": 202},
                {"visit_source_value": "synthetic-revised"},
            ),
            OmopMutation.tombstone(
                "condition_occurrence",
                {"condition_occurrence_id": 303},
            ),
        )
    )
    snapshot = VocabularySnapshot(
        {"SYNTHETIC": "synthetic-release"},
        (VocabularyConcept(101, "SYNTHETIC", standard_concept="S"),),
    )
    manifest = build_omop_rollback_manifest(
        batch,
        snapshot,
        (
            OmopRollbackInstruction(
                0,
                RollbackStrategy.DELETE_INSERTED_ROW,
                _digest("a"),
            ),
            OmopRollbackInstruction(
                1,
                RollbackStrategy.RESTORE_BEFORE_IMAGE,
                _digest("b"),
            ),
            OmopRollbackInstruction(
                2,
                RollbackStrategy.REINSERT_TOMBSTONED_ROW,
                _digest("c"),
            ),
        ),
    )
    return manifest.to_dict()


def _validator() -> Draft202012Validator:
    schema = export_omop_rollback_manifest_schema()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _refs(value: Any) -> list[str]:
    if isinstance(value, dict):
        refs = [value["$ref"]] if "$ref" in value else []
        for nested in value.values():
            refs.extend(_refs(nested))
        return refs
    if isinstance(value, list):
        return [ref for nested in value for ref in _refs(nested)]
    return []


def test_schema_is_draft_2020_12_and_resolves_entirely_locally() -> None:
    schema = export_omop_rollback_manifest_schema()

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"] == OMOP_ROLLBACK_MANIFEST_JSON_SCHEMA_ID
    assert _refs(schema)
    assert all(ref.startswith("#/$defs/") for ref in _refs(schema))
    _validator().validate(_manifest())


def test_schema_export_is_byte_stable_and_has_a_drift_guard() -> None:
    first = export_omop_rollback_manifest_schema_json()
    second = export_omop_rollback_manifest_schema_json()

    assert first == second
    assert json.loads(first) == export_omop_rollback_manifest_schema()
    assert hashlib.sha256(first.encode("utf-8")).hexdigest() == (
        "c414ac74534673a38b29548d0a54fd724fdc6b9cc29f111313353c37a16a0017"
    )


def test_schema_tracks_runtime_enums_and_count_bounds() -> None:
    schema = export_omop_rollback_manifest_schema()
    definitions = schema["$defs"]

    assert definitions["entry"]["properties"]["operation"]["enum"] == [
        operation.value for operation in MutationOperation
    ]
    assert definitions["entry"]["properties"]["strategy"]["enum"] == [
        strategy.value for strategy in RollbackStrategy
    ]
    assert definitions["table"]["enum"] == list(OMOP_ROLLBACK_MANIFEST_TABLES)
    assert definitions["positiveCount"]["maximum"] == MAX_BATCH_MUTATIONS
    assert definitions["ordinal"]["maximum"] == MAX_BATCH_MUTATIONS - 1
    assert schema["properties"]["schema"] == {"const": OMOP_ROLLBACK_MANIFEST_SCHEMA}


@pytest.mark.parametrize("field", ["entries", "tables", "operation_counts"])
def test_schema_rejects_incomplete_manifests(field: str) -> None:
    payload = _manifest()
    payload.pop(field)

    assert not _validator().is_valid(payload)


@pytest.mark.parametrize("field", ["entries", "tables"])
def test_schema_rejects_empty_manifest_coverage(field: str) -> None:
    payload = _manifest()
    payload[field] = []

    assert not _validator().is_valid(payload)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("entries", 0, "row"), {"person_id": "synthetic-private-key"}),
        (("entries", 0, "sql"), "DELETE FROM person"),
        (("entries", 0, "mutation_digest"), "not-a-digest"),
        (("entries", 0, "table"), "custom_patient_table"),
        (("entries", 0, "operation"), "replace"),
        (("mutation_count",), 0),
        (("vocabulary_snapshot_digest",), _digest("A")),
    ],
)
def test_schema_rejects_row_sql_and_malformed_content(
    path: tuple[str | int, ...],
    value: Any,
) -> None:
    payload = _manifest()
    target: Any = payload
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value

    assert not _validator().is_valid(payload)


def test_schema_binds_each_operation_to_its_rollback_strategy() -> None:
    payload = _manifest()
    payload["entries"][0]["strategy"] = RollbackStrategy.DELETE_INSERTED_ROW.value

    assert payload["entries"][0]["operation"] == MutationOperation.TOMBSTONE.value
    assert not _validator().is_valid(payload)


def test_every_schema_object_is_closed() -> None:
    schema = export_omop_rollback_manifest_schema()

    assert schema["additionalProperties"] is False
    assert schema["$defs"]["entry"]["additionalProperties"] is False
    assert schema["$defs"]["operationCounts"]["additionalProperties"] is False
    assert schema["$defs"]["tableSummary"]["additionalProperties"] is False
