from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from openmed.interop.omop import (
    OmopMutation,
    OmopMutationBatch,
    VocabularyConcept,
    VocabularySnapshot,
)
from openmed.interop.omop_rollback_manifest import (
    OmopRollbackInstruction,
    OmopRollbackManifestError,
    RollbackStrategy,
    build_omop_rollback_manifest,
    validate_omop_rollback_manifest,
)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _snapshot(version: str = "synthetic-release-current") -> VocabularySnapshot:
    return VocabularySnapshot(
        {"SYNTHETIC": version},
        (VocabularyConcept(101, "SYNTHETIC", standard_concept="S"),),
    )


def _instruction(
    ordinal: int,
    strategy: RollbackStrategy,
    character: str,
) -> OmopRollbackInstruction:
    return OmopRollbackInstruction(ordinal, strategy, _digest(character))


def test_insert_manifest_has_strict_content_free_metadata() -> None:
    private_identifier = "synthetic-private-person-key"
    private_value = "synthetic-private-source-value"
    batch = OmopMutationBatch(
        (
            OmopMutation.insert(
                "person",
                {
                    "person_id": private_identifier,
                    "person_source_value": private_value,
                },
            ),
        )
    )

    manifest = build_omop_rollback_manifest(
        batch,
        _snapshot(),
        (
            _instruction(
                0,
                RollbackStrategy.DELETE_INSERTED_ROW,
                "a",
            ),
        ),
    )

    assert manifest.mutation_count == 1
    assert manifest.operation_counts == (("insert", 1),)
    assert manifest.tables[0].to_dict() == {
        "mutation_count": 1,
        "operation_counts": {"insert": 1},
        "table": "person",
    }
    assert manifest.entries[0].strategy is RollbackStrategy.DELETE_INSERTED_ROW
    assert manifest.entries[0].mutation_digest == batch.mutations[0].row_digest
    manifest.validate(batch, _snapshot())

    rendered = manifest.to_json()
    assert private_identifier not in rendered
    assert private_value not in rendered


def test_update_manifest_requires_before_image_strategy() -> None:
    batch = OmopMutationBatch(
        (
            OmopMutation.update(
                "person",
                {"person_id": "synthetic-private-person-key"},
                {"year_of_birth": 2000},
            ),
        )
    )

    manifest = build_omop_rollback_manifest(
        batch,
        _snapshot(),
        (_instruction(0, RollbackStrategy.RESTORE_BEFORE_IMAGE, "b"),),
    )

    assert manifest.operation_counts == (("update", 1),)
    assert manifest.entries[0].strategy is RollbackStrategy.RESTORE_BEFORE_IMAGE

    with pytest.raises(
        OmopRollbackManifestError,
        match="strategy_operation_mismatch",
    ):
        build_omop_rollback_manifest(
            batch,
            _snapshot(),
            (_instruction(0, RollbackStrategy.DELETE_INSERTED_ROW, "b"),),
        )


def test_mixed_manifest_is_deterministic_and_reverses_mutation_order() -> None:
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
    instructions = (
        _instruction(0, RollbackStrategy.DELETE_INSERTED_ROW, "a"),
        _instruction(1, RollbackStrategy.RESTORE_BEFORE_IMAGE, "b"),
        _instruction(2, RollbackStrategy.REINSERT_TOMBSTONED_ROW, "c"),
    )

    forward = build_omop_rollback_manifest(batch, _snapshot(), instructions)
    shuffled = build_omop_rollback_manifest(
        batch,
        _snapshot(),
        (instructions[2], instructions[0], instructions[1]),
    )

    assert forward == shuffled
    assert forward.to_json() == shuffled.to_json()
    assert [entry.mutation_ordinal for entry in forward.entries] == [2, 1, 0]
    assert [entry.rollback_ordinal for entry in forward.entries] == [0, 1, 2]
    assert [summary.table for summary in forward.tables] == [
        "condition_occurrence",
        "person",
        "visit_occurrence",
    ]
    assert forward.operation_counts == (
        ("insert", 1),
        ("tombstone", 1),
        ("update", 1),
    )


def test_incomplete_and_duplicate_coverage_are_rejected() -> None:
    batch = OmopMutationBatch(
        (
            OmopMutation.insert("person", {"person_id": 101}),
            OmopMutation.insert("person", {"person_id": 202}),
        )
    )
    first = _instruction(0, RollbackStrategy.DELETE_INSERTED_ROW, "a")

    with pytest.raises(OmopRollbackManifestError, match="incomplete_coverage"):
        build_omop_rollback_manifest(batch, _snapshot(), (first,))
    with pytest.raises(
        OmopRollbackManifestError,
        match="duplicate_mutation_coverage",
    ):
        build_omop_rollback_manifest(batch, _snapshot(), (first, first))


def test_unknown_mutation_coverage_is_rejected() -> None:
    batch = OmopMutationBatch((OmopMutation.insert("person", {"person_id": 101}),))

    with pytest.raises(OmopRollbackManifestError, match="unknown_mutation"):
        build_omop_rollback_manifest(
            batch,
            _snapshot(),
            (_instruction(1, RollbackStrategy.DELETE_INSERTED_ROW, "a"),),
        )


def test_vocabulary_snapshot_mismatch_fails_closed() -> None:
    batch = OmopMutationBatch((OmopMutation.insert("person", {"person_id": 101}),))
    manifest = build_omop_rollback_manifest(
        batch,
        _snapshot(),
        (_instruction(0, RollbackStrategy.DELETE_INSERTED_ROW, "a"),),
    )

    with pytest.raises(
        OmopRollbackManifestError,
        match="vocabulary_snapshot_mismatch",
    ):
        validate_omop_rollback_manifest(
            manifest,
            batch,
            _snapshot("synthetic-release-changed"),
        )


def test_batch_mismatch_and_tampered_digest_fail_closed() -> None:
    batch = OmopMutationBatch((OmopMutation.insert("person", {"person_id": 101}),))
    manifest = build_omop_rollback_manifest(
        batch,
        _snapshot(),
        (_instruction(0, RollbackStrategy.DELETE_INSERTED_ROW, "a"),),
    )
    changed_batch = OmopMutationBatch(
        (OmopMutation.insert("person", {"person_id": 202}),)
    )

    with pytest.raises(OmopRollbackManifestError, match="batch_mismatch"):
        manifest.validate(changed_batch, _snapshot())
    with pytest.raises(OmopRollbackManifestError, match="manifest_digest_mismatch"):
        replace(manifest, manifest_digest=_digest("f")).validate(batch, _snapshot())


def test_manifest_omits_sql_connection_strings_and_rollback_values() -> None:
    private_values = (
        "synthetic-patient-identifier",
        "DELETE FROM person WHERE person_id = 'synthetic-patient-identifier'",
        "postgresql://private-user:private-password@localhost/openmed",
        "synthetic-before-image-value",
    )
    batch = OmopMutationBatch(
        (
            OmopMutation.update(
                "person",
                {"person_id": private_values[0]},
                {"person_source_value": private_values[3]},
            ),
        )
    )
    manifest = build_omop_rollback_manifest(
        batch,
        _snapshot(),
        (_instruction(0, RollbackStrategy.RESTORE_BEFORE_IMAGE, "d"),),
    )

    rendered = manifest.to_json()
    for private_value in private_values:
        assert private_value not in rendered
        assert private_value not in repr(manifest)


@pytest.mark.parametrize(
    ("factory", "code"),
    [
        (
            lambda: OmopRollbackInstruction(
                -1,
                RollbackStrategy.DELETE_INSERTED_ROW,
                _digest("a"),
            ),
            "invalid_non_negative_integer",
        ),
        (
            lambda: OmopRollbackInstruction(
                0,
                RollbackStrategy.DELETE_INSERTED_ROW,
                "synthetic-not-a-digest",
            ),
            "invalid_digest",
        ),
        (
            lambda: OmopRollbackInstruction(0, "delete_inserted_row", _digest("a")),
            "unknown_strategy",
        ),
    ],
)
def test_instruction_metadata_is_strict_and_errors_are_value_free(
    factory: Any,
    code: str,
) -> None:
    with pytest.raises(OmopRollbackManifestError) as captured:
        factory()

    assert captured.value.code == code
    assert "synthetic-not-a-digest" not in str(captured.value)
