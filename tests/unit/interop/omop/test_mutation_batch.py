from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from openmed.interop.omop import OmopMutationBatch as PublicOmopMutationBatch
from openmed.interop.omop.mutation_batch import (
    CommitStatus,
    MutationOperation,
    OmopApprovalBinding,
    OmopMutation,
    OmopMutationBatch,
    OmopMutationError,
    OmopRowKey,
)

RECEIPT_DIGEST = "sha256:" + "a" * 64


def test_batch_is_exported_from_omop_package() -> None:
    assert PublicOmopMutationBatch is OmopMutationBatch


def _valid_batch() -> tuple[OmopMutationBatch, tuple[OmopRowKey, ...]]:
    existing = (OmopRowKey("person", {"person_id": 101}),)
    mutations = (
        OmopMutation.insert(
            "visit_occurrence",
            {
                "visit_occurrence_id": 201,
                "person_id": 101,
                "visit_source_value": "synthetic-visit-source",
            },
        ),
        OmopMutation.insert(
            "condition_occurrence",
            {
                "condition_occurrence_id": 301,
                "person_id": 101,
                "visit_occurrence_id": 201,
                "condition_concept_id": 0,
                "condition_source_value": "synthetic-condition-source",
            },
        ),
        OmopMutation.update(
            "condition_occurrence",
            {"condition_occurrence_id": 301},
            {"condition_source_value": "synthetic-revised-source"},
        ),
        OmopMutation.tombstone(
            "condition_occurrence",
            {"condition_occurrence_id": 301},
        ),
    )
    return OmopMutationBatch(mutations), existing


def test_preview_is_deterministic_ordered_and_value_free() -> None:
    batch, existing = _valid_batch()

    first = batch.preview(existing_rows=existing)
    second = batch.preview(existing_rows=existing)

    assert first == second
    assert first.is_valid
    assert first.batch_digest == batch.batch_digest
    assert first.operation_counts == (("insert", 2), ("tombstone", 1), ("update", 1))
    assert [item.operation for item in first.mutations] == [
        MutationOperation.INSERT,
        MutationOperation.INSERT,
        MutationOperation.UPDATE,
        MutationOperation.TOMBSTONE,
    ]
    rendered = first.to_json()
    for source_value in (
        "synthetic-visit-source",
        "synthetic-condition-source",
        "synthetic-revised-source",
    ):
        assert source_value not in rendered
        assert source_value not in repr(batch)
    assert "101" not in rendered
    assert "201" not in rendered
    assert "301" not in rendered


def test_mutation_and_row_key_reprs_hide_values() -> None:
    mutation = OmopMutation.insert(
        "custom_table",
        {"opaque_id": "synthetic-private-key", "source_value": "synthetic-private"},
        key={"opaque_id": "synthetic-private-key"},
    )

    assert "synthetic-private" not in repr(mutation)
    assert "synthetic-private-key" not in repr(mutation.key)
    assert mutation.values == {
        "opaque_id": "synthetic-private-key",
        "source_value": "synthetic-private",
    }


def test_row_and_batch_digests_change_with_order_or_values() -> None:
    first = OmopMutation.insert("person", {"person_id": 1})
    second = OmopMutation.insert("person", {"person_id": 2})
    changed = OmopMutation.insert("person", {"person_id": 3})

    assert first.row_digest != changed.row_digest
    assert (
        OmopMutationBatch((first, second)).batch_digest
        != OmopMutationBatch((second, first)).batch_digest
    )


def test_referential_checks_report_only_safe_metadata() -> None:
    private_reference = "synthetic-missing-person"
    batch = OmopMutationBatch(
        (
            OmopMutation.insert(
                "visit_occurrence",
                {
                    "visit_occurrence_id": 201,
                    "person_id": private_reference,
                },
            ),
        )
    )

    preview = batch.preview()

    assert not preview.is_valid
    assert [issue.code for issue in preview.issues] == ["missing_reference"]
    assert preview.issues[0].field_name == "person_id"
    assert private_reference not in preview.to_json()


def test_ordered_checks_reject_child_before_parent() -> None:
    batch = OmopMutationBatch(
        (
            OmopMutation.insert(
                "visit_occurrence",
                {"visit_occurrence_id": 201, "person_id": 101},
            ),
            OmopMutation.insert("person", {"person_id": 101}),
        )
    )

    preview = batch.preview()

    assert [issue.code for issue in preview.issues] == ["missing_reference"]


def test_tombstone_rejects_a_parent_referenced_by_staged_row() -> None:
    batch = OmopMutationBatch(
        (
            OmopMutation.insert("person", {"person_id": 101}),
            OmopMutation.insert(
                "visit_occurrence",
                {"visit_occurrence_id": 201, "person_id": 101},
            ),
            OmopMutation.update(
                "visit_occurrence",
                {"visit_occurrence_id": 201},
                {"visit_source_value": "synthetic-source"},
            ),
            OmopMutation.tombstone("person", {"person_id": 101}),
        )
    )

    preview = batch.preview()

    assert [issue.code for issue in preview.issues] == ["referenced_tombstone"]


def test_updates_and_tombstones_require_a_live_target() -> None:
    update = OmopMutation.update("person", {"person_id": 101}, {"year_of_birth": 2000})
    tombstone = OmopMutation.tombstone("person", {"person_id": 102})

    preview = OmopMutationBatch((update, tombstone)).preview()

    assert [issue.code for issue in preview.issues] == [
        "missing_target",
        "missing_target",
    ]


def test_invalid_preview_cannot_be_bound_for_approval() -> None:
    batch = OmopMutationBatch(
        (
            OmopMutation.insert(
                "visit_occurrence",
                {"visit_occurrence_id": 201, "person_id": 101},
            ),
        )
    )
    preview = batch.preview()

    with pytest.raises(OmopMutationError, match="reference_check_failed"):
        batch.bind_approval(
            preview,
            approved_preview_digest=preview.preview_digest,
            approval_receipt_digest=RECEIPT_DIGEST,
        )


def test_approval_is_bound_to_exact_preview_and_batch() -> None:
    batch, existing = _valid_batch()
    preview = batch.preview(existing_rows=existing)
    changed_batch = OmopMutationBatch(
        (
            *batch.mutations,
            OmopMutation.insert("person", {"person_id": 999}),
        )
    )

    with pytest.raises(OmopMutationError, match="preview_changed"):
        batch.bind_approval(
            preview,
            approved_preview_digest="sha256:" + "b" * 64,
            approval_receipt_digest=RECEIPT_DIGEST,
        )
    with pytest.raises(OmopMutationError, match="batch_changed"):
        changed_batch.bind_approval(
            preview,
            approved_preview_digest=preview.preview_digest,
            approval_receipt_digest=RECEIPT_DIGEST,
        )


def test_approval_rejects_tampered_preview_metadata() -> None:
    batch, existing = _valid_batch()
    preview = batch.preview(existing_rows=existing)
    tampered = replace(preview, mutations=preview.mutations[:-1])

    with pytest.raises(OmopMutationError, match="invalid_preview"):
        batch.bind_approval(
            tampered,
            approved_preview_digest=preview.preview_digest,
            approval_receipt_digest=RECEIPT_DIGEST,
        )


class _RecordingCommitter:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def commit_batch(
        self,
        mutations: tuple[OmopMutation, ...],
        *,
        batch_digest: str,
        approval: OmopApprovalBinding,
    ) -> None:
        self.calls.append(
            {
                "mutations": mutations,
                "batch_digest": batch_digest,
                "approval": approval,
            }
        )


def test_commit_returns_explicit_result_and_preserves_order() -> None:
    batch, existing = _valid_batch()
    preview = batch.preview(existing_rows=existing)
    approval = batch.bind_approval(
        preview,
        approved_preview_digest=preview.preview_digest,
        approval_receipt_digest=RECEIPT_DIGEST,
    )
    committer = _RecordingCommitter()

    result = batch.commit(committer, approval=approval)

    assert result.status is CommitStatus.COMMITTED
    assert result.error_code is None
    assert result.mutation_count == len(batch.mutations)
    assert result.to_dict()["status"] == "committed"
    assert committer.calls == [
        {
            "mutations": batch.mutations,
            "batch_digest": batch.batch_digest,
            "approval": approval,
        }
    ]


def test_commit_failure_does_not_expose_adapter_exception() -> None:
    private_error = "synthetic-private-adapter-detail"

    class FailingCommitter:
        def commit_batch(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(private_error)

    batch, existing = _valid_batch()
    preview = batch.preview(existing_rows=existing)
    approval = batch.bind_approval(
        preview,
        approved_preview_digest=preview.preview_digest,
        approval_receipt_digest=RECEIPT_DIGEST,
    )

    result = batch.commit(FailingCommitter(), approval=approval)

    assert result.status is CommitStatus.FAILED
    assert result.error_code == "committer_error"
    assert result.mutation_count == 0
    assert private_error not in str(result.to_dict())


@pytest.mark.parametrize(
    ("factory", "code"),
    [
        (lambda: OmopMutationBatch(()), "empty_batch"),
        (
            lambda: OmopMutation.insert("custom_table", {"id": 1}),
            "key_required",
        ),
        (
            lambda: OmopMutation.update("person", {"person_id": 1}, {}),
            "empty_update",
        ),
        (
            lambda: OmopMutation.update("person", {"person_id": 1}, {"person_id": 2}),
            "key_field_changed",
        ),
        (
            lambda: OmopMutation.insert(
                "custom_table",
                {"row_id": 1},
                key={"row_id": 2},
            ),
            "key_mismatch",
        ),
        (
            lambda: OmopMutation.update(
                "person",
                {"not_person_id": 1},
                {"year_of_birth": 2000},
            ),
            "invalid_primary_key",
        ),
    ],
)
def test_invalid_mutations_fail_with_closed_value_free_codes(
    factory: Any,
    code: str,
) -> None:
    with pytest.raises(OmopMutationError) as exc_info:
        factory()

    assert exc_info.value.code == code
    assert "synthetic" not in str(exc_info.value)
