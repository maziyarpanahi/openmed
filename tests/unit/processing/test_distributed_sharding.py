"""Tests for deterministic distributed batch sharding."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from openmed.processing import (
    DuplicateDocumentIDError,
    MissingDocumentIDError,
    plan_document_shards,
)


@dataclass(frozen=True)
class ClinicalDocument:
    document_id: str
    text: str
    external_key: str


def _documents() -> list[dict[str, str]]:
    return [
        {
            "id": "note-001",
            "text": "Patient Jane Doe called from 555-0100.",
        },
        {
            "id": "note-002",
            "text": "Patient John Roe emailed john.roe@example.org.",
        },
        {
            "id": "note-003",
            "text": "Patient Alex Kim lives at 12 Oak Street.",
        },
        {
            "id": "note-004",
            "text": "Patient Maria Lee has diabetes.",
        },
    ]


def test_shard_plan_is_independent_of_input_order() -> None:
    docs = _documents()

    plan = plan_document_shards(docs, shard_count=7)
    reversed_plan = plan_document_shards(list(reversed(docs)), shard_count=7)

    assert plan.document_to_shard() == reversed_plan.document_to_shard()
    assert [shard.document_ids for shard in plan.shards] == [
        shard.document_ids for shard in reversed_plan.shards
    ]
    assert plan.fingerprint == reversed_plan.fingerprint


def test_worker_count_does_not_change_membership_for_fixed_shard_count() -> None:
    docs = _documents()

    single_worker = plan_document_shards(docs, shard_count=8, worker_count=1)
    eight_workers = plan_document_shards(docs, shard_count=8, worker_count=8)

    assert single_worker.document_to_shard() == eight_workers.document_to_shard()
    assert single_worker.fingerprint == eight_workers.fingerprint


def test_duplicate_document_ids_are_rejected_without_document_text() -> None:
    docs = [
        {"id": "same-id", "text": "Patient Jane Doe."},
        {"id": "same-id", "text": "Patient John Roe."},
    ]

    with pytest.raises(DuplicateDocumentIDError) as exc_info:
        plan_document_shards(docs, shard_count=2)

    message = str(exc_info.value)
    assert "Duplicate document id" in message
    assert "Jane Doe" not in message
    assert "John Roe" not in message


def test_default_extraction_supports_objects_with_document_id_field() -> None:
    docs = [
        ClinicalDocument(
            document_id="obj-001",
            text="Patient Jane Doe.",
            external_key="source-a",
        ),
        ClinicalDocument(
            document_id="obj-002",
            text="Patient John Roe.",
            external_key="source-b",
        ),
    ]

    plan = plan_document_shards(docs, shard_count=4)

    assert set(plan.document_to_shard()) == {"obj-001", "obj-002"}


def test_custom_id_extractor_overrides_default_fields() -> None:
    docs = [
        ClinicalDocument(
            document_id="internal-001",
            text="Patient Jane Doe.",
            external_key="external-a",
        ),
        ClinicalDocument(
            document_id="internal-002",
            text="Patient John Roe.",
            external_key="external-b",
        ),
    ]

    plan = plan_document_shards(
        docs,
        shard_count=4,
        id_extractor=lambda document: document.external_key,
    )

    assert set(plan.document_to_shard()) == {"external-a", "external-b"}


def test_missing_document_id_error_does_not_include_document_text() -> None:
    docs = [{"text": "Patient Jane Doe called from 555-0100."}]

    with pytest.raises(MissingDocumentIDError) as exc_info:
        plan_document_shards(docs, shard_count=2)

    message = str(exc_info.value)
    assert "Document at index 0" in message
    assert "Jane Doe" not in message
    assert "555-0100" not in message


def test_plan_metadata_excludes_document_text_by_default() -> None:
    docs = _documents()

    plan = plan_document_shards(docs, shard_count=3)
    rendered = json.dumps(plan.to_dict(), sort_keys=True)

    assert plan.document_count == len(docs)
    assert "document_hashes" in rendered
    assert "document_ids" not in rendered
    assert "Jane Doe" not in rendered
    assert "555-0100" not in rendered
    assert "john.roe@example.org" not in rendered
    assert "12 Oak Street" not in rendered


def test_plan_can_emit_document_ids_when_requested_without_document_text() -> None:
    docs = _documents()

    plan = plan_document_shards(docs, shard_count=3)
    rendered = json.dumps(plan.to_dict(include_document_ids=True), sort_keys=True)

    assert "note-001" in rendered
    assert "note-004" in rendered
    assert "Jane Doe" not in rendered
    assert "john.roe@example.org" not in rendered
