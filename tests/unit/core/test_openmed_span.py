"""Tests for the canonical OpenMedSpan record."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from jsonschema import validate

from openmed.core.labels import ID_NUM
from openmed.core.schemas import (
    SCHEMA_NAMES,
    OpenMedSpan,
    build_schema_snapshot,
    compare_schema_drift,
    hmac_text_hash,
    load_all_schemas,
    load_schema,
    load_schema_snapshot,
    schema_aliases,
)


def test_hmac_rejects_empty_explicit_keys(subtests):
    for key in ["", b""]:
        with subtests.test(key=key):
            with pytest.raises(ValueError, match="non-empty"):
                hmac_text_hash("synthetic", key)


def test_hmac_preserves_explicit_utf8_and_byte_keys():
    import hashlib
    import hmac

    key = "synthetic-key-é"
    expected = (
        "hmac-sha256:"
        + hmac.new(key.encode(), b"synthetic", hashlib.sha256).hexdigest()
    )
    assert hmac_text_hash("synthetic", key) == expected
    assert hmac_text_hash(b"synthetic", key.encode()) == expected


def _span() -> OpenMedSpan:
    return OpenMedSpan(
        doc_id="doc-1",
        start=10,
        end=18,
        text_hash=hmac_text_hash("MRN-1234", "test-secret"),
        entity_type="medical_record_number",
        canonical_label=ID_NUM,
        regulatory_tags=("HIPAA",),
        score=0.91,
        detector="safety_sweep",
        evidence={"pattern": "MRN"},
        action="keep",
        replacement=None,
        reversible_id=None,
        section="Assessment",
        metadata={"source": "fixture"},
    )


def test_openmed_span_importable_and_round_trips_through_schema() -> None:
    span = _span()
    payload = span.to_dict()

    validate(payload, load_schema("span"))
    restored = OpenMedSpan.from_json(span.to_json())

    assert restored == span
    assert restored.policy_label == "DIRECT_IDENTIFIER"
    assert restored.regulatory_tags == ("HIPAA",)


def test_invalid_canonical_label_raises_at_construction() -> None:
    with pytest.raises(ValueError, match="canonical_label"):
        OpenMedSpan(
            doc_id="doc-1",
            start=0,
            end=1,
            text_hash=hmac_text_hash("x", "secret"),
            entity_type="made_up",
            canonical_label="NOT_A_CANONICAL_LABEL",
        )


def test_span_is_frozen_and_does_not_store_surface_text() -> None:
    span = _span()

    with pytest.raises(FrozenInstanceError):
        span.doc_id = "other"  # type: ignore[misc]

    serialized = span.to_json()
    assert "MRN-1234" not in serialized
    assert span.text_hash.startswith("hmac-sha256:")


def test_span_validates_offsets_score_action_and_hash() -> None:
    with pytest.raises(ValueError, match="start/end"):
        OpenMedSpan(
            doc_id="doc-1",
            start=3,
            end=1,
            text_hash=hmac_text_hash("x", "secret"),
            entity_type="id_num",
            canonical_label=ID_NUM,
        )
    with pytest.raises(ValueError, match="score"):
        OpenMedSpan(
            doc_id="doc-1",
            start=0,
            end=1,
            text_hash=hmac_text_hash("x", "secret"),
            entity_type="id_num",
            canonical_label=ID_NUM,
            score=2.0,
        )
    with pytest.raises(ValueError, match="action"):
        OpenMedSpan(
            doc_id="doc-1",
            start=0,
            end=1,
            text_hash=hmac_text_hash("x", "secret"),
            entity_type="id_num",
            canonical_label=ID_NUM,
            action="delete",
        )
    with pytest.raises(ValueError, match="text_hash"):
        OpenMedSpan(
            doc_id="doc-1",
            start=0,
            end=1,
            text_hash="cleartext",
            entity_type="id_num",
            canonical_label=ID_NUM,
        )


def test_audit_event_alias_loads_the_same_document_as_audit() -> None:
    event = load_schema("audit_event")

    assert event == load_schema("audit")
    assert event["title"] == "OpenMedAuditEvent"
    assert load_schema("audit_event.schema.json") == event
    assert load_schema("audit_event.schema") == event


def test_schema_aliases_names_both_audit_event_names() -> None:
    aliases = schema_aliases()

    assert aliases == {"audit_event": "audit"}
    assert all(target in SCHEMA_NAMES for target in aliases.values())
    assert not set(aliases) & set(SCHEMA_NAMES)

    aliases["audit_event"] = "span"
    assert schema_aliases() == {"audit_event": "audit"}
    assert load_schema("audit_event")["title"] == "OpenMedAuditEvent"


def test_alias_leaves_bundle_names_and_fingerprint_snapshot_unchanged() -> None:
    schemas = load_all_schemas()

    assert tuple(schemas) == SCHEMA_NAMES
    assert "audit_event" not in schemas
    assert build_schema_snapshot(schemas) == load_schema_snapshot()

    drift = compare_schema_drift("audit_event")
    assert drift.schema_name == "audit"
    assert drift.fingerprint == compare_schema_drift("audit").fingerprint
    assert drift.breaking_change is False


@pytest.mark.parametrize(
    "name",
    ["audit_events", "Audit_Event", "audit-event", "", "event", "audit_event.json"],
)
def test_unknown_schema_names_raise_key_error_with_only_the_name(name: str) -> None:
    with pytest.raises(KeyError) as excinfo:
        load_schema(name)

    assert excinfo.value.args == (f"unknown schema: {name!r}",)
