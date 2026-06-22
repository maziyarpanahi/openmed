"""Tests for the canonical OpenMedSpan record."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from jsonschema import validate

from openmed.core.labels import ID_NUM
from openmed.core.schemas import OpenMedSpan, hmac_text_hash, load_schema


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
