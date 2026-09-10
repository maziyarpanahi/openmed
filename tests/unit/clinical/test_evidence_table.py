"""Tests for value-free clinical evidence table rendering."""

from __future__ import annotations

import json
import math
import socket
import traceback

import pytest

from openmed.clinical.evidence_table import (
    EVIDENCE_TABLE_DISCLAIMER,
    EVIDENCE_TABLE_SCHEMA_VERSION,
    AssertionStatus,
    EvidenceRecord,
    EvidenceTable,
)

RAW_MARKER = "raw-phi-marker"


class LeakyInt(int):
    def __repr__(self) -> str:
        return RAW_MARKER

    def __format__(self, format_spec: str) -> str:
        return RAW_MARKER


class LeakyFloat(float):
    def __float__(self) -> float:
        raise RuntimeError(RAW_MARKER)


class LeakyStr(str):
    def __repr__(self) -> str:
        return RAW_MARKER

    def __format__(self, format_spec: str) -> str:
        return RAW_MARKER

    def encode(self, encoding: str = "utf-8", errors: str = "strict") -> bytes:
        raise RuntimeError(RAW_MARKER)


class ExplodingStr(str):
    def __str__(self) -> str:
        raise RuntimeError(RAW_MARKER)

    def __hash__(self) -> int:
        raise RuntimeError(RAW_MARKER)


def record(
    start: int,
    end: int,
    assertion: AssertionStatus = AssertionStatus.AFFIRMED,
    confidence: float = 0.9,
    review_required: bool = False,
) -> EvidenceRecord:
    return EvidenceRecord.from_extraction(
        source_start=start,
        source_end=end,
        assertion_status=assertion,
        confidence=confidence,
        review_required=review_required,
    )


def test_table_sorts_records_and_counts_review_state() -> None:
    table = EvidenceTable(
        records=(
            record(20, 30, AssertionStatus.NEGATED, 0.8, True),
            record(4, 9, AssertionStatus.AFFIRMED, 0.95),
            record(20, 25, AssertionStatus.UNCERTAIN, 0.6, True),
        )
    )

    payload = table.to_dict()

    assert [item["source_offsets"] for item in payload["records"]] == [
        {"start": 4, "end": 9},
        {"start": 20, "end": 25},
        {"start": 20, "end": 30},
    ]
    assert payload["record_count"] == 3
    assert payload["review_required_count"] == 2
    assert payload["assertion_counts"] == {
        "affirmed": 1,
        "negated": 1,
        "uncertain": 1,
    }


def test_json_is_compact_and_deterministic() -> None:
    table = EvidenceTable(records=(record(3, 7),))

    expected = {
        "schema_version": 1,
        "record_count": 1,
        "review_required_count": 0,
        "assertion_counts": {"affirmed": 1},
        "records": [
            {
                "source_offsets": {"start": 3, "end": 7},
                "assertion_status": "affirmed",
                "confidence": 0.9,
                "review_required": False,
            }
        ],
        "disclaimer": (
            "Clinical evidence tables are value-free review aids, not clinical "
            "decisions or compliance certifications."
        ),
    }

    assert json.loads(table.to_json()) == expected
    assert table.to_json() == table.to_json()
    assert '": ' not in table.to_json()
    assert ', "' not in table.to_json()


def test_json_uses_canonical_key_order_and_public_schema_constants() -> None:
    table = EvidenceTable(records=(record(3, 7),))

    assert EVIDENCE_TABLE_SCHEMA_VERSION == 1
    assert "value-free review aids" in EVIDENCE_TABLE_DISCLAIMER
    assert table.to_json().startswith(
        '{"assertion_counts":{"affirmed":1},"disclaimer":'
    )


def test_markdown_contains_offsets_and_review_flags_without_values() -> None:
    table = EvidenceTable(
        records=(
            record(1, 5, review_required=True),
            record(8, 13, AssertionStatus.HISTORICAL, 0.75),
        )
    )

    markdown = table.to_markdown()

    assert "| 1 | 1 | 5 | affirmed | 0.900000 | yes | omitted |" in markdown
    assert "| 2 | 8 | 13 | historical | 0.750000 | no | omitted |" in markdown
    assert "Records: 2" in markdown
    assert "Review required: 1" in markdown


def test_protected_value_hashing_is_opt_in_and_never_stores_raw_value() -> None:
    protected_value = "synthetic-sensitive-finding"

    omitted = EvidenceRecord.from_extraction(
        source_start=1,
        source_end=5,
        assertion_status=AssertionStatus.AFFIRMED,
        confidence=0.9,
        review_required=False,
        protected_value=protected_value,
    )
    hashed = EvidenceRecord.from_extraction(
        source_start=1,
        source_end=5,
        assertion_status=AssertionStatus.AFFIRMED,
        confidence=0.9,
        review_required=False,
        protected_value=protected_value,
        include_value_hash=True,
    )

    assert omitted.value_hash is None
    assert hashed.value_hash is not None
    assert hashed.value_hash.startswith("sha256:")
    rendered = EvidenceTable(records=(omitted, hashed)).to_json()
    assert protected_value not in rendered
    assert protected_value not in repr(omitted)
    assert protected_value not in repr(hashed)


def test_hashing_requires_a_protected_value_without_echoing_other_fields() -> None:
    with pytest.raises(ValueError, match="protected value is required"):
        EvidenceRecord.from_extraction(
            source_start=1,
            source_end=5,
            assertion_status=AssertionStatus.AFFIRMED,
            confidence=0.9,
            review_required=False,
            include_value_hash=True,
        )


@pytest.mark.parametrize(
    ("start", "end"),
    ((-1, 2), (2, 2), (4, 3), (True, 2), (1, False)),
)
def test_invalid_offsets_fail_closed(start: object, end: object) -> None:
    with pytest.raises((TypeError, ValueError), match="source offsets"):
        EvidenceRecord.from_extraction(
            source_start=start,
            source_end=end,
            assertion_status=AssertionStatus.AFFIRMED,
            confidence=0.9,
            review_required=False,
        )


@pytest.mark.parametrize(
    "confidence",
    (-0.01, 1.01, math.nan, math.inf, True, "secret-confidence"),
)
def test_invalid_confidence_fails_without_echoing_value(confidence: object) -> None:
    with pytest.raises((TypeError, ValueError)) as exc_info:
        EvidenceRecord.from_extraction(
            source_start=1,
            source_end=5,
            assertion_status=AssertionStatus.AFFIRMED,
            confidence=confidence,
            review_required=False,
        )

    assert "secret-confidence" not in str(exc_info.value)


def test_unknown_assertion_fails_without_echoing_value() -> None:
    with pytest.raises(ValueError) as exc_info:
        EvidenceRecord.from_extraction(
            source_start=1,
            source_end=5,
            assertion_status="secret-assertion",
            confidence=0.9,
            review_required=False,
        )

    assert "secret-assertion" not in str(exc_info.value)


def test_review_required_must_be_boolean() -> None:
    with pytest.raises(TypeError, match="review_required must be a boolean"):
        EvidenceRecord.from_extraction(
            source_start=1,
            source_end=5,
            assertion_status=AssertionStatus.AFFIRMED,
            confidence=0.9,
            review_required=1,
        )


def test_direct_value_hash_must_have_the_stable_digest_shape() -> None:
    with pytest.raises(ValueError, match="value_hash is invalid"):
        EvidenceRecord(
            source_start=1,
            source_end=5,
            assertion_status=AssertionStatus.AFFIRMED,
            confidence=0.9,
            review_required=False,
            value_hash="synthetic-sensitive-finding",
        )


def test_scalar_subclasses_are_normalized_before_storage_and_rendering() -> None:
    record = EvidenceRecord(
        source_start=LeakyInt(1),
        source_end=LeakyInt(5),
        assertion_status=AssertionStatus.AFFIRMED,
        confidence=0.9,
        review_required=False,
        value_hash=LeakyStr(f"sha256:{'a' * 64}"),
    )
    table = EvidenceTable(records=(record,))

    assert type(record.source_start) is int
    assert type(record.source_end) is int
    assert type(record.value_hash) is str
    assert RAW_MARKER not in repr(record)
    assert RAW_MARKER not in table.to_json()
    assert RAW_MARKER not in table.to_markdown()


def test_hashing_bypasses_a_string_subclass_encode_override() -> None:
    record = EvidenceRecord.from_extraction(
        source_start=1,
        source_end=5,
        assertion_status=AssertionStatus.AFFIRMED,
        confidence=0.9,
        review_required=False,
        protected_value=LeakyStr("synthetic-sensitive-finding"),
        include_value_hash=True,
    )

    assert record.value_hash is not None
    assert RAW_MARKER not in repr(record)


def test_scalar_normalization_errors_do_not_reflect_subclass_content() -> None:
    with pytest.raises(TypeError) as exc_info:
        EvidenceRecord.from_extraction(
            source_start=1,
            source_end=5,
            assertion_status=AssertionStatus.AFFIRMED,
            confidence=LeakyFloat(0.9),
            review_required=False,
        )

    assert RAW_MARKER not in str(exc_info.value)


def test_hashing_error_traceback_does_not_reflect_subclass_content() -> None:
    with pytest.raises(TypeError) as exc_info:
        EvidenceRecord.from_extraction(
            source_start=1,
            source_end=5,
            assertion_status=AssertionStatus.AFFIRMED,
            confidence=0.9,
            review_required=False,
            protected_value=ExplodingStr("synthetic-sensitive-finding"),
            include_value_hash=True,
        )

    rendered_traceback = "".join(
        traceback.format_exception(
            exc_info.type,
            exc_info.value,
            exc_info.tb,
        )
    )
    assert RAW_MARKER not in rendered_traceback


def test_assertion_error_traceback_does_not_reflect_subclass_content() -> None:
    with pytest.raises(ValueError) as exc_info:
        EvidenceRecord.from_extraction(
            source_start=1,
            source_end=5,
            assertion_status=ExplodingStr("synthetic-sensitive-assertion"),
            confidence=0.9,
            review_required=False,
        )

    rendered_traceback = "".join(
        traceback.format_exception(
            exc_info.type,
            exc_info.value,
            exc_info.tb,
        )
    )
    assert RAW_MARKER not in rendered_traceback


def test_empty_table_has_stable_zero_counts() -> None:
    assert EvidenceTable(records=()).to_dict() == {
        "schema_version": 1,
        "record_count": 0,
        "review_required_count": 0,
        "assertion_counts": {},
        "records": [],
        "disclaimer": (
            "Clinical evidence tables are value-free review aids, not clinical "
            "decisions or compliance certifications."
        ),
    }


def test_rendering_performs_no_network_calls(monkeypatch) -> None:
    def fail_network(*args, **kwargs):
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "create_connection", fail_network)

    table = EvidenceTable(records=(record(1, 5),))
    table.to_json()
    table.to_markdown()
