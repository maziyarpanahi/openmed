"""Tests for versioned annotation interchange and access controls."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema.validators import validator_for

from openmed.eval.annotation import (
    ANNOTATION_INTERCHANGE_SCHEMA_NAMES,
    MAX_ANNOTATION_EMBEDDING_DIMENSIONS,
    AnnotationAccessPolicy,
    AnnotationCatalog,
    AnnotationInterchangeError,
    AnnotationQuery,
    AnnotationState,
    CoordinateConvention,
    convert_record_offsets,
    export_annotation_tsv,
    import_annotation_tsv,
    load_annotation_interchange_schema,
)

FIXTURE = Path("tests/fixtures/annotation/interchange.tsv")


def test_checked_in_tsv_round_trip_and_schema_parity() -> None:
    source = FIXTURE.read_text(encoding="utf-8")
    envelope = import_annotation_tsv(source)

    exported = export_annotation_tsv(envelope)

    assert exported.text == source
    assert exported.report.state is AnnotationState.SUCCESS
    records = {
        "record": envelope.records[0].to_dict(),
        "envelope": envelope.to_dict(),
        "loss_report": exported.report.to_dict(),
        "page": AnnotationCatalog(envelope).list(AnnotationQuery()).to_dict(),
    }
    assert set(records) == set(ANNOTATION_INTERCHANGE_SCHEMA_NAMES)
    for name, payload in records.items():
        schema = load_annotation_interchange_schema(name)
        validator = validator_for(schema)
        validator.check_schema(schema)
        assert not tuple(validator(schema).iter_errors(payload))


def test_persisted_json_round_trip_is_client_compatible() -> None:
    envelope = import_annotation_tsv(FIXTURE.read_text(encoding="utf-8"))

    restored = type(envelope).from_json(envelope.to_json())

    assert restored == envelope
    assert restored.envelope_digest == envelope.envelope_digest


def test_embedding_omission_is_declared_as_loss() -> None:
    envelope = import_annotation_tsv(FIXTURE.read_bytes())

    exported = export_annotation_tsv(envelope, include_embeddings=False)
    restored = import_annotation_tsv(exported.text)

    assert exported.report.state is AnnotationState.PARTIAL
    assert exported.report.entries[0].kind.value == "embedding_omitted"
    assert restored.records[0].embedding is None


def test_unicode_offset_conversions_are_explicit_and_exact() -> None:
    envelope = import_annotation_tsv(FIXTURE.read_text(encoding="utf-8"))
    source = "A🩺BC"
    record = replace(envelope.records[0], start=1, end=2)

    utf8, utf8_report = convert_record_offsets(
        record,
        source_text=source,
        target=CoordinateConvention.UTF8_BYTE,
    )
    utf16, _ = convert_record_offsets(
        record,
        source_text=source,
        target=CoordinateConvention.UTF16_CODE_UNIT,
    )
    restored, _ = convert_record_offsets(
        utf8,
        source_text=source,
        target=CoordinateConvention.UNICODE_CODEPOINT,
    )

    assert (utf8.start, utf8.end) == (1, 5)
    assert (utf16.start, utf16.end) == (1, 3)
    assert (restored.start, restored.end) == (1, 2)
    assert utf8_report.state is AnnotationState.SUCCESS


def test_catalog_paginates_and_binds_cursor_to_access_query() -> None:
    envelope = import_annotation_tsv(FIXTURE.read_text(encoding="utf-8"))
    catalog = AnnotationCatalog(envelope)
    query = AnnotationQuery(first=1)

    first = catalog.list(query)
    assert first.state is AnnotationState.SUCCESS
    assert first.next_cursor is not None
    second = catalog.list(replace(query, after=first.next_cursor))
    assert second.records[0].annotation_id != first.records[0].annotation_id

    mismatched = catalog.list(
        replace(query, purpose="training_review", after=first.next_cursor)
    )
    assert mismatched.state is AnnotationState.FAILURE
    assert mismatched.code == "cursor_query_mismatch"


def test_catalog_denies_access_without_returning_records() -> None:
    envelope = import_annotation_tsv(FIXTURE.read_text(encoding="utf-8"))
    policy = AnnotationAccessPolicy(allowed_roles=frozenset({"privacy_officer"}))

    page = AnnotationCatalog(envelope).list(AnnotationQuery(), policy=policy)

    assert page.state is AnnotationState.DENIED
    assert page.code == "role_denied"
    assert page.records == ()


def test_abuse_limits_and_strict_json_fail_closed() -> None:
    envelope = import_annotation_tsv(FIXTURE.read_text(encoding="utf-8"))
    with pytest.raises(AnnotationInterchangeError, match="dimensions"):
        replace(
            envelope.records[0],
            embedding=(0.0,) * (MAX_ANNOTATION_EMBEDDING_DIMENSIONS + 1),
        )
    with pytest.raises(AnnotationInterchangeError, match="columns"):
        import_annotation_tsv("source_text\nsynthetic-canary\n")
    broken = FIXTURE.read_text(encoding="utf-8").replace(
        '"{""label"":""condition"",',
        '"{""label"":""condition"",""label"":""duplicate"",',
        1,
    )
    with pytest.raises(AnnotationInterchangeError, match="invalid JSON"):
        import_annotation_tsv(broken)

    recovered = import_annotation_tsv(FIXTURE.read_text(encoding="utf-8"))
    assert len(recovered.records) == 3
