"""Focused tests for the deterministic document-intake quarantine policy."""

from __future__ import annotations

import hashlib
import io
import zipfile

from openmed.interop.document_quarantine import (
    DEFAULT_POLICY,
    REASON_ACCEPTED,
    REASON_ARCHIVE_DEPTH_EXCEEDED,
    REASON_DECLARED_MIME_SNIFF_MISMATCH,
    REASON_SIZE_LIMIT_EXCEEDED,
    Disposition,
    DocumentQuarantinePolicy,
    classify_document,
)


def _zip_with_member(name: str, payload: bytes) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, payload)
    return stream.getvalue()


def test_matching_pdf_is_accepted_with_only_safe_report_fields() -> None:
    payload = b"%PDF-1.7\nsynthetic fixture"

    result = classify_document(
        payload,
        declared_mime="application/pdf",
        filename="synthetic-report.pdf",
    )

    assert result.disposition is Disposition.ACCEPTED
    assert result.reason_codes == (REASON_ACCEPTED,)
    assert result.sha256 == hashlib.sha256(payload).hexdigest()
    assert result.to_dict() == {
        "disposition": "accepted",
        "reason_codes": ["accepted"],
        "sha256": result.sha256,
    }
    assert "synthetic-report.pdf" not in repr(result)
    assert payload.decode() not in repr(result)


def test_declared_mime_conflict_is_quarantined_without_echoing_input() -> None:
    payload = b"%PDF-1.7\nsynthetic fixture"

    result = classify_document(
        payload,
        declared_mime="image/png",
        filename="synthetic-report.pdf",
    )

    assert result.disposition is Disposition.QUARANTINED
    assert REASON_DECLARED_MIME_SNIFF_MISMATCH in result.reason_codes
    assert result.to_dict().keys() == {"disposition", "reason_codes", "sha256"}
    assert "image/png" not in repr(result)


def test_size_limit_rejects_before_parser_or_archive_work() -> None:
    policy = DocumentQuarantinePolicy(max_size_bytes=4)

    result = classify_document(
        b"012345",
        declared_mime="text/plain",
        filename="synthetic.txt",
        policy=policy,
    )

    assert result.disposition is Disposition.REJECTED
    assert result.reason_codes == (REASON_SIZE_LIMIT_EXCEEDED,)


def test_nested_archive_over_depth_limit_is_rejected_deterministically() -> None:
    inner = _zip_with_member("synthetic.txt", b"offline fixture")
    outer = _zip_with_member("nested.zip", inner)

    result = classify_document(
        outer,
        declared_mime="application/zip",
        filename="synthetic-bundle.zip",
        policy=DocumentQuarantinePolicy(max_archive_depth=1),
    )

    assert result.disposition is Disposition.REJECTED
    assert result.reason_codes == (REASON_ARCHIVE_DEPTH_EXCEEDED,)
    assert result == classify_document(
        outer,
        declared_mime="application/zip",
        filename="synthetic-bundle.zip",
        policy=DocumentQuarantinePolicy(max_archive_depth=1),
    )


def test_default_policy_is_local_and_supports_a_single_zip_layer() -> None:
    payload = _zip_with_member("synthetic.txt", b"offline fixture")

    result = classify_document(
        payload,
        declared_mime="application/zip",
        filename="synthetic-bundle.zip",
        policy=DEFAULT_POLICY,
    )

    assert result.disposition is Disposition.ACCEPTED


def test_malformed_archive_is_rejected_without_exposing_parser_details() -> None:
    payload = b"PK\x03\x04synthetic-invalid-archive"

    result = classify_document(
        payload,
        declared_mime="application/zip",
        filename="synthetic-bundle.zip",
    )

    assert result.disposition is Disposition.REJECTED
    assert result.reason_codes == ("archive_invalid",)
    assert "synthetic-invalid-archive" not in repr(result)
