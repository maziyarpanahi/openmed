"""Synthetic offline integration of inventory and existing asset preflight."""

import hashlib
import io

import pytest

from openmed.multimodal import (
    AbstentionReason,
    PdfContentProfile,
    PreflightError,
    PreflightStatus,
    preflight_asset,
    preflight_pdf_asset,
)
from tests.fixtures.pdf_inventory import SENTINEL, category_pdf, plain_pdf

pytestmark = pytest.mark.integration


def manifest(source):
    return {
        "asset_id": "synthetic-3824",
        "media_type": "application/pdf",
        "sha256": hashlib.sha256(source).hexdigest(),
        "byte_size": len(source),
        "pages": 1,
    }


@pytest.mark.parametrize(
    "category,reason",
    [
        ("forms", AbstentionReason.PHI_UNCERTAINTY),
        ("attachments", AbstentionReason.UNSUPPORTED_MEDIA),
        ("javascript", AbstentionReason.UNSUPPORTED_MEDIA),
    ],
)
def test_hidden_content_abstains_after_asset_checks(category, reason):
    source = category_pdf(category)
    report = preflight_pdf_asset(manifest(source), io.BytesIO(source))
    assert report.status is PreflightStatus.ABSTAIN
    assert report.abstention.reason is AbstentionReason.RESOURCE_LIMIT
    from openmed.multimodal import read_pdf_inventory

    assert read_pdf_inventory(source).abstention.reason is reason
    assert report.digest.byte_count == len(source)
    assert report.findings[-1].check == "pdf_content"
    assert SENTINEL not in report.to_json()
    assert preflight_asset(manifest(source), source).status is PreflightStatus.ABSTAIN


def test_plain_pdf_preserves_raster_budget_abstention_and_stream_is_restored():
    source = plain_pdf()
    stream = io.BytesIO(b"prefix" + source)
    stream.seek(6)
    report = preflight_pdf_asset(manifest(source), stream)
    assert report.status is PreflightStatus.ABSTAIN
    assert report.to_dict() == preflight_asset(manifest(source), source).to_dict()
    assert not any(f.check == "pdf_content" for f in report.findings)
    assert stream.tell() == 6 and not stream.closed


def test_review_policy_requires_review_for_attachments():
    source = category_pdf("attachments")
    report = preflight_pdf_asset(
        manifest(source), source, content_profile=PdfContentProfile.REVIEW
    )
    assert report.status is PreflightStatus.ABSTAIN
    assert report.abstention.reason is AbstentionReason.RESOURCE_LIMIT
    assert report.findings[-1].reason_code == "pdf_embedded_files"


def test_digest_mismatch_precedes_inventory():
    source = category_pdf("forms")
    declared = manifest(source)
    declared["sha256"] = "0" * 64
    report = preflight_pdf_asset(declared, source)
    assert [
        (f.check, f.reason_code) for f in report.findings if f.check != "limits"
    ] == [("digest", "sha256_mismatch")]


def test_nonseekable_input_is_consumed_once_with_one_probe_byte_bound():
    source = plain_pdf()

    class Stream:
        def __init__(self):
            self.buffer = io.BytesIO(source + b"oversized")

        def read(self, size):
            return self.buffer.read(size)

    stream = Stream()
    report = preflight_pdf_asset(manifest(source), stream)
    assert report.status is PreflightStatus.ABSTAIN
    assert report.abstention.reason is AbstentionReason.RESOURCE_LIMIT
    assert stream.buffer.tell() == len(source) + 1


def test_source_errors_are_value_free():
    class Broken:
        def read(self, size):
            raise OSError(SENTINEL)

    with pytest.raises(PreflightError, match="preflight_source_read_error"):
        preflight_pdf_asset(manifest(plain_pdf()), Broken())
