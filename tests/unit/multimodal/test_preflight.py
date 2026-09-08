"""Tests for the privacy-safe multimodal preflight report."""

from __future__ import annotations

import hashlib
import io
import json

import pytest

from openmed.multimodal.abstention import (
    AbstentionReason,
    AbstentionRecord,
    AbstentionStage,
)
from openmed.multimodal.asset_limits import DESKTOP_V1, MOBILE_V1, LimitFinding
from openmed.multimodal.asset_manifest import AssetManifest
from openmed.multimodal.digest import AssetDigest
from openmed.multimodal.manifest_profiles import (
    AUDIO_V1,
    DICOM_V1,
    IMAGE_V1,
    PDF_V1,
    ValidationFinding,
)
from openmed.multimodal.media_type import (
    MAX_MEDIA_TYPE_PREFIX_BYTES,
    MediaTypeStatus,
    validate_media_type,
)
from openmed.multimodal.preflight import (
    PREFLIGHT_CHECKS,
    PREFLIGHT_SCHEMA_VERSION,
    PreflightError,
    PreflightFinding,
    PreflightReport,
    PreflightStatus,
    preflight_asset,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
PDF = b"%PDF-1.7\n" + b"\x00" * 63
DICOM = b"\x00" * 128 + b"DICM" + b"\x00" * 60
WAV = b"RIFF\x10\x00\x00\x00WAVE" + b"\x00" * 60


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def manifest(source: bytes, media_type: str, **fields):
    data = {
        "asset_id": "asset-1",
        "media_type": media_type,
        "sha256": sha(source),
        "byte_size": len(source),
    }
    data.update(fields)
    return data


IMAGE_MANIFEST = manifest(PNG, "image/png", width=8, height=8)
PDF_MANIFEST = manifest(PDF, "application/pdf", pages=2)
DICOM_MANIFEST = manifest(DICOM, "application/dicom", width=4, height=4, frames=2)
AUDIO_MANIFEST = manifest(WAV, "audio/wav", duration_seconds=1.5)


class NonSeekableStream:
    def __init__(self, payload: bytes) -> None:
        self._stream = io.BytesIO(payload)
        self.closed = False
        self.read_sizes: list[int] = []

    def seekable(self) -> bool:
        return False

    def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        return self._stream.read(size)


def checks(report: PreflightReport) -> list[tuple[str, str]]:
    return [(f.check, f.reason_code) for f in report.findings]


# --- happy paths ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "source", "modality", "profile", "detected"),
    [
        pytest.param(IMAGE_MANIFEST, PNG, "image", IMAGE_V1, "image/png", id="image"),
        pytest.param(
            DICOM_MANIFEST, DICOM, "dicom", DICOM_V1, "application/dicom", id="dicom"
        ),
        pytest.param(AUDIO_MANIFEST, WAV, "audio", AUDIO_V1, "audio/wav", id="audio"),
    ],
)
def test_synthetic_happy_paths_are_accepted(data, source, modality, profile, detected):
    report = preflight_asset(data, source)

    assert report.status is PreflightStatus.ACCEPT
    assert report.findings == ()
    assert report.abstention is None
    assert report.manifest == AssetManifest.from_dict(data)
    assert report.modality == modality
    assert report.metadata_profile is profile
    assert report.limit_profile is DESKTOP_V1
    assert report.detected_media_type == detected
    assert report.media_type_status is MediaTypeStatus.MATCH
    assert report.digest == AssetDigest(sha(source), len(source))


def test_accepted_image_report_serialises_byte_stably():
    report = preflight_asset(IMAGE_MANIFEST, PNG)
    expected = (
        '{"schema_version":1,"status":"accept","abstention":null,'
        '"manifest":{"version":1,"asset_id":"asset-1","media_type":"image/png",'
        f'"sha256":"{sha(PNG)}","byte_size":72,"width":8,"height":8}},'
        '"modality":"image","metadata_profile":{"modality":"image","version":"1.0"},'
        '"limit_profile":{"name":"desktop","version":"1.0"},'
        '"media_type":{"detected":"image/png","status":"match"},'
        f'"digest":{{"sha256":"{sha(PNG)}","byte_count":72}},"findings":[]}}'
    )
    assert report.to_json() == expected
    assert preflight_asset(IMAGE_MANIFEST, io.BytesIO(PNG)).to_json() == expected
    assert json.loads(expected) == report.to_dict()


def test_accepted_dicom_and_audio_reports_serialise_byte_stably():
    dicom = preflight_asset(DICOM_MANIFEST, DICOM).to_json()
    assert dicom == (
        '{"schema_version":1,"status":"accept","abstention":null,'
        '"manifest":{"version":1,"asset_id":"asset-1","media_type":"application/dicom",'
        f'"sha256":"{sha(DICOM)}","byte_size":192,"width":4,"height":4,"frames":2}},'
        '"modality":"dicom","metadata_profile":{"modality":"dicom","version":"1.0"},'
        '"limit_profile":{"name":"desktop","version":"1.0"},'
        '"media_type":{"detected":"application/dicom","status":"match"},'
        f'"digest":{{"sha256":"{sha(DICOM)}","byte_count":192}},"findings":[]}}'
    )
    audio = preflight_asset(AUDIO_MANIFEST, WAV).to_json()
    assert audio == (
        '{"schema_version":1,"status":"accept","abstention":null,'
        '"manifest":{"version":1,"asset_id":"asset-1","media_type":"audio/wav",'
        f'"sha256":"{sha(WAV)}","byte_size":72,"duration_seconds":1.5}},'
        '"modality":"audio","metadata_profile":{"modality":"audio","version":"1.0"},'
        '"limit_profile":{"name":"desktop","version":"1.0"},'
        '"media_type":{"detected":"audio/wav","status":"match"},'
        f'"digest":{{"sha256":"{sha(WAV)}","byte_count":72}},"findings":[]}}'
    )


def test_pdf_happy_path_abstains_on_unavailable_pixel_metadata():
    # Under the current manifest contract a PDF carries no raster geometry, so
    # its two pixel rules are unevaluable even when bytes, pages, media type,
    # and digest all pass. Missing evidence never becomes acceptance.
    report = preflight_asset(PDF_MANIFEST, PDF)

    assert report.status is PreflightStatus.ABSTAIN
    assert report.abstention == AbstentionRecord(
        AbstentionStage.PREFLIGHT, AbstentionReason.RESOURCE_LIMIT
    )
    assert report.media_type_status is MediaTypeStatus.MATCH
    assert report.digest == AssetDigest(sha(PDF), len(PDF))
    assert report.findings == (
        PreflightFinding(
            "limits", "insufficient_metadata", "pixels", DESKTOP_V1.max_pixels
        ),
        PreflightFinding(
            "limits",
            "insufficient_metadata",
            "total_pixels",
            DESKTOP_V1.max_total_pixels,
        ),
    )
    assert report.to_json() == (
        '{"schema_version":1,"status":"abstain",'
        '"abstention":{"schema_version":1,"stage":"preflight","reason":"resource_limit"},'
        '"manifest":{"version":1,"asset_id":"asset-1","media_type":"application/pdf",'
        f'"sha256":"{sha(PDF)}","byte_size":72,"pages":2}},'
        '"modality":"pdf","metadata_profile":{"modality":"pdf","version":"1.0"},'
        '"limit_profile":{"name":"desktop","version":"1.0"},'
        '"media_type":{"detected":"application/pdf","status":"match"},'
        f'"digest":{{"sha256":"{sha(PDF)}","byte_count":72}},'
        '"findings":['
        '{"check":"limits","reason_code":"insufficient_metadata","field_name":"pixels","limit":40000000,"observed":null},'
        '{"check":"limits","reason_code":"insufficient_metadata","field_name":"total_pixels","limit":100000000,"observed":null}'
        "]}"
    )


def test_accepts_a_manifest_instance_and_a_caller_limit_profile():
    instance = AssetManifest.from_dict(IMAGE_MANIFEST)
    report = preflight_asset(instance, PNG, limit_profile=MOBILE_V1)

    assert report.status is PreflightStatus.ACCEPT
    assert report.manifest is instance
    assert report.to_dict()["limit_profile"] == {"name": "mobile", "version": "1.0"}


# --- malformed manifest ----------------------------------------------------------


@pytest.mark.parametrize(
    "data",
    [
        pytest.param({}, id="empty"),
        pytest.param({**IMAGE_MANIFEST, "path": "x"}, id="unknown_field"),
        pytest.param({**IMAGE_MANIFEST, "asset_id": "../scan"}, id="path_like_id"),
        pytest.param({**IMAGE_MANIFEST, "byte_size": 0}, id="zero_bytes"),
        pytest.param({**IMAGE_MANIFEST, "media_type": "text/plain"}, id="unsupported"),
        pytest.param({**IMAGE_MANIFEST, "version": 2}, id="bad_version"),
    ],
)
def test_malformed_manifest_abstains_before_the_source_is_touched(data):
    stream = NonSeekableStream(PNG)
    report = preflight_asset(data, stream)

    assert report.status is PreflightStatus.ABSTAIN
    assert report.findings == (PreflightFinding("manifest", "malformed_manifest"),)
    assert report.abstention.reason is AbstentionReason.UNSUPPORTED_MEDIA
    assert report.manifest is None
    assert report.modality is None
    assert report.metadata_profile is None
    assert report.media_type_status is None
    assert report.digest is None
    assert stream.read_sizes == []
    assert report.to_json() == (
        '{"schema_version":1,"status":"abstain",'
        '"abstention":{"schema_version":1,"stage":"preflight","reason":"unsupported_media"},'
        '"manifest":null,"modality":null,"metadata_profile":null,'
        '"limit_profile":{"name":"desktop","version":"1.0"},'
        '"media_type":null,"digest":null,"findings":['
        '{"check":"manifest","reason_code":"malformed_manifest","field_name":null,"limit":null,"observed":null}'
        "]}"
    )


# --- media type -------------------------------------------------------------------


def test_media_mismatch_is_a_fail_closed_finding():
    report = preflight_asset(manifest(PDF, "image/png", width=8, height=8), PDF)

    assert report.status is PreflightStatus.ABSTAIN
    assert report.abstention.reason is AbstentionReason.UNSUPPORTED_MEDIA
    assert report.detected_media_type == "application/pdf"
    assert report.media_type_status is MediaTypeStatus.MISMATCH
    assert checks(report) == [("media_type", "mismatch")]
    assert report.digest == AssetDigest(sha(PDF), len(PDF))


@pytest.mark.parametrize(
    ("source", "media_type"),
    [
        pytest.param(b"\x89PN", "image/png", id="truncated_prefix"),
        pytest.param(
            b"RIFF\x00\x00\x00\x00WEBPVP8 " + b"\x00" * 60,
            "image/webp",
            id="unsupported_signature",
        ),
        pytest.param(b"\x00" * 200, "image/png", id="zero_bytes"),
    ],
)
def test_undetectable_media_is_unknown_never_a_match(source, media_type):
    report = preflight_asset(manifest(source, media_type, width=1, height=1), source)

    assert report.status is PreflightStatus.ABSTAIN
    assert report.detected_media_type is None
    assert report.media_type_status is MediaTypeStatus.UNKNOWN
    assert checks(report)[0] == ("media_type", "unknown")


def test_media_type_status_agrees_with_the_media_type_validator():
    for source, declared in (
        (PNG, "image/png"),
        (PDF, "image/png"),
        (b"?", "image/png"),
    ):
        report = preflight_asset(manifest(source, declared, width=1, height=1), source)
        assert report.media_type_status is validate_media_type(
            source[:MAX_MEDIA_TYPE_PREFIX_BYTES], declared
        )


# --- metadata profiles ---------------------------------------------------------------


def test_profile_findings_precede_the_limit_findings_they_cause():
    report = preflight_asset(manifest(PNG, "image/png", width=8), PNG)

    assert report.findings == (
        PreflightFinding("metadata", "missing_required", "height"),
        PreflightFinding(
            "limits", "insufficient_metadata", "pixels", DESKTOP_V1.max_pixels
        ),
        PreflightFinding(
            "limits",
            "insufficient_metadata",
            "total_pixels",
            DESKTOP_V1.max_total_pixels,
        ),
    )
    assert report.abstention.reason is AbstentionReason.UNSUPPORTED_MEDIA
    assert report.digest is not None


def test_inapplicable_fields_are_profile_findings():
    report = preflight_asset(manifest(PDF, "application/pdf", pages=1, frames=3), PDF)

    assert checks(report)[0] == ("metadata", "inapplicable_present")
    assert report.findings[0].field_name == "frames"


def test_unsupported_modality_cannot_be_evaluated_or_hashed():
    data = manifest(DICOM, "application/dicom+json")
    stream = io.BytesIO(DICOM)
    report = preflight_asset(data, stream)

    assert report.status is PreflightStatus.ABSTAIN
    assert report.abstention.reason is AbstentionReason.UNSUPPORTED_MEDIA
    assert report.modality is None
    assert report.metadata_profile is None
    assert report.digest is None
    assert checks(report) == [
        ("media_type", "mismatch"),
        ("metadata", "unsupported_modality"),
        ("digest", "not_evaluated"),
    ]
    assert stream.tell() == 0


# --- every resource-limit class -------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "source", "field_name", "observed"),
    [
        pytest.param(
            manifest(PDF, "application/pdf", pages=MOBILE_V1.max_pages + 1),
            PDF,
            "pages",
            MOBILE_V1.max_pages + 1,
            id="pages",
        ),
        pytest.param(
            manifest(PNG, "image/png", width=2_501, height=4_000),
            PNG,
            "pixels",
            2_501 * 4_000,
            id="pixels",
        ),
        pytest.param(
            manifest(DICOM, "application/dicom", width=1_000, height=1_000, frames=26),
            DICOM,
            "total_pixels",
            26_000_000,
            id="total_pixels",
        ),
        pytest.param(
            manifest(
                DICOM,
                "application/dicom",
                width=2,
                height=2,
                frames=MOBILE_V1.max_frames + 1,
            ),
            DICOM,
            "frames",
            MOBILE_V1.max_frames + 1,
            id="frames",
        ),
        pytest.param(
            manifest(WAV, "audio/wav", duration_seconds=300.5),
            WAV,
            "duration_seconds",
            300.5,
            id="duration_seconds",
        ),
    ],
)
def test_each_exceeded_limit_abstains_for_a_resource_limit(
    data, source, field_name, observed
):
    report = preflight_asset(data, source, limit_profile=MOBILE_V1)

    assert report.status is PreflightStatus.ABSTAIN
    assert report.abstention == AbstentionRecord(
        AbstentionStage.PREFLIGHT, AbstentionReason.RESOURCE_LIMIT
    )
    exceeded = [f for f in report.findings if f.reason_code == "limit_exceeded"]
    assert exceeded == [
        PreflightFinding(
            "limits",
            "limit_exceeded",
            field_name,
            MOBILE_V1.limit_for(field_name),
            observed,
        )
    ]
    assert report.digest is not None


def test_byte_size_limit_skips_the_digest_and_reads_only_the_prefix():
    oversized = MOBILE_V1.max_byte_size + 1
    padded = PNG + b"\x00" * 200
    data = manifest(padded, "image/png", width=8, height=8, byte_size=oversized)
    stream = NonSeekableStream(padded)
    report = preflight_asset(data, stream, limit_profile=MOBILE_V1)

    assert report.status is PreflightStatus.ABSTAIN
    assert report.abstention.reason is AbstentionReason.RESOURCE_LIMIT
    assert report.findings == (
        PreflightFinding(
            "limits", "limit_exceeded", "byte_size", MOBILE_V1.max_byte_size, oversized
        ),
        PreflightFinding("digest", "not_evaluated"),
    )
    assert report.digest is None
    assert stream.read_sizes == [MAX_MEDIA_TYPE_PREFIX_BYTES]


def test_missing_limit_inputs_abstain_as_insufficient_metadata():
    report = preflight_asset(manifest(WAV, "audio/wav"), WAV)

    assert checks(report) == [
        ("metadata", "missing_required"),
        ("limits", "insufficient_metadata"),
    ]
    assert report.findings[1] == PreflightFinding(
        "limits",
        "insufficient_metadata",
        "duration_seconds",
        DESKTOP_V1.max_duration_seconds,
    )


# --- digest -------------------------------------------------------------------------


def test_sha256_mismatch_is_a_fail_closed_finding():
    data = manifest(PNG, "image/png", width=8, height=8, sha256="f" * 64)
    report = preflight_asset(data, PNG)

    assert report.status is PreflightStatus.ABSTAIN
    assert report.abstention.reason is AbstentionReason.UNSUPPORTED_MEDIA
    assert report.findings == (PreflightFinding("digest", "sha256_mismatch", "sha256"),)
    assert report.digest == AssetDigest(sha(PNG), len(PNG))


def test_short_source_reports_the_bytes_hashed():
    data = manifest(PNG, "image/png", width=8, height=8, byte_size=len(PNG) + 10)
    report = preflight_asset(data, io.BytesIO(PNG))

    assert report.findings == (
        PreflightFinding(
            "digest", "byte_count_mismatch", "byte_size", len(PNG) + 10, len(PNG)
        ),
    )
    assert report.digest == AssetDigest(sha(PNG), len(PNG))


def test_long_source_stops_at_the_declared_size_and_reports_no_count():
    data = manifest(PNG, "image/png", width=8, height=8, byte_size=len(PNG) - 10)
    stream = NonSeekableStream(PNG)
    report = preflight_asset(data, stream)

    assert report.findings == (
        PreflightFinding("digest", "byte_count_mismatch", "byte_size", len(PNG) - 10),
    )
    assert report.digest is None
    assert sum(stream.read_sizes) <= MAX_MEDIA_TYPE_PREFIX_BYTES + len(PNG) - 10 + 1


def test_seekable_streams_are_restored_and_never_closed():
    payload = b"prefix-" + PNG
    stream = io.BytesIO(payload)
    stream.seek(7)

    report = preflight_asset(IMAGE_MANIFEST, stream)

    assert report.status is PreflightStatus.ACCEPT
    assert stream.tell() == 7
    assert not stream.closed

    stream.seek(7)
    failing = preflight_asset(manifest(PNG, "image/png", width=8), stream)
    assert failing.status is PreflightStatus.ABSTAIN
    assert stream.tell() == 7
    assert not stream.closed


def test_non_seekable_streams_are_hashed_through_the_replayed_prefix():
    stream = NonSeekableStream(PNG)
    report = preflight_asset(IMAGE_MANIFEST, stream)

    assert report.status is PreflightStatus.ACCEPT
    assert report.digest == AssetDigest(sha(PNG), len(PNG))
    assert not stream.closed


def test_bytes_like_sources_are_hashed_in_memory():
    for source in (bytearray(PNG), memoryview(PNG)):
        assert preflight_asset(IMAGE_MANIFEST, source).status is PreflightStatus.ACCEPT


@pytest.mark.parametrize("failure", ["read", "position", "restore"])
def test_source_failures_are_categorical_and_unchained(failure):
    secret = "/private/patient-name.dcm contains secret bytes"

    class BrokenStream(NonSeekableStream):
        def seekable(self) -> bool:
            return failure != "read"

        def tell(self) -> int:
            if failure == "position":
                raise OSError(secret)
            return 0

        def seek(self, position: int) -> int:
            raise OSError(secret)

        def read(self, size: int) -> bytes:
            if failure == "read":
                raise OSError(secret)
            return super().read(size)

    with pytest.raises(PreflightError) as raised:
        preflight_asset(IMAGE_MANIFEST, BrokenStream(PNG))

    assert str(raised.value) == f"preflight_source_{failure}_error"
    assert secret not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_digest_read_failure_after_the_prefix_is_categorical():
    class LateFailure(NonSeekableStream):
        def read(self, size: int) -> bytes:
            if self.read_sizes:
                raise OSError("/private/patient-name.dcm")
            return super().read(size)

    with pytest.raises(PreflightError, match="preflight_source_read_error") as raised:
        preflight_asset(IMAGE_MANIFEST, LateFailure(PNG))
    assert raised.value.__cause__ is None
    assert "patient" not in str(raised.value)


def test_source_contract_violations_are_categorical():
    class TextStream:
        def read(self, size: int) -> str:
            return "not bytes"

    with pytest.raises(PreflightError, match="preflight_source_contract_error"):
        preflight_asset(IMAGE_MANIFEST, TextStream())


# --- ordering and the coarse reason --------------------------------------------------


def test_findings_follow_the_documented_check_order():
    # A PNG declared as a DICOM with no geometry, a wrong digest, and an
    # oversized frame count fails four checks at once.
    data = manifest(
        PNG,
        "application/dicom",
        width=1,
        height=1,
        frames=DESKTOP_V1.max_frames + 1,
        sha256="0" * 64,
    )
    report = preflight_asset(data, PNG)

    assert checks(report) == [
        ("media_type", "mismatch"),
        ("limits", "limit_exceeded"),
        ("digest", "sha256_mismatch"),
    ]
    positions = [PREFLIGHT_CHECKS.index(f.check) for f in report.findings]
    assert positions == sorted(positions)
    assert report.abstention.reason is AbstentionReason.UNSUPPORTED_MEDIA


def test_coarse_reason_follows_the_earliest_failing_check():
    limits_first = preflight_asset(
        manifest(PNG, "image/png", width=2_501, height=4_000, sha256="0" * 64),
        PNG,
        limit_profile=MOBILE_V1,
    )
    assert checks(limits_first) == [
        ("limits", "limit_exceeded"),
        ("digest", "sha256_mismatch"),
    ]
    assert limits_first.abstention.reason is AbstentionReason.RESOURCE_LIMIT

    media_first = preflight_asset(
        manifest(PDF, "image/png", width=2_501, height=4_000),
        PDF,
        limit_profile=MOBILE_V1,
    )
    assert checks(media_first)[0] == ("media_type", "mismatch")
    assert media_first.abstention.reason is AbstentionReason.UNSUPPORTED_MEDIA


def test_reports_are_deterministic_across_runs():
    first = preflight_asset(PDF_MANIFEST, io.BytesIO(PDF))
    second = preflight_asset(PDF_MANIFEST, PDF)

    assert first == second
    assert first.to_json() == second.to_json()


# --- finding and report contracts ------------------------------------------------------


def test_finding_allowlists_reuse_the_underlying_contracts():
    assert PreflightFinding(
        "limits", "limit_exceeded", "pages", 10, 11
    ) == PreflightFinding(
        "limits",
        LimitFinding("pages", "limit_exceeded", 10, 11).reason_code,
        "pages",
        10,
        11,
    )
    ValidationFinding("width", "invalid_type")
    assert PreflightFinding("metadata", "invalid_type", "width").field_name == "width"
    assert PREFLIGHT_SCHEMA_VERSION == 1
    assert PREFLIGHT_CHECKS == (
        "manifest",
        "media_type",
        "metadata",
        "limits",
        "digest",
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(("upload", "malformed_manifest"), id="unknown_check"),
        pytest.param(("manifest", "too_big"), id="unknown_reason"),
        pytest.param(
            ("manifest", "malformed_manifest", "asset_id"), id="unexpected_field"
        ),
        pytest.param(("media_type", "mismatch", None, 1), id="unexpected_limit"),
        pytest.param(
            ("metadata", "missing_required", "path"), id="metadata_field_not_allowed"
        ),
        pytest.param(
            ("metadata", "missing_required", "width", 1), id="metadata_with_limit"
        ),
        pytest.param(
            ("metadata", "unsupported_modality", "width"), id="modality_with_field"
        ),
        pytest.param(
            ("limits", "limit_exceeded", "path", 1, 2), id="limit_field_not_allowed"
        ),
        pytest.param(
            ("limits", "insufficient_metadata", "pages", 1, 2),
            id="unevaluable_with_observed",
        ),
        pytest.param(
            ("limits", "limit_exceeded", None, 1, 2), id="limit_without_field"
        ),
        pytest.param(
            ("digest", "sha256_mismatch", "byte_size"), id="digest_wrong_field"
        ),
        pytest.param(
            ("digest", "byte_count_mismatch", "byte_size", 0), id="zero_declared_size"
        ),
        pytest.param(
            ("digest", "byte_count_mismatch", "byte_size", 5, -1), id="negative_count"
        ),
        pytest.param(
            ("digest", "byte_count_mismatch", "byte_size", 5, 1.5), id="float_count"
        ),
        pytest.param(
            ("digest", "byte_count_mismatch", None, 5, 1), id="count_without_field"
        ),
        pytest.param(
            ("digest", "not_evaluated", None, 5), id="not_evaluated_with_limit"
        ),
    ],
)
def test_findings_cannot_be_constructed_outside_the_allowlists(args):
    with pytest.raises(PreflightError):
        PreflightFinding(*args)


def test_finding_serialises_with_a_fixed_shape():
    assert PreflightFinding(
        "digest", "byte_count_mismatch", "byte_size", 5, 3
    ).to_dict() == {
        "check": "digest",
        "reason_code": "byte_count_mismatch",
        "field_name": "byte_size",
        "limit": 5,
        "observed": 3,
    }


def test_reports_validate_their_own_consistency():
    accepted = preflight_asset(IMAGE_MANIFEST, PNG)
    abstained = preflight_asset(PDF_MANIFEST, PDF)
    finding = PreflightFinding("manifest", "malformed_manifest")

    def build(base: PreflightReport, **overrides):
        values = {
            name: getattr(base, name)
            for name in (
                "status",
                "findings",
                "limit_profile",
                "manifest",
                "modality",
                "metadata_profile",
                "detected_media_type",
                "media_type_status",
                "digest",
                "abstention",
                "schema_version",
            )
        }
        values.update(overrides)
        return PreflightReport(**values)

    with pytest.raises(PreflightError):
        build(accepted, status="maybe")
    with pytest.raises(PreflightError):
        build(accepted, schema_version=2)
    with pytest.raises(PreflightError):
        build(accepted, findings=(finding,))
    with pytest.raises(PreflightError):
        build(accepted, digest=None)
    with pytest.raises(PreflightError):
        build(accepted, media_type_status=MediaTypeStatus.UNKNOWN)
    with pytest.raises(PreflightError):
        build(accepted, limit_profile={"max_pages": 1})
    with pytest.raises(PreflightError):
        build(accepted, modality="video")
    with pytest.raises(PreflightError):
        build(abstained, findings=())
    with pytest.raises(PreflightError):
        build(abstained, abstention=None)
    with pytest.raises(PreflightError):
        build(abstained, findings=abstained.findings + (finding,))
    with pytest.raises(PreflightError):
        build(
            abstained,
            abstention=AbstentionRecord(
                AbstentionStage.DECODE, AbstentionReason.RESOURCE_LIMIT
            ),
        )
    with pytest.raises(PreflightError):
        build(abstained, findings=("limits",))
    assert build(accepted, status="accept") == accepted


# --- caller misuse and the privacy sentinel ---------------------------------------------


def test_caller_misuse_raises_type_errors():
    with pytest.raises(TypeError):
        preflight_asset(IMAGE_MANIFEST, PNG, limit_profile={"max_pages": 1})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        preflight_asset(42, PNG)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        preflight_asset(IMAGE_MANIFEST, "not-bytes")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        preflight_asset(IMAGE_MANIFEST, None)  # type: ignore[arg-type]


def test_sentinel_values_never_appear_in_reports_or_exceptions():
    sentinels = (
        "/Users/clinician/Desktop/patient-scan.dcm",
        "C:\\exports\\jane-doe-mammogram.png",
        "synthetic OCR text: DOB 01/02/1960",
        "synthetic transcript: patient states chest pain",
        "PatientName^Jane^Doe",
        "Bearer sk-live-synthetic-credential",
    )
    raw_prefix = b"\x89PNG\r\n\x1a\n" + b"PatientName^Jane^Doe DOB 01/02/1960 " * 3
    rendered: list[str] = []

    malformed = {
        **manifest(raw_prefix, "image/png", width=8, height=8),
        "asset_id": sentinels[0],
        "source_path": sentinels[1],
        "ocr_text": sentinels[2],
        "transcript": sentinels[3],
        "dicom_patient_name": sentinels[4],
        "authorization": sentinels[5],
    }
    report = preflight_asset(malformed, raw_prefix)
    rendered.extend((report.to_json(), repr(report), str(report.to_dict())))

    for data in (
        manifest(raw_prefix, "image/png", width=8, height=8),
        manifest(raw_prefix, "image/png", width=8),
        manifest(raw_prefix, "application/pdf", pages=1),
        manifest(raw_prefix, "image/png", width=8, height=8, sha256="a" * 64),
    ):
        report = preflight_asset(data, raw_prefix)
        rendered.extend((report.to_json(), repr(report)))
        for finding in report.findings:
            rendered.append(repr(finding))

    class LeakyStream(NonSeekableStream):
        def read(self, size: int) -> bytes:
            raise OSError(sentinels[0])

    with pytest.raises(PreflightError) as raised:
        preflight_asset(
            manifest(raw_prefix, "image/png", width=8, height=8), LeakyStream(b"")
        )
    rendered.append(str(raised.value))
    with pytest.raises(PreflightError) as raised:
        PreflightFinding("manifest", sentinels[2])
    rendered.append(str(raised.value))

    text = "\n".join(rendered)
    for sentinel in sentinels:
        assert sentinel not in text
    assert "DOB" not in text
    assert raw_prefix.decode("latin-1")[8:40] not in text
    assert "\\x89PNG" not in text and "PNG" not in text


def test_preflight_contract_is_available_from_public_multimodal_api():
    import openmed.multimodal as multimodal

    assert multimodal.preflight_asset is preflight_asset
    assert multimodal.PreflightReport is PreflightReport
    assert multimodal.PreflightFinding is PreflightFinding
    assert multimodal.PreflightStatus is PreflightStatus
    assert multimodal.PreflightError is PreflightError
    assert multimodal.PREFLIGHT_SCHEMA_VERSION == PREFLIGHT_SCHEMA_VERSION
    assert multimodal.PREFLIGHT_CHECKS == PREFLIGHT_CHECKS
