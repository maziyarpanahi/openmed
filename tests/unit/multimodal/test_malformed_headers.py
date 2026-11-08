"""Tests for the synthetic malformed PDF and DICOM header fixtures (#3094).

The fixtures pin the two *pre-decode* boundaries this repository already
implements: the dependency-free PDF header/page-tree reader
(``read_pdf_geometry``) and the bounded media-type preflight
(``preflight_asset``/``detect_media_type``). Every assertion is offline and
byte-exact so a fixture cannot silently drift away from the boundary it
documents.
"""

from __future__ import annotations

import hashlib
import importlib
import socket

import pytest

from openmed.multimodal.asset_manifest import AssetManifest
from openmed.multimodal.media_type import MAX_MEDIA_TYPE_PREFIX_BYTES, detect_media_type
from openmed.multimodal.pdf_geometry import (
    PDF_REASON_CODES,
    PdfGeometryStatus,
    read_pdf_geometry,
)
from openmed.multimodal.preflight import PreflightStatus, preflight_asset
from tests.fixtures.multimodal.malformed import (
    CORRUPTION_CLASSES,
    DICOM_BOUNDARY,
    MALFORMED_HEADER_CASES,
    MAX_FIXTURE_BYTES,
    MODALITIES,
    PAYLOAD_SHA256,
    PDF_BOUNDARY,
    PHI_MARKERS,
    MalformedHeaderCase,
    case_by_name,
    cases_for,
    corruption_matrix,
)

PDF_CASES = cases_for("pdf")
DICOM_CASES = cases_for("dicom")
ALL_CASES = MALFORMED_HEADER_CASES
IDS = [case.name for case in ALL_CASES]
READER_LIMIT_NAMES = frozenset(
    {"max_bytes", "max_pages", "max_objects", "max_decompressed_bytes"}
)
DICOM_MAGIC = b"DICM"


def _manifest(case: MalformedHeaderCase) -> AssetManifest:
    """Build the matching manifest for a DICOM-boundary fixture."""
    return AssetManifest(
        asset_id="fixture-asset",
        sha256=case.sha256(),
        byte_size=len(case.payload),
        **case.declaration(),
    )


def test_fixture_table_covers_every_modality_and_corruption_class() -> None:
    assert len(ALL_CASES) == len(MODALITIES) * len(CORRUPTION_CLASSES)
    assert tuple(case.modality for case in PDF_CASES) == ("pdf",) * len(
        CORRUPTION_CLASSES
    )
    assert tuple(case.modality for case in DICOM_CASES) == ("dicom",) * len(
        CORRUPTION_CLASSES
    )
    assert tuple(case.corruption for case in PDF_CASES) == CORRUPTION_CLASSES
    assert tuple(case.corruption for case in DICOM_CASES) == CORRUPTION_CLASSES
    assert corruption_matrix() == {
        "pdf": CORRUPTION_CLASSES,
        "dicom": CORRUPTION_CLASSES,
    }


def test_fixture_names_are_unique_and_kebab_case() -> None:
    assert len(set(IDS)) == len(IDS)
    for case in ALL_CASES:
        assert case.name == case.name.lower()
        assert "-" in case.name
        assert case.name.startswith(f"{case.modality}-")


def test_boundary_matches_modality() -> None:
    for case in PDF_CASES:
        assert case.boundary == PDF_BOUNDARY
    for case in DICOM_CASES:
        assert case.boundary == DICOM_BOUNDARY


def test_payloads_are_tiny_and_carry_no_phi_markers() -> None:
    for case in ALL_CASES:
        assert 0 < len(case.payload) <= MAX_FIXTURE_BYTES
        for marker in PHI_MARKERS:
            assert marker not in case.payload


def test_payload_digests_match_the_pinned_table() -> None:
    assert set(PAYLOAD_SHA256) == set(IDS)
    for case in ALL_CASES:
        digest = hashlib.sha256(case.payload).hexdigest()
        assert digest == case.sha256()
        assert PAYLOAD_SHA256[case.name] == digest


def test_fixture_module_is_byte_stable_across_reloads() -> None:
    module = importlib.import_module("tests.fixtures.multimodal.malformed")
    reloaded = importlib.reload(module)
    try:
        assert tuple(case.payload for case in reloaded.MALFORMED_HEADER_CASES) == tuple(
            case.payload for case in ALL_CASES
        )
        assert reloaded.PAYLOAD_SHA256 == PAYLOAD_SHA256
    finally:
        importlib.reload(module)


@pytest.mark.parametrize("case", PDF_CASES, ids=[c.name for c in PDF_CASES])
def test_pdf_fixture_fails_at_the_geometry_boundary(case: MalformedHeaderCase) -> None:
    report = read_pdf_geometry(case.payload, **case.limits())
    assert case.expected_reason in PDF_REASON_CODES
    assert report.reason_codes == (case.expected_reason,)
    if case.rejected:
        assert report.status is PdfGeometryStatus.REJECTED
        assert report.pages == ()
    else:
        assert report.status is PdfGeometryStatus.REVIEW
        assert report.pages


def test_reader_limits_are_declared_only_where_a_budget_is_exercised() -> None:
    limited = [case for case in PDF_CASES if case.limits()]
    assert [case.name for case in limited] == ["pdf-header-oversized"]
    for case in PDF_CASES:
        assert set(case.limits()) <= READER_LIMIT_NAMES
        assert case.declared_media_type == ""
        assert case.declaration() == {"media_type": ""}


@pytest.mark.parametrize("case", DICOM_CASES, ids=[c.name for c in DICOM_CASES])
def test_dicom_fixture_fails_at_the_media_type_boundary(
    case: MalformedHeaderCase,
) -> None:
    detected = detect_media_type(case.payload)
    assert detected == case.expected_detected
    report = preflight_asset(_manifest(case), case.payload)
    assert report.status is PreflightStatus.ABSTAIN
    assert report.detected_media_type == case.expected_detected
    assert report.abstention is not None
    assert [(finding.check, finding.reason_code) for finding in report.findings] == [
        ("media_type", case.expected_reason)
    ]


@pytest.mark.parametrize("case", DICOM_CASES, ids=[c.name for c in DICOM_CASES])
def test_dicom_magic_position_matches_the_bounded_prefix(
    case: MalformedHeaderCase,
) -> None:
    bounded = case.payload[:MAX_MEDIA_TYPE_PREFIX_BYTES]
    magic = bounded[128:132] if len(bounded) >= 132 else b""
    if case.expected_detected:
        assert magic == DICOM_MAGIC
    else:
        assert magic != DICOM_MAGIC


def test_mismatch_fixture_contradicts_a_supported_declaration() -> None:
    case = case_by_name("dicom-header-inconsistent")
    assert case.declaration() == {"media_type": "image/bmp", "width": 8, "height": 8}
    assert detect_media_type(case.payload) == "application/dicom"
    assert case.expected_reason == "mismatch"


def test_fixture_lookup_helpers_reject_unknown_keys() -> None:
    assert case_by_name("pdf-header-cyclic").corruption == "cyclic"
    with pytest.raises(KeyError):
        case_by_name("pdf-header-missing")
    with pytest.raises(ValueError):
        cases_for("nifti")
    with pytest.raises(ValueError):
        cases_for("")


def test_case_validation_rejects_out_of_contract_values() -> None:
    base: dict[str, object] = {
        "name": "pdf-header-synthetic",
        "modality": "pdf",
        "corruption": "truncated",
        "payload": b"%PDF-1.7\n",
        "boundary": PDF_BOUNDARY,
        "expected_reason": "pdf_catalog_missing",
    }
    MalformedHeaderCase(**base)  # type: ignore[arg-type]
    for override in (
        {"modality": "nifti"},
        {"corruption": "encrypted"},
        {"boundary": "decoder"},
        {"payload": b""},
        {"payload": b"\x00" * (MAX_FIXTURE_BYTES + 1)},
        {"declared_media_type": "application/dicom"},
        {"name": "dicom-header-synthetic", "boundary": DICOM_BOUNDARY},
    ):
        with pytest.raises(ValueError):
            MalformedHeaderCase(**{**base, **override})  # type: ignore[arg-type]


def test_every_fixture_runs_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("the fixtures must not touch the network")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    for case in PDF_CASES:
        report = read_pdf_geometry(case.payload, **case.limits())
        assert case.expected_reason in report.reason_codes
    for case in DICOM_CASES:
        report = preflight_asset(_manifest(case), case.payload)
        assert report.status is PreflightStatus.ABSTAIN
