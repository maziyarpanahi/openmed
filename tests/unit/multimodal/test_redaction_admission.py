"""Offline negative controls for resource admission at redaction entry points."""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace

import pytest

from openmed.multimodal import (
    MOBILE_V1,
    RedactionAdmissionError,
    base,
    documents_pdf,
    image,
    redact_image,
)
from openmed.multimodal import dicom as dicom_mod
from openmed.multimodal.documents_pdf_tables import extract_pdf_regions
from openmed.multimodal.redaction_admission import (
    _HeaderStream,
    admit_dicom,
    admit_image,
    admit_pdf,
    redaction_limit_profile,
)
from tests.fixtures.multimodal import redaction_assets as assets


def forbidden(*args, **kwargs):
    pytest.fail("decoder or provider was invoked before admission")


@pytest.mark.parametrize(
    ("payload", "profile", "code"),
    [
        (assets.png(5000, 5000), MOBILE_V1, "image_pixel_limit_exceeded"),
        (assets.png(frames=129), MOBILE_V1, "limit_exceeded"),
        (
            assets.tiff(((8, 8),) * 3),
            MOBILE_V1.with_limits(max_frames=2),
            "limit_exceeded",
        ),
        (assets.tiff(((8, 8), (5000, 5000))), MOBILE_V1, "tiff_pixel_limit_exceeded"),
        (
            assets.tiff(((8, 8),) * 2),
            MOBILE_V1.with_limits(max_total_pixels=100, max_pixels=100),
            "limit_exceeded",
        ),
        (assets.tiff(((8, 8),), cycle=True), MOBILE_V1, "tiff_ifd_offset_invalid"),
    ],
)
def test_image_rejected_before_pillow_or_ocr(monkeypatch, payload, profile, code):
    monkeypatch.setattr(image, "_import_pillow", forbidden)
    monkeypatch.setattr(image, "ocr", forbidden)
    with pytest.raises(RedactionAdmissionError) as raised:
        redact_image(BytesIO(payload), policy={"asset_limit_profile": profile})
    assert raised.value.reason_code == code


@pytest.mark.parametrize("order", ["<", ">"])
def test_classic_tiff_counts_and_policy_override(order):
    source = BytesIO(assets.tiff(((8, 8),) * 3, order=order))
    source.seek(4)
    with pytest.raises(RedactionAdmissionError):
        admit_image(source, MOBILE_V1.with_limits(max_frames=2))
    assert source.tell() == 4
    assert admit_image(source, MOBILE_V1.with_limits(max_frames=3)) == 3
    assert source.tell() == 4 and not source.closed


def test_mpo_frame_count_and_secondary_geometry_checked_before_pillow(monkeypatch):
    Image = pytest.importorskip("PIL.Image")
    payload = BytesIO()
    Image.new("RGB", (8, 8)).save(
        payload,
        format="MPO",
        save_all=True,
        append_images=[Image.new("RGB", (20, 20))],
    )
    assert admit_image(payload, MOBILE_V1) == 2
    monkeypatch.setattr(image, "_import_pillow", forbidden)
    for profile, field in (
        (MOBILE_V1.with_limits(max_frames=1), "frames"),
        (MOBILE_V1.with_limits(max_pixels=100), None),
    ):
        with pytest.raises(RedactionAdmissionError) as raised:
            redact_image(payload, policy={"asset_limit_profile": profile})
        assert raised.value.field_name == field


def test_apng_default_image_counts_toward_frame_bound():
    Image = pytest.importorskip("PIL.Image")
    payload = BytesIO()
    Image.new("RGB", (8, 8)).save(
        payload,
        format="PNG",
        save_all=True,
        default_image=True,
        append_images=[
            Image.new("RGB", (8, 8), "white"),
            Image.new("RGB", (8, 8), "black"),
        ],
    )
    assert admit_image(payload, MOBILE_V1) == 3
    with pytest.raises(RedactionAdmissionError) as raised:
        admit_image(payload, MOBILE_V1.with_limits(max_frames=2))
    assert raised.value.field_name == "frames"


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (assets.pdf(11), "pdf_page_limit"),
        (assets.pdf(width=5000, height=5000), "limit_exceeded"),
        (assets.pdf(2, declared=1), "page_count_mismatch"),
    ],
)
def test_pdf_rejected_before_parser_or_renderer(monkeypatch, payload, code):
    monkeypatch.setattr(documents_pdf, "_import_pdfplumber", forbidden)
    for handler in (
        documents_pdf._pdf_handler,
        lambda source: documents_pdf._render_redacted_pdf(source, ()),
    ):
        with pytest.raises(RedactionAdmissionError) as raised:
            handler(BytesIO(payload))
        assert raised.value.reason_code == code


def test_pdf_total_pixel_and_dpi_limits():
    profile = MOBILE_V1.with_limits(max_pixels=30000, max_total_pixels=30000)
    with pytest.raises(RedactionAdmissionError) as raised:
        admit_pdf(assets.pdf(2), profile)
    assert raised.value.field_name == "total_pixels"
    assert admit_pdf(assets.pdf(), profile, resolution=72) == 1
    with pytest.raises(RedactionAdmissionError):
        admit_pdf(assets.pdf(), profile, resolution=300)


def test_pdf_crop_does_not_hide_full_media_box_allocation(monkeypatch):
    payload = assets.pdf(width=5000, height=5000).replace(
        b"/MediaBox", b"/CropBox [0 0 72 72] /MediaBox"
    )
    monkeypatch.setattr(documents_pdf, "_import_pdfplumber", forbidden)
    with pytest.raises(RedactionAdmissionError) as raised:
        documents_pdf._render_redacted_pdf(BytesIO(payload), ())
    assert raised.value.field_name == "pixels"


@pytest.mark.parametrize("entry", ["extract", "regions", "render"])
def test_underdeclared_pdf_runtime_bound_precedes_page_work(monkeypatch, entry):
    profile = MOBILE_V1.with_limits(max_pages=1)
    visits = []

    class Page:
        width = height = 72

        def extract_words(self, **kwargs):
            visits.append("words")
            return []

        def to_image(self, **kwargs):
            visits.append("render")
            return SimpleNamespace(
                original=pytest.importorskip("PIL.Image").new("RGB", (150, 150))
            )

    class Pdf:
        pages = [Page(), Page()]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(
        documents_pdf,
        "_import_pdfplumber",
        lambda: SimpleNamespace(open=lambda _: Pdf()),
    )
    monkeypatch.setattr(
        "openmed.multimodal.documents_pdf_tables._import_pdfplumber",
        lambda: SimpleNamespace(open=lambda _: Pdf()),
    )
    monkeypatch.setattr(
        "openmed.multimodal.documents_pdf_tables._find_tables",
        lambda page: visits.append("tables") or (),
    )
    with pytest.raises(RedactionAdmissionError) as raised:
        if entry == "extract":
            documents_pdf.extract_pdf(BytesIO(assets.pdf()), _limit_profile=profile)
        elif entry == "regions":
            extract_pdf_regions(
                BytesIO(assets.pdf()),
                document=base.ExtractedDocument(""),
                _limit_profile=profile,
            )
        else:
            documents_pdf._render_redacted_pdf(
                BytesIO(assets.pdf()), (), policy={"asset_limit_profile": profile}
            )
    assert raised.value.field_name == "pages"
    assert visits == [
        {"extract": "words", "regions": "tables", "render": "render"}[entry]
    ]


def test_underdeclared_image_count_rejected_before_copy(monkeypatch):
    class Opened:
        n_frames = 129
        format = "TIFF"
        copy = seek = forbidden

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(image, "_import_pillow", lambda: (None, None, None, None))
    monkeypatch.setattr(image, "_open_image", lambda *args: Opened())
    with pytest.raises(RedactionAdmissionError) as raised:
        redact_image(BytesIO(assets.tiff(((8, 8),))))
    assert raised.value.field_name == "frames"


@pytest.mark.parametrize(("width", "height", "frames"), [(8, 8, 129), (5000, 5000, 1)])
def test_dicom_rejected_before_full_read_or_pixel_decode(
    monkeypatch, tmp_path, width, height, frames
):
    pydicom = pytest.importorskip("pydicom")
    source = tmp_path / "synthetic-private-name.dcm"
    source.write_bytes(assets.dicom(width, height, frames=frames))
    original = pydicom.dcmread
    calls = []

    def read(stream, **kwargs):
        assert kwargs["stop_before_pixels"] is True
        calls.append(kwargs)
        return original(stream, **kwargs)

    monkeypatch.setattr(pydicom, "dcmread", read)
    monkeypatch.setattr(dicom_mod, "_decompress_pixel_data", forbidden)
    monkeypatch.setattr(dicom_mod, "_copy_pixel_array", forbidden)
    with pytest.raises(RedactionAdmissionError) as raised:
        dicom_mod.redact_dicom_pixels(source, output_path=tmp_path / "output.dcm")
    assert raised.value.reason_code == "limit_exceeded"
    assert len(calls) == 1
    assert not (tmp_path / "output.dcm").exists()
    assert "synthetic-private-name" not in str(raised.value)


def test_dicom_actual_shape_is_bounded_without_silent_truncation():
    np = pytest.importorskip("numpy")
    dataset = SimpleNamespace(NumberOfFrames=1, SamplesPerPixel=1)
    with pytest.raises(RedactionAdmissionError) as raised:
        dicom_mod._iter_pixel_frames(np.zeros((129, 2, 2), dtype=np.uint8), dataset)
    assert raised.value.field_name == "frames"
    with pytest.raises(RedactionAdmissionError, match="frame_count_mismatch"):
        dicom_mod._iter_pixel_frames(np.zeros((2, 2, 2), dtype=np.uint8), dataset)


def test_underdeclared_dicom_payload_refused_before_ocr_or_output(
    monkeypatch, tmp_path
):
    pytest.importorskip("pydicom")
    payload = assets.dicom()
    # Two synthetic frames despite a declaration of one.
    payload = payload[:-68] + b"\x80\0\0\0" + b"\0" * 128
    source = tmp_path / "underdeclared.dcm"
    source.write_bytes(payload)
    monkeypatch.setattr(dicom_mod, "_detect_frame_pixel_findings", forbidden)
    with pytest.warns(UserWarning, match="frames"):
        with pytest.raises(RedactionAdmissionError) as raised:
            dicom_mod.redact_dicom_pixels(
                source,
                output_path=tmp_path / "output.dcm",
                policy={"asset_limit_profile": MOBILE_V1.with_limits(max_frames=1)},
            )
    assert raised.value.field_name == "frames"
    assert raised.value.observed == 2
    assert source.read_bytes() == payload
    assert not (tmp_path / "output.dcm").exists()


@pytest.mark.parametrize(
    "admit,payload", [(admit_image, assets.png()), (admit_pdf, assets.pdf())]
)
def test_byte_limit_precedes_header_read_and_restores_stream(admit, payload):
    source = BytesIO(payload)
    source.seek(4)
    with pytest.raises(RedactionAdmissionError) as raised:
        admit(source, MOBILE_V1.with_limits(max_byte_size=len(payload) - 1))
    assert raised.value.field_name == "byte_size"
    assert raised.value.observed == len(payload)
    assert source.tell() == 4 and not source.closed


def test_bounded_dicom_metadata_reader_refuses_unbounded_request():
    reader = _HeaderStream(BytesIO(b"synthetic"), 9)
    with pytest.raises(RedactionAdmissionError, match="header_byte_limit"):
        reader.read()
    with pytest.raises(RedactionAdmissionError, match="header_offset_invalid"):
        reader.seek(10000)


@pytest.mark.parametrize("payload", [b"private-patient-payload", assets.pdf()])
def test_image_type_and_read_errors_do_not_reflect_source(payload, tmp_path):
    source = tmp_path / "private-patient-path.png"
    source.write_bytes(payload)
    with pytest.raises(RedactionAdmissionError) as raised:
        admit_image(source, MOBILE_V1)
    assert raised.value.reason_code in {"unknown", "mismatch"}
    assert "private" not in repr(raised.value)
    assert raised.value.args == (raised.value.reason_code, None, None, None)


def test_policy_cannot_disable_limits_or_echo_values():
    assert redaction_limit_profile(None) == MOBILE_V1
    assert redaction_limit_profile({"asset_limit_profile": MOBILE_V1}) == MOBILE_V1
    with pytest.raises(
        RedactionAdmissionError, match="limit_profile_invalid"
    ) as raised:
        redaction_limit_profile({"asset_limit_profile": "private-patient-value"})
    assert "private" not in str(raised.value)


def test_dispatch_ocr_only_image_path_checks_limits(monkeypatch, tmp_path):
    source = tmp_path / "image.png"
    source.write_bytes(assets.png(5000, 5000))
    monkeypatch.setattr(base, "ensure_multimodal_available", lambda: None)
    monkeypatch.setattr(image, "ocr", forbidden)
    with pytest.raises(RedactionAdmissionError):
        base.redact_document(source)


def test_dicom_missing_numeric_evidence_fails_closed():
    pydicom = pytest.importorskip("pydicom")
    payload = assets.dicom().replace(b"\x28\x00\x11\x00US", b"\x28\x00\x12\x00US")
    with pytest.raises(RedactionAdmissionError, match="insufficient_metadata"):
        admit_dicom(payload, MOBILE_V1, pydicom)
