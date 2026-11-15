"""Real DICOM roundtrips and independent synthetic pixel/mask privacy controls."""

import hashlib
import json
import re
import socket
from dataclasses import replace
from io import BytesIO

import pytest
from PIL import Image, ImageChops, ImageFont
from pydicom import dcmread, dcmwrite
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage

from openmed.training.synthetic.burned_in import BurnedInFont
from openmed.training.synthetic.dicom_overlays import (
    DicomCanvas,
    DicomOverlayError,
    DicomOverlayLimits,
    DicomOverlaySpec,
    DicomPrivacyField,
    OverlayRole,
    generate_dicom_overlays,
    reproduce_dicom_overlays,
)


@pytest.fixture(scope="module")
def fixture():
    return generate_dicom_overlays(spec=DicomOverlaySpec(seed=42))


@pytest.mark.parametrize("canvas", list(DicomCanvas))
@pytest.mark.parametrize("seed", [0, 42, 77])
def test_all_canvases_have_real_dicom_pixels_matching_the_png(canvas, seed):
    result = generate_dicom_overlays(spec=DicomOverlaySpec(seed=seed, canvas=canvas))
    dataset = dcmread(BytesIO(result.dicom_bytes))
    assert dataset.file_meta.TransferSyntaxUID == ExplicitVRLittleEndian
    assert dataset.SOPClassUID == SecondaryCaptureImageStorage
    assert dataset.file_meta.MediaStorageSOPInstanceUID == dataset.SOPInstanceUID
    assert dataset.BurnedInAnnotation == "YES"
    assert dataset.PatientIdentityRemoved == "NO"
    assert dataset.ImageType == ["DERIVED", "SECONDARY"]
    assert dataset.Rows == result.spec.size[1]
    assert dataset.Columns == result.spec.size[0]
    with Image.open(BytesIO(result.image_png)) as image:
        assert dataset.pixel_array.tobytes() == image.tobytes()
        assert dataset.pixel_array.shape == (image.height, image.width)
    assert result.lineage.accepted
    assert (
        result.lineage.source.source_digest
        == hashlib.sha256(result.dicom_bytes).hexdigest()
    )


def test_dicom_metadata_and_painted_identifiers_agree(fixture):
    dataset = dcmread(BytesIO(fixture.dicom_bytes))
    fields = {item.keyword: item for item in fixture.metadata}
    assert set(fields) == set(DicomPrivacyField)
    for item in fixture.metadata:
        assert str(getattr(dataset, item.keyword.value)) == item.value
    identifiers = [
        item for item in fixture.annotations if item.role is OverlayRole.IDENTIFIER
    ]
    assert len(identifiers) == 5
    for item in identifiers:
        field = fields[item.field_code]
        assert fixture.text[slice(*item.span)] == field.value
        assert item.label == field.label
    for key in ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"):
        assert getattr(dataset, key).is_valid


def test_exact_glyph_masks_match_painted_pixels_and_identity_lineage(fixture):
    with (
        Image.open(BytesIO(fixture.image_png)) as image,
        Image.open(BytesIO(fixture.background_png)) as background,
    ):
        for item in fixture.annotations:
            with (
                Image.open(BytesIO(item.mask_png)) as mask,
                background.crop(item.box) as expected,
                image.crop(item.box) as actual,
            ):
                assert mask.getbbox() == (0, 0, *mask.size)
                expected.paste(255, (0, 0, *mask.size), mask)
                assert ImageChops.difference(expected, actual).getbbox() is None
            if item.role is OverlayRole.IDENTIFIER:
                source = next(
                    node
                    for node in fixture.lineage.source_nodes
                    if node.node_id == item.annotation_id[:32]
                )
                projected = next(
                    mask
                    for mask in fixture.lineage.masks
                    if mask.node_id == source.node_id
                )
                assert source.region == item.box
                assert projected.raster_box == item.box


def test_masks_remove_all_identifier_ink_and_omission_leaves_ink(fixture):
    identifiers = [
        item for item in fixture.annotations if item.role is OverlayRole.IDENTIFIER
    ]
    with (
        Image.open(BytesIO(fixture.image_png)) as image,
        Image.open(BytesIO(fixture.background_png)) as background,
        Image.open(BytesIO(fixture.phi_mask_png)) as mask,
    ):
        restored = image.copy()
        restored.paste(
            background, (0, 0), mask.point(lambda value: 255 if value else 0)
        )
        for item in identifiers:
            with restored.crop(item.box) as region, background.crop(item.box) as clean:
                assert ImageChops.difference(region, clean).getbbox() is None
        omitted = image.copy()
        for item in identifiers[1:]:
            with (
                Image.open(BytesIO(item.mask_png)) as tile,
                background.crop(item.box) as clean,
            ):
                omitted.paste(
                    clean, item.box, tile.point(lambda value: 255 if value else 0)
                )
        with (
            omitted.crop(identifiers[0].box) as exposed,
            background.crop(identifiers[0].box) as clean,
        ):
            assert ImageChops.difference(exposed, clean).getbbox() is not None
        restored.close()
        omitted.close()


def test_ruler_and_acquisition_annotations_are_actual_pixels(fixture):
    assert {item.role for item in fixture.annotations} == set(OverlayRole)
    ruler = fixture.ruler
    with (
        Image.open(BytesIO(ruler.mask_png)) as mask,
        Image.open(BytesIO(fixture.image_png)) as image,
    ):
        assert mask.getbbox() == (0, 0, *mask.size)
        x = ruler.box[2] - 1
        assert len(ruler.ticks) > 2
        for index, y in enumerate(ruler.ticks):
            length = 12 if index % 4 == 0 else 7
            assert all(
                image.getpixel((position, y)) == 255
                for position in range(x - length, x + 1)
            )
    assert all(
        b - a == ruler.tick_spacing for a, b in zip(ruler.ticks, ruler.ticks[1:])
    )


def test_reproduction_checks_actual_files_not_only_spec(fixture):
    assert reproduce_dicom_overlays(fixture) == fixture
    changed = generate_dicom_overlays(spec=replace(fixture.spec, seed=43))
    assert changed.dicom_bytes != fixture.dicom_bytes
    assert changed.metadata != fixture.metadata
    assert changed.image_png != fixture.image_png
    assert changed.digest != fixture.digest


def test_odd_pixel_count_pads_dicom_only_and_large_font_fits():
    result = generate_dicom_overlays(
        spec=DicomOverlaySpec(seed=42, size=(385, 385), font_size=24)
    )
    dataset = dcmread(BytesIO(result.dicom_bytes))
    assert len(dataset.PixelData) == 385 * 385 + 1
    assert dataset.PixelData[-1:] == b"\0"
    assert dataset.pixel_array.shape == (385, 385)
    assert any(item.font_size < 24 for item in result.annotations)


def test_custom_local_font_bytes_reproduce_without_path_in_receipt(tmp_path, fixture):
    path = tmp_path / "private-path.ttf"
    path.write_bytes(ImageFont.load_default(size=18).font_bytes)
    font = BurnedInFont(path=path, license="CC0-1.0")
    result = generate_dicom_overlays(spec=fixture.spec, font=font)
    assert result == fixture
    assert reproduce_dicom_overlays(result, font=font) == result
    assert str(path) not in result.to_json()


def test_offline_generation_does_not_create_files(monkeypatch, tmp_path, fixture):
    def reject_network(*args, **kwargs):
        raise AssertionError("unexpected network")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket, "create_connection", reject_network)
    assert generate_dicom_overlays(spec=fixture.spec) == fixture
    assert list(tmp_path.iterdir()) == []


def test_reports_and_repr_exclude_generated_identifiers(fixture):
    report = fixture.to_json()
    data = json.loads(report)
    assert data["synthetic"] and not data["clinical_validation"]
    assert data["ruler"]["units"] == "pixels"
    for item in fixture.metadata:
        assert item.value not in report
        assert item.value not in repr(fixture)
        assert item.value not in repr(item)
    assert "data:image" not in report


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": True},
        {"seed": -1},
        {"seed": 0, "size": (100, 512)},
        {"seed": 0, "size": (1025, 512)},
        {"seed": 0, "font_size": 25},
        {"seed": 0, "canvas": "private"},
    ],
)
def test_invalid_specs_have_fixed_errors(kwargs):
    with pytest.raises(DicomOverlayError, match="^synthetic_dicom_overlay_rejected$"):
        DicomOverlaySpec(**kwargs)


@pytest.mark.parametrize(
    "limits",
    [
        DicomOverlayLimits(max_pixels=1),
        DicomOverlayLimits(max_mask_work_pixels=1),
        DicomOverlayLimits(max_payload_bytes=1),
    ],
)
def test_resource_limits_refuse_whole_fixture(limits):
    with pytest.raises(DicomOverlayError, match="^synthetic_dicom_overlay_rejected$"):
        generate_dicom_overlays(spec=DicomOverlaySpec(seed=42), limits=limits)


def test_missing_font_cannot_fall_back_or_retain_backend_exception(tmp_path):
    with pytest.raises(DicomOverlayError) as error:
        generate_dicom_overlays(
            spec=DicomOverlaySpec(seed=42),
            font=BurnedInFont(path=tmp_path / "private.ttf", license="OFL-1.1"),
        )
    assert str(error.value) == "synthetic_dicom_overlay_rejected"
    assert error.value.__context__ is None and error.value.__cause__ is None


@pytest.mark.parametrize(
    "keyword,value",
    [
        ("PatientName", "PRIVATE_SENTINEL"),
        ("Columns", 400),
        ("BitsAllocated", 16),
        ("BurnedInAnnotation", "NO"),
    ],
)
def test_metadata_or_pixel_geometry_tampering_is_rejected(fixture, keyword, value):
    dataset = dcmread(BytesIO(fixture.dicom_bytes))
    setattr(dataset, keyword, value)
    stream = BytesIO()
    dcmwrite(stream, dataset, enforce_file_format=True)
    altered = replace(fixture, dicom_bytes=stream.getvalue())
    with pytest.raises(DicomOverlayError) as error:
        altered.to_json()
    assert error.value.__context__ is None
    assert "PRIVATE" not in str(error.value)
    with pytest.raises(DicomOverlayError):
        reproduce_dicom_overlays(altered)


@pytest.mark.parametrize("field", ["font_digest", "runtime_digest"])
def test_reproduction_refuses_changed_runtime_and_font(fixture, field):
    with pytest.raises(DicomOverlayError):
        reproduce_dicom_overlays(replace(fixture, **{field: "0" * 64}))


def test_dropped_identifier_mask_and_forged_labels_fail(fixture):
    with Image.new("L", fixture.spec.size) as image:
        stream = BytesIO()
        image.save(stream, format="PNG")
    with pytest.raises(DicomOverlayError):
        replace(fixture, phi_mask_png=stream.getvalue()).to_json()
    index = next(
        i
        for i, item in enumerate(fixture.annotations)
        if item.role is OverlayRole.IDENTIFIER
    )
    changed = list(fixture.annotations)
    changed[index] = replace(changed[index], label="PRIVATE_SENTINEL")
    with pytest.raises(DicomOverlayError):
        replace(fixture, annotations=tuple(changed)).to_json()


def test_structurally_edited_ruler_cannot_pass_regeneration(fixture):
    ruler = replace(
        fixture.ruler,
        tick_spacing=fixture.ruler.tick_spacing * 2,
        ticks=fixture.ruler.ticks[::2],
    )
    with pytest.raises(DicomOverlayError):
        reproduce_dicom_overlays(replace(fixture, ruler=ruler))
    with pytest.raises(DicomOverlayError):
        replace(fixture, ruler=ruler).to_json()


def test_executable_public_guide():
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    source = (root / "docs/training/synthetic/dicom-overlays.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", source, re.DOTALL)
    assert blocks
    for block in blocks:
        exec(compile(block, "dicom-overlays-guide", "exec"), {})
