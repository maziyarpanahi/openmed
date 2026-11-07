"""Actual synthetic pixels, container carriers and all-or-nothing privacy gates."""

import hashlib
import io
import json
import logging
import socket
from dataclasses import replace

import pytest

Image = pytest.importorskip("PIL.Image")
np = pytest.importorskip("numpy")
pydicom = pytest.importorskip("pydicom")
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian

from openmed.multimodal.page_rotation import PageRotation
from openmed.multimodal.privacy.pixel_coverage import (
    LocalPixelDetector,
    PixelCoverageError,
    PixelCoverageLimits,
    PixelFramePlan,
    PixelInspection,
    audit_pixel_redaction,
    inventory_pixel_frames,
)
from openmed.multimodal.vision.dicom.series_sampler import (
    DisplayWindow,
    SamplingPolicy,
    sample_dicom_series,
)

PRIVATE = "SYNTHETIC-PRIVATE-2795"
WINDOW = DisplayWindow(center=127.5, width=255)
MARK = (5, 3, 9, 6)


def image_bytes(*, count=1, mark=MARK, format="PNG", metadata=False):
    frames = []
    for _ in range(count):
        image = Image.new("RGB", (16, 12), (30, 30, 30))
        image.paste((255, 255, 255), mark)
        frames.append(image)
    stream = io.BytesIO()
    options = {}
    if metadata:
        from PIL.PngImagePlugin import PngInfo

        info = PngInfo()
        info.add_text("Comment", PRIVATE)
        options["pnginfo"] = info
    if count > 1:
        options.update(save_all=True, append_images=frames[1:])
    frames[0].save(stream, format=format, **options)
    for frame in frames:
        frame.close()
    return stream.getvalue()


class BrightRegionDetector:
    """Deterministic synthetic marker detector, not a clinical PHI detector."""

    def __init__(self):
        self.sizes = []
        self.complete = True

    def __call__(self, image):
        self.sizes.append(image.size)
        mask = np.asarray(image).min(axis=2) > 250
        if mask.any():
            ys, xs = np.where(mask)
            boxes = (
                (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
            )
        else:
            boxes = ()
        return PixelInspection(boxes, self.complete)


def provider(callback=None, digest="a"):
    return LocalPixelDetector(
        digest=digest * 64, inspect=callback or BrightRegionDetector(), local_only=True
    )


def plans_for(source, *, kind="raster", masks=(MARK,), **kwargs):
    frames = inventory_pixel_frames(
        source, source_kind=kind, window=WINDOW if kind == "dicom" else None
    )
    return tuple(
        PixelFramePlan(reference=frame.reference, masks=masks, **kwargs)
        for frame in frames
    )


def audit(source, plans=None, *, kind="raster", **kwargs):
    return audit_pixel_redaction(
        source,
        plans if plans is not None else plans_for(source, kind=kind),
        detector=kwargs.pop("detector", provider()),
        residual_detector=kwargs.pop("residual_detector", provider(digest="b")),
        source_kind=kind,
        window=WINDOW if kind == "dicom" else None,
        **kwargs,
    )


def test_real_raster_detection_masks_output_verification_and_metadata_exclusion():
    source = image_bytes(metadata=True)
    result = audit(source)
    assert result.accepted, result.to_json()
    assert result == audit(source)
    report = json.loads(result.to_json())
    row = report["records"][0]
    assert row["detected_critical_regions"] == [list(MARK)]
    assert row["unresolved_regions"] == row["residual_regions"] == []
    assert row["complete"] and row["pixel_verification"] == "exact_decoded_raster"
    assert report["source_digest"] == hashlib.sha256(source).hexdigest()
    assert not report["source_container_released"]
    assert PRIVATE not in repr(result) + result.to_json()
    with Image.open(io.BytesIO(result.outputs[0])) as image:
        assert image.mode == "RGB" and image.size == (16, 12) and image.info == {}
        assert image.getpixel((6, 4)) == (0, 0, 0)
        assert image.getpixel((0, 0)) == (30, 30, 30)
    assert PRIVATE.encode() not in result.outputs[0]


@pytest.mark.parametrize(
    "regions",
    [
        (),
        ((1, 0, 16, 12),),
        ((0, 0, 16, 5), (0, 6, 16, 12)),
        ((0, 0, 8, 12), (0, 0, 8, 12)),
    ],
)
def test_missing_borders_holes_and_duplicate_area_do_not_count_as_coverage(regions):
    source = image_bytes()
    detector = BrightRegionDetector()
    result = audit(
        source,
        plans_for(source, inspected_regions=regions),
        detector=provider(detector),
    )
    assert result.failure == "regional_coverage_incomplete"
    assert not result.outputs and not detector.sizes


def test_crop_coordinates_and_tiled_union_are_bound_to_actual_detector_inputs():
    source = image_bytes()
    detector = BrightRegionDetector()
    result = audit(
        source,
        plans_for(source, inspected_regions=((0, 0, 10, 12), (8, 0, 16, 12))),
        detector=provider(detector),
    )
    assert result.accepted and detector.sizes == [(10, 12), (8, 12)]
    assert json.loads(result.to_json())["records"][0]["detected_critical_regions"] == [
        [5, 3, 9, 6],
        [8, 3, 9, 6],
    ]


def test_ignored_regions_hold_even_when_other_crops_cover_them():
    source = image_bytes()
    result = audit(source, plans_for(source, ignored_regions=((0, 0, 1, 12),)))
    assert result.failure == "regional_coverage_incomplete" and not result.outputs


@pytest.mark.parametrize("masks", [(), ((5, 3, 8, 6),), ((5, 3, 7, 6), (8, 3, 9, 6))])
def test_unresolved_critical_regions_never_release_partial_redactions(masks):
    source = image_bytes()
    result = audit(source, plans_for(source, masks=masks))
    assert result.failure == "critical_region_unresolved" and not result.outputs
    assert json.loads(result.to_json())["records"][0]["unresolved_regions"] == [
        list(MARK)
    ]


def test_union_of_masks_resolves_one_identifier_without_requiring_one_big_box():
    source = image_bytes()
    assert audit(source, plans_for(source, masks=((5, 3, 7, 6), (7, 3, 9, 6)))).accepted


def test_known_critical_region_remains_required_when_detector_misses_it():
    source = image_bytes()
    empty = provider(lambda _: PixelInspection())
    result = audit(
        source, plans_for(source, masks=(), critical_regions=(MARK,)), detector=empty
    )
    assert result.failure == "critical_region_unresolved"


def test_uncertain_or_missing_detector_is_not_empty_clean_evidence():
    source = image_bytes()
    incomplete = BrightRegionDetector()
    incomplete.complete = False
    assert audit(source, detector=provider(incomplete)).failure == "detector_incomplete"
    result = audit_pixel_redaction(
        source, plans_for(source), detector=None, residual_detector=provider()
    )
    assert not result.accepted and not result.outputs
    with pytest.raises(PixelCoverageError):
        LocalPixelDetector(digest="a" * 64, inspect=incomplete, local_only=False)


@pytest.mark.parametrize("rotation", list(PageRotation))
def test_real_transform_matches_projected_mask_for_all_quarter_turns(rotation):
    source = image_bytes()
    result = audit(
        source,
        plans_for(source, crop=(2, 1, 14, 11), rotation=rotation, output_size=(24, 20)),
    )
    assert result.accepted, result.to_json()
    row = json.loads(result.to_json())["records"][0]
    assert len(row["transforms"]) == (2 if rotation == 0 else 3)
    with Image.open(io.BytesIO(result.outputs[0])) as image:
        assert image.size == (24, 20)
        for box in row["output_masks"]:
            with image.crop(box) as masked:
                assert masked.getextrema() == ((0, 0), (0, 0), (0, 0))
        assert not BrightRegionDetector()(image).critical_regions


def test_crop_cannot_discard_protected_mask_support():
    source = image_bytes()
    result = audit(source, plans_for(source, crop=(6, 0, 16, 12)))
    assert result.failure == "mask_lineage_rejected" and not result.outputs


def test_source_change_stale_plan_duplicate_missing_or_invented_frame_fails():
    source = image_bytes(count=2, format="TIFF")
    plans = plans_for(source)
    for bad in (
        plans[:1],
        plans + plans[:1],
        (replace(plans[0], reference="f" * 64), plans[1]),
    ):
        assert audit(source, bad).failure == "frame_coverage_incomplete"
    changed = image_bytes(count=2, format="TIFF", mark=(1, 1, 2, 2))
    assert audit(changed, plans).failure == "frame_coverage_incomplete"


def test_late_frame_residual_or_cancellation_withholds_all_earlier_outputs():
    source = image_bytes(count=2, format="TIFF")
    calls = []

    def residual(image):
        calls.append(image.size)
        return PixelInspection(((0, 0, 1, 1),) if len(calls) == 2 else ())

    result = audit(source, residual_detector=provider(residual))
    assert result.failure == "residual_identifier" and not result.outputs
    assert json.loads(result.to_json())["records"][0]["complete"]
    calls.clear()

    def cancelled():
        return len(calls) >= 2

    result = audit(source, residual_detector=provider(residual), cancel_check=cancelled)
    assert result.failure == "cancelled" and not result.outputs


@pytest.mark.parametrize(
    "limit",
    [
        dict(max_source_bytes=1),
        dict(max_frames=1),
        dict(max_frame_pixels=10),
        dict(max_total_pixels=200),
        dict(max_output_bytes=1),
    ],
)
def test_resource_limits_fail_closed_with_no_export(limit):
    source = image_bytes(count=2, format="TIFF")
    result = audit(source, limits=PixelCoverageLimits(**limit))
    assert not result.accepted and not result.outputs


def test_no_network_no_disk_write_and_fixed_backend_error_evidence(monkeypatch, caplog):
    source = image_bytes()
    plans = plans_for(source)

    def forbidden(*args, **kwargs):
        raise AssertionError("external side effect")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    assert audit(source, plans).accepted

    def fail(image):
        raise RuntimeError(PRIVATE)

    with caplog.at_level(logging.WARNING):
        result = audit(source, plans, detector=provider(fail))
    assert result.failure == "pixel_audit_failed" and not result.outputs
    assert PRIVATE not in repr(result) + result.to_json() + caplog.text


def dicom(*, frames=1, icons=False, overlay=False):
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = "1.2.826.0.1.3680043.10.543.2795"
    ds.SeriesInstanceUID = "1.2.826.0.1.3680043.10.543.2796"
    ds.FrameOfReferenceUID = "1.2.826.0.1.3680043.10.543.2797"
    ds.PatientName = PRIVATE
    ds.Modality = "CT"
    ds.Rows, ds.Columns = 12, 16
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.PixelSpacing = [1, 1]
    ds.AcquisitionNumber = 1
    if frames == 1:
        ds.ImagePositionPatient = [0, 0, 0]
    else:
        ds.NumberOfFrames = frames
        groups = []
        for i in range(frames):
            group, position = Dataset(), Dataset()
            position.ImagePositionPatient = [0, 0, i * 2]
            group.PlanePositionSequence = [position]
            groups.append(group)
        ds.PerFrameFunctionalGroupsSequence = groups
    pixels = np.full((frames, 12, 16), 30, dtype="uint8")
    pixels[:, MARK[1] : MARK[3], MARK[0] : MARK[2]] = 255
    ds.PixelData = pixels.tobytes()
    if icons:
        icon = Dataset()
        for name in (
            "Rows",
            "Columns",
            "SamplesPerPixel",
            "BitsAllocated",
            "BitsStored",
            "HighBit",
            "PixelRepresentation",
            "PhotometricInterpretation",
        ):
            setattr(icon, name, getattr(ds, name))
        icon.PixelData = pixels[0].tobytes()
        ds.IconImageSequence = [icon]
    if overlay:
        mask = np.zeros((12, 16), dtype="uint8")
        mask[MARK[1] : MARK[3], MARK[0] : MARK[2]] = 1
        for offset, vr, value in (
            (0x0010, "US", 12),
            (0x0011, "US", 16),
            (0x0040, "CS", "G"),
            (0x0050, "SS", [-2, 1]),
            (0x0100, "US", 1),
            (0x0102, "US", 0),
            (0x3000, "OW", np.packbits(mask, bitorder="little").tobytes()),
        ):
            ds.add_new((0x6000, offset), vr, value)
    return ds


def encode(ds):
    stream = io.BytesIO()
    pydicom.dcmwrite(stream, ds, enforce_file_format=True)
    return stream.getvalue()


def test_dicom_inventory_scans_frames_icons_and_off_image_overlay_planes():
    source = encode(dicom(frames=3, icons=True, overlay=True))
    frames = inventory_pixel_frames(source, source_kind="dicom", window=WINDOW)
    assert [f.carrier for f in frames] == ["image", "image", "image", "icon", "overlay"]
    result = audit(source, kind="dicom")
    assert result.accepted and len(result.outputs) == 5, result.to_json()
    assert PRIVATE not in result.to_json() + repr(frames)
    for output in result.outputs:
        assert PRIVATE.encode() not in output
        with Image.open(io.BytesIO(output)) as image:
            assert (
                image.info == {} and not BrightRegionDetector()(image).critical_regions
            )
    plans = plans_for(source, kind="dicom")
    assert (
        audit(source, plans[:-1], kind="dicom").failure == "frame_coverage_incomplete"
    )
    assert (
        audit(source, (*plans[:-1], replace(plans[-1], masks=())), kind="dicom").failure
        == "critical_region_unresolved"
    )


def test_vlm_sampled_frames_do_not_grant_privacy_coverage_to_unexamined_frames():
    source = encode(dicom(frames=4))
    sampling = sample_dicom_series(
        (source,),
        window=WINDOW,
        policy=SamplingPolicy(max_selected=2, max_gap_mm=10, min_signal_range=0),
    )
    assert sampling.unexamined
    result = audit(source, kind="dicom", sampling_plan=sampling)
    assert result.accepted, result.to_json()
    rows = json.loads(result.to_json())["records"]
    assert len(rows) == 4 and all(row["complete"] for row in rows)
    assert any(row["vlm_selected"] is False for row in rows)
    assert (
        audit(
            source,
            plans_for(source, kind="dicom")[:1],
            kind="dicom",
            sampling_plan=sampling,
        ).failure
        == "frame_coverage_incomplete"
    )
    changed = dicom(frames=4)
    changed.PatientName = "OTHER-SYNTHETIC"
    assert not audit(encode(changed), kind="dicom", sampling_plan=sampling).accepted


@pytest.mark.parametrize(
    "mutation",
    [
        "compressed",
        "embedded_overlay",
        "nested_pixels",
        "float_pixels",
        "truncated",
        "fractional_count",
    ],
)
def test_unsupported_or_ambiguous_dicom_never_claims_coverage(mutation):
    ds = dicom(overlay=True)
    if mutation == "compressed":
        from pydicom.encaps import encapsulate

        ds.file_meta.TransferSyntaxUID = "1.2.840.10008.1.2.4.50"
        ds.PixelData = encapsulate([ds.PixelData])
    elif mutation == "embedded_overlay":
        del ds[(0x6000, 0x3000)]
    elif mutation == "nested_pixels":
        item = Dataset()
        item.add_new(0x7FE00010, "OB", b"00")
        ds.ReferencedImageSequence = [item]
    elif mutation == "float_pixels":
        ds.FloatPixelData = b"0000"
    elif mutation == "truncated":
        ds.PixelData = b"00"
    else:
        with pytest.warns(UserWarning):
            ds.NumberOfFrames = "1.5"
    source = encode(ds)
    with pytest.raises(PixelCoverageError, match="pixel_inventory_rejected") as caught:
        inventory_pixel_frames(source, source_kind="dicom", window=WINDOW)
    assert caught.value.__context__ is None
    assert not audit(source, (), kind="dicom").accepted


def test_invalid_detector_boxes_and_exception_payload_never_enter_reports():
    source = image_bytes()
    for inspection in (
        PixelInspection(((0, 0, 99, 99),)),
        PixelInspection(((PRIVATE, 0, 1, 1),)),
        PRIVATE,
        PixelInspection(complete=PRIVATE),
    ):
        result = audit(source, detector=provider(lambda image, reply=inspection: reply))
        assert not result.accepted and not result.outputs
        assert PRIVATE not in result.to_json() + repr(result)


def test_decoder_debug_metadata_is_suppressed_and_filters_are_restored(caplog):
    with Image.open(io.BytesIO(image_bytes())) as image:
        stream = io.BytesIO()
        image.save(stream, format="TIFF", tiffinfo={270: PRIVATE})
    source = stream.getvalue()
    logger = logging.getLogger("PIL.TiffImagePlugin")
    filters = list(logger.filters)
    with caplog.at_level(logging.DEBUG):
        assert audit(source).accepted
    assert not any(record.name == "PIL.TiffImagePlugin" for record in caplog.records)
    assert PRIVATE not in caplog.text
    assert logger.filters == filters


def test_exporter_failure_withholds_bytes_even_with_clean_residual_detector(
    monkeypatch,
):
    import openmed.multimodal.privacy.pixel_coverage as module

    def fail(*args, **kwargs):
        raise RuntimeError(PRIVATE)

    source = image_bytes()
    monkeypatch.setattr(module, "render_redacted_raster_pages", fail)
    result = audit(source)
    assert not result.accepted and not result.outputs
    assert PRIVATE not in result.to_json()


def test_signed_stored_bits_and_monochrome_one_are_inspected_after_windowing():
    ds = dicom()
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 1
    ds.PhotometricInterpretation = "MONOCHROME1"
    ds.RescaleSlope = 1
    ds.RescaleIntercept = 128
    values = np.full((12, 16), 100, dtype="<i2")
    values[MARK[1] : MARK[3], MARK[0] : MARK[2]] = -128
    ds.PixelData = values.tobytes()
    result = audit(encode(ds), kind="dicom")
    assert result.accepted, result.to_json()
    assert json.loads(result.to_json())["records"][0]["detected_critical_regions"] == [
        list(MARK)
    ]


def test_rgb_icon_and_multiframe_overlay_are_independently_inspected():
    ds = dicom(icons=True, overlay=True)
    icon = ds.IconImageSequence[0]
    icon.SamplesPerPixel = 3
    icon.PhotometricInterpretation = "RGB"
    icon.PlanarConfiguration = 0
    icon.PixelData = np.repeat(
        np.frombuffer(icon.PixelData, dtype="uint8"), 3
    ).tobytes()
    ds.add_new((0x6000, 0x0015), "IS", 2)
    ds.add_new((0x6000, 0x0051), "US", 1)
    ds[(0x6000, 0x3000)].value *= 2
    source = encode(ds)
    result = audit(source, kind="dicom")
    assert result.accepted and len(result.outputs) == 4, result.to_json()
    assert [r["frame"]["carrier"] for r in json.loads(result.to_json())["records"]] == [
        "image",
        "icon",
        "overlay",
        "overlay",
    ]


def test_transparent_hidden_color_is_not_retained_in_export():
    with Image.new("RGBA", (16, 12), (255, 0, 0, 0)) as image:
        stream = io.BytesIO()
        image.save(stream, format="PNG")
    source = stream.getvalue()
    # Transparent storage becomes a white flattened raster; mask all of it.
    result = audit(source, plans_for(source, masks=((0, 0, 16, 12),)))
    assert result.accepted
    with Image.open(io.BytesIO(result.outputs[0])) as image:
        assert image.mode == "RGB" and image.getextrema() == ((0, 0), (0, 0), (0, 0))


def test_nested_overlay_is_not_silently_omitted():
    ds = dicom(icons=True)
    ds.IconImageSequence[0].add_new((0x6000, 0x0010), "US", 12)
    result = audit(encode(ds), (), kind="dicom")
    assert result.failure == "unsupported_pixel_carrier" and not result.outputs
