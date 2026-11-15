"""Actual synthetic native/RLE/JPEG bytes and reproducible regional VLM evidence."""

import hashlib
import io
import json
import logging
import socket
from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")
pydicom = pytest.importorskip("pydicom")
Image = pytest.importorskip("PIL.Image")
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.encaps import encapsulate
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    RLELossless,
)

from openmed.multimodal.abstention import (
    AbstentionReason,
    AbstentionRecord,
    AbstentionStage,
)
from openmed.multimodal.page_rotation import PageRotation
from openmed.multimodal.vision.dicom.pixel_provenance import (
    PixelDecoder,
    PixelProvenanceError,
    PixelProvenanceLimits,
    PixelRenderSpec,
    PixelStage,
    ResizeFilter,
    bind_region_provenance,
    render_dicom_frame,
    reproduce_dicom_frame,
    run_provenanced_dicom_vlm,
    verify_region_provenance,
)
from openmed.multimodal.vision.dicom.series_sampler import (
    DisplayWindow,
    SamplingPolicy,
    sample_dicom_series,
)
from openmed.multimodal.vision.providers.base import (
    ImageRegion,
    LocalProviderMetadata,
    Prompt,
    RuntimeBackend,
    RuntimeMetadata,
    VisualEvidence,
    VLMResult,
)

PRIVATE = "SYNTHETIC-PRIVATE-2801"
WINDOW = DisplayWindow(center=128, width=256)
SPEC = PixelRenderSpec(frame_index=0, window=WINDOW)


def dataset(
    *,
    count=1,
    signed=False,
    photo="MONOCHROME2",
    planar=0,
    syntax=ExplicitVRLittleEndian,
):
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = syntax
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = "1.2.826.0.1.3680043.10.543.2801"
    ds.SeriesInstanceUID = "1.2.826.0.1.3680043.10.543.2802"
    ds.FrameOfReferenceUID = "1.2.826.0.1.3680043.10.543.2803"
    ds.PatientName = PRIVATE
    ds.PatientID = PRIVATE
    ds.Modality = "CT"
    ds.ImagePositionPatient = [0, 0, 0]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.PixelSpacing = [1, 1]
    ds.AcquisitionNumber = 1
    ds.Rows, ds.Columns = 12, 16
    color = photo in ("RGB", "YBR_FULL", "YBR_FULL_422")
    ds.SamplesPerPixel = 3 if color else 1
    ds.BitsAllocated = ds.BitsStored = 8 if color else 16
    ds.HighBit = ds.BitsStored - 1
    ds.PixelRepresentation = int(signed)
    ds.PhotometricInterpretation = photo
    dtype = "uint8" if color else "<i2" if signed else "<u2"
    pixels = np.arange(count * 192, dtype=dtype).reshape(count, 12, 16)
    if count > 1:
        ds.NumberOfFrames = count
    if color:
        ds.PlanarConfiguration = planar
        channels = (pixels, np.full_like(pixels, 128), np.full_like(pixels, 128))
        pixels = np.stack(channels, axis=-1)
        if photo == "YBR_FULL_422":
            pixels = np.stack(
                (
                    pixels[:, :, ::2, 0],
                    pixels[:, :, 1::2, 0],
                    pixels[:, :, ::2, 1],
                    pixels[:, :, ::2, 2],
                ),
                axis=-1,
            )
        elif planar:
            pixels = pixels.transpose(0, 3, 1, 2)
    ds.PixelData = pixels.tobytes()
    return ds


def encode(ds):
    stream = io.BytesIO()
    pydicom.dcmwrite(stream, ds, enforce_file_format=True)
    return stream.getvalue()


def pixels(frame):
    with Image.open(io.BytesIO(frame.image.data)) as image:
        assert image.info == {}
        return np.asarray(image).copy()


def region(frame, box=(1, 1, 5, 4)):
    return ImageRegion(
        image_digest=frame.image.digest, x0=box[0], y0=box[1], x1=box[2], y1=box[3]
    )


def test_native_pipeline_records_every_actual_step_and_reproduces_exact_bytes():
    source = encode(dataset())
    frame = render_dicom_frame(source, spec=SPEC)
    proof = frame.provenance
    assert proof.source_digest == hashlib.sha256(source).hexdigest()
    assert [stage.operation for stage in proof.stages] == [
        "transfer_syntax",
        "frame",
        "decode",
        "rescale",
        "window",
        "color",
        "orientation",
        "resize",
        "encode",
    ]
    previous = proof.source_digest
    for stage in proof.stages:
        assert stage.previous_digest == previous
        previous = stage.digest
    expected = np.floor(np.arange(192).reshape(12, 16) / 256 * 255).astype("uint8")
    np.testing.assert_array_equal(
        pixels(frame), np.repeat(expected[:, :, None], 3, axis=2)
    )
    assert reproduce_dicom_frame(source, proof) == frame
    assert json.loads(proof.to_json())["spec"]["frame_index"] == 0
    assert PRIVATE not in proof.to_json() + repr(frame)
    assert PRIVATE.encode() not in frame.image.data


@pytest.mark.parametrize("syntax", [ExplicitVRLittleEndian, ImplicitVRLittleEndian])
def test_native_transfer_syntax_and_selected_frame_are_independently_bound(syntax):
    ds = dataset(count=2, syntax=syntax)
    source = encode(ds)
    first = render_dicom_frame(source, spec=SPEC)
    second = render_dicom_frame(source, spec=replace(SPEC, frame_index=1))
    assert first.provenance.header.transfer_syntax == syntax
    assert first.provenance.frame_reference != second.provenance.frame_reference
    assert (
        first.provenance.stages[1].output_digest
        != second.provenance.stages[1].output_digest
    )
    assert first.provenance.pixels_digest != second.provenance.pixels_digest


def test_frame_reference_joins_actual_sampler_records_without_patient_uids():
    source = encode(dataset())
    sampled = sample_dicom_series(
        (source,), window=WINDOW, policy=SamplingPolicy(max_selected=1)
    )
    frame = render_dicom_frame(source, spec=SPEC)
    assert frame.provenance.frame_reference == sampled.frames[0].reference
    assert frame.provenance.source_digest == sampled.frames[0].source_digest


def test_per_frame_rescale_overrides_shared_and_root_as_one_complete_pair():
    ds = dataset(count=2, signed=True)
    ds.RescaleSlope, ds.RescaleIntercept = 9, 10
    shared, shared_scale = Dataset(), Dataset()
    shared_scale.RescaleSlope, shared_scale.RescaleIntercept = 2, -4
    shared.PixelValueTransformationSequence = [shared_scale]
    ds.SharedFunctionalGroupsSequence = [shared]
    first, second, second_scale = Dataset(), Dataset(), Dataset()
    second_scale.RescaleSlope, second_scale.RescaleIntercept = 0.5, 3
    second.PixelValueTransformationSequence = [second_scale]
    ds.PerFrameFunctionalGroupsSequence = [first, second]
    source = encode(ds)
    a = render_dicom_frame(source, spec=SPEC)
    b = render_dicom_frame(source, spec=replace(SPEC, frame_index=1))
    assert (a.provenance.header.slope, a.provenance.header.intercept) == (2, -4)
    assert (b.provenance.header.slope, b.provenance.header.intercept) == (0.5, 3)
    assert (
        a.provenance.stages[3].parameters_digest
        != b.provenance.stages[3].parameters_digest
    )
    del second_scale.RescaleIntercept
    with pytest.raises(PixelProvenanceError):
        render_dicom_frame(encode(ds), spec=replace(SPEC, frame_index=1))


def test_signed_stored_bits_and_monochrome_one_inversion_are_recorded():
    ds = dataset(signed=True, photo="MONOCHROME1")
    ds.BitsStored, ds.HighBit = 12, 11
    ds.RescaleSlope, ds.RescaleIntercept = 1, 128
    values = np.arange(-96, 96, dtype="<i2").reshape(12, 16)
    ds.PixelData = values.tobytes()
    frame = render_dicom_frame(encode(ds), spec=SPEC)
    expected = 255 - np.floor((values + 128) / 256 * 255).astype("uint8")
    np.testing.assert_array_equal(pixels(frame)[:, :, 0], expected)
    assert frame.provenance.header.bits_stored == 12
    assert frame.provenance.header.signed == 1


@pytest.mark.parametrize(
    "photo,planar",
    [("RGB", 0), ("RGB", 1), ("YBR_FULL", 0), ("YBR_FULL", 1), ("YBR_FULL_422", 0)],
)
def test_color_conversion_and_planar_subsampling_are_actual_pixel_operations(
    photo, planar
):
    frame = render_dicom_frame(
        encode(dataset(photo=photo, planar=planar)), spec=PixelRenderSpec(frame_index=0)
    )
    actual = pixels(frame)
    ramp = np.arange(192, dtype="uint8").reshape(12, 16)
    np.testing.assert_array_equal(actual[:, :, 0], ramp)
    if photo != "RGB":
        np.testing.assert_array_equal(actual[:, :, 1], ramp)
        np.testing.assert_array_equal(actual[:, :, 2], ramp)
    else:
        assert (actual[:, :, 1:] == 128).all()


@pytest.mark.parametrize("rotation", list(PageRotation))
@pytest.mark.parametrize("filter", list(ResizeFilter))
def test_rotation_and_noninteger_resize_match_actual_declared_renderer(
    rotation, filter
):
    source = encode(dataset())
    base = render_dicom_frame(source, spec=SPEC)
    spec = replace(SPEC, rotation=rotation, output_size=(19, 7), resize_filter=filter)
    frame = render_dicom_frame(source, spec=spec)
    with Image.fromarray(np.rot90(pixels(base), k=-int(rotation) // 90)) as oriented:
        method = (
            Image.Resampling.NEAREST
            if filter is ResizeFilter.NEAREST
            else Image.Resampling.BILINEAR
        )
        with oriented.resize((19, 7), method) as expected:
            np.testing.assert_array_equal(pixels(frame), np.asarray(expected))
    assert reproduce_dicom_frame(source, frame.provenance) == frame
    assert frame.provenance.size == (19, 7)


def test_real_rle_decompression_reproduces_native_pixels_and_records_different_chain():
    ds = dataset(count=2)
    native = render_dicom_frame(encode(ds), spec=replace(SPEC, frame_index=1))
    ds.compress(RLELossless, generate_instance_uid=False)
    source = encode(ds)
    frame = render_dicom_frame(
        source, spec=replace(SPEC, frame_index=1, decoder=PixelDecoder.RLE)
    )
    np.testing.assert_array_equal(pixels(frame), pixels(native))
    assert frame.provenance.digest != native.provenance.digest
    assert (
        frame.provenance.stages[1].output_digest
        != native.provenance.stages[1].output_digest
    )
    assert reproduce_dicom_frame(source, frame.provenance) == frame
    with pytest.raises(PixelProvenanceError):
        render_dicom_frame(source, spec=SPEC)


def jpeg_source(*, size=(16, 12)):
    ds = dataset()
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    with Image.fromarray(
        np.arange(size[0] * size[1], dtype="uint8").reshape(size[1], size[0])
    ) as image:
        stream = io.BytesIO()
        image.save(stream, format="JPEG", quality=91)
    payload = stream.getvalue()
    ds.file_meta.TransferSyntaxUID = "1.2.840.10008.1.2.4.50"
    ds.PixelData = encapsulate([payload])
    return encode(ds), payload


def test_real_grayscale_jpeg_decoder_is_explicit_and_reproduced():
    source, payload = jpeg_source()
    frame = render_dicom_frame(source, spec=replace(SPEC, decoder=PixelDecoder.JPEG))
    with Image.open(io.BytesIO(payload)) as decoded:
        expected = np.floor(np.asarray(decoded).astype(float) / 256 * 255).astype(
            "uint8"
        )
    np.testing.assert_array_equal(pixels(frame)[:, :, 0], expected)
    assert reproduce_dicom_frame(source, frame.provenance) == frame


def test_compressed_dimensions_and_rle_expansion_are_checked_before_decoder(
    monkeypatch,
):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(1)
        raise AssertionError("decoder must not run")

    monkeypatch.setattr(pydicom.pixels, "pixel_array", forbidden)
    source, _ = jpeg_source(size=(32, 32))
    with pytest.raises(PixelProvenanceError):
        render_dicom_frame(source, spec=replace(SPEC, decoder=PixelDecoder.JPEG))
    ds = dataset()
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    ds.file_meta.TransferSyntaxUID = RLELossless
    # One segment claiming 256 expanded pixels when the declared frame has 192.
    payload = (
        (1).to_bytes(4, "little")
        + (64).to_bytes(4, "little")
        + bytes(56)
        + bytes([129, 1, 129, 1])
    )
    ds.PixelData = encapsulate([payload])
    with pytest.raises(PixelProvenanceError):
        render_dicom_frame(encode(ds), spec=replace(SPEC, decoder=PixelDecoder.RLE))
    assert not calls


def test_region_bindings_require_reproduction_and_exact_crop_pixels():
    source = encode(dataset())
    frame = render_dicom_frame(source, spec=SPEC)
    regions = (region(frame), region(frame, (6, 3, 9, 8)))
    bound = bind_region_provenance(source, frame.provenance, regions)
    assert verify_region_provenance(source, frame.provenance, bound) == regions
    for bad in (
        replace(bound[0], pixels_digest="0" * 64),
        replace(bound[0], provenance_digest="0" * 64),
        replace(bound[0], region=replace(regions[0], image_digest="0" * 64)),
        replace(bound[0], region=replace(regions[0], x0=2)),
    ):
        with pytest.raises(PixelProvenanceError):
            verify_region_provenance(source, frame.provenance, (bad,))


def test_relinked_forged_chain_and_stale_source_are_rejected_by_actual_reproduction():
    ds = dataset()
    source = encode(ds)
    frame = render_dicom_frame(source, spec=SPEC)
    stages = []
    previous = frame.provenance.source_digest
    for old in frame.provenance.stages:
        stage = PixelStage(
            old.operation,
            "f" * 64 if old.operation == "window" else old.parameters_digest,
            old.output_digest,
            previous,
        )
        stages.append(stage)
        previous = stage.digest
    forged = replace(frame.provenance, stages=tuple(stages))
    with pytest.raises(PixelProvenanceError):
        reproduce_dicom_frame(source, forged)
    ds.PatientName = "OTHER-SYNTHETIC"
    with pytest.raises(PixelProvenanceError):
        reproduce_dicom_frame(encode(ds), frame.provenance)


def test_decoder_drift_is_detected_even_when_window_clips_to_identical_final_pixels(
    monkeypatch,
):
    source = encode(dataset())
    spec = replace(SPEC, window=DisplayWindow(center=10000, width=1))
    frame = render_dicom_frame(source, spec=spec)
    original = pydicom.pixels.pixel_array

    def changed(*args, **kwargs):
        return original(*args, **kwargs) + 1

    monkeypatch.setattr(pydicom.pixels, "pixel_array", changed)
    new = render_dicom_frame(source, spec=spec)
    assert new.image.data == frame.image.data
    with pytest.raises(PixelProvenanceError):
        reproduce_dicom_frame(source, frame.provenance)


class Provider:
    metadata = LocalProviderMetadata(
        backend=RuntimeBackend.SYNTHETIC,
        model_digest="a" * 64,
        preprocessing_digest="b" * 64,
    )

    def __init__(self):
        self.requests = []
        self.abstain = False
        self.fail = False

    def infer(self, request):
        self.requests.append(request)
        if self.fail:
            raise RuntimeError(PRIVATE)
        selected = (
            request.regions[0]
            if request.regions
            else ImageRegion(image_digest=request.image.digest, x0=0, y0=0, x1=4, y1=4)
        )
        return VLMResult(
            request_digest=request.digest,
            runtime=RuntimeMetadata(
                provider=self.metadata,
                duration_ms=1,
                output_tokens=0 if self.abstain else 1,
            ),
            text="" if self.abstain else PRIVATE,
            evidence=()
            if self.abstain
            else (VisualEvidence(region=selected, start=0, end=4),),
            abstention=AbstentionRecord(
                AbstentionStage.INFERENCE, AbstentionReason.LOW_QUALITY
            )
            if self.abstain
            else None,
        )


def test_actual_local_vlm_evidence_is_bound_and_reproduction_guards_later_access():
    source = encode(dataset())
    provider = Provider()
    result = run_provenanced_dicom_vlm(
        source, provider, Prompt(text=PRIVATE), spec=SPEC, regions=((1, 2, 5, 6),)
    )
    assert len(provider.requests) == 1
    assert result.require_evidence(source) == result.result.evidence
    assert result.regions[0].region == provider.requests[0].regions[0]
    assert result.regions[0].provenance_digest == result.provenance.digest
    assert PRIVATE not in repr(result) + result.to_json()
    with pytest.raises(PixelProvenanceError):
        result.require_evidence(encode(dataset(photo="MONOCHROME1")))
    provider.abstain = True
    abstained = run_provenanced_dicom_vlm(
        source, provider, Prompt(text=PRIVATE), spec=SPEC
    )
    with pytest.raises(PixelProvenanceError):
        abstained.require_evidence(source)


@pytest.mark.parametrize(
    "limits",
    [
        PixelProvenanceLimits(max_source_bytes=132),
        PixelProvenanceLimits(max_header_bytes=132),
        PixelProvenanceLimits(max_frame_pixels=10),
        PixelProvenanceLimits(max_output_bytes=1),
    ],
)
def test_resource_limits_reject_without_partial_output(limits):
    with pytest.raises(PixelProvenanceError):
        render_dicom_frame(encode(dataset()), spec=SPEC, limits=limits)


def test_offline_fixed_errors_and_decoder_log_privacy(monkeypatch, caplog):
    source = encode(dataset())

    def forbidden(*args, **kwargs):
        raise AssertionError(PRIVATE)

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    with caplog.at_level(logging.DEBUG):
        frame = render_dicom_frame(source, spec=SPEC)
        assert reproduce_dicom_frame(source, frame.provenance) == frame
    assert PRIVATE not in caplog.text
    monkeypatch.setattr(pydicom.pixels, "pixel_array", forbidden)
    with pytest.raises(PixelProvenanceError) as caught:
        render_dicom_frame(source, spec=SPEC)
    assert str(caught.value) == "dicom_pixel_provenance_rejected"
    assert caught.value.__context__ is None


def test_cancellation_and_provider_failure_return_no_partial_evidence():
    source = encode(dataset())
    with pytest.raises(PixelProvenanceError):
        render_dicom_frame(source, spec=SPEC, cancel_check=lambda: True)
    provider = Provider()
    provider.fail = True
    with pytest.raises(PixelProvenanceError) as caught:
        run_provenanced_dicom_vlm(source, provider, Prompt(text=PRIVATE), spec=SPEC)
    assert caught.value.__context__ is None and PRIVATE not in str(caught.value)


def test_readable_syntax_uid_does_not_bypass_a_failed_file_meta_preflight(monkeypatch):
    from openmed.multimodal.dicom_transfer_syntax import read_dicom_transfer_syntax

    source = bytearray(encode(dataset()))
    assert source[132:138] == b"\x02\x00\x00\x00UL"
    declared_length = int.from_bytes(source[140:144], "little")
    source[140:144] = (declared_length + 2).to_bytes(4, "little")
    source = bytes(source)
    preflight = read_dicom_transfer_syntax(source)
    assert not preflight.ok
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(1)
        raise AssertionError("preflight must stop decoding")

    monkeypatch.setattr(pydicom.pixels, "pixel_array", forbidden)
    with pytest.raises(PixelProvenanceError):
        render_dicom_frame(source, spec=SPEC)
    assert not calls


@pytest.mark.parametrize("photo", ["RGB", "YBR_FULL"])
def test_real_color_rle_preserves_declared_conversion(photo):
    ds = dataset(photo=photo)
    spec = PixelRenderSpec(frame_index=0)
    native = render_dicom_frame(encode(ds), spec=spec)
    ds.compress(RLELossless, generate_instance_uid=False)
    source = encode(ds)
    compressed = render_dicom_frame(
        source, spec=replace(spec, decoder=PixelDecoder.RLE)
    )
    np.testing.assert_array_equal(pixels(compressed), pixels(native))
    assert reproduce_dicom_frame(source, compressed.provenance) == compressed


@pytest.mark.parametrize(
    "change",
    [
        "lut",
        "presentation",
        "partial_rescale",
        "bad_frame",
        "missing_window",
        "color_window",
        "bad_dtype",
    ],
)
def test_ambiguous_or_unsupported_display_requests_fail_before_inference(change):
    ds = dataset()
    spec = SPEC
    if change == "lut":
        ds.ModalityLUTSequence = [Dataset()]
    elif change == "presentation":
        ds.PresentationLUTShape = "INVERSE"
    elif change == "partial_rescale":
        ds.RescaleSlope = 2
    elif change == "bad_frame":
        spec = replace(spec, frame_index=1)
    elif change == "missing_window":
        spec = replace(spec, window=None)
    elif change == "color_window":
        ds = dataset(photo="RGB")
    else:
        ds.BitsAllocated = 32
    provider = Provider()
    with pytest.raises(PixelProvenanceError):
        run_provenanced_dicom_vlm(encode(ds), provider, Prompt(text=PRIVATE), spec=spec)
    assert not provider.requests


def test_declared_frame_and_target_allocation_limits_are_enforced():
    with pytest.raises(PixelProvenanceError):
        render_dicom_frame(
            encode(dataset(count=2)),
            spec=SPEC,
            limits=PixelProvenanceLimits(max_frames=1),
        )
    with pytest.raises(PixelProvenanceError):
        render_dicom_frame(
            encode(dataset()), spec=replace(SPEC, output_size=(5000, 5000))
        )


def test_runtime_fingerprint_changes_require_a_new_receipt(monkeypatch):
    import openmed.multimodal.vision.dicom.pixel_provenance as module

    source = encode(dataset())
    frame = render_dicom_frame(source, spec=SPEC)
    monkeypatch.setattr(module, "_runtime_digest", lambda: "0" * 64)
    assert render_dicom_frame(source, spec=SPEC).image.data == frame.image.data
    with pytest.raises(PixelProvenanceError):
        reproduce_dicom_frame(source, frame.provenance)


def test_mutated_model_evidence_cannot_reuse_another_region_binding():
    source = encode(dataset())
    result = run_provenanced_dicom_vlm(
        source, Provider(), Prompt(text=PRIVATE), spec=SPEC
    )
    visual = result.result.evidence[0]
    changed = replace(visual, region=replace(visual.region, x0=1))
    mismatched = replace(result, result=replace(result.result, evidence=(changed,)))
    with pytest.raises(PixelProvenanceError):
        mismatched.require_evidence(source)
