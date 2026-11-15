"""Real synthetic DICOM bytes, coverage controls and sampled-only VLM evidence."""

import hashlib
import io
import json
import logging
import socket
from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")
pydicom = pytest.importorskip("pydicom")
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, ImplicitVRLittleEndian

from openmed.multimodal.abstention import (
    AbstentionReason,
    AbstentionRecord,
    AbstentionStage,
)
from openmed.multimodal.vision.dicom.series_sampler import (
    DisplayWindow,
    SamplingPolicy,
    SeriesSamplingError,
    run_sampled_dicom_vlm,
    sample_dicom_series,
)
from openmed.multimodal.vision.providers.base import (
    LocalProviderMetadata,
    Prompt,
    RuntimeBackend,
    RuntimeMetadata,
    VLMResult,
)

PRIVATE = "SYNTHETIC-PRIVATE-987"
WINDOW = DisplayWindow(center=128, width=256)
POLICY = SamplingPolicy(max_selected=4, max_gap_mm=4)


def encode(ds):
    stream = io.BytesIO()
    pydicom.dcmwrite(stream, ds, enforce_file_format=True)
    return stream.getvalue()


def dataset(
    index, *, acquisition=1, z=None, pattern=None, syntax=ExplicitVRLittleEndian
):
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = syntax
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = f"1.2.826.0.1.3680043.10.543.{index + 1}"
    ds.SeriesInstanceUID = "1.2.826.0.1.3680043.10.543.1000"
    ds.FrameOfReferenceUID = "1.2.826.0.1.3680043.10.543.2000"
    ds.PatientName = PRIVATE
    ds.PatientID = PRIVATE
    ds.Modality = "CT"
    ds.Rows = ds.Columns = 16
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.ImagePositionPatient = [0, 0, index * 2 if z is None else z]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.PixelSpacing = [1, 1]
    ds.AcquisitionNumber = acquisition
    pixels = np.arange(256, dtype="<u2").reshape(16, 16) if pattern is None else pattern
    ds.PixelData = pixels.astype("<u2").tobytes()
    return ds


def sources(count=8, **kwargs):
    return tuple(encode(dataset(i, **kwargs)) for i in range(count))


def frame_ref(data, index=0):
    payload = json.dumps(
        [hashlib.sha256(data).hexdigest(), index], sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def sample(data, **kwargs):
    return sample_dicom_series(
        data, window=WINDOW, policy=kwargs.pop("policy", POLICY), **kwargs
    )


class Provider:
    metadata = LocalProviderMetadata(
        backend=RuntimeBackend.SYNTHETIC,
        model_digest="a" * 64,
        preprocessing_digest="b" * 64,
    )

    def __init__(self):
        self.requests = []
        self.abstain = False
        self.bad = False

    def infer(self, request):
        self.requests.append(request)
        if self.bad:
            raise RuntimeError(PRIVATE)
        return VLMResult(
            request_digest=request.digest,
            runtime=RuntimeMetadata(
                provider=self.metadata,
                duration_ms=1,
                output_tokens=0 if self.abstain else 1,
            ),
            text="" if self.abstain else PRIVATE,
            abstention=AbstentionRecord(
                AbstentionStage.INFERENCE, AbstentionReason.LOW_QUALITY
            )
            if self.abstain
            else None,
        )


def test_actual_dicom_selection_is_order_independent_and_covers_every_frame():
    data = sources()
    first = sample(data)
    assert first == sample(tuple(reversed(data)))
    assert 0 < len(first.selected) <= POLICY.max_selected
    assert set(first.selected) | set(first.unexamined) == {frame_ref(d) for d in data}
    assert not set(first.selected) & set(first.unexamined)
    assert first.maximum_gap_mm <= POLICY.max_gap_mm
    selected_positions = [
        i * 2 for i, d in enumerate(data) if frame_ref(d) in first.selected
    ]
    assert all(
        min(abs(i * 2 - p) for p in selected_positions) <= POLICY.max_gap_mm
        for i in range(len(data))
    )
    assert PRIVATE not in first.to_json() + repr(first)
    assert "ImagePositionPatient" not in first.to_json()
    assert {f.source_digest for f in first.frames} == {
        hashlib.sha256(d).hexdigest() for d in data
    }
    assert all(f.frame_index == 0 for f in first.frames)
    assert {f.reference for f in first.frames} == {frame_ref(d) for d in data}


def test_distinct_acquisition_groups_cannot_be_covered_by_one_another():
    data = tuple(
        encode(dataset(i, acquisition=1 if i < 4 else 2, z=i % 4)) for i in range(8)
    )
    plan = sample(data, policy=replace(POLICY, max_gap_mm=20))
    assert len(plan.groups) == 2
    for group in (data[:4], data[4:]):
        assert set(plan.selected) & {frame_ref(d) for d in group}
    with pytest.raises(SeriesSamplingError):
        sample(data, policy=replace(POLICY, max_selected=1))


def test_diversity_adds_observed_different_pixels_after_spatial_coverage():
    plain = np.arange(256, dtype="<u2").reshape(16, 16)
    data = (encode(dataset(0, pattern=plain)), encode(dataset(1, pattern=255 - plain)))
    plan = sample(data, policy=replace(POLICY, max_gap_mm=100, max_selected=2))
    assert len(plan.selected) == 2
    same = (encode(dataset(0, pattern=plain)), encode(dataset(1, pattern=plain)))
    assert (
        len(
            sample(
                same, policy=replace(POLICY, max_gap_mm=100, max_selected=2)
            ).selected
        )
        == 1
    )


def test_low_signal_frames_remain_explicit_and_cannot_be_the_only_coverage():
    data = (
        encode(dataset(0)),
        encode(dataset(1, pattern=np.zeros((16, 16), dtype="<u2"))),
    )
    plan = sample(data)
    assert plan.low_signal == (frame_ref(data[1]),)
    assert plan.low_signal[0] in plan.unexamined
    with pytest.raises(SeriesSamplingError):
        sample(data, policy=replace(POLICY, max_gap_mm=0))


def test_insufficient_frame_budget_abstains_instead_of_omitting_distant_positions():
    with pytest.raises(SeriesSamplingError):
        sample(sources(), policy=replace(POLICY, max_selected=1, max_gap_mm=1))


@pytest.mark.parametrize(
    "change",
    [
        {"max_frames": 1, "max_selected": 1},
        {"max_instances": 1},
        {"max_source_bytes": 100},
        {"max_total_bytes": 100},
        {"max_header_bytes": 140},
        {"max_frame_pixels": 1},
    ],
)
def test_resource_bounds_reject_before_expensive_model_execution(change):
    with pytest.raises(SeriesSamplingError):
        sample(sources(2), policy=replace(POLICY, **change))


@pytest.mark.parametrize(
    "key",
    [
        "ImagePositionPatient",
        "ImageOrientationPatient",
        "PixelSpacing",
        "FrameOfReferenceUID",
        "SeriesInstanceUID",
        "AcquisitionNumber",
    ],
)
def test_missing_metadata_is_not_replaced_with_fabricated_geometry_or_acquisition(key):
    ds = dataset(0)
    delattr(ds, key)
    with pytest.raises(SeriesSamplingError):
        sample((encode(ds),))


def test_duplicate_instances_and_mixed_series_are_rejected():
    data = sources(2)
    for invalid in ((data[0], data[0]),):
        with pytest.raises(SeriesSamplingError):
            sample(invalid)
    ds = dataset(1)
    ds.SeriesInstanceUID = "1.2.3.4.5"
    with pytest.raises(SeriesSamplingError):
        sample((data[0], encode(ds)))


def test_unknown_acquisition_requires_explicit_policy_and_remains_distinct():
    data = []
    for index in range(2):
        ds = dataset(index, z=0)
        del ds.AcquisitionNumber
        data.append(encode(ds))
    policy = replace(POLICY, require_acquisition=False)
    plan = sample(tuple(data), policy=policy)
    assert len(plan.selected) == len(plan.groups) == 2
    assert plan.unknown_acquisition == plan.selected
    with pytest.raises(SeriesSamplingError):
        sample(tuple(data), policy=replace(policy, max_selected=1))


@pytest.mark.parametrize("uid", ["X.2.826", "1.02.26", "1..826"])
def test_uid_validation_does_not_depend_on_application_warning_settings(uid):
    original = b"1.2.826.0.1.3680043.10.543.1000"
    replacement = (uid + "." + "1" * (len(original) - len(uid) - 1)).encode()
    raw = sources(1)[0].replace(original, replacement)
    with pydicom.config.disable_value_validation(), pytest.raises(SeriesSamplingError):
        sample((raw,))


def test_fractional_acquisition_cannot_be_truncated_to_an_integer():
    ds = dataset(0, acquisition=123)
    raw = encode(ds).replace(b"123 ", b"1.5 ")
    with pydicom.config.disable_value_validation(), pytest.raises(SeriesSamplingError):
        sample((raw,))


@pytest.mark.parametrize("orientation", [[1, 0, 0, 1, 0, 0], [2, 0, 0, 0, 1, 0]])
def test_degenerate_or_nonunit_direction_cosines_are_rejected(orientation):
    ds = dataset(0)
    ds.ImageOrientationPatient = orientation
    with pytest.raises(SeriesSamplingError):
        sample((encode(ds),))


def test_implicit_native_syntax_is_supported():
    assert sample(sources(3, syntax=ImplicitVRLittleEndian)).selected


def test_enhanced_multiframe_uses_each_frames_actual_position_and_acquisition():
    ds = dataset(0)
    del ds.ImagePositionPatient
    ds.NumberOfFrames = 4
    groups = []
    for index in range(4):
        fg = Dataset()
        plane = Dataset()
        plane.ImagePositionPatient = [0, 0, index * 5]
        fg.PlanePositionSequence = [plane]
        content = Dataset()
        content.FrameAcquisitionNumber = 1
        content.TemporalPositionIndex = 1 if index < 2 else 2
        fg.FrameContentSequence = [content]
        groups.append(fg)
    ds.PerFrameFunctionalGroupsSequence = groups
    ds.PixelData = np.tile(np.arange(256, dtype="<u2"), (4, 1)).tobytes()
    raw = encode(ds)
    plan = sample((raw,), policy=replace(POLICY, max_gap_mm=6))
    assert len(plan.groups) == 2
    assert set(plan.selected) | set(plan.unexamined) == {
        frame_ref(raw, i) for i in range(4)
    }
    assert {f.frame_index for f in plan.frames} == {0, 1, 2, 3}


def test_multiframe_missing_per_frame_positions_cannot_invent_slice_spacing():
    ds = dataset(0)
    ds.NumberOfFrames = 2
    ds.PixelData *= 2
    with pytest.raises(SeriesSamplingError):
        sample((encode(ds),))


def test_multiframe_content_groups_do_not_promote_one_legacy_position_to_every_frame():
    ds = dataset(0)
    ds.NumberOfFrames = 2
    groups = []
    for index in range(2):
        fg, content = Dataset(), Dataset()
        content.FrameAcquisitionNumber = index + 1
        fg.FrameContentSequence = [content]
        groups.append(fg)
    ds.PerFrameFunctionalGroupsSequence = groups
    ds.PixelData *= 2
    with pytest.raises(SeriesSamplingError):
        sample((encode(ds),))


def test_selected_vlm_inputs_are_actual_images_and_unexamined_frames_cannot_be_cited():
    from PIL import Image

    data = sources()
    plan = sample(data)
    provider = Provider()
    result = run_sampled_dicom_vlm(
        data,
        plan,
        provider,
        Prompt(text="Review this selected frame"),
        window=WINDOW,
        policy=POLICY,
    )
    assert len(provider.requests) == len(plan.selected)
    for item, request in zip(result.frames, provider.requests, strict=True):
        with Image.open(io.BytesIO(request.image.data)) as image:
            assert image.size == (16, 16) and image.mode == "L"
            assert np.asarray(image).ravel().tolist() == [0, *range(255)]
        assert item.result.request_digest == request.digest
        assert result.require_evidence(item.frame_reference) is item.result
    with pytest.raises(SeriesSamplingError):
        result.require_evidence(plan.unexamined[0])
    report = json.loads(result.to_json())
    assert report["evidence_scope"] == "selected_frames_only"
    assert set(report["unexamined"]) == set(plan.unexamined)
    assert PRIVATE not in repr(result) + result.to_json()


def test_provider_abstention_never_counts_as_examined_evidence():
    data = sources(2)
    plan = sample(data)
    provider = Provider()
    provider.abstain = True
    result = run_sampled_dicom_vlm(
        data,
        plan,
        provider,
        Prompt(text="Review selected frame"),
        window=WINDOW,
        policy=POLICY,
    )
    assert json.loads(result.to_json())["examined"] == []
    with pytest.raises(SeriesSamplingError):
        result.require_evidence(plan.selected[0])


@pytest.mark.parametrize("changed", ["source", "window", "policy", "manifest"])
def test_stale_or_forged_bindings_never_reach_provider(changed):
    data = sources(3)
    plan = sample(data)
    provider = Provider()
    window = WINDOW
    policy = POLICY
    if changed == "source":
        modified = dataset(0)
        modified.PatientID = "SYNTHETIC-OTHER"
        data = (encode(modified), *data[1:])
    elif changed == "window":
        window = DisplayWindow(center=0, width=256)
    elif changed == "policy":
        policy = replace(POLICY, max_gap_mm=5)
    else:
        plan = replace(plan, frames_digest="f" * 64)
    with pytest.raises(SeriesSamplingError) as error:
        run_sampled_dicom_vlm(
            data, plan, provider, Prompt(text=PRIVATE), window=window, policy=policy
        )
    assert error.value.__context__ is None and not provider.requests


def test_backend_errors_and_parser_warnings_never_escape_as_phi(caplog):
    caplog.set_level(logging.DEBUG, logger="pydicom")
    data = sources(2)
    plan = sample(data)
    caplog.clear()
    provider = Provider()
    provider.bad = True
    with pytest.raises(SeriesSamplingError) as error:
        run_sampled_dicom_vlm(
            data, plan, provider, Prompt(text=PRIVATE), window=WINDOW, policy=POLICY
        )
    assert PRIVATE not in caplog.text + str(error.value)
    assert error.value.__context__ is None
    raw = data[0].replace(
        b"1.2.826.0.1.3680043.10.543.1", b"X.2.826.0.1.3680043.10.543.1"
    )
    with pytest.raises(SeriesSamplingError):
        sample((raw,))
    assert "X.2.826" not in caplog.text


def test_offline_execution_never_opens_a_socket(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket, "socket", forbidden)
    data = sources(2)
    plan = sample(data)
    assert run_sampled_dicom_vlm(
        data, plan, Provider(), Prompt(text="Synthetic"), window=WINDOW, policy=POLICY
    ).frames
