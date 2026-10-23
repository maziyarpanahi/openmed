"""Hand-checked waveform values and deterministic privacy negative controls."""

from __future__ import annotations

import dataclasses
import json
import logging
import struct
import traceback
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest

from openmed.multimodal import dicom_waveform as waveform
from openmed.multimodal.exceptions import MissingDependencyError

pydicom = pytest.importorskip("pydicom")

from tests.fixtures.multimodal.dicom_waveform_synthetic import (  # noqa: E402
    LEAD_CODES,
    PHI_TOKENS,
    build_synthetic_ecg,
)


@pytest.mark.parametrize("channels", [1, 12])
@pytest.mark.parametrize("bits", [16, 32])
@pytest.mark.parametrize("endian", ["<", ">"])
def test_hand_checked_physical_values(channels, bits, endian):
    ds = build_synthetic_ecg(channels=channels, bits=bits, endian=endian)
    record = waveform.read_dicom_ecg_waveform(ds)
    group = record.groups[0]
    assert group.sampling_frequency_hz == 250
    assert group.sample_count == group.source_sample_count == 4
    for i, channel in enumerate(group.channels):
        assert channel.source_scheme == "MDC"
        assert channel.source_code == LEAD_CODES[i]
        assert channel.unit == "uV"
        assert channel.samples == (9 - i, 10, 11 + i, 12 + 2 * i)
        assert channel.first_sample_offset_seconds == pytest.approx(0.5 + i / 1000)


def test_windows_and_multiple_groups_preserve_skew():
    ds = build_synthetic_ecg(channels=1)
    second = deepcopy(ds.WaveformSequence[0])
    second.MultiplexGroupTimeOffset = "2500"
    del second.ChannelDefinitionSequence[0].ChannelTimeSkew
    second.ChannelDefinitionSequence[0].ChannelSampleSkew = "0.5"
    second.ChannelDefinitionSequence[0].ChannelOffset = "0.1"
    ds.WaveformSequence.append(second)
    record = waveform.read_dicom_ecg_waveform(ds, start_sample=2, sample_count=100)
    assert [g.sample_count for g in record.groups] == [2, 2]
    assert [g.group_index for g in record.groups] == [0, 1]
    assert record.groups[1].channels[0].samples == (11, 12)
    assert record.groups[1].channels[0].first_sample_offset_seconds == pytest.approx(
        2.610
    )


@pytest.mark.parametrize("uid", ["1.2.840.10008.1.2", "1.2.840.10008.1.2.1"])
@pytest.mark.parametrize(
    "sop", ["1.2.840.10008.5.1.4.1.1.9.1.2", "1.2.840.10008.5.1.4.1.1.9.1.3"]
)
def test_general_and_ambulatory_with_uncompressed_syntaxes(uid, sop):
    ds = build_synthetic_ecg(channels=1)
    ds.SOPClassUID = sop
    ds.file_meta.TransferSyntaxUID = uid
    assert waveform.read_dicom_ecg_waveform(ds).groups[0].channels[0].samples == (
        9,
        10,
        11,
        12,
    )


def test_padding_is_missing_not_an_amplitude():
    ds = build_synthetic_ecg(channels=1)
    ds.WaveformSequence[0].WaveformPaddingValue = struct.pack("<h", -2)
    record = waveform.read_dicom_ecg_waveform(ds)
    assert record.groups[0].channels[0].samples == (None, 10, 11, 12)
    assert record.to_report()["groups"][0]["missing_sample_count"] == 1


def test_source_is_unchanged_and_all_text_is_withheld(caplog, capsys):
    ds = build_synthetic_ecg()
    before = deepcopy(ds)
    with caplog.at_level(logging.DEBUG, logger="pydicom"):
        record = waveform.read_dicom_ecg_waveform(ds)
    assert ds == before
    assert record.annotations == waveform.DicomWaveformAnnotations(1, 1, 1)
    assert record.annotations.device_interpretation_present
    output = (
        repr(record)
        + json.dumps(dataclasses.asdict(record))
        + json.dumps(record.to_report())
    )
    output += caplog.text + str(capsys.readouterr())
    for token in PHI_TOKENS:
        assert token not in output
    report = json.dumps(record.to_report())
    assert "samples" not in report
    assert "sensitivity" not in report
    assert "frequency" not in report
    assert "first_sample_offset" not in report
    assert waveform.DICOM_ECG_NOTICE == record.notice
    assert record.review_required
    assert json.dumps(record.to_report(), sort_keys=True) == json.dumps(
        waveform.read_dicom_ecg_waveform(ds).to_report(), sort_keys=True
    )


def test_uncoded_device_text_is_reported_as_possible_interpretation():
    ds = build_synthetic_ecg(channels=1)
    ds.WaveformAnnotationSequence = [ds.WaveformAnnotationSequence[2]]
    ds.WaveformAnnotationSequence[0].UnformattedTextValue = PHI_TOKENS[2]
    record = waveform.read_dicom_ecg_waveform(ds)
    assert record.annotations.free_text_count == 1
    assert record.annotations.device_interpretation_present
    assert record.to_report()["annotations"]["values_withheld"]


@pytest.mark.parametrize("confirmation", [False, None, 1, "true", PHI_TOKENS[0]])
def test_consequential_use_requires_explicit_confirmation(confirmation):
    record = waveform.read_dicom_ecg_waveform(build_synthetic_ecg())
    with pytest.raises(waveform.DicomWaveformError, match="^review_required$"):
        record.confirm_for_consequential_use(reviewer_confirmed=confirmation)
    record.confirm_for_consequential_use(reviewer_confirmed=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.review_required = False


@pytest.mark.parametrize(
    "start,size",
    [(-1, 1), (True, 1), (0, False), (0, 0), (0, 10001), (4, 1), ("PHI", 1)],
)
def test_invalid_windows(start, size):
    with pytest.raises(waveform.DicomWaveformError, match="^invalid_window$"):
        waveform.read_dicom_ecg_waveform(
            build_synthetic_ecg(), start_sample=start, sample_count=size
        )


@pytest.mark.parametrize(
    "scope,keyword,value,reason",
    [
        ("ds", "SOPClassUID", "1.2.3", "unsupported_sop_class"),
        (
            "meta",
            "TransferSyntaxUID",
            "1.2.840.10008.1.2.4.50",
            "unsupported_transfer_syntax",
        ),
        (
            "meta",
            "TransferSyntaxUID",
            "1.2.840.10008.1.2.1.99",
            "unsupported_transfer_syntax",
        ),
        (
            "group",
            "WaveformSampleInterpretation",
            "US",
            "unsupported_sample_interpretation",
        ),
        (
            "group",
            "WaveformSampleInterpretation",
            "AB",
            "unsupported_sample_interpretation",
        ),
        (
            "group",
            "WaveformSampleInterpretation",
            "SL",
            "unsupported_sample_interpretation",
        ),
        ("group", "WaveformBitsAllocated", 8, "unsupported_sample_interpretation"),
        ("group", "NumberOfWaveformSamples", 5, "inconsistent_sample_count"),
        ("group", "NumberOfWaveformSamples", 0, "inconsistent_sample_count"),
        ("group", "NumberOfWaveformChannels", 2, "inconsistent_sample_count"),
        ("group", "NumberOfWaveformChannels", 65, "unsupported_multiplex_layout"),
        ("group", "WaveformData", b"", "inconsistent_sample_count"),
        ("group", "ChannelDefinitionSequence", [], "missing_channel_definitions"),
        ("group", "SamplingFrequency", 0, "invalid_timing"),
        ("channel", "ChannelSensitivity", 0, "invalid_calibration"),
        ("channel", "ChannelSensitivityCorrectionFactor", -1, "invalid_calibration"),
        ("channel", "ChannelSensitivityUnitsSequence", [], "unsupported_units"),
        ("channel", "ChannelSourceSequence", [], "unsupported_channel_source"),
        ("channel", "WaveformBitsStored", 17, "unsupported_sample_interpretation"),
        ("channel", "WaveformBitsStored", 2, "invalid_sample_value"),
        ("channel", "ChannelSampleSkew", "1", "invalid_timing"),
        ("group", "WaveformPaddingValue", b"x", "unsupported_multiplex_layout"),
    ],
)
def test_malformed_layouts_fail_closed(scope, keyword, value, reason, caplog):
    ds = build_synthetic_ecg(channels=1)
    target = {
        "ds": ds,
        "meta": ds.file_meta,
        "group": ds.WaveformSequence[0],
        "channel": ds.WaveformSequence[0].ChannelDefinitionSequence[0],
    }[scope]
    setattr(target, keyword, value)
    with pytest.raises(waveform.DicomWaveformError) as caught:
        waveform.read_dicom_ecg_waveform(ds)
    assert str(caught.value) == caught.value.reason_code == reason
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    for token in PHI_TOKENS:
        assert (
            token not in "".join(traceback.format_exception(caught.value)) + caplog.text
        )


@pytest.mark.parametrize(
    "keyword,reason",
    [
        ("ChannelBaseline", "invalid_calibration"),
        ("ChannelSensitivity", "invalid_calibration"),
        ("ChannelSensitivityCorrectionFactor", "invalid_calibration"),
        ("ChannelTimeSkew", "invalid_timing"),
        ("WaveformBitsStored", "unsupported_sample_interpretation"),
    ],
)
def test_missing_channel_metadata(keyword, reason):
    ds = build_synthetic_ecg(channels=1)
    del ds.WaveformSequence[0].ChannelDefinitionSequence[0][keyword]
    with pytest.raises(waveform.DicomWaveformError, match=f"^{reason}$"):
        waveform.read_dicom_ecg_waveform(ds)


@pytest.mark.parametrize(
    "keyword,scope",
    [
        ("CodeValue", "source"),
        ("CodingSchemeDesignator", "source"),
        ("CodeValue", "unit"),
    ],
)
def test_untrusted_codes_are_not_echoed(keyword, scope):
    ds = build_synthetic_ecg(channels=1)
    channel = ds.WaveformSequence[0].ChannelDefinitionSequence[0]
    code = (
        channel.ChannelSourceSequence[0]
        if scope == "source"
        else channel.ChannelSensitivityUnitsSequence[0]
    )
    setattr(code, keyword, PHI_TOKENS[1])
    with pytest.raises(waveform.DicomWaveformError) as caught:
        waveform.read_dicom_ecg_waveform(ds)
    assert PHI_TOKENS[1] not in "".join(traceback.format_exception(caught.value))


def test_resource_caps(monkeypatch):
    ds = build_synthetic_ecg(channels=1)
    monkeypatch.setattr(waveform, "_MAX_BYTES", 1)
    with pytest.raises(waveform.DicomWaveformError, match="^resource_limit$"):
        waveform.read_dicom_ecg_waveform(ds)
    monkeypatch.setattr(waveform, "_MAX_BYTES", 1000)
    monkeypatch.setattr(waveform, "_MAX_OUTPUT_VALUES", 1)
    with pytest.raises(waveform.DicomWaveformError, match="^resource_limit$"):
        waveform.read_dicom_ecg_waveform(ds)
    monkeypatch.setattr(waveform, "_MAX_OUTPUT_VALUES", 1000)
    monkeypatch.setattr(waveform, "_MAX_ANNOTATIONS", 1)
    with pytest.raises(waveform.DicomWaveformError, match="^resource_limit$"):
        waveform.read_dicom_ecg_waveform(ds)


def test_missing_optional_dependency(monkeypatch):
    def unavailable(name):
        raise ImportError(PHI_TOKENS[0])

    monkeypatch.setattr(waveform.importlib, "import_module", unavailable)
    with pytest.raises(MissingDependencyError) as caught:
        waveform.read_dicom_ecg_waveform(None)
    assert caught.value.__context__ is None
    assert PHI_TOKENS[0] not in "".join(traceback.format_exception(caught.value))


def test_empty_groups_and_invalid_dataset():
    ds = build_synthetic_ecg()
    ds.WaveformSequence = []
    with pytest.raises(
        waveform.DicomWaveformError, match="^unsupported_multiplex_layout$"
    ):
        waveform.read_dicom_ecg_waveform(ds)
    with pytest.raises(waveform.DicomWaveformError, match="^invalid_dataset$"):
        waveform.read_dicom_ecg_waveform(PHI_TOKENS[0])


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "1e308"])
def test_nonfinite_and_overflowing_calibration(value):
    ds = build_synthetic_ecg(channels=1)
    definition = ds.WaveformSequence[0].ChannelDefinitionSequence[0]
    if value != "1e308":
        with pytest.warns(UserWarning, match="Invalid value for VR DS"):
            definition.ChannelSensitivity = value
    else:
        definition.ChannelSensitivity = value
    with pytest.raises(waveform.DicomWaveformError, match="^invalid_calibration$"):
        waveform.read_dicom_ecg_waveform(ds)


@pytest.mark.parametrize("unit", ["V", "mV", "uV"])
def test_units_are_preserved_without_normalization(unit):
    ds = build_synthetic_ecg(channels=1)
    ds.WaveformSequence[0].ChannelDefinitionSequence[0].ChannelSensitivityUnitsSequence[
        0
    ].CodeValue = unit
    channel = waveform.read_dicom_ecg_waveform(ds).groups[0].channels[0]
    assert channel.unit == unit
    assert channel.samples == (9, 10, 11, 12)


@pytest.mark.parametrize("extra", [b"x", b"\x00\x00"])
def test_extra_sample_bytes_are_never_ignored(extra):
    ds = build_synthetic_ecg(channels=1)
    ds.WaveformSequence[0].WaveformData += extra
    with pytest.raises(
        waveform.DicomWaveformError, match="^inconsistent_sample_count$"
    ):
        waveform.read_dicom_ecg_waveform(ds)


def test_missing_definitions_and_nonempty_source_modifiers():
    ds = build_synthetic_ecg(channels=1)
    definition = ds.WaveformSequence[0].ChannelDefinitionSequence[0]
    definition.ChannelSourceModifiersSequence = deepcopy(
        definition.ChannelSourceSequence
    )
    with pytest.raises(
        waveform.DicomWaveformError, match="^unsupported_channel_source$"
    ):
        waveform.read_dicom_ecg_waveform(ds)
    del ds.WaveformSequence[0].ChannelDefinitionSequence
    with pytest.raises(
        waveform.DicomWaveformError, match="^missing_channel_definitions$"
    ):
        waveform.read_dicom_ecg_waveform(ds)


def test_parser_failures_and_diagnostics_never_echo_values(caplog, recwarn):
    from pydicom.dataelem import RawDataElement
    from pydicom.tag import Tag

    ds = build_synthetic_ecg(channels=1)
    definition = ds.WaveformSequence[0].ChannelDefinitionSequence[0]
    tag = Tag(0x003A, 0x0210)
    payload = PHI_TOKENS[1].encode("ascii")
    definition[tag] = RawDataElement(tag, "DS", len(payload), payload, 0, False, True)
    logger = logging.getLogger("pydicom")
    previous_filters = tuple(logger.filters)
    with caplog.at_level(logging.DEBUG, logger="pydicom"):
        with pytest.raises(waveform.DicomWaveformError) as caught:
            waveform.read_dicom_ecg_waveform(ds)
        assert tuple(logger.filters) == previous_filters
        logger.warning("synthetic parser filter restoration check")
    assert str(caught.value) in {"invalid_dataset", "invalid_calibration"}
    assert caught.value.__context__ is None
    assert not recwarn.list
    assert "synthetic parser filter restoration check" in caplog.text
    for token in PHI_TOKENS:
        assert (
            token not in "".join(traceback.format_exception(caught.value)) + caplog.text
        )


def test_concurrent_malformed_reads_keep_diagnostics_private(caplog, recwarn):
    from pydicom.dataelem import RawDataElement
    from pydicom.tag import Tag

    def read(_):
        ds = build_synthetic_ecg(channels=1)
        definition = ds.WaveformSequence[0].ChannelDefinitionSequence[0]
        tag = Tag(0x003A, 0x0210)
        payload = PHI_TOKENS[1].encode("ascii")
        definition[tag] = RawDataElement(
            tag, "DS", len(payload), payload, 0, False, True
        )
        try:
            waveform.read_dicom_ecg_waveform(ds)
        except waveform.DicomWaveformError as error:
            return "".join(traceback.format_exception(error))
        pytest.fail("Malformed calibration was accepted")

    logger = logging.getLogger("pydicom")
    previous_filters = tuple(logger.filters)
    with caplog.at_level(logging.DEBUG, logger="pydicom"):
        with ThreadPoolExecutor(max_workers=4) as pool:
            outputs = list(pool.map(read, range(16)))
    assert tuple(logger.filters) == previous_filters
    assert not recwarn.list
    for token in PHI_TOKENS:
        assert token not in "".join(outputs) + caplog.text


def test_no_annotations_and_later_group_failure_are_fail_closed():
    ds = build_synthetic_ecg(channels=1)
    del ds.WaveformAnnotationSequence
    record = waveform.read_dicom_ecg_waveform(ds)
    assert record.annotations == waveform.DicomWaveformAnnotations(0, 0, 0)
    assert not record.annotations.device_interpretation_present
    second = deepcopy(ds.WaveformSequence[0])
    second.WaveformData = b""
    ds.WaveformSequence.append(second)
    with pytest.raises(
        waveform.DicomWaveformError, match="^inconsistent_sample_count$"
    ):
        waveform.read_dicom_ecg_waveform(ds)
