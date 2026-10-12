"""Offline, in-memory DICOM serialization/reader integration."""

from __future__ import annotations

from io import BytesIO

import pytest

from openmed.multimodal.dicom_waveform import read_dicom_ecg_waveform

pydicom = pytest.importorskip("pydicom")

from tests.fixtures.multimodal.dicom_waveform_synthetic import (  # noqa: E402
    build_synthetic_ecg,
)


@pytest.mark.integration
@pytest.mark.parametrize("channels,bits", [(12, 16), (1, 16), (1, 32)])
@pytest.mark.parametrize(
    "syntax", ["1.2.840.10008.1.2", "1.2.840.10008.1.2.1", "1.2.840.10008.1.2.2"]
)
def test_serialized_ecg_roundtrip(channels, bits, syntax):
    endian = ">" if syntax.endswith(".2.2") else "<"
    ds = build_synthetic_ecg(channels=channels, bits=bits, endian=endian)
    ds.file_meta.TransferSyntaxUID = syntax
    stream = BytesIO()
    pydicom.dcmwrite(stream, ds, enforce_file_format=True)
    stream.seek(0)
    parsed = pydicom.dcmread(stream)
    record = read_dicom_ecg_waveform(parsed, start_sample=1, sample_count=2)
    for i, channel in enumerate(record.groups[0].channels):
        assert channel.samples == (10, 11 + i)
    assert record.annotations.device_interpretation_present
