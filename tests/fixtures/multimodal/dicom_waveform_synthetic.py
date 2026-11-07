"""Small, entirely synthetic ECG datasets; no binary or patient assets."""

from __future__ import annotations

import struct
from typing import Any

PHI_TOKENS = (
    "SYNTHETIC^TEST",
    "TEST-ID-3825",
    "SYNTHETIC DEVICE DIAGNOSIS WITHHELD",
    "SYNTHETIC ANNOTATION 李明 123-45-6789",
)
LEAD_CODES = (
    "2:1",
    "2:2",
    "2:61",
    "2:62",
    "2:63",
    "2:64",
    "2:3",
    "2:4",
    "2:5",
    "2:6",
    "2:7",
    "2:8",
)


def build_synthetic_ecg(
    *, channels: int = 12, bits: int = 16, endian: str = "<"
) -> Any:
    """Build four interleaved samples per lead with distinct calibrations.

    Each lead encodes (-2, 0, 2, 4). Channel i has sensitivity i+1 uV,
    correction factor 0.5, baseline 10 uV and time skew i/1000 seconds.
    Expected physical samples are (9-i, 10, 11+i, 12+2*i) uV.
    """
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.sequence import Sequence

    def code(value: str, scheme: str, meaning: str) -> Any:
        item = Dataset()
        item.CodeValue = value
        item.CodingSchemeDesignator = scheme
        item.CodeMeaning = meaning
        return item

    ds = Dataset()
    ds.SOPClassUID = (
        "1.2.840.10008.5.1.4.1.1.9.1.4"
        if bits == 32
        else "1.2.840.10008.5.1.4.1.1.9.1.1"
        if channels == 12
        else "1.2.840.10008.5.1.4.1.1.9.1.2"
    )
    ds.SOPInstanceUID = "1.2.826.0.1.3680043.10.3825.1"
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = (
        "1.2.840.10008.1.2.1" if endian == "<" else "1.2.840.10008.1.2.2"
    )
    ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.PatientName = PHI_TOKENS[0]
    ds.SpecificCharacterSet = "ISO_IR 192"
    ds.PatientID = PHI_TOKENS[1]
    ds.Modality = "ECG"
    group = Dataset()
    group.NumberOfWaveformChannels = channels
    group.NumberOfWaveformSamples = 4
    group.SamplingFrequency = "250"
    group.WaveformOriginality = "ORIGINAL"
    group.MultiplexGroupTimeOffset = "500"
    group.WaveformBitsAllocated = bits
    group.WaveformSampleInterpretation = "SS" if bits == 16 else "SL"
    group.MultiplexGroupLabel = PHI_TOKENS[1]
    definitions = []
    for i in range(channels):
        ch = Dataset()
        ch.ChannelSourceSequence = Sequence([code(LEAD_CODES[i], "MDC", PHI_TOKENS[0])])
        ch.ChannelLabel = PHI_TOKENS[0]
        ch.ChannelSensitivity = str(i + 1)
        ch.ChannelSensitivityCorrectionFactor = "0.5"
        ch.ChannelBaseline = "10"
        ch.ChannelTimeSkew = str(i / 1000)
        ch.WaveformBitsStored = bits
        ch.ChannelSensitivityUnitsSequence = Sequence(
            [code("uV", "UCUM", PHI_TOKENS[0])]
        )
        definitions.append(ch)
    group.ChannelDefinitionSequence = Sequence(definitions)
    fmt = "h" if bits == 16 else "i"
    group.WaveformData = struct.pack(
        endian + fmt * (4 * channels),
        *(value for value in (-2, 0, 2, 4) for _ in range(channels)),
    )
    ds.WaveformSequence = Sequence([group])
    measurement = Dataset()
    measurement.NumericValue = "60"
    measurement.ConceptNameCodeSequence = Sequence(
        [code("10:2432", "MDC", PHI_TOKENS[0])]
    )
    statement = Dataset()
    statement.ConceptNameCodeSequence = Sequence(
        [code("10:8192", "MDC", PHI_TOKENS[0])]
    )
    statement.ConceptCodeSequence = Sequence([code("10:8208", "MDC", PHI_TOKENS[2])])
    text = Dataset()
    text.UnformattedTextValue = PHI_TOKENS[3]
    for item in (measurement, statement, text):
        item.ReferencedWaveformChannels = [1, 0]
    ds.WaveformAnnotationSequence = Sequence([measurement, statement, text])
    return ds
