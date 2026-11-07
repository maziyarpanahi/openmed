"""Bounded, local DICOM ECG decoding without source text or interpretation.

The reader accepts a pydicom Dataset, never writes a de-identified copy, and
does not normalize leads. Physical amplitudes remain protected signal data.
"""

from __future__ import annotations

import importlib
import logging
import math
import struct
import threading
import warnings
from dataclasses import dataclass, field
from typing import Any

from .exceptions import MissingDependencyError

DICOM_ECG_NOTICE = (
    "Non-diagnostic waveform decoding for review only. Device statements are "
    "withheld. Explicit qualified reviewer confirmation is required before "
    "consequential use; never auto-trigger clinical decisions."
)

# ECG Storage SOP classes: twelve-lead, general, ambulatory, general 32-bit.
_SOPS = {
    "1.2.840.10008.5.1.4.1.1.9.1.1": ("twelve_lead", {"SS": (16, "h")}),
    "1.2.840.10008.5.1.4.1.1.9.1.2": ("general", {"SS": (16, "h")}),
    "1.2.840.10008.5.1.4.1.1.9.1.3": ("ambulatory", {"SS": (16, "h")}),
    "1.2.840.10008.5.1.4.1.1.9.1.4": (
        "general_32_bit",
        {"SS": (16, "h"), "SL": (32, "i")},
    ),
}
_SYNTAXES = {
    "1.2.840.10008.1.2": "<",
    "1.2.840.10008.1.2.1": "<",
    "1.2.840.10008.1.2.2": ">",
}
_REASONS = frozenset(
    {
        "invalid_dataset",
        "unsupported_sop_class",
        "unsupported_transfer_syntax",
        "unsupported_multiplex_layout",
        "missing_channel_definitions",
        "unsupported_channel_source",
        "unsupported_units",
        "invalid_calibration",
        "invalid_timing",
        "unsupported_sample_interpretation",
        "inconsistent_sample_count",
        "invalid_sample_value",
        "invalid_window",
        "resource_limit",
        "review_required",
    }
)
_MAX_BYTES = 128 * 1024 * 1024
_MAX_OUTPUT_VALUES = 1_000_000
_MAX_ANNOTATIONS = 10_000
# Python 3.10-3.13 warning filters are process-global. Serialize these reader
# contexts so concurrent reads cannot restore filters underneath another read.
_PARSER_LOCK = threading.RLock()


class DicomWaveformError(ValueError):
    """Fail-closed rejection containing only a controlled reason code."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code if reason_code in _REASONS else "invalid_dataset"
        super().__init__(self.reason_code)


@dataclass(frozen=True)
class DicomWaveformAnnotations:
    """Counts only; all annotation values, including measurements, are withheld.

    Coded nonnumeric observations are conservatively treated as possible
    device interpretation statements, without asserting their authorship.
    Unformatted text may also contain device statements.
    """

    measurement_count: int
    device_interpretation_count: int
    free_text_count: int

    @property
    def device_interpretation_present(self) -> bool:
        """Whether annotations may contain device interpretation statements."""
        return bool(self.device_interpretation_count or self.free_text_count)


@dataclass(frozen=True)
class DicomWaveformChannel:
    """One source-coded channel, in its declared UCUM voltage unit.

    Source codes are validated against pydicom's CID 3001 vocabulary; source
    labels and code meanings are never copied. Timing is relative, in seconds.
    """

    channel_index: int
    source_scheme: str
    source_code: str
    unit: str
    sensitivity: float
    correction_factor: float
    baseline: float
    first_sample_offset_seconds: float
    samples: tuple[float | None, ...] = field(repr=False)


@dataclass(frozen=True)
class DicomWaveformGroup:
    """A bounded window from one interleaved multiplex group."""

    group_index: int
    sampling_frequency_hz: float
    source_sample_count: int
    start_sample: int
    sample_count: int
    sample_interpretation: str
    channels: tuple[DicomWaveformChannel, ...]


@dataclass(frozen=True)
class DicomWaveformRecord:
    """Reader-local signal record with no patient headers or source text."""

    sop_category: str
    groups: tuple[DicomWaveformGroup, ...]
    annotations: DicomWaveformAnnotations
    notice: str = field(default=DICOM_ECG_NOTICE, init=False)
    review_required: bool = field(default=True, init=False)

    def to_report(self) -> dict[str, Any]:
        """Return counts and controlled codes, excluding amplitudes and timing."""
        return {
            "type": "dicom_ecg_waveform",
            "sop_category": self.sop_category,
            "groups": [
                {
                    "group_index": group.group_index,
                    "source_sample_count": group.source_sample_count,
                    "start_sample": group.start_sample,
                    "sample_count": group.sample_count,
                    "channel_count": len(group.channels),
                    "missing_sample_count": sum(
                        value is None
                        for channel in group.channels
                        for value in channel.samples
                    ),
                    "sample_interpretation": group.sample_interpretation,
                }
                for group in self.groups
            ],
            "annotations": {
                "measurement_count": self.annotations.measurement_count,
                "device_interpretation_count": (
                    self.annotations.device_interpretation_count
                ),
                "free_text_count": self.annotations.free_text_count,
                "device_interpretation_present": (
                    self.annotations.device_interpretation_present
                ),
                "values_withheld": True,
            },
            "notice": self.notice,
            "review_required": True,
        }

    def confirm_for_consequential_use(self, *, reviewer_confirmed: bool) -> None:
        """Require an explicit boolean confirmation; confer no clinical validity.

        Args:
            reviewer_confirmed: True only after qualified human review.

        Raises:
            DicomWaveformError: If explicit confirmation is absent.
        """
        if reviewer_confirmed is not True:
            raise DicomWaveformError("review_required")


class _QuietParser(logging.Filter):
    """Prevent optional parser diagnostics from echoing source values here."""

    def __init__(self) -> None:
        super().__init__()
        self.thread_id = threading.get_ident()

    def filter(self, record: logging.LogRecord) -> bool:
        return record.thread != self.thread_id


def read_dicom_ecg_waveform(
    dataset: Any, *, start_sample: int = 0, sample_count: int = 2500
) -> DicomWaveformRecord:
    """Decode bounded windows from all ECG multiplex groups, entirely locally.

    Args:
        dataset: A caller-supplied pydicom Dataset with file_meta transfer syntax.
            File parsing/transport is caller-owned; no files or network are used.
        start_sample: Zero-based start, applied separately to each group.
        sample_count: Maximum samples per channel, 1 through 10000. Short final
            windows are clipped; a start beyond any group's end is rejected.

    Returns:
        Immutable source-coded channels in original voltage units, with relative
        timing and annotation counts. Padding samples become None.

    Raises:
        MissingDependencyError: If the optional pydicom dependency is absent.
        DicomWaveformError: For unsupported or malformed inputs, using only
            controlled codes and suppressing parser diagnostics and exceptions.
    """
    dependency_available = True
    try:
        pydicom = importlib.import_module("pydicom")
        vocab = importlib.import_module("pydicom.sr.codedict")
    except ImportError:
        dependency_available = False
    if not dependency_available:
        raise MissingDependencyError(
            "pydicom", 'Install with: pip install "openmed[multimodal]".'
        )
    if not isinstance(dataset, pydicom.dataset.Dataset):
        raise DicomWaveformError("invalid_dataset")
    if (
        type(start_sample) is not int
        or start_sample < 0
        or type(sample_count) is not int
        or not 1 <= sample_count <= 10_000
    ):
        raise DicomWaveformError("invalid_window")

    logger = logging.getLogger("pydicom")
    quiet = _QuietParser()
    logger.addFilter(quiet)
    try:
        with _PARSER_LOCK, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            known_sources = {
                (code.scheme_designator, code.value)
                for code in vocab.codes.cid3001.concepts.values()
            }
            return _read(dataset, known_sources, start_sample, sample_count)
    except DicomWaveformError as error:
        reason = error.reason_code
    except Exception:  # noqa: BLE001 - never reflect parser errors or payloads.
        reason = "invalid_dataset"
    finally:
        logger.removeFilter(quiet)
    # Raise outside the exception handler to discard even hidden __context__
    # chains, which may contain PHI in third-party conversion errors.
    raise DicomWaveformError(reason)


def _number(value: Any, reason: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise DicomWaveformError(reason)
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise DicomWaveformError(reason) from None
    if not math.isfinite(result) or (positive and result <= 0):
        raise DicomWaveformError(reason)
    return result


def _count(value: Any, reason: str) -> int:
    result = _number(value, reason, positive=True)
    if result != int(result):
        raise DicomWaveformError(reason)
    return int(result)


def _read(
    ds: Any, known_sources: set[tuple[str, str]], start: int, size: int
) -> DicomWaveformRecord:
    sop = _SOPS.get(getattr(ds, "SOPClassUID", None))
    if sop is None:
        raise DicomWaveformError("unsupported_sop_class")
    endian = _SYNTAXES.get(
        getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
    )
    if endian is None:
        raise DicomWaveformError("unsupported_transfer_syntax")
    sequence = getattr(ds, "WaveformSequence", ())
    if not 1 <= len(sequence) <= 16:
        raise DicomWaveformError("unsupported_multiplex_layout")
    output_values = 0
    input_bytes = 0
    groups = []
    for index, group in enumerate(sequence):
        channels = _count(
            getattr(group, "NumberOfWaveformChannels", None),
            "unsupported_multiplex_layout",
        )
        total = _count(
            getattr(group, "NumberOfWaveformSamples", None),
            "inconsistent_sample_count",
        )
        if channels > 64:
            raise DicomWaveformError("unsupported_multiplex_layout")
        if start >= total:
            raise DicomWaveformError("invalid_window")
        count = min(size, total - start)
        output_values += channels * count
        data = getattr(group, "WaveformData", None)
        if not isinstance(data, bytes):
            raise DicomWaveformError("unsupported_multiplex_layout")
        input_bytes += len(data)
        if input_bytes > _MAX_BYTES or output_values > _MAX_OUTPUT_VALUES:
            raise DicomWaveformError("resource_limit")
        groups.append(
            _group(
                group,
                index,
                channels,
                total,
                start,
                count,
                sop[1],
                endian,
                known_sources,
                data,
            )
        )
    annotations = getattr(ds, "WaveformAnnotationSequence", ())
    if len(annotations) > _MAX_ANNOTATIONS:
        raise DicomWaveformError("resource_limit")
    measurement = statement = text = 0
    for item in annotations:
        # Never read text, numeric values, code meanings or interpretation codes.
        # Mixed/malformed items are classified conservatively as statements.
        if "UnformattedTextValue" in item or "TextValue" in item:
            text += 1
        elif "NumericValue" in item and "ConceptCodeSequence" not in item:
            measurement += 1
        else:
            statement += 1
    return DicomWaveformRecord(
        sop[0], tuple(groups), DicomWaveformAnnotations(measurement, statement, text)
    )


def _group(
    group: Any,
    index: int,
    n: int,
    total: int,
    start: int,
    count: int,
    interpretations: dict[str, tuple[int, str]],
    endian: str,
    known_sources: set[tuple[str, str]],
    data: bytes,
) -> DicomWaveformGroup:
    interpretation = getattr(group, "WaveformSampleInterpretation", None)
    encoding = interpretations.get(interpretation)
    if encoding is None or getattr(group, "WaveformBitsAllocated", None) != encoding[0]:
        raise DicomWaveformError("unsupported_sample_interpretation")
    bits, fmt = encoding
    width = bits // 8
    if len(data) != total * n * width:
        raise DicomWaveformError("inconsistent_sample_count")
    definitions = getattr(group, "ChannelDefinitionSequence", ())
    if len(definitions) != n:
        raise DicomWaveformError("missing_channel_definitions")
    frequency = _number(
        getattr(group, "SamplingFrequency", None), "invalid_timing", positive=True
    )
    group_offset = (
        _number(getattr(group, "MultiplexGroupTimeOffset", 0), "invalid_timing") / 1000
    )
    padding = getattr(group, "WaveformPaddingValue", None)
    if padding is not None:
        if not isinstance(padding, bytes) or len(padding) != width:
            raise DicomWaveformError("unsupported_multiplex_layout")
        padding = struct.unpack(endian + fmt, padding)[0]
    channels = []
    for channel_index, definition in enumerate(definitions):
        sources = getattr(definition, "ChannelSourceSequence", ())
        if len(sources) != 1:
            raise DicomWaveformError("unsupported_channel_source")
        source = (
            getattr(sources[0], "CodingSchemeDesignator", None),
            getattr(sources[0], "CodeValue", None),
        )
        if source not in known_sources:
            raise DicomWaveformError("unsupported_channel_source")
        # Modifiers require a separate semantics contract; do not drop them.
        if getattr(definition, "ChannelSourceModifiersSequence", ()):
            raise DicomWaveformError("unsupported_channel_source")
        units = getattr(definition, "ChannelSensitivityUnitsSequence", ())
        if (
            len(units) != 1
            or getattr(units[0], "CodingSchemeDesignator", None) != "UCUM"
            or getattr(units[0], "CodeValue", None) not in {"V", "mV", "uV"}
        ):
            raise DicomWaveformError("unsupported_units")
        sensitivity = _number(
            getattr(definition, "ChannelSensitivity", None),
            "invalid_calibration",
            positive=True,
        )
        correction = _number(
            getattr(definition, "ChannelSensitivityCorrectionFactor", None),
            "invalid_calibration",
            positive=True,
        )
        baseline = _number(
            getattr(definition, "ChannelBaseline", None), "invalid_calibration"
        )
        stored = _count(
            getattr(definition, "WaveformBitsStored", None),
            "unsupported_sample_interpretation",
        )
        if stored > bits:
            raise DicomWaveformError("unsupported_sample_interpretation")
        time_skew = getattr(definition, "ChannelTimeSkew", None)
        sample_skew = getattr(definition, "ChannelSampleSkew", None)
        if time_skew is None and sample_skew is None:
            raise DicomWaveformError("invalid_timing")
        skew = (
            _number(time_skew, "invalid_timing")
            if time_skew is not None
            else _number(sample_skew, "invalid_timing") / frequency
        )
        if sample_skew is not None and time_skew is not None:
            if not math.isclose(
                skew,
                _number(sample_skew, "invalid_timing") / frequency,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise DicomWaveformError("invalid_timing")
        offset = (
            group_offset
            + skew
            + start / frequency
            + _number(getattr(definition, "ChannelOffset", 0), "invalid_timing")
        )
        if not math.isfinite(offset) or not math.isfinite(offset + count / frequency):
            raise DicomWaveformError("invalid_timing")
        values = []
        for position in range(start, start + count):
            raw = struct.unpack_from(
                endian + fmt, data, (position * n + channel_index) * width
            )[0]
            if raw == padding:
                values.append(None)
                continue
            if not -(2 ** (stored - 1)) <= raw < 2 ** (stored - 1):
                raise DicomWaveformError("invalid_sample_value")
            value = raw * sensitivity * correction + baseline
            if not math.isfinite(value):
                raise DicomWaveformError("invalid_calibration")
            values.append(value)
        channels.append(
            DicomWaveformChannel(
                channel_index,
                source[0],
                source[1],
                units[0].CodeValue,
                sensitivity,
                correction,
                baseline,
                offset,
                tuple(values),
            )
        )
    return DicomWaveformGroup(
        index, frequency, total, start, count, interpretation, tuple(channels)
    )
