"""Bounded local EDF/EDF+ reading with value-safe metadata and explicit gaps."""

from __future__ import annotations

import io
import math
import re
import struct
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import BinaryIO

EDF_NOTICE = (
    "Non-diagnostic signals for review only. Explicit reviewer confirmation is "
    "required before consequential use. Header withholding does not de-identify "
    "waveform samples or establish clinical validity."
)
_LABELS = frozenset(
    ["ECG", "EEG", "EMG", "EOG", "Temp rectal", "Body temp", "SaO2", "SpO2"]
    + [
        f"ECG {lead}"
        for lead in (
            "I",
            "II",
            "III",
            "aVR",
            "aVL",
            "aVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        )
    ]
    + ["EEG Fpz-Cz", "EEG Pz-Oz"]
)
_UNITS = frozenset(("V", "mV", "uV", "nV", "degreeC", "%", "Ohm", "mmHg"))
_NUMBER = re.compile(rb"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[Ee][+-]?[0-9]+)?")
_OFFSET = re.compile(rb"[+-][0-9]+(?:\.[0-9]+)?")
_DURATION = re.compile(rb"[0-9]+(?:\.[0-9]+)?")


class EdfError(ValueError):
    """Failure containing a stable controlled reason code, never source values."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class EdfLimits:
    """Caller-selected resource budgets enforced before decoding/retention."""

    max_bytes: int = 64 * 1024 * 1024
    max_signals: int = 64
    max_records: int = 100_000
    max_record_bytes: int = 1024 * 1024
    max_duration_seconds: int = 86_400
    max_window_seconds: int = 3600
    max_output_samples: int = 1_000_000
    max_annotation_lists: int = 100_000


@dataclass(frozen=True, slots=True)
class EdfIdentityStatus:
    """Presence and placeholder status; never a de-identification attestation."""

    present: bool
    anonymization_status: str


@dataclass(frozen=True, slots=True)
class EdfSignal:
    """Controlled technical metadata; arbitrary labels and units are withheld."""

    index: int
    label: str
    physical_dimension: str
    physical_minimum: float
    physical_maximum: float
    digital_minimum: int
    digital_maximum: int
    samples_per_record: int
    sampling_rate_hz: float | None


@dataclass(frozen=True, slots=True)
class EdfSignalWindow:
    """Samples of one signal in one record, without bridging record gaps."""

    signal_index: int
    first_sample_index: int
    digital_samples: tuple[int, ...] = field(repr=False)
    physical_samples: tuple[float, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class EdfRecordWindow:
    """One intersecting data record with offset timing and sample windows."""

    record_index: int
    onset_seconds: float
    duration_seconds: float
    signals: tuple[EdfSignalWindow, ...]


@dataclass(frozen=True, slots=True)
class EdfAnnotation:
    """One TAL's timing and nonempty annotation count, without any text."""

    record_index: int
    signal_index: int
    onset_seconds: float
    duration_seconds: float | None
    count: int


@dataclass(frozen=True, slots=True)
class EdfGap:
    """Missing acquisition interval before a discontinuous record."""

    before_record_index: int
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True, slots=True)
class EdfRecording:
    """Windowed non-diagnostic signals with content-free reporting."""

    format: str
    patient: EdfIdentityStatus
    recording: EdfIdentityStatus
    signals: tuple[EdfSignal, ...]
    record_count: int
    record_onsets_seconds: tuple[float, ...]
    record_duration_seconds: float
    window_start_seconds: float
    window_end_seconds: float
    records: tuple[EdfRecordWindow, ...]
    annotations: tuple[EdfAnnotation, ...]
    gaps: tuple[EdfGap, ...]
    notice: str = field(default=EDF_NOTICE, init=False)
    reviewer_confirmed: bool = False

    def reviewed(self, *, confirmed: bool) -> EdfRecording:
        """Bind explicit human confirmation before consequential handoff."""
        if confirmed is not True:
            raise EdfError("edf_review_required")
        return replace(self, reviewer_confirmed=True)

    def report(self) -> dict[str, object]:
        """Return codes/counts only; omit samples, labels and source headers."""
        return {
            "format": self.format,
            "signal_count": len(self.signals),
            "record_count": self.record_count,
            "window_record_count": len(self.records),
            "annotation_count": sum(a.count for a in self.annotations),
            "gap_count": len(self.gaps),
            "patient": {
                "present": self.patient.present,
                "anonymization_status": self.patient.anonymization_status,
            },
            "recording": {
                "present": self.recording.present,
                "anonymization_status": self.recording.anonymization_status,
            },
            "notice": self.notice,
            "review_required": not self.reviewer_confirmed,
        }


def read_edf(
    source: bytes | BinaryIO,
    *,
    start_seconds: float = 0,
    end_seconds: float | None = None,
    limits: EdfLimits = EdfLimits(),
) -> EdfRecording:
    """Read EDF, EDF+C or EDF+D from bytes or a caller-owned binary stream.

    Args:
        source: Bounded bytes or stream at the beginning of an EDF payload.
        start_seconds: Inclusive offset from the withheld header start second.
        end_seconds: Exclusive offset; None requests the entire bounded recording.
        limits: Resource budgets, including retained window samples.

    Returns:
        Immutable metadata and window samples. All record times and gaps are
        validated even outside the window. Annotation text is never returned.

    Raises:
        EdfError: Stable value-free code for invalid input or exceeded budgets.

    Seekable streams are restored on success/failure and never closed. Other
    streams are consumed sequentially. Reads are capped at 64 KiB. No paths,
    network, logs, caches, diagnosis, normalization or clock alignment are used.
    """
    if not isinstance(limits, EdfLimits) or any(
        type(value) is not int or value <= 0
        for value in (getattr(limits, name) for name in limits.__dataclass_fields__)
    ):
        raise EdfError("edf_limits_invalid")
    start = _window_number(start_seconds)
    end = None if end_seconds is None else _window_number(end_seconds)
    if start < 0 or (
        end is not None and (end <= start or end - start > limits.max_window_seconds)
    ):
        raise EdfError("edf_window_invalid")
    if isinstance(source, bytes):
        if len(source) > limits.max_bytes:
            raise EdfError("edf_byte_limit")
        source = io.BytesIO(source)
    position = None
    try:
        if source.seekable():
            position = source.tell()
    except Exception:
        pass
    result = None
    failure = None
    try:
        result = _parse(_Reader(source, limits.max_bytes), start, end, limits)
    except EdfError as exc:
        failure = exc.code
    finally:
        if position is not None:
            try:
                source.seek(position)
            except Exception:
                failure = "edf_stream_restore_error"
    if failure is not None:
        raise EdfError(failure)
    assert result is not None
    return result


class _Reader:
    def __init__(self, source: BinaryIO, limit: int) -> None:
        self.source = source
        self.limit = limit
        self.count = 0
        self.pending = b""

    def read(self, size: int, *, eof: bool = False) -> bytes:
        if size > self.limit - self.count + len(self.pending):
            raise EdfError("edf_byte_limit")
        chunks = [self.pending[:size]]
        remaining = size - len(chunks[0])
        self.pending = self.pending[size:]
        while remaining:
            request = min(remaining, 65536)
            chunk = None
            try:
                chunk = self.source.read(request)
            except Exception:
                pass
            if not isinstance(chunk, bytes) or len(chunk) > request:
                raise EdfError("edf_stream_read_error")
            if not chunk:
                if eof and remaining == size:
                    return b""
                raise EdfError("edf_truncated")
            chunks.append(chunk)
            self.count += len(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def at_end(self) -> bool:
        # A single bounded EOF probe is allowed at the exact byte budget.
        if self.pending:
            return False
        self.limit += 1
        try:
            self.pending = self.read(1, eof=True)
            return not self.pending
        finally:
            self.limit -= 1


def _integer(raw: bytes) -> int:
    raw = raw.strip(b" ")
    if not re.fullmatch(rb"-?[0-9]+", raw):
        raise EdfError("edf_header_invalid")
    return int(raw)


def _number(raw: bytes, pattern: re.Pattern[bytes] = _NUMBER) -> Fraction:
    raw = raw.strip(b" ") if pattern is _NUMBER else raw
    # Bound exponent and precision before constructing a rational number.
    if len(raw) > 28 or not pattern.fullmatch(raw):
        raise EdfError("edf_numeric_invalid")
    if b"e" in raw.lower() and abs(int(raw.lower().split(b"e")[1])) > 12:
        raise EdfError("edf_numeric_invalid")
    return Fraction(raw.decode("ascii"))


def _window_number(value: float) -> Fraction:
    if type(value) not in (int, float) or abs(value) > 1e12 or not math.isfinite(value):
        raise EdfError("edf_window_invalid")
    return Fraction(str(value))


def _identity(raw: bytes, placeholder: bytes) -> EdfIdentityStatus:
    text = raw.strip(b" ")
    return EdfIdentityStatus(
        bool(text),
        "absent"
        if not text
        else "placeholder_only"
        if text == placeholder
        else "not_verified",
    )


def _header_date_valid(raw: bytes, time: bytes) -> bool:
    if not re.fullmatch(rb"[0-9]{2}\.[0-9]{2}\.(?:[0-9]{2}|yy)", raw):
        return False
    if not re.fullmatch(rb"[0-9]{2}\.[0-9]{2}\.[0-9]{2}", time):
        return False
    day, month = int(raw[:2]), int(raw[3:5])
    year = (
        2000
        if raw[6:] == b"yy"
        else int(raw[6:]) + (1900 if int(raw[6:]) >= 85 else 2000)
    )
    days = (
        31,
        29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28,
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    )
    return (
        1 <= month <= 12
        and 1 <= day <= days[month - 1]
        and int(time[:2]) < 24
        and int(time[3:5]) < 60
        and int(time[6:]) < 60
    )


def _parse(
    reader: _Reader, start: Fraction, end: Fraction | None, limits: EdfLimits
) -> EdfRecording:
    header = reader.read(256)
    if any(byte < 32 or byte > 126 for byte in header) or header[:8] != b"0       ":
        raise EdfError("edf_header_invalid")
    if not _header_date_valid(header[168:176], header[176:184]):
        raise EdfError("edf_header_invalid")
    signal_count = _integer(header[252:256])
    if signal_count < 1 or signal_count > limits.max_signals:
        raise EdfError("edf_signal_limit")
    header_size = _integer(header[184:192])
    if header_size != 256 * (signal_count + 1):
        raise EdfError("edf_header_size_invalid")
    declared_records = _integer(header[236:244])
    if declared_records < -1 or declared_records == 0:
        raise EdfError("edf_record_count_invalid")
    if declared_records > limits.max_records:
        raise EdfError("edf_record_limit")
    duration = _number(header[244:252])
    if (
        duration < 0
        or duration > limits.max_duration_seconds
        or duration * max(0, declared_records) > limits.max_duration_seconds
    ):
        raise EdfError("edf_duration_limit")
    kind = header[192:197]
    format_name = kind.decode("ascii") if kind in (b"EDF+C", b"EDF+D") else "EDF"
    fields = reader.read(header_size - 256)
    if any(byte < 32 or byte > 126 for byte in fields):
        raise EdfError("edf_header_invalid")
    columns = []
    cursor = 0
    for width in (16, 80, 8, 8, 8, 8, 8, 80, 8, 32):
        columns.append(
            [
                fields[cursor + index * width : cursor + (index + 1) * width]
                for index in range(signal_count)
            ]
        )
        cursor += width * signal_count
    signals = []
    sizes = []
    annotation_channels = []
    for index in range(signal_count):
        label = columns[0][index].strip(b" ").decode("ascii")
        unit = columns[2][index].strip(b" ").decode("ascii")
        pmin, pmax = (_number(columns[col][index]) for col in (3, 4))
        dmin, dmax = (_integer(columns[col][index]) for col in (5, 6))
        size = _integer(columns[8][index])
        if (
            pmin == pmax
            or not -32768 <= dmin < dmax <= 32767
            or abs(pmin) > 1e12
            or abs(pmax) > 1e12
        ):
            raise EdfError("edf_range_invalid")
        if size <= 0:
            raise EdfError("edf_samples_invalid")
        sizes.append(size)
        if label == "EDF Annotations":
            if format_name == "EDF" or (dmin, dmax) != (-32768, 32767):
                raise EdfError("edf_annotation_header_invalid")
            annotation_channels.append(index)
        else:
            if duration == 0 and (format_name != "EDF+D" or size != 1):
                raise EdfError("edf_duration_invalid")
            signals.append(
                EdfSignal(
                    index,
                    label if label in _LABELS else "withheld",
                    unit if unit in _UNITS else "withheld",
                    float(pmin),
                    float(pmax),
                    dmin,
                    dmax,
                    size,
                    float(size / duration) if duration else None,
                )
            )
    if format_name != "EDF" and not annotation_channels:
        raise EdfError("edf_annotation_channel_missing")
    record_bytes = sum(sizes) * 2
    if record_bytes > limits.max_record_bytes:
        raise EdfError("edf_record_byte_limit")
    if (
        declared_records > 0
        and record_bytes * declared_records > limits.max_bytes - reader.count
    ):
        raise EdfError("edf_byte_limit")
    if end is None:
        end = start + limits.max_window_seconds
        entire = True
    else:
        entire = False
    records, onsets, annotations, gaps = [], [], [], []
    output_count = tal_count = 0
    previous_end = Fraction(0)
    while declared_records == -1 or len(onsets) < declared_records:
        if reader.at_end():
            if declared_records != -1:
                raise EdfError("edf_record_count_invalid")
            break
        if len(onsets) >= limits.max_records:
            raise EdfError("edf_record_limit")
        data = reader.read(record_bytes)
        record_index = len(onsets)
        onset = duration * record_index
        cursor = 0
        channels = []
        for size in sizes:
            channels.append(data[cursor : cursor + size * 2])
            cursor += size * 2
        for channel in annotation_channels:
            tals = _annotations(
                channels[channel],
                limits.max_duration_seconds,
                limits.max_annotation_lists - tal_count,
            )
            tal_count += len(tals)
            if tal_count > limits.max_annotation_lists:
                raise EdfError("edf_annotation_limit")
            if channel == annotation_channels[0]:
                if not tals or tals[0][1] is not None or not tals[0][3]:
                    raise EdfError("edf_timekeeping_invalid")
                onset = tals[0][0]
            for event_onset, event_duration, count, _ in tals:
                if count and (
                    start <= event_onset < end
                    or (
                        event_duration
                        and event_onset < end
                        and event_onset + event_duration > start
                    )
                ):
                    annotations.append(
                        (
                            EdfAnnotation(
                                record_index,
                                channel,
                                float(event_onset),
                                None
                                if event_duration is None
                                else float(event_duration),
                                count,
                            ),
                            event_onset,
                            event_duration,
                        )
                    )
        if record_index == 0 and not 0 <= onset < 1:
            raise EdfError("edf_timekeeping_invalid")
        if record_index and (
            onset < previous_end or (format_name != "EDF+D" and onset != previous_end)
        ):
            raise EdfError("edf_record_timing_invalid")
        if onset + duration > limits.max_duration_seconds:
            raise EdfError("edf_duration_limit")
        if record_index and onset > previous_end:
            gaps.append(EdfGap(record_index, float(previous_end), float(onset)))
        previous_end = onset + duration
        onsets.append(float(onset))
        if (duration and onset < end and onset + duration > start) or (
            not duration and start <= onset < end
        ):
            windows = []
            for signal in signals:
                if duration:
                    # Rational ceiling preserves exact half-open sample boundaries.
                    first = max(
                        0,
                        min(
                            signal.samples_per_record,
                            math.ceil(
                                (start - onset) * signal.samples_per_record / duration
                            ),
                        ),
                    )
                    stop = max(
                        0,
                        min(
                            signal.samples_per_record,
                            math.ceil(
                                (end - onset) * signal.samples_per_record / duration
                            ),
                        ),
                    )
                else:
                    first, stop = 0, 1
                output_count += stop - first
                if output_count > limits.max_output_samples:
                    raise EdfError("edf_output_sample_limit")
                samples = tuple(
                    item[0]
                    for item in struct.iter_unpack(
                        "<h", channels[signal.index][first * 2 : stop * 2]
                    )
                )
                if any(
                    value < signal.digital_minimum or value > signal.digital_maximum
                    for value in samples
                ):
                    raise EdfError("edf_sample_range_invalid")
                scale = (signal.physical_maximum - signal.physical_minimum) / (
                    signal.digital_maximum - signal.digital_minimum
                )
                physical = tuple(
                    signal.physical_minimum + (value - signal.digital_minimum) * scale
                    for value in samples
                )
                windows.append(EdfSignalWindow(signal.index, first, samples, physical))
            records.append(
                EdfRecordWindow(
                    record_index, float(onset), float(duration), tuple(windows)
                )
            )
    if not onsets or not reader.at_end():
        raise EdfError("edf_record_count_invalid")
    if entire:
        end = previous_end + (1 if duration == 0 else 0)
        if end <= start or end - start > limits.max_window_seconds:
            raise EdfError("edf_window_invalid")
    return EdfRecording(
        format_name,
        _identity(header[8:88], b"X X X X"),
        _identity(header[88:168], b"Startdate X X X X"),
        tuple(signals),
        len(onsets),
        tuple(onsets),
        float(duration),
        float(start),
        float(end),
        tuple(records),
        tuple(
            annotation
            for annotation, onset, event_duration in annotations
            if start <= onset < end
            or (event_duration and onset < end and onset + event_duration > start)
        ),
        tuple(gaps),
    )


def _annotations(
    data: bytes, max_duration: int, max_lists: int
) -> list[tuple[Fraction, Fraction | None, int, bool]]:
    results = []
    cursor = 0
    while cursor < len(data) and data[cursor] != 0:
        stop = data.find(b"\x00", cursor)
        if stop == -1:
            raise EdfError("edf_annotation_invalid")
        tal = data[cursor:stop]
        parts = tal.split(b"\x14")
        if len(parts) < 3 or parts[-1] != b"":
            raise EdfError("edf_annotation_invalid")
        timing = parts[0].split(b"\x15")
        if len(timing) > 2:
            raise EdfError("edf_annotation_invalid")
        onset = _number(timing[0], _OFFSET)
        duration = _number(timing[1], _DURATION) if len(timing) == 2 else None
        if abs(onset) > max_duration or (
            duration is not None and duration > max_duration
        ):
            raise EdfError("edf_duration_limit")
        count = 0
        for text in parts[1:-1]:
            valid = False
            try:
                text.decode("utf-8")
                valid = True
            except UnicodeDecodeError:
                pass
            if not valid or any(byte < 32 and byte not in (9, 10, 13) for byte in text):
                raise EdfError("edf_annotation_invalid")
            count += bool(text)
        if len(results) >= max_lists:
            raise EdfError("edf_annotation_limit")
        results.append((onset, duration, count, parts[1] == b""))
        cursor = stop + 1
    if any(data[cursor:]):
        raise EdfError("edf_annotation_invalid")
    return results
