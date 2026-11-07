"""Bounded, local-only WFDB decoding with value-safe metadata and diagnostics."""

from __future__ import annotations

import math
import re
import struct
from contextlib import ExitStack
from dataclasses import dataclass
from typing import BinaryIO, Sequence

__all__ = [
    "WFDB_NOTICE",
    "WfdbAnnotations",
    "WfdbError",
    "WfdbLimits",
    "WfdbRecord",
    "WfdbSignal",
    "read_wfdb_record",
]

WFDB_NOTICE = (
    "Non-diagnostic waveform data for human review only. "
    "No clinical validation or autonomous clinical action is provided. "
    "Explicit reviewer confirmation is required for consequential use."
)
_CHUNK = 8192
_NUMBER = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
_FORMAT = re.compile(r"([0-9]+)(?:x([0-9]+))?(?::([0-9]+))?(?:\+([0-9]+))?")
_GAIN = re.compile(rf"({_NUMBER})(?:\(([+-]?[0-9]+)\))?(?:/([^\s]+))?")
_RATE = re.compile(rf"({_NUMBER})(?:/{_NUMBER}(?:\({_NUMBER}\))?)?")
# Descriptions and unit strings are untrusted free text. Preserve only declared
# ECG labels and voltage units, without mapping or normalizing them (#2807).
_LEADS = frozenset(
    (
        "I",
        "II",
        "III",
        "aVR",
        "aVL",
        "aVF",
        "AVR",
        "AVL",
        "AVF",
        "MLI",
        "MLII",
        "MLIII",
        "MCL1",
        "MCL6",
        "ECG",
    )
    + tuple(f"V{i}" for i in range(1, 10))
)
_UNITS = frozenset(("mV", "uV", "µV", "μV", "V"))


class WfdbError(ValueError):
    """A controlled reason code with no input values or transport exception."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


@dataclass(frozen=True, slots=True)
class WfdbLimits:
    """Resource budgets checked before allocating outputs or reading payloads."""

    max_header_bytes: int = 65536
    max_signals: int = 32
    max_samples: int = 20_000_000
    max_window_samples: int = 100_000
    max_file_bytes: int = 256 * 1024 * 1024
    max_annotation_bytes: int = 4 * 1024 * 1024
    max_annotations: int = 100_000

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if type(value) is not int or value <= 0:
                raise WfdbError("wfdb_limits_invalid")


@dataclass(frozen=True, slots=True)
class WfdbSignal:
    """Declared calibration and an integer window, without source identifiers."""

    lead_label: str | None
    gain: float
    baseline: int
    unit: str | None
    format_code: int
    samples: tuple[int, ...]
    checksum_verified: bool


@dataclass(frozen=True, slots=True)
class WfdbAnnotations:
    """MIT annotation counts and absolute window positions; no types or text."""

    total_count: int
    sample_positions: tuple[int, ...]
    auxiliary_text_present: bool


@dataclass(frozen=True, slots=True)
class WfdbRecord:
    """Review-only raw waveform window and bounded, allow-listed metadata.

    Samples remain potentially sensitive physiological data. This record is
    not a de-identification, lead-normalization, or quality-gate result.
    """

    signals: tuple[WfdbSignal, ...]
    sample_rate_hz: float
    total_samples: int
    start_sample: int
    sample_count: int
    annotations: WfdbAnnotations | None
    comments_present: bool
    record_name_present: bool
    path_fields_present: bool
    timing_fields_present: bool
    descriptions_withheld: bool
    units_withheld: bool

    @property
    def notice(self) -> str:
        """Return the mandatory non-diagnostic boundary."""
        return WFDB_NOTICE

    @property
    def start_seconds(self) -> float:
        """Return the window start relative to sample zero, never a wall clock."""
        return self.start_sample / self.sample_rate_hz

    @property
    def duration_seconds(self) -> float:
        """Return window duration in seconds."""
        return self.sample_count / self.sample_rate_hz

    def require_reviewer_confirmation(self, *, confirmed: bool = False) -> None:
        """Gate consequential downstream use on an explicit human confirmation."""
        if confirmed is not True:
            raise WfdbError("wfdb_reviewer_confirmation_required")

    def report(self) -> dict[str, object]:
        """Return a content-free summary without samples or annotation positions."""
        return {
            "signal_count": len(self.signals),
            "total_samples": self.total_samples,
            "start_sample": self.start_sample,
            "sample_count": self.sample_count,
            "annotation_count": self.annotations.total_count if self.annotations else 0,
            "checksums_verified": sum(s.checksum_verified for s in self.signals),
            "comments_present": self.comments_present,
            "record_name_present": self.record_name_present,
            "path_fields_present": self.path_fields_present,
            "timing_fields_present": self.timing_fields_present,
            "descriptions_withheld": self.descriptions_withheld,
            "units_withheld": self.units_withheld,
            "annotation_text_present": bool(
                self.annotations and self.annotations.auxiliary_text_present
            ),
            "review_required": True,
            "notice": self.notice,
        }


@dataclass(slots=True)
class _SignalSpec:
    group: int
    format_code: int
    offset: int
    gain: float
    baseline: int
    unit: str | None
    lead_label: str | None
    checksum: int | None
    description_withheld: bool
    unit_withheld: bool


class _Source:
    """Bounded sequential reads; restore seekable caller-owned streams."""

    def __init__(self, source: bytes | BinaryIO, limit: int) -> None:
        self.source = source
        self.limit = limit
        self.position = 0
        self.initial: int | None = None
        self.length: int | None = None
        if isinstance(source, bytes):
            self.length = len(source)
        else:
            try:
                if source.seekable():
                    initial = source.tell()
                    if type(initial) is not int or initial < 0:
                        raise ValueError
                    self.initial = initial
                    source.seek(0, 2)
                    self.length = source.tell() - initial
                    source.seek(initial)
                    if type(self.length) is not int or self.length < 0:
                        raise ValueError
                elif not callable(source.read):
                    raise ValueError
            except Exception:
                pass
            else:
                return self._check_size()
            if self.initial is not None:
                self.close()
            # Raise outside the handler so private transport errors are not
            # retained in __context__ or displayed by traceback formatting.
            raise WfdbError("wfdb_stream_contract_error")
        self._check_size()

    def _check_size(self) -> None:
        if self.length is not None and self.length > self.limit:
            raise WfdbError("wfdb_file_limit_exceeded")

    def close(self) -> None:
        if self.initial is not None:
            try:
                self.source.seek(self.initial)
            except Exception:
                pass
            else:
                return
            raise WfdbError("wfdb_stream_restore_error")

    def read(self, size: int) -> bytes:
        if isinstance(self.source, bytes):
            chunk = self.source[self.position : self.position + size]
        else:
            try:
                chunk = self.source.read(size)
            except Exception:
                pass
            else:
                if isinstance(chunk, bytes) and len(chunk) <= size:
                    return self._advance(chunk)
                raise WfdbError("wfdb_stream_contract_error")
            raise WfdbError("wfdb_stream_read_error")
        return self._advance(chunk)

    def _advance(self, chunk: bytes) -> bytes:
        self.position += len(chunk)
        if self.position > self.limit:
            raise WfdbError("wfdb_file_limit_exceeded")
        return chunk

    def exact(self, size: int, code: str) -> bytes:
        result = bytearray()
        while len(result) < size:
            part = self.read(size - len(result))
            if not part:
                raise WfdbError(code)
            result.extend(part)
        return bytes(result)

    def discard(self, size: int, code: str) -> None:
        while size:
            chunk = self.exact(min(size, _CHUNK), code)
            size -= len(chunk)

    def finish(self) -> None:
        # Check trailing file size too, including for forward-only sources.
        while self.read(min(_CHUNK, self.limit - self.position + 1)):
            pass


def _integer(token: str) -> int:
    if not re.fullmatch(r"[+-]?[0-9]{1,12}", token):
        raise WfdbError("wfdb_header_invalid")
    return int(token)


def _parse_header(payload: bytes, limits: WfdbLimits) -> tuple:
    lines: list[str] = []
    comments = False
    for raw in payload.splitlines():
        raw = raw.strip()
        if raw.startswith(b"#"):
            comments = True
            continue
        if not raw:
            continue
        if len(raw) > 255:
            raise WfdbError("wfdb_header_invalid")
        try:
            text = raw.decode("utf-8")
        except UnicodeError:
            pass
        else:
            lines.append(text)
            continue
        raise WfdbError("wfdb_header_invalid")
    if not lines:
        raise WfdbError("wfdb_header_invalid")
    record = lines[0].split()
    if "/" in record[0]:
        raise WfdbError("wfdb_multisegment_unsupported")
    if len(record) < 4:
        raise WfdbError("wfdb_sample_count_required")
    count, total = _integer(record[1]), _integer(record[3])
    if count < 1 or count > limits.max_signals:
        raise WfdbError("wfdb_signal_limit_exceeded")
    if total < 1 or total > limits.max_samples:
        raise WfdbError("wfdb_sample_limit_exceeded")
    rate_match = _RATE.fullmatch(record[2])
    if rate_match is None:
        raise WfdbError("wfdb_header_invalid")
    rate = float(rate_match[1])
    if not math.isfinite(rate) or rate <= 0:
        raise WfdbError("wfdb_rate_invalid")
    if len(lines) != count + 1:
        raise WfdbError("wfdb_header_invalid")
    specs: list[_SignalSpec] = []
    files: list[str] = []
    for line in lines[1:]:
        fields = line.split(maxsplit=8)
        if len(fields) < 2:
            raise WfdbError("wfdb_header_invalid")
        filename = fields[0]
        if not files or filename != files[-1]:
            if filename in files:
                raise WfdbError("wfdb_signal_group_invalid")
            files.append(filename)
        match = _FORMAT.fullmatch(fields[1])
        if match is None:
            raise WfdbError("wfdb_format_unsupported")
        fmt = _integer(match[1])
        if fmt not in (16, 212, 80):
            raise WfdbError("wfdb_format_unsupported")
        if (match[2] is not None and _integer(match[2]) != 1) or (
            match[3] is not None and _integer(match[3]) != 0
        ):
            raise WfdbError("wfdb_layout_unsupported")
        offset = _integer(match[4]) if match[4] else 0
        if offset > limits.max_file_bytes:
            raise WfdbError("wfdb_file_limit_exceeded")
        gain, baseline, unit = 200.0, None, "mV"
        if len(fields) > 2:
            calibration = _GAIN.fullmatch(fields[2])
            if calibration is None:
                raise WfdbError("wfdb_header_invalid")
            gain = float(calibration[1])
            baseline = _integer(calibration[2]) if calibration[2] else None
            unit = calibration[3] or "mV"
        if not math.isfinite(gain) or gain == 0:
            raise WfdbError("wfdb_gain_invalid")
        numeric = [_integer(token) for token in fields[3:8]]
        if baseline is None:
            baseline = numeric[1] if len(numeric) > 1 else 0
        checksum = numeric[3] if len(numeric) > 3 else None
        if checksum is not None and not -32768 <= checksum <= 32767:
            raise WfdbError("wfdb_header_invalid")
        if len(numeric) > 4 and numeric[4] != 0:
            raise WfdbError("wfdb_layout_unsupported")
        label = fields[8] if len(fields) > 8 else None
        spec = _SignalSpec(
            len(files) - 1,
            fmt,
            offset,
            gain,
            baseline,
            unit if unit in _UNITS else None,
            label if label in _LEADS else None,
            checksum,
            label is not None and label not in _LEADS,
            unit not in _UNITS,
        )
        if (
            specs
            and specs[-1].group == spec.group
            and (specs[-1].format_code != fmt or specs[-1].offset != offset)
        ):
            raise WfdbError("wfdb_signal_group_invalid")
        specs.append(spec)
    return specs, rate, total, comments, len(record) > 4, len(files)


def _decoded(source: _Source, fmt: int, count: int):
    if fmt in (16, 80):
        width = 2 if fmt == 16 else 1
        while count:
            take = min(count, _CHUNK // width)
            chunk = source.exact(take * width, "wfdb_signal_truncated")
            if fmt == 16:
                yield from (s[0] for s in struct.iter_unpack("<h", chunk))
            else:
                yield from (s - 128 for s in chunk)
            count -= take
    else:
        while count >= 2:
            pairs = min(count // 2, _CHUNK // 3)
            chunk = source.exact(pairs * 3, "wfdb_signal_truncated")
            for i in range(0, len(chunk), 3):
                a = chunk[i] | ((chunk[i + 1] & 15) << 8)
                b = chunk[i + 2] | ((chunk[i + 1] & 240) << 4)
                yield a - 4096 if a & 2048 else a
                yield b - 4096 if b & 2048 else b
            count -= pairs * 2
        if count:
            # An odd final sample needs only its low byte and high-nibble byte.
            chunk = source.exact(2, "wfdb_signal_truncated")
            value = chunk[0] | ((chunk[1] & 15) << 8)
            yield value - 4096 if value & 2048 else value


def _annotations(
    source: _Source,
    total: int,
    start: int,
    end: int,
    limits: WfdbLimits,
) -> WfdbAnnotations:
    count, position, aux = 0, 0, False
    positions: list[int] = []
    while True:
        word = int.from_bytes(source.exact(2, "wfdb_annotation_truncated"), "little")
        if word == 0:
            source.finish()
            break
        code, interval = word >> 10, word & 1023
        if code == 59:
            if interval:
                raise WfdbError("wfdb_annotation_invalid")
            high, low = struct.unpack(
                "<HH", source.exact(4, "wfdb_annotation_truncated")
            )
            delta = (high << 16) | low
            position += delta - (1 << 32) if delta & (1 << 31) else delta
        elif code == 63:
            if not count:
                raise WfdbError("wfdb_annotation_invalid")
            aux = aux or interval > 0
            source.discard(interval + (interval & 1), "wfdb_annotation_truncated")
        elif code in (60, 61, 62):
            if not count:
                raise WfdbError("wfdb_annotation_invalid")
        elif 1 <= code <= 49:
            position += interval
            if not 0 <= position < total:
                raise WfdbError("wfdb_annotation_position_invalid")
            count += 1
            if count > limits.max_annotations:
                raise WfdbError("wfdb_annotation_limit_exceeded")
            if start <= position < end:
                positions.append(position)
        else:
            raise WfdbError("wfdb_annotation_format_unsupported")
    return WfdbAnnotations(count, tuple(positions), aux)


def read_wfdb_record(
    header: bytes | BinaryIO,
    signal_sources: Sequence[bytes | BinaryIO],
    *,
    start_sample: int = 0,
    sample_count: int | None = None,
    annotations: bytes | BinaryIO | None = None,
    limits: WfdbLimits = WfdbLimits(),
) -> WfdbRecord:
    """Read a bounded WFDB window without resolving names, paths, or URLs.

    Args:
        header: Bounded UTF-8 header bytes or a binary stream.
        signal_sources: Caller-supplied sources in first-file-appearance order.
            Consecutive signals sharing a file use a single source.
        start_sample: Zero-based first frame in the requested window.
        sample_count: Window length; None requests the remaining record.
        annotations: Optional MIT-format bytes or stream. Auxiliary text and
            annotation types are discarded; only counts and positions survive.
        limits: Positive budgets for inputs, decoding and output allocation.

    Returns:
        A review-only raw integer window with relative timing and safe metadata.

    Raises:
        WfdbError: A stable, value-free reason code for unsupported, malformed,
            uncalibrated or oversized inputs, failed checksums or stream errors.

    All declared samples are scanned in fixed chunks to check truncation and
    whole-record checksums, retaining only the window. Seekable streams are
    restored; forward-only streams are consumed. Caller streams are never closed.
    """
    if not isinstance(limits, WfdbLimits):
        raise WfdbError("wfdb_limits_invalid")
    with ExitStack() as stack:

        def wrap(source, budget):
            reader = _Source(source, budget)
            stack.callback(reader.close)
            return reader

        header_source = wrap(header, limits.max_header_bytes)
        parts = bytearray()
        while chunk := header_source.read(
            min(_CHUNK, limits.max_header_bytes - len(parts) + 1)
        ):
            parts.extend(chunk)
        specs, rate, total, comments, timing, group_count = _parse_header(
            bytes(parts), limits
        )
        if type(start_sample) is not int or not 0 <= start_sample <= total:
            raise WfdbError("wfdb_window_invalid")
        if sample_count is None:
            sample_count = total - start_sample
        if (
            type(sample_count) is not int
            or not 0 <= sample_count <= total - start_sample
        ):
            raise WfdbError("wfdb_window_invalid")
        if sample_count > limits.max_window_samples:
            raise WfdbError("wfdb_window_limit_exceeded")
        if (
            not isinstance(signal_sources, (list, tuple))
            or len(signal_sources) != group_count
        ):
            raise WfdbError("wfdb_source_count_invalid")
        outputs: list[WfdbSignal] = []
        for group in range(group_count):
            group_specs = [s for s in specs if s.group == group]
            source = wrap(signal_sources[group], limits.max_file_bytes)
            first = group_specs[0]
            scalar_count = total * len(group_specs)
            encoded_bytes = (
                (scalar_count * 3 + 1) // 2
                if first.format_code == 212
                else scalar_count * (2 if first.format_code == 16 else 1)
            )
            if first.offset + encoded_bytes > limits.max_file_bytes:
                raise WfdbError("wfdb_file_limit_exceeded")
            if (
                source.length is not None
                and first.offset + encoded_bytes > source.length
            ):
                raise WfdbError("wfdb_signal_truncated")
            source.discard(first.offset, "wfdb_signal_truncated")
            windows: list[list[int]] = [[] for _ in group_specs]
            sums = [0] * len(group_specs)
            for index, value in enumerate(
                _decoded(source, first.format_code, scalar_count)
            ):
                frame, channel = divmod(index, len(group_specs))
                sums[channel] = (sums[channel] + value) & 65535
                if start_sample <= frame < start_sample + sample_count:
                    windows[channel].append(value)
            source.finish()
            for spec, values, checksum in zip(group_specs, windows, sums):
                if spec.checksum is not None and spec.checksum & 65535 != checksum:
                    raise WfdbError("wfdb_checksum_mismatch")
                outputs.append(
                    WfdbSignal(
                        spec.lead_label,
                        spec.gain,
                        spec.baseline,
                        spec.unit,
                        spec.format_code,
                        tuple(values),
                        spec.checksum is not None,
                    )
                )
        ann = None
        if annotations is not None:
            ann = _annotations(
                wrap(annotations, limits.max_annotation_bytes),
                total,
                start_sample,
                start_sample + sample_count,
                limits,
            )
        return WfdbRecord(
            tuple(outputs),
            rate,
            total,
            start_sample,
            sample_count,
            ann,
            comments,
            True,
            True,
            timing,
            any(s.description_withheld for s in specs),
            any(s.unit_withheld for s in specs),
        )
