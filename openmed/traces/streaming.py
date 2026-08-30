"""Bounded-memory, value-free streaming redaction for trace records.

The public iterator in this module accepts structured trace records and emits
redacted copies in input order. It never puts a source record, a redacted
value, or a callback exception in progress state or reports. A caller can
inject a local text redactor (for example, one backed by a preloaded OpenMed
model); the built-in fallback handles common structured identifiers without
making a network request.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TextIO

DEFAULT_RECORD_BATCH_SIZE = 128
DEFAULT_BYTE_BATCH_SIZE = 1024 * 1024

_DEFAULT_HMAC_SECRET = b"openmed-trace-redaction-v1"
_SUPPORTED_METHODS = frozenset({"mask", "remove", "replace", "hash"})

DEFAULT_TRACE_FIELDS = (
    "message",
    "body",
    "description",
    "error.message",
    "error.stacktrace",
    "exception.message",
    "exception.stacktrace",
    "attributes.message",
    "attributes.body",
    "attributes.description",
    "attributes.error.message",
    "attributes.error.stacktrace",
    "attributes.exception.message",
    "attributes.exception.stacktrace",
    "attributes.user.email",
    "attributes.user.name",
    "attributes.patient.id",
    "attributes.patient.name",
    "attributes.mrn",
    "attributes.email",
    "attributes.phone",
    "attributes.ssn",
    "events.*.attributes.message",
    "events.*.attributes.exception.message",
    "events.*.attributes.exception.stacktrace",
)

_EMAIL_PATTERN = re.compile(
    r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+"
    r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![A-Za-z0-9._%+-])"
)
_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_DATE_PATTERN = re.compile(
    r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b"
)
_PHONE_PATTERN = re.compile(
    r"(?<!\w)(?:\+\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)"
    r"\d{3,4}[\s.-]?\d{3,4}(?!\w)"
)
_IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_UUID_PATTERN = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_MRN_PATTERN = re.compile(
    r"\b(?:MRN|medical\s+record(?:\s+number)?)\s*[:#-]?\s*"
    r"([A-Z0-9][A-Z0-9-]{3,})\b",
    re.IGNORECASE,
)
_CUE_NAME_PATTERN = re.compile(
    r"\b(?:patient|subject|user)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b",
    re.IGNORECASE,
)

_PATTERNS = (
    ("EMAIL", _EMAIL_PATTERN, 0),
    ("SSN", _SSN_PATTERN, 0),
    ("DATE", _DATE_PATTERN, 0),
    ("PHONE", _PHONE_PATTERN, 0),
    ("IP_ADDRESS", _IP_PATTERN, 0),
    ("UUID", _UUID_PATTERN, 0),
    ("ID", _MRN_PATTERN, 1),
    ("NAME", _CUE_NAME_PATTERN, 1),
)


TextRedactor = Callable[[str], Any]
ProgressCallback = Callable[["TraceProgress"], Any]
CancellationCheck = Callable[["TraceProgress"], Any]


def _secret_bytes(secret: str | bytes) -> bytes:
    if isinstance(secret, str):
        encoded = secret.encode("utf-8")
    elif isinstance(secret, bytes):
        encoded = secret
    else:
        raise TypeError("hmac_secret must be a string or bytes")
    if not encoded:
        raise ValueError("hmac_secret must be non-empty")
    return encoded


def _validate_positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class TraceRedactionConfig:
    """Configuration for one bounded trace-redaction run.

    The record batch size limits the number of records retained before a
    batch is processed. The byte batch size limits the estimated UTF-8 size
    of that batch. The byte limit is applied to each individual record too;
    an oversized record is rejected rather than silently exceeding the
    caller's bound.
    """

    record_batch_size: int = DEFAULT_RECORD_BATCH_SIZE
    byte_batch_size: int = DEFAULT_BYTE_BATCH_SIZE
    text_fields: tuple[str, ...] = DEFAULT_TRACE_FIELDS
    method: str = "mask"
    seed: int = 0
    hmac_secret: str | bytes = field(
        default=_DEFAULT_HMAC_SECRET,
        repr=False,
    )
    preserve_unmatched_text: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "record_batch_size",
            _validate_positive_integer(
                self.record_batch_size,
                "record_batch_size",
            ),
        )
        object.__setattr__(
            self,
            "byte_batch_size",
            _validate_positive_integer(self.byte_batch_size, "byte_batch_size"),
        )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        method = str(self.method).strip().lower()
        if method not in _SUPPORTED_METHODS:
            supported = ", ".join(sorted(_SUPPORTED_METHODS))
            raise ValueError(f"method must be one of: {supported}")
        object.__setattr__(self, "method", method)
        fields = tuple(self.text_fields)
        if not fields or any(
            not isinstance(item, str) or not item.strip() for item in fields
        ):
            raise ValueError("text_fields must contain non-empty strings")
        object.__setattr__(
            self,
            "text_fields",
            tuple(item.strip() for item in fields),
        )
        if not isinstance(self.preserve_unmatched_text, bool):
            raise ValueError("preserve_unmatched_text must be a boolean")
        _secret_bytes(self.hmac_secret)

    @property
    def max_records(self) -> int:
        """Alias for the record batch bound."""

        return self.record_batch_size

    @property
    def max_bytes(self) -> int:
        """Alias for the byte batch bound."""

        return self.byte_batch_size


@dataclass(frozen=True)
class TraceProgress:
    """PHI-free progress counters for a trace-redaction run."""

    records_seen: int = 0
    records_emitted: int = 0
    bytes_seen: int = 0
    bytes_emitted: int = 0
    batches_completed: int = 0
    redacted_fields: int = 0
    cancelled: bool = False

    @property
    def records_processed(self) -> int:
        """Return the number of records emitted so far."""

        return self.records_emitted

    @property
    def records_redacted(self) -> int:
        """Return the number of records emitted so far."""

        return self.records_emitted

    @property
    def bytes_read(self) -> int:
        """Return the number of estimated input bytes observed."""

        return self.bytes_seen

    @property
    def bytes_written(self) -> int:
        """Return the number of estimated output bytes emitted."""

        return self.bytes_emitted

    def to_dict(self) -> dict[str, int | bool]:
        """Return counters suitable for logs and metrics."""

        return {
            "records_seen": self.records_seen,
            "records_emitted": self.records_emitted,
            "bytes_seen": self.bytes_seen,
            "bytes_emitted": self.bytes_emitted,
            "batches_completed": self.batches_completed,
            "redacted_fields": self.redacted_fields,
            "cancelled": self.cancelled,
        }


@dataclass(frozen=True)
class TraceRedactionReport:
    """Aggregate PHI-free result counters for a trace-redaction run."""

    records_seen: int = 0
    records_emitted: int = 0
    bytes_seen: int = 0
    bytes_emitted: int = 0
    batches_completed: int = 0
    redacted_fields: int = 0
    cancelled: bool = False

    @property
    def records_processed(self) -> int:
        """Return the number of records emitted."""

        return self.records_emitted

    @property
    def records_redacted(self) -> int:
        """Return the number of records emitted."""

        return self.records_emitted

    @property
    def bytes_read(self) -> int:
        """Return the estimated input byte count."""

        return self.bytes_seen

    @property
    def bytes_written(self) -> int:
        """Return the estimated output byte count."""

        return self.bytes_emitted

    def to_dict(self) -> dict[str, int | bool]:
        """Return a value-free report for persistence or telemetry."""

        return {
            "records_seen": self.records_seen,
            "records_emitted": self.records_emitted,
            "bytes_seen": self.bytes_seen,
            "bytes_emitted": self.bytes_emitted,
            "batches_completed": self.batches_completed,
            "redacted_fields": self.redacted_fields,
            "cancelled": self.cancelled,
        }


@dataclass(frozen=True)
class TraceRedactionContext:
    """Deterministic, value-free pseudonym context for custom redactors."""

    method: str
    seed: int
    _hmac_secret: bytes = field(repr=False)

    def pseudonym(self, value: str, *, label: str = "VALUE") -> str:
        """Return a stable replacement without retaining the source value.

        Custom text redactors can use this helper when they need the same
        pseudonym for an identifier appearing in different batches.
        """

        if not isinstance(value, str):
            raise TypeError("pseudonym value must be a string")
        normalized_label = _normalize_label(label)
        material = (
            f"openmed.trace.redaction.v1|{self.seed}|{normalized_label}|{value}"
        ).encode("utf-8")
        digest = (
            hmac.new(
                self._hmac_secret,
                material,
                hashlib.sha256,
            )
            .hexdigest()[:16]
            .upper()
        )
        if self.method == "mask":
            return f"[{normalized_label}]"
        if self.method == "remove":
            return ""
        return f"{normalized_label}_{digest}"

    def redact_value(self, value: str, *, label: str = "VALUE") -> str:
        """Alias for pseudonym with redaction-oriented naming."""

        return self.pseudonym(value, label=label)


class TraceRedactionError(RuntimeError):
    """Raised when a trace stream cannot be safely redacted."""


class TraceRecordTooLargeError(TraceRedactionError):
    """Raised when one trace record exceeds the configured byte bound."""


class CancellationToken:
    """Cooperative cancellation signal for a trace-redaction iterator."""

    __slots__ = ("_cancelled",)

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation at the next deterministic batch boundary."""

        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        """Return whether cancellation has been requested."""

        return self._cancelled

    @property
    def is_cancelled(self) -> bool:
        """Alias for cancelled."""

        return self._cancelled


class TraceRedactor:
    """Redact structured trace records with bounded batch memory.

    The iterator processes a complete bounded batch before checking
    cancellation. This makes cancellation deterministic: a batch is either
    fully emitted or not started. The optional text_redactor is called only
    with configured string values; its return value may be a string or an
    object exposing deidentified_text.
    """

    def __init__(
        self,
        *,
        record_batch_size: int = DEFAULT_RECORD_BATCH_SIZE,
        byte_batch_size: int = DEFAULT_BYTE_BATCH_SIZE,
        max_records: int | None = None,
        max_bytes: int | None = None,
        batch_size: int | None = None,
        max_batch_bytes: int | None = None,
        text_fields: Sequence[str] = DEFAULT_TRACE_FIELDS,
        fields: Sequence[str] | None = None,
        method: str = "mask",
        seed: int = 0,
        hmac_secret: str | bytes = _DEFAULT_HMAC_SECRET,
        preserve_unmatched_text: bool = True,
        text_redactor: TextRedactor | None = None,
        redactor: TextRedactor | None = None,
        on_progress: ProgressCallback | None = None,
        cancellation: CancellationToken | CancellationCheck | None = None,
        cancel: CancellationToken | CancellationCheck | None = None,
    ) -> None:
        resolved_record_batch_size = record_batch_size
        if max_records is not None:
            resolved_record_batch_size = max_records
        if batch_size is not None:
            resolved_record_batch_size = batch_size
        resolved_byte_batch_size = byte_batch_size
        if max_bytes is not None:
            resolved_byte_batch_size = max_bytes
        if max_batch_bytes is not None:
            resolved_byte_batch_size = max_batch_bytes
        if fields is not None:
            text_fields = fields
        if text_redactor is not None and redactor is not None:
            raise ValueError("pass only one of text_redactor or redactor")
        if cancellation is not None and cancel is not None:
            raise ValueError("pass only one of cancellation or cancel")

        self.config = TraceRedactionConfig(
            record_batch_size=resolved_record_batch_size,
            byte_batch_size=resolved_byte_batch_size,
            text_fields=tuple(text_fields),
            method=method,
            seed=seed,
            hmac_secret=hmac_secret,
            preserve_unmatched_text=preserve_unmatched_text,
        )
        self.context = TraceRedactionContext(
            method=self.config.method,
            seed=self.config.seed,
            _hmac_secret=_secret_bytes(self.config.hmac_secret),
        )
        self._text_redactor = text_redactor or redactor
        self._on_progress = on_progress
        self._cancellation = cancellation or cancel
        self._internal_cancelled = False
        self._running = False
        self._report: TraceRedactionReport | None = None
        self._progress = TraceProgress()
        self._reset_counters()

    @property
    def report(self) -> TraceRedactionReport | None:
        """Return the completed or latest PHI-free report."""

        return self._report

    @property
    def progress(self) -> TraceProgress:
        """Return the latest PHI-free progress snapshot."""

        return self._progress

    def cancel(self) -> None:
        """Request cancellation at the next deterministic batch boundary."""

        self._internal_cancelled = True

    def __call__(
        self,
        records: Iterable[Mapping[str, Any]],
    ) -> Iterator[dict[str, Any]]:
        """Return an iterator over redacted records."""

        return self.iter_records(records)

    def iter_records(
        self,
        records: Iterable[Mapping[str, Any]],
    ) -> Iterator[dict[str, Any]]:
        """Yield redacted copies in source order."""

        if self._running:
            raise TraceRedactionError("a trace redaction run is already active")
        self._running = True
        self._reset_counters()
        batch: list[tuple[Mapping[str, Any], int]] = []
        batch_bytes = 0
        try:
            try:
                iterator = iter(records)
            except Exception:
                raise TraceRedactionError("trace input is not iterable") from None

            while True:
                if not batch and self._check_cancellation():
                    break
                try:
                    record = next(iterator)
                except StopIteration:
                    break
                except Exception:
                    raise TraceRedactionError("trace input could not be read") from None

                if not isinstance(record, Mapping):
                    raise TraceRedactionError("trace records must be mappings")
                try:
                    record_bytes = _estimate_bytes(record)
                except Exception:
                    raise TraceRedactionError(
                        "trace record size could not be measured"
                    ) from None
                if record_bytes > self.config.byte_batch_size:
                    raise TraceRecordTooLargeError(
                        "trace record exceeds byte_batch_size"
                    )
                if batch and (
                    len(batch) >= self.config.record_batch_size
                    or batch_bytes + record_bytes > self.config.byte_batch_size
                ):
                    yield from self._emit_batch(batch)
                    batch = []
                    batch_bytes = 0
                    if self._cancelled:
                        break

                batch.append((record, record_bytes))
                batch_bytes += record_bytes
                self._records_seen += 1
                self._bytes_seen += record_bytes

                if (
                    len(batch) >= self.config.record_batch_size
                    or batch_bytes >= self.config.byte_batch_size
                ):
                    yield from self._emit_batch(batch)
                    batch = []
                    batch_bytes = 0
                    if self._cancelled:
                        break

            if batch and not self._cancelled:
                yield from self._emit_batch(batch)
        finally:
            self._progress = self._snapshot()
            self._report = TraceRedactionReport(
                records_seen=self._records_seen,
                records_emitted=self._records_emitted,
                bytes_seen=self._bytes_seen,
                bytes_emitted=self._bytes_emitted,
                batches_completed=self._batches_completed,
                redacted_fields=self._redacted_fields,
                cancelled=self._cancelled,
            )
            self._running = False

    def _reset_counters(self) -> None:
        self._records_seen = 0
        self._records_emitted = 0
        self._bytes_seen = 0
        self._bytes_emitted = 0
        self._batches_completed = 0
        self._redacted_fields = 0
        self._cancelled = False
        self._progress = TraceProgress()
        self._report = None

    def _snapshot(self) -> TraceProgress:
        return TraceProgress(
            records_seen=self._records_seen,
            records_emitted=self._records_emitted,
            bytes_seen=self._bytes_seen,
            bytes_emitted=self._bytes_emitted,
            batches_completed=self._batches_completed,
            redacted_fields=self._redacted_fields,
            cancelled=self._cancelled,
        )

    def _check_cancellation(self) -> bool:
        if self._cancelled:
            return True
        if self._internal_cancelled:
            self._cancelled = True
            return True
        cancellation = self._cancellation
        if isinstance(cancellation, CancellationToken) and cancellation.cancelled:
            self._cancelled = True
            return True
        if callable(cancellation):
            try:
                requested = bool(cancellation(self._snapshot()))
            except Exception:
                raise TraceRedactionError("trace cancellation check failed") from None
            if requested:
                self._cancelled = True
        return self._cancelled

    def _emit_batch(
        self,
        batch: list[tuple[Mapping[str, Any], int]],
    ) -> Iterator[dict[str, Any]]:
        if not batch:
            return

        for index, (record, _) in enumerate(batch):
            # Release each source record before yielding its redacted copy. This
            # keeps the iterator incremental instead of retaining both complete
            # input and output batches at once.
            batch[index] = ({}, 0)
            try:
                redacted_record, field_count = self._redact_record(record)
                redacted_bytes = _estimate_bytes(redacted_record)
            except TraceRedactionError:
                raise
            except Exception:
                raise TraceRedactionError("trace record redaction failed") from None
            if redacted_bytes > self.config.byte_batch_size:
                raise TraceRecordTooLargeError(
                    "redacted trace record exceeds byte_batch_size"
                )

            self._records_emitted += 1
            self._bytes_emitted += redacted_bytes
            self._redacted_fields += field_count
            yield redacted_record

        self._batches_completed += 1
        self._progress = self._snapshot()
        if self._notify_progress(self._progress):
            self._cancelled = True

    def _notify_progress(self, progress: TraceProgress) -> bool:
        callback = self._on_progress
        if callback is None:
            return self._check_cancellation()
        try:
            requested = bool(callback(progress))
        except Exception:
            raise TraceRedactionError("trace progress callback failed") from None
        if requested:
            return True
        return self._check_cancellation()

    def _redact_record(
        self,
        record: Mapping[str, Any],
    ) -> tuple[dict[str, Any], int]:
        try:
            redacted: Any = copy.deepcopy(dict(record))
        except Exception:
            raise TraceRedactionError("trace record could not be copied") from None

        redacted_fields = 0
        for field_name in self.config.text_fields:
            parts = tuple(part for part in field_name.split(".") if part)
            redacted, count = _apply_field(
                redacted,
                parts,
                field_name=field_name,
                redact_text=self._redact_text,
            )
            redacted_fields += count
        if not isinstance(redacted, dict):
            raise TraceRedactionError("trace redaction returned an invalid record")
        return redacted, redacted_fields

    def _redact_text(self, text: str, *, field_name: str) -> str:
        callback = self._text_redactor
        if callback is not None:
            try:
                result = callback(text)
                if not isinstance(result, str):
                    result = getattr(result, "deidentified_text", result)
                if not isinstance(result, str):
                    raise TypeError
                return result
            except Exception:
                raise TraceRedactionError("trace text redaction failed") from None
        return _offline_redact_text(
            text,
            context=self.context,
            field_name=field_name,
            preserve_unmatched_text=self.config.preserve_unmatched_text,
        )


def redact_trace_records(
    records: Iterable[Mapping[str, Any]],
    *,
    record_batch_size: int = DEFAULT_RECORD_BATCH_SIZE,
    byte_batch_size: int = DEFAULT_BYTE_BATCH_SIZE,
    max_records: int | None = None,
    max_bytes: int | None = None,
    batch_size: int | None = None,
    max_batch_bytes: int | None = None,
    text_fields: Sequence[str] = DEFAULT_TRACE_FIELDS,
    fields: Sequence[str] | None = None,
    method: str = "mask",
    seed: int = 0,
    hmac_secret: str | bytes = _DEFAULT_HMAC_SECRET,
    preserve_unmatched_text: bool = True,
    text_redactor: TextRedactor | None = None,
    redactor: TextRedactor | None = None,
    on_progress: ProgressCallback | None = None,
    cancellation: CancellationToken | CancellationCheck | None = None,
    cancel: CancellationToken | CancellationCheck | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield redacted trace records in source order.

    The record and byte bounds are independent. A batch is flushed as soon as
    either bound is reached. Use TraceRedactor directly when the final report
    is needed.
    """

    runner = TraceRedactor(
        record_batch_size=record_batch_size,
        byte_batch_size=byte_batch_size,
        max_records=max_records,
        max_bytes=max_bytes,
        batch_size=batch_size,
        max_batch_bytes=max_batch_bytes,
        text_fields=text_fields,
        fields=fields,
        method=method,
        seed=seed,
        hmac_secret=hmac_secret,
        preserve_unmatched_text=preserve_unmatched_text,
        text_redactor=text_redactor,
        redactor=redactor,
        on_progress=on_progress,
        cancellation=cancellation,
        cancel=cancel,
    )
    yield from runner.iter_records(records)


def redact_trace_lines(
    lines: Iterable[str],
    *,
    record_batch_size: int = DEFAULT_RECORD_BATCH_SIZE,
    byte_batch_size: int = DEFAULT_BYTE_BATCH_SIZE,
    max_records: int | None = None,
    max_bytes: int | None = None,
    batch_size: int | None = None,
    max_batch_bytes: int | None = None,
    text_fields: Sequence[str] = DEFAULT_TRACE_FIELDS,
    fields: Sequence[str] | None = None,
    method: str = "mask",
    seed: int = 0,
    hmac_secret: str | bytes = _DEFAULT_HMAC_SECRET,
    preserve_unmatched_text: bool = True,
    text_redactor: TextRedactor | None = None,
    redactor: TextRedactor | None = None,
    on_progress: ProgressCallback | None = None,
    cancellation: CancellationToken | CancellationCheck | None = None,
    cancel: CancellationToken | CancellationCheck | None = None,
) -> Iterator[str]:
    """Yield redacted NDJSON lines without materializing the input."""

    runner = TraceRedactor(
        record_batch_size=record_batch_size,
        byte_batch_size=byte_batch_size,
        max_records=max_records,
        max_bytes=max_bytes,
        batch_size=batch_size,
        max_batch_bytes=max_batch_bytes,
        text_fields=text_fields,
        fields=fields,
        method=method,
        seed=seed,
        hmac_secret=hmac_secret,
        preserve_unmatched_text=preserve_unmatched_text,
        text_redactor=text_redactor,
        redactor=redactor,
        on_progress=on_progress,
        cancellation=cancellation,
        cancel=cancel,
    )
    for item in runner.iter_records(_iter_ndjson_records(lines)):
        try:
            yield (
                json.dumps(
                    item,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
        except Exception:
            raise TraceRedactionError("trace record could not be serialized") from None


def redact_ndjson_stream(
    input_stream: TextIO,
    output_stream: TextIO,
    *,
    record_batch_size: int = DEFAULT_RECORD_BATCH_SIZE,
    byte_batch_size: int = DEFAULT_BYTE_BATCH_SIZE,
    max_records: int | None = None,
    max_bytes: int | None = None,
    batch_size: int | None = None,
    max_batch_bytes: int | None = None,
    text_fields: Sequence[str] = DEFAULT_TRACE_FIELDS,
    fields: Sequence[str] | None = None,
    method: str = "mask",
    seed: int = 0,
    hmac_secret: str | bytes = _DEFAULT_HMAC_SECRET,
    preserve_unmatched_text: bool = True,
    text_redactor: TextRedactor | None = None,
    redactor: TextRedactor | None = None,
    on_progress: ProgressCallback | None = None,
    cancellation: CancellationToken | CancellationCheck | None = None,
    cancel: CancellationToken | CancellationCheck | None = None,
) -> TraceRedactionReport:
    """Redact an NDJSON trace stream and return a value-free report."""

    runner = TraceRedactor(
        record_batch_size=record_batch_size,
        byte_batch_size=byte_batch_size,
        max_records=max_records,
        max_bytes=max_bytes,
        batch_size=batch_size,
        max_batch_bytes=max_batch_bytes,
        text_fields=text_fields,
        fields=fields,
        method=method,
        seed=seed,
        hmac_secret=hmac_secret,
        preserve_unmatched_text=preserve_unmatched_text,
        text_redactor=text_redactor,
        redactor=redactor,
        on_progress=on_progress,
        cancellation=cancellation,
        cancel=cancel,
    )
    try:
        for record in runner.iter_records(_iter_ndjson_records(input_stream)):
            try:
                output_stream.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            except Exception:
                raise TraceRedactionError(
                    "trace record could not be serialized"
                ) from None
    except TraceRedactionError:
        raise
    except Exception:
        raise TraceRedactionError("trace output could not be written") from None
    return runner.report or TraceRedactionReport()


def _iter_ndjson_records(lines: Iterable[str]) -> Iterator[Mapping[str, Any]]:
    try:
        iterator = iter(lines)
    except Exception:
        raise TraceRedactionError("trace input is not iterable") from None
    line_number = 0
    while True:
        try:
            line = next(iterator)
        except StopIteration:
            return
        except Exception:
            raise TraceRedactionError("trace input could not be read") from None
        line_number += 1
        if not isinstance(line, str):
            raise TraceRedactionError("trace input lines must be strings")
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            raise TraceRedactionError(
                f"invalid trace JSON at input line {line_number}"
            ) from None
        if not isinstance(value, Mapping):
            raise TraceRedactionError(
                f"trace input line {line_number} must contain an object"
            )
        yield value


def _normalize_label(label: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", str(label).strip()).strip("_")
    return normalized.upper() or "VALUE"


def _field_label(field_name: str) -> str | None:
    tail = field_name.rsplit(".", 1)[-1].strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", tail)
    if normalized in {"email", "user_email"}:
        return "EMAIL"
    if normalized in {"phone", "telephone", "mobile"}:
        return "PHONE"
    if normalized in {"name", "user_name", "patient_name"}:
        return "NAME"
    if normalized in {"id", "mrn", "patient_id", "subject_id", "ssn"}:
        return "ID"
    if normalized in {"dob", "date_of_birth", "birth_date"}:
        return "DATE"
    if normalized in {"stacktrace", "stack_trace"}:
        return "STACKTRACE"
    return None


def _offline_redact_text(
    text: str,
    *,
    context: TraceRedactionContext,
    field_name: str,
    preserve_unmatched_text: bool,
) -> str:
    result = text
    matched = False
    for label, pattern, group in _PATTERNS:
        pattern_matched = False

        def replace(match: re.Match[str]) -> str:
            nonlocal pattern_matched
            pattern_matched = True
            source = match.group(group)
            replacement = context.redact_value(source, label=label)
            if group == 0:
                return replacement
            start = match.start(group) - match.start()
            end = match.end(group) - match.start()
            whole = match.group(0)
            return f"{whole[:start]}{replacement}{whole[end:]}"

        result = pattern.sub(replace, result)
        matched = matched or pattern_matched

    field_label = _field_label(field_name)
    if (
        matched
        or (preserve_unmatched_text and field_label is None)
        or not result.strip()
    ):
        return result
    label = field_label or "TEXT"
    leading = len(result) - len(result.lstrip())
    trailing = len(result) - len(result.rstrip())
    end = len(result) - trailing if trailing else len(result)
    return (
        result[:leading]
        + context.redact_value(result[leading:end], label=label)
        + (result[end:] if trailing else "")
    )


def _apply_field(
    value: Any,
    parts: Sequence[str],
    *,
    field_name: str,
    redact_text: Callable[..., str],
) -> tuple[Any, int]:
    if not parts:
        return _redact_target(
            value,
            field_name=field_name,
            redact_text=redact_text,
        )

    if isinstance(value, Mapping):
        mapping_result = dict(value)
        if parts[0] == "*":
            count = 0
            for key in tuple(mapping_result):
                mapping_result[key], changed = _apply_field(
                    mapping_result[key],
                    parts[1:],
                    field_name=field_name,
                    redact_text=redact_text,
                )
                count += changed
            return mapping_result, count

        count = 0
        direct_key = ".".join(parts)
        if direct_key in mapping_result:
            mapping_result[direct_key], changed = _redact_target(
                mapping_result[direct_key],
                field_name=field_name,
                redact_text=redact_text,
            )
            count += changed

        if len(parts) == 1:
            return mapping_result, count

        key = parts[0]
        if key not in mapping_result:
            return mapping_result, count
        mapping_result[key], changed = _apply_field(
            mapping_result[key],
            parts[1:],
            field_name=field_name,
            redact_text=redact_text,
        )
        count += changed
        return mapping_result, count

    if isinstance(value, list) and parts[0] == "*":
        list_result = list(value)
        count = 0
        for index, item in enumerate(list_result):
            list_result[index], changed = _apply_field(
                item,
                parts[1:],
                field_name=field_name,
                redact_text=redact_text,
            )
            count += changed
        return list_result, count

    if isinstance(value, tuple) and parts[0] == "*":
        tuple_result = list(value)
        count = 0
        for index, item in enumerate(tuple_result):
            tuple_result[index], changed = _apply_field(
                item,
                parts[1:],
                field_name=field_name,
                redact_text=redact_text,
            )
            count += changed
        return tuple(tuple_result), count

    return value, 0


def _redact_target(
    value: Any,
    *,
    field_name: str,
    redact_text: Callable[..., str],
) -> tuple[Any, int]:
    if isinstance(value, str):
        if not value:
            return value, 0
        return redact_text(value, field_name=field_name), 1
    if isinstance(value, list):
        result = list(value)
        count = 0
        for index, item in enumerate(result):
            result[index], changed = _redact_target(
                item,
                field_name=field_name,
                redact_text=redact_text,
            )
            count += changed
        return result, count
    if isinstance(value, tuple):
        result = list(value)
        count = 0
        for index, item in enumerate(result):
            result[index], changed = _redact_target(
                item,
                field_name=field_name,
                redact_text=redact_text,
            )
            count += changed
        return tuple(result), count
    return value, 0


def _estimate_bytes(value: Any) -> int:
    """Estimate UTF-8 JSON size without retaining a serialized copy."""

    if value is None:
        return 4
    if isinstance(value, bool):
        return 4 if value else 5
    if isinstance(value, str):
        total = 2
        for character in value:
            codepoint = ord(character)
            if character in {'"', "\\", "\b", "\f", "\n", "\r", "\t"}:
                total += 2
            elif codepoint < 0x20:
                total += 6
            else:
                total += len(character.encode("utf-8"))
        return total
    if isinstance(value, bytes):
        return len(value) + 2
    if isinstance(value, int):
        return len(str(value))
    if isinstance(value, float):
        return len(json.dumps(value))
    if isinstance(value, Mapping):
        size = 2
        for index, (key, item) in enumerate(value.items()):
            if index:
                size += 1
            size += _estimate_bytes(str(key)) + 1 + _estimate_bytes(item)
        return size
    if isinstance(value, (list, tuple)):
        size = 2
        for index, item in enumerate(value):
            if index:
                size += 1
            size += _estimate_bytes(item)
        return size
    return len(type(value).__name__) + 8


StreamingTraceRedactor = TraceRedactor
TraceCancellationToken = CancellationToken
stream_redact_trace_records = redact_trace_records
stream_redact_trace_lines = redact_trace_lines


__all__ = [
    "CancellationToken",
    "DEFAULT_BYTE_BATCH_SIZE",
    "DEFAULT_RECORD_BATCH_SIZE",
    "DEFAULT_TRACE_FIELDS",
    "StreamingTraceRedactor",
    "TextRedactor",
    "TraceCancellationToken",
    "TraceProgress",
    "TraceRecordTooLargeError",
    "TraceRedactionConfig",
    "TraceRedactionContext",
    "TraceRedactionError",
    "TraceRedactionReport",
    "TraceRedactor",
    "redact_ndjson_stream",
    "redact_trace_lines",
    "redact_trace_records",
    "stream_redact_trace_lines",
    "stream_redact_trace_records",
]
