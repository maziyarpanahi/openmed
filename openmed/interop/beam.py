"""Local-first Apache Beam redaction transform contract.

The Apache Beam SDK is optional.  This module can be imported, configured, and
used with :func:`run_synthetic_harness` without Beam installed, which keeps
serialization and retry behavior testable in an offline environment.  When
Beam is available, :class:`BeamRedactionTransform` is a regular ``PTransform``
whose worker-local ``DoFn`` uses the same redaction contract.

The contract is deliberately small and bounded: a run has explicit record and
byte limits, redaction attempts are capped, and state is limited to counters
and byte totals.  Reports contain only schema metadata, fingerprints, and
value-free counters.  Input records are never interpolated into logs or
exceptions.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

try:
    _beam = import_module("apache_beam")
except ModuleNotFoundError as exc:
    if exc.name != "apache_beam":  # pragma: no cover - broken installation
        raise
    _beam = None

_DoFnBase = _beam.DoFn if _beam is not None else object
_PTransformBase = _beam.PTransform if _beam is not None else object

_ARTIFACT_TYPE = "openmed.interop.beam.redaction"
_SCHEMA_VERSION = 1
_DEFAULT_TEXT_FIELD = "text"
_DEFAULT_POLICY = "hipaa_safe_harbor"
_DEFAULT_METHOD = "mask"
_DEFAULT_MAX_RECORDS = 10_000
_DEFAULT_MAX_INPUT_BYTES = 10 * 1024 * 1024
_DEFAULT_MAX_RECORD_BYTES = 1024 * 1024
_DEFAULT_MAX_ATTEMPTS = 3
_MAX_ATTEMPTS = 10
_MAX_RETRY_BACKOFF_SECONDS = 60.0

Record = str | Mapping[str, Any]
Deidentifier = Callable[..., Any]


class BeamRedactionError(RuntimeError):
    """Raised for a safe, deterministic Beam redaction contract failure."""


@dataclass(frozen=True)
class BeamRedactionSpec:
    """Bounded, serializable configuration for one redaction transform.

    The input and output schema is ``str | Mapping[str, Any]``.  Mapping
    records retain their outer shape and have only ``text_field`` transformed;
    a ``None`` value in that field is preserved without invoking a redactor.
    ``extra_kwargs`` are forwarded to the injected or default deidentifier,
    but their values are intentionally omitted from public metadata.
    """

    text_field: str = _DEFAULT_TEXT_FIELD
    policy: str = _DEFAULT_POLICY
    method: str = _DEFAULT_METHOD
    max_records: int = _DEFAULT_MAX_RECORDS
    max_input_bytes: int = _DEFAULT_MAX_INPUT_BYTES
    max_record_bytes: int = _DEFAULT_MAX_RECORD_BYTES
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS
    retry_backoff_seconds: float = 0.0
    extra_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize configuration before a worker is created."""

        _require_non_empty_string(self.text_field, "text_field")
        _require_non_empty_string(self.policy, "policy")
        _require_non_empty_string(self.method, "method")
        _require_positive_int(self.max_records, "max_records")
        _require_positive_int(self.max_input_bytes, "max_input_bytes")
        _require_positive_int(self.max_record_bytes, "max_record_bytes")
        _require_positive_int(self.max_attempts, "max_attempts")
        if self.max_attempts > _MAX_ATTEMPTS:
            raise ValueError("max_attempts exceeds the bounded retry limit")
        if isinstance(self.retry_backoff_seconds, bool) or not isinstance(
            self.retry_backoff_seconds, (int, float)
        ):
            raise TypeError("retry_backoff_seconds must be a real number")
        if not 0 <= float(self.retry_backoff_seconds) <= _MAX_RETRY_BACKOFF_SECONDS:
            raise ValueError("retry_backoff_seconds is outside the bounded range")
        if not isinstance(self.extra_kwargs, Mapping):
            raise TypeError("extra_kwargs must be a mapping")
        collisions = {"loader"} & set(self.extra_kwargs)
        if collisions:
            raise ValueError("extra_kwargs cannot override the worker-local loader")
        object.__setattr__(self, "extra_kwargs", dict(self.extra_kwargs))

    @property
    def input_schema(self) -> str:
        """Return the stable input schema identifier."""

        return "string_or_mapping"

    @property
    def output_schema(self) -> str:
        """Return the stable output schema identifier."""

        return "same_as_input"

    def to_deidentify_kwargs(self) -> dict[str, Any]:
        """Return explicit, deterministic options for the deidentifier."""

        kwargs: dict[str, Any] = {
            "method": self.method,
            "policy": self.policy,
        }
        collisions = sorted(set(kwargs) & set(self.extra_kwargs))
        if collisions:
            fields = ", ".join(collisions)
            raise ValueError(f"extra_kwargs cannot override named fields: {fields}")
        kwargs.update(self.extra_kwargs)
        return kwargs

    def to_dict(self) -> dict[str, Any]:
        """Return PHI-free schema and bound metadata for this specification."""

        return {
            "artifact_type": _ARTIFACT_TYPE,
            "schema_version": _SCHEMA_VERSION,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "text_field": self.text_field,
            "policy": self.policy,
            "method": self.method,
            "max_records": self.max_records,
            "max_input_bytes": self.max_input_bytes,
            "max_record_bytes": self.max_record_bytes,
            "max_attempts": self.max_attempts,
            "retry_backoff_seconds": float(self.retry_backoff_seconds),
            "extra_keys": tuple(sorted(str(key) for key in self.extra_kwargs)),
        }

    def fingerprint(self) -> str:
        """Return a deterministic fingerprint without exposing option values."""

        payload = {
            **self.to_dict(),
            "extra_values": _fingerprint_value(self.extra_kwargs),
        }
        return _digest_bytes(_canonical_json(payload))

    def __repr__(self) -> str:
        """Return configuration metadata without extra-option values."""

        return f"BeamRedactionSpec({self.to_dict()!r})"


@dataclass(frozen=True)
class BeamRedactionCounters:
    """Aggregate counters safe to expose to Beam metrics or pipeline reports."""

    records_processed: int = 0
    records_changed: int = 0
    records_failed: int = 0
    attempts: int = 0
    retries: int = 0
    spans_redacted: int = 0
    input_bytes: int = 0
    output_bytes: int = 0

    def __post_init__(self) -> None:
        """Reject invalid counter values instead of serializing them."""

        for name in self.to_dict():
            _require_non_negative_int(getattr(self, name), name)

    def to_dict(self) -> dict[str, int]:
        """Return deterministic counts without identifiers or source values."""

        return {
            "attempts": self.attempts,
            "input_bytes": self.input_bytes,
            "output_bytes": self.output_bytes,
            "records_changed": self.records_changed,
            "records_failed": self.records_failed,
            "records_processed": self.records_processed,
            "retries": self.retries,
            "spans_redacted": self.spans_redacted,
        }

    def __getitem__(self, key: str) -> int:
        """Allow mapping-style access to a counter."""

        return self.to_dict()[key]


@dataclass
class BeamRedactionState:
    """Mutable bounded state for one worker or direct synthetic run.

    Only record and byte totals are retained.  No input, output, identifier,
    or exception value is stored in the state object.
    """

    max_records: int = _DEFAULT_MAX_RECORDS
    max_input_bytes: int = _DEFAULT_MAX_INPUT_BYTES
    max_record_bytes: int = _DEFAULT_MAX_RECORD_BYTES
    records_seen: int = 0
    input_bytes: int = 0

    def __post_init__(self) -> None:
        """Validate state bounds and counters."""

        _require_positive_int(self.max_records, "max_records")
        _require_positive_int(self.max_input_bytes, "max_input_bytes")
        _require_positive_int(self.max_record_bytes, "max_record_bytes")
        _require_non_negative_int(self.records_seen, "records_seen")
        _require_non_negative_int(self.input_bytes, "input_bytes")
        if self.records_seen > self.max_records:
            raise ValueError("records_seen exceeds max_records")
        if self.input_bytes > self.max_input_bytes:
            raise ValueError("input_bytes exceeds max_input_bytes")

    def accept(self, serialized_record: bytes) -> None:
        """Account for one serialized record, enforcing all state bounds."""

        if len(serialized_record) > self.max_record_bytes:
            raise BeamRedactionError("record exceeds the configured byte limit")
        if self.records_seen >= self.max_records:
            raise BeamRedactionError("record batch exceeds the configured limit")
        if self.input_bytes + len(serialized_record) > self.max_input_bytes:
            raise BeamRedactionError("record batch exceeds the configured byte limit")
        self.records_seen += 1
        self.input_bytes += len(serialized_record)

    def to_dict(self) -> dict[str, int]:
        """Return only bounded state counters."""

        return {
            "input_bytes": self.input_bytes,
            "max_input_bytes": self.max_input_bytes,
            "max_record_bytes": self.max_record_bytes,
            "max_records": self.max_records,
            "records_seen": self.records_seen,
        }


@dataclass(frozen=True)
class BeamRedactionResult:
    """Redacted direct-run output plus a PHI-free aggregate report."""

    redacted_records: tuple[Record, ...]
    counters: BeamRedactionCounters
    input_fingerprint: str
    output_fingerprint: str
    spec_fingerprint: str
    serialized_output: bytes

    def report(self) -> dict[str, Any]:
        """Return a report containing no record values or exception details."""

        return {
            "artifact_type": _ARTIFACT_TYPE,
            "schema_version": _SCHEMA_VERSION,
            "input_fingerprint": self.input_fingerprint,
            "output_fingerprint": self.output_fingerprint,
            "spec_fingerprint": self.spec_fingerprint,
            "counters": self.counters.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Alias for :meth:`report` for JSON-oriented callers."""

        return self.report()

    def __repr__(self) -> str:
        """Return a safe summary without output records or serialized bytes."""

        return (
            "BeamRedactionResult("
            f"records={len(self.redacted_records)}, "
            f"counters={self.counters.to_dict()!r}, "
            f"input_fingerprint={self.input_fingerprint!r}, "
            f"output_fingerprint={self.output_fingerprint!r}, "
            f"spec_fingerprint={self.spec_fingerprint!r})"
        )


class _CounterAccumulator:
    """Mutable implementation detail for one bounded run."""

    def __init__(self) -> None:
        self.records_processed = 0
        self.records_changed = 0
        self.records_failed = 0
        self.attempts = 0
        self.retries = 0
        self.spans_redacted = 0
        self.input_bytes = 0
        self.output_bytes = 0

    def freeze(self) -> BeamRedactionCounters:
        """Return immutable, value-free counters."""

        return BeamRedactionCounters(
            records_processed=self.records_processed,
            records_changed=self.records_changed,
            records_failed=self.records_failed,
            attempts=self.attempts,
            retries=self.retries,
            spans_redacted=self.spans_redacted,
            input_bytes=self.input_bytes,
            output_bytes=self.output_bytes,
        )


class _BeamRedactionDoFn(_DoFnBase):  # type: ignore[misc,valid-type]
    """Worker-local Beam ``DoFn`` implementing :class:`BeamRedactionSpec`."""

    def __init__(
        self,
        *,
        spec: BeamRedactionSpec,
        deidentifier: Deidentifier | None = None,
        loader_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._spec = spec
        self._deidentifier = deidentifier
        self._loader_factory = loader_factory
        self._loader: Any = None
        self._state = BeamRedactionState(
            max_records=spec.max_records,
            max_input_bytes=spec.max_input_bytes,
            max_record_bytes=spec.max_record_bytes,
        )
        self._counters = _CounterAccumulator()
        self._metrics: dict[str, Any] = {}

    def setup(self) -> None:
        """Initialize one local loader and value-free Beam metrics."""

        if self._loader is None and self._deidentifier is None:
            self._loader = (self._loader_factory or _new_model_loader)()
        if _beam is not None and not self._metrics:
            metrics = _beam.metrics.Metrics
            self._metrics = {
                name: metrics.counter("openmed", name)
                for name in (
                    "records_processed",
                    "records_changed",
                    "records_failed",
                    "attempts",
                    "retries",
                    "spans_redacted",
                )
            }

    def process(self, element: Record) -> Iterator[Record]:
        """Redact one element and yield the same outer schema."""

        if self._loader is None and self._deidentifier is None:
            raise RuntimeError("BeamRedactionTransform.setup() must run first")

        normalized, serialized = _validate_and_serialize_record(
            element,
            self._spec,
        )
        self._state.accept(serialized)
        self._counters.records_processed += 1
        self._counters.input_bytes += len(serialized)
        self._inc("records_processed")

        try:
            redacted, spans, attempts, retries = _redact_with_retries(
                normalized,
                spec=self._spec,
                deidentifier=self._deidentifier,
                loader=self._loader,
            )
        except BeamRedactionError:
            self._counters.records_failed += 1
            self._inc("records_failed")
            raise

        output_bytes = serialize_record(redacted)
        self._counters.attempts += attempts
        self._counters.retries += retries
        self._counters.spans_redacted += spans
        self._counters.output_bytes += len(output_bytes)
        self._inc("attempts", attempts)
        self._inc("retries", retries)
        self._inc("spans_redacted", spans)
        if redacted != normalized:
            self._counters.records_changed += 1
            self._inc("records_changed")
        yield redacted

    def _inc(self, name: str, amount: int = 1) -> None:
        metric = self._metrics.get(name)
        if metric is not None:
            metric.inc(amount)


class BeamRedactionTransform(_PTransformBase):  # type: ignore[misc,valid-type]
    """Apply a bounded, local-first redaction contract to a Beam collection.

    ``spec`` can be supplied as a complete :class:`BeamRedactionSpec`, or the
    common fields can be passed directly.  The optional ``deidentifier`` is
    useful for a worker-preloaded model or an offline test.  It receives the
    text plus the explicit ``method`` and ``policy`` options; a default
    deidentifier receives a worker-local loader and cache-only configuration.
    """

    def __init__(
        self,
        spec: BeamRedactionSpec | None = None,
        *,
        text_field: str = _DEFAULT_TEXT_FIELD,
        policy: str = _DEFAULT_POLICY,
        method: str = _DEFAULT_METHOD,
        max_records: int = _DEFAULT_MAX_RECORDS,
        max_input_bytes: int = _DEFAULT_MAX_INPUT_BYTES,
        max_record_bytes: int = _DEFAULT_MAX_RECORD_BYTES,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
        retry_backoff_seconds: float = 0.0,
        extra_kwargs: Mapping[str, Any] | None = None,
        deidentifier: Deidentifier | None = None,
        loader_factory: Callable[[], Any] | None = None,
    ) -> None:
        if _beam is not None:
            super().__init__()
        if spec is not None:
            if any(
                value != default
                for value, default in (
                    (text_field, _DEFAULT_TEXT_FIELD),
                    (policy, _DEFAULT_POLICY),
                    (method, _DEFAULT_METHOD),
                    (max_records, _DEFAULT_MAX_RECORDS),
                    (max_input_bytes, _DEFAULT_MAX_INPUT_BYTES),
                    (max_record_bytes, _DEFAULT_MAX_RECORD_BYTES),
                    (max_attempts, _DEFAULT_MAX_ATTEMPTS),
                    (retry_backoff_seconds, 0.0),
                    (extra_kwargs, None),
                )
            ):
                raise TypeError("spec cannot be combined with direct options")
            self.spec = spec
        else:
            self.spec = BeamRedactionSpec(
                text_field=text_field,
                policy=policy,
                method=method,
                max_records=max_records,
                max_input_bytes=max_input_bytes,
                max_record_bytes=max_record_bytes,
                max_attempts=max_attempts,
                retry_backoff_seconds=retry_backoff_seconds,
                extra_kwargs=extra_kwargs or {},
            )
        self._deidentifier = deidentifier
        self._loader_factory = loader_factory

    def expand(self, pcoll: Any) -> Any:
        """Return a Beam ``PCollection`` with the configured transform applied."""

        beam = _require_beam()
        return pcoll | beam.ParDo(
            _BeamRedactionDoFn(
                spec=self.spec,
                deidentifier=self._deidentifier,
                loader_factory=self._loader_factory,
            )
        )


def run_synthetic_harness(
    records: Iterable[Record],
    *,
    spec: BeamRedactionSpec | None = None,
    deidentifier: Deidentifier | None = None,
    loader_factory: Callable[[], Any] | None = None,
) -> BeamRedactionResult:
    """Run the Beam contract directly on bounded synthetic records.

    The harness performs canonical JSON serialization, enforces the same
    record and byte bounds as the Beam worker, retries failed redactions up to
    ``spec.max_attempts``, and returns a counts-only report alongside the
    redacted records.  It performs no network operation itself; the default
    OpenMed path is configured for cache-only loading.
    """

    resolved_spec = spec or BeamRedactionSpec()
    state = BeamRedactionState(
        max_records=resolved_spec.max_records,
        max_input_bytes=resolved_spec.max_input_bytes,
        max_record_bytes=resolved_spec.max_record_bytes,
    )
    counters = _CounterAccumulator()
    input_serialized: list[bytes] = []
    output_serialized: list[bytes] = []
    redacted_records: list[Record] = []
    loader: Any = None

    for record in records:
        normalized, serialized = _validate_and_serialize_record(
            record,
            resolved_spec,
        )
        state.accept(serialized)
        input_serialized.append(serialized)
        counters.records_processed += 1
        counters.input_bytes += len(serialized)
        if (
            deidentifier is None
            and loader is None
            and _record_text(normalized, resolved_spec.text_field) is not None
        ):
            loader = (loader_factory or _new_model_loader)()

        redacted, spans, attempts, retries = _redact_with_retries(
            normalized,
            spec=resolved_spec,
            deidentifier=deidentifier,
            loader=loader,
        )
        output = serialize_record(redacted)
        redacted_records.append(redacted)
        output_serialized.append(output)
        counters.attempts += attempts
        counters.retries += retries
        counters.spans_redacted += spans
        counters.output_bytes += len(output)
        counters.records_changed += int(redacted != normalized)

    serialized_input = _serialize_lines(input_serialized)
    serialized_output = _serialize_lines(output_serialized)
    return BeamRedactionResult(
        redacted_records=tuple(redacted_records),
        counters=counters.freeze(),
        input_fingerprint=_digest_bytes(serialized_input),
        output_fingerprint=_digest_bytes(serialized_output),
        spec_fingerprint=resolved_spec.fingerprint(),
        serialized_output=serialized_output,
    )


def serialize_record(record: Record) -> bytes:
    """Return one record in deterministic canonical JSON bytes."""

    return _canonical_json(record)


def serialize_records(records: Iterable[Record]) -> bytes:
    """Return a deterministic newline-delimited serialization of records."""

    return _serialize_lines(serialize_record(record) for record in records)


def _redact_with_retries(
    record: Record,
    *,
    spec: BeamRedactionSpec,
    deidentifier: Deidentifier | None,
    loader: Any,
) -> tuple[Record, int, int, int]:
    text = _record_text(record, spec.text_field)
    if text is None:
        return record, 0, 0, 0

    kwargs = spec.to_deidentify_kwargs()
    if deidentifier is None:
        deidentifier = _default_deidentifier
        kwargs.setdefault("config", _offline_config())
    if loader is not None:
        kwargs["loader"] = loader

    attempts = 0
    retries = 0
    while attempts < spec.max_attempts:
        attempts += 1
        try:
            result = deidentifier(text, **kwargs)
            redacted, spans = _result_text_and_spans(result, text)
            break
        except Exception:
            if attempts >= spec.max_attempts:
                fingerprint = _digest_bytes(text.encode("utf-8"))
                raise BeamRedactionError(
                    "redaction failed after the configured attempts; "
                    f"record_fingerprint={fingerprint}"
                ) from None
            retries += 1
            if spec.retry_backoff_seconds:
                time.sleep(float(spec.retry_backoff_seconds))
    else:  # pragma: no cover - loop is bounded by max_attempts
        raise BeamRedactionError("redaction attempts were exhausted") from None

    if isinstance(record, str):
        return redacted, spans, attempts, retries
    output = dict(record)
    output[spec.text_field] = redacted
    return output, spans, attempts, retries


def _validate_and_serialize_record(
    record: Record,
    spec: BeamRedactionSpec,
) -> tuple[Record, bytes]:
    if isinstance(record, str):
        normalized: Record = record
    elif isinstance(record, Mapping):
        if not all(isinstance(key, str) for key in record):
            raise BeamRedactionError("record mapping keys must be strings")
        normalized = dict(record)
        if spec.text_field not in normalized:
            raise BeamRedactionError("record is missing the configured text field")
        value = normalized[spec.text_field]
        if value is not None and not isinstance(value, str):
            raise BeamRedactionError(
                "configured record field must contain text or null"
            )
    else:
        raise BeamRedactionError("records must contain strings or mappings")
    try:
        serialized = serialize_record(normalized)
    except (TypeError, ValueError, OverflowError):
        raise BeamRedactionError("record is not JSON serializable") from None
    return normalized, serialized


def _record_text(record: Record, text_field: str) -> str | None:
    if isinstance(record, str):
        return record
    value = record[text_field]
    return value


def _result_text_and_spans(result: Any, original: str) -> tuple[str, int]:
    if isinstance(result, str):
        return result, int(result != original)
    redacted = getattr(result, "deidentified_text", None)
    if not isinstance(redacted, str):
        raise TypeError("deidentifier must return text or deidentified_text")
    entities = getattr(result, "pii_entities", ())
    try:
        spans = len(entities)
    except (TypeError, ValueError):
        spans = int(redacted != original)
    if isinstance(entities, (str, bytes, bytearray)):
        spans = int(redacted != original)
    return redacted, max(0, int(spans))


def _serialize_lines(records: Iterable[bytes]) -> bytes:
    values = tuple(records)
    return b"\n".join(values) + (b"\n" if values else b"")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _fingerprint_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return {"type": "str", "sha256": _digest_bytes(value.encode("utf-8"))}
    if isinstance(value, bytes):
        return {"type": "bytes", "sha256": _digest_bytes(value)}
    if isinstance(value, Mapping):
        return {
            str(key): _fingerprint_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_fingerprint_value(item) for item in value]
    return {"type": type(value).__name__}


def _require_beam() -> Any:
    if _beam is None:
        raise ImportError(
            "Apache Beam support requires the optional dependency; install "
            "openmed[beam] to use BeamRedactionTransform"
        )
    return _beam


def _new_model_loader() -> Any:
    from openmed import ModelLoader

    return ModelLoader(config=_offline_config())


def _default_deidentifier(text: str, **kwargs: Any) -> Any:
    from openmed.core.pii import deidentify

    return deidentify(text, **kwargs)


def _offline_config() -> Any:
    from openmed.core.config import OpenMedConfig

    return OpenMedConfig(local_only=True)


def _require_non_empty_string(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _require_positive_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_non_negative_int(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


RedactionTransformSpec = BeamRedactionSpec
BeamTransformSpec = BeamRedactionSpec
RedactionCounters = BeamRedactionCounters
BeamTransformResult = BeamRedactionResult
run_direct = run_synthetic_harness
run_direct_harness = run_synthetic_harness


__all__ = [
    "BeamRedactionCounters",
    "BeamRedactionError",
    "BeamRedactionResult",
    "BeamRedactionSpec",
    "BeamRedactionState",
    "BeamRedactionTransform",
    "BeamTransformResult",
    "BeamTransformSpec",
    "RedactionCounters",
    "RedactionTransformSpec",
    "run_direct",
    "run_direct_harness",
    "run_synthetic_harness",
    "serialize_record",
    "serialize_records",
]
