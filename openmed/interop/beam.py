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
import math
import re
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, cast

from openmed.core.policy import canonical_policy_name

try:
    _beam: Any = import_module("apache_beam")
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
_DEFAULT_MAX_OUTPUT_BYTES = 10 * 1024 * 1024
_DEFAULT_MAX_RECORD_BYTES = 1024 * 1024
_DEFAULT_MAX_ATTEMPTS = 3
_MAX_ATTEMPTS = 10
_MAX_CONFIG_TEXT_CHARS = 256
_MAX_EXTRA_DEPTH = 16
_MAX_EXTRA_ITEMS = 1_000
_MAX_EXTRA_INT_BITS = 4_096
_MAX_EXTRA_KEY_CHARS = 128
_MAX_EXTRA_KWARGS = 64
_MAX_EXTRA_STRING_CHARS = 64 * 1024
_MAX_INPUT_BYTES = 256 * 1024 * 1024
_MIN_OUTPUT_CHARS = 4_096
_MAX_OUTPUT_BYTES = 256 * 1024 * 1024
_MAX_OUTPUT_EXPANSION = 8
_MAX_OUTPUT_RECORD_BYTES = 64 * 1024 * 1024
_MAX_RECORD_BYTES = 16 * 1024 * 1024
_MAX_RECORD_DEPTH = 32
_MAX_RECORD_INT_BITS = 4_096
_MAX_RECORD_ITEMS = 10_000
_MAX_RECORD_KEY_CHARS = 4_096
_MAX_RECORDS = 1_000_000
_MAX_RETRY_BACKOFF_SECONDS = 60.0
_MAX_SPANS_PER_RECORD = 10_000
_SAFE_METADATA_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_METADATA_IDENTIFIER_RE = re.compile(
    r"(?:^|[_.:-])(?:account|case|encounter|member|mrn|patient|record|subject)"
    r"[_.:-]?(?:\d{2,}|[a-f0-9]{8,})(?:$|[_.:-])|"
    r"(?<![A-Za-z0-9])\d{6,}(?![A-Za-z0-9])"
)
_DEIDENTIFICATION_METHODS = frozenset(
    {
        "aadhaar_mask",
        "format_preserve",
        "hash",
        "mask",
        "remove",
        "replace",
        "shift_dates",
    }
)
_RESERVED_EXTRA_KEYS = frozenset(
    {
        "audit",
        "config",
        "keep_mapping",
        "loader",
        "method",
        "policy",
        "use_safety_sweep",
    }
)
_MISSING = object()

Record = str | Mapping[str, Any]
Deidentifier = Callable[..., Any]


class BeamRedactionError(RuntimeError):
    """Raised for a safe, deterministic Beam redaction contract failure."""


class _BoundaryError(Exception):
    """Internal marker for a rejected untrusted input boundary."""


class _FrozenList(tuple[Any, ...]):
    """Pickle-friendly marker preserving a caller-supplied list shape."""


@dataclass(frozen=True)
class _FrozenOptions(Mapping[str, Any]):
    """Pickle-friendly immutable mapping with a value-free representation."""

    _items: tuple[tuple[str, Any], ...]

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, key: str) -> Any:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __repr__(self) -> str:
        return f"_FrozenOptions(keys={tuple(self)!r})"


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
    max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES
    max_record_bytes: int = _DEFAULT_MAX_RECORD_BYTES
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS
    retry_backoff_seconds: float = 0.0
    extra_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize configuration before a worker is created."""

        object.__setattr__(
            self,
            "text_field",
            _normalize_text_field(self.text_field),
        )
        object.__setattr__(self, "policy", _normalize_policy(self.policy))
        object.__setattr__(
            self,
            "method",
            _normalize_method(self.method),
        )
        _require_positive_int(
            self.max_records,
            "max_records",
            maximum=_MAX_RECORDS,
        )
        _require_positive_int(
            self.max_input_bytes,
            "max_input_bytes",
            maximum=_MAX_INPUT_BYTES,
        )
        _require_positive_int(
            self.max_output_bytes,
            "max_output_bytes",
            maximum=_MAX_OUTPUT_BYTES,
        )
        _require_positive_int(
            self.max_record_bytes,
            "max_record_bytes",
            maximum=_MAX_RECORD_BYTES,
        )
        _require_positive_int(self.max_attempts, "max_attempts")
        if self.max_attempts > _MAX_ATTEMPTS:
            raise ValueError("max_attempts exceeds the bounded retry limit")
        if type(self.retry_backoff_seconds) not in (int, float):
            raise TypeError("retry_backoff_seconds must be a real number")
        normalized_backoff = float(self.retry_backoff_seconds)
        if (
            not math.isfinite(normalized_backoff)
            or not 0 <= normalized_backoff <= _MAX_RETRY_BACKOFF_SECONDS
        ):
            raise ValueError("retry_backoff_seconds is outside the bounded range")
        object.__setattr__(self, "retry_backoff_seconds", normalized_backoff)
        if not isinstance(self.extra_kwargs, Mapping):
            raise TypeError("extra_kwargs must be a mapping")
        object.__setattr__(
            self,
            "extra_kwargs",
            _snapshot_extra_kwargs(self.extra_kwargs),
        )

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

        safe = _validated_spec(self)
        kwargs: dict[str, Any] = {
            "method": safe.method,
            "policy": safe.policy,
        }
        kwargs.update(
            {key: _clone_extra_value(value) for key, value in safe.extra_kwargs.items()}
        )
        return kwargs

    def to_dict(self) -> dict[str, Any]:
        """Return PHI-free schema and bound metadata for this specification."""

        return _validated_spec(self)._to_dict_unchecked()

    def _to_dict_unchecked(self) -> dict[str, Any]:
        """Return metadata after the specification has been reconstructed."""

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
            "max_output_bytes": self.max_output_bytes,
            "max_record_bytes": self.max_record_bytes,
            "max_attempts": self.max_attempts,
            "retry_backoff_seconds": self.retry_backoff_seconds,
            "extra_key_count": len(self.extra_kwargs),
            "extra_keys_fingerprint": _digest_bytes(
                _canonical_json(tuple(sorted(self.extra_kwargs)))
            ),
        }

    def fingerprint(self) -> str:
        """Return a deterministic fingerprint without exposing option values."""

        try:
            safe = _validated_spec(self)
            payload = {
                **safe._to_dict_unchecked(),
                "extra_values": _fingerprint_value(safe.extra_kwargs),
            }
            return _digest_bytes(_canonical_json(payload))
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise BeamRedactionError(
                "redaction specification could not be fingerprinted safely"
            ) from None

    def __repr__(self) -> str:
        """Return configuration metadata without extra-option values."""

        try:
            return f"BeamRedactionSpec({self.to_dict()!r})"
        except BaseException:
            return "BeamRedactionSpec(<invalid>)"


@dataclass(frozen=True, slots=True)
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

        maxima = {
            "attempts": _MAX_RECORDS * _MAX_ATTEMPTS,
            "input_bytes": _MAX_INPUT_BYTES,
            "output_bytes": _MAX_OUTPUT_BYTES,
            "records_changed": _MAX_RECORDS,
            "records_failed": _MAX_RECORDS,
            "records_processed": _MAX_RECORDS,
            "retries": _MAX_RECORDS * (_MAX_ATTEMPTS - 1),
            "spans_redacted": _MAX_RECORDS * _MAX_SPANS_PER_RECORD,
        }
        for name, maximum in maxima.items():
            _require_non_negative_int(getattr(self, name), name, maximum=maximum)
        if self.records_changed + self.records_failed > self.records_processed:
            raise ValueError("record outcome counters exceed records_processed")
        if self.retries > self.attempts:
            raise ValueError("retries cannot exceed attempts")

    def to_dict(self) -> dict[str, int]:
        """Return deterministic counts without identifiers or source values."""

        return _validated_counters(self)._to_dict_unchecked()

    def _to_dict_unchecked(self) -> dict[str, int]:
        """Return counters after the value has been reconstructed."""

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
    max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES
    max_record_bytes: int = _DEFAULT_MAX_RECORD_BYTES
    records_seen: int = 0
    input_bytes: int = 0
    output_bytes: int = 0

    def __post_init__(self) -> None:
        """Validate state bounds and counters."""

        _require_positive_int(
            self.max_records,
            "max_records",
            maximum=_MAX_RECORDS,
        )
        _require_positive_int(
            self.max_input_bytes,
            "max_input_bytes",
            maximum=_MAX_INPUT_BYTES,
        )
        _require_positive_int(
            self.max_output_bytes,
            "max_output_bytes",
            maximum=_MAX_OUTPUT_BYTES,
        )
        _require_positive_int(
            self.max_record_bytes,
            "max_record_bytes",
            maximum=_MAX_RECORD_BYTES,
        )
        _require_non_negative_int(self.records_seen, "records_seen")
        _require_non_negative_int(self.input_bytes, "input_bytes")
        _require_non_negative_int(self.output_bytes, "output_bytes")
        if self.records_seen > self.max_records:
            raise ValueError("records_seen exceeds max_records")
        if self.input_bytes > self.max_input_bytes:
            raise ValueError("input_bytes exceeds max_input_bytes")
        if self.output_bytes > self.max_output_bytes:
            raise ValueError("output_bytes exceeds max_output_bytes")

    def accept(self, serialized_record: bytes) -> None:
        """Account for one serialized record, enforcing all state bounds."""

        self.__post_init__()
        if type(serialized_record) is not bytes:
            raise TypeError("serialized_record must be bytes")
        if len(serialized_record) > self.max_record_bytes:
            raise BeamRedactionError("record exceeds the configured byte limit")
        if self.records_seen >= self.max_records:
            raise BeamRedactionError("record batch exceeds the configured limit")
        if self.input_bytes + len(serialized_record) > self.max_input_bytes:
            raise BeamRedactionError("record batch exceeds the configured byte limit")
        self.records_seen += 1
        self.input_bytes += len(serialized_record)

    def accept_output(self, serialized_record: bytes) -> None:
        """Account for one output record without retaining its value."""

        if type(serialized_record) is not bytes:
            raise TypeError("serialized output must be bytes")
        if len(serialized_record) > self.max_record_bytes:
            raise BeamRedactionError("output record exceeds the configured byte limit")
        if self.output_bytes + len(serialized_record) > self.max_output_bytes:
            raise BeamRedactionError("record batch exceeds the configured output limit")
        self.output_bytes += len(serialized_record)

    def to_dict(self) -> dict[str, int]:
        """Return only bounded state counters."""

        self.__post_init__()
        return {
            "input_bytes": self.input_bytes,
            "max_input_bytes": self.max_input_bytes,
            "max_output_bytes": self.max_output_bytes,
            "max_record_bytes": self.max_record_bytes,
            "max_records": self.max_records,
            "output_bytes": self.output_bytes,
            "records_seen": self.records_seen,
        }


@dataclass(frozen=True, slots=True)
class BeamRedactionResult:
    """Redacted direct-run output plus a PHI-free aggregate report."""

    redacted_records: tuple[Record, ...]
    counters: BeamRedactionCounters
    input_fingerprint: str
    output_fingerprint: str
    spec_fingerprint: str
    serialized_output: bytes

    def __post_init__(self) -> None:
        """Validate and detach bounded public result state."""

        if type(self.redacted_records) is not tuple:
            raise TypeError("redacted_records must be a tuple")
        if len(self.redacted_records) > _MAX_RECORDS:
            raise ValueError("redacted_records exceed the bounded maximum")
        if type(self.counters) is not BeamRedactionCounters:
            raise TypeError("counters must be BeamRedactionCounters")
        counters = _validated_counters(self.counters)
        if len(self.redacted_records) != counters.records_processed:
            raise ValueError("redacted record count does not match counters")
        _require_digest(self.input_fingerprint, "input_fingerprint")
        _require_digest(self.output_fingerprint, "output_fingerprint")
        _require_digest(self.spec_fingerprint, "spec_fingerprint")
        if type(self.serialized_output) is not bytes:
            raise TypeError("serialized_output must be bytes")
        expected_bytes = counters.output_bytes + counters.records_processed
        if len(self.serialized_output) != expected_bytes:
            raise ValueError("serialized_output size does not match counters")
        if _digest_bytes(self.serialized_output) != self.output_fingerprint:
            raise ValueError("output_fingerprint does not match serialized_output")
        detached_records = tuple(
            _copy_record(record, maximum_text_chars=_MAX_OUTPUT_RECORD_BYTES)
            for record in self.redacted_records
        )
        object.__setattr__(self, "redacted_records", detached_records)
        object.__setattr__(self, "counters", counters)

    def report(self) -> dict[str, Any]:
        """Return a report containing no record values or exception details."""

        return _validated_result(self)._report_unchecked()

    def _report_unchecked(self) -> dict[str, Any]:
        """Return aggregate metadata after reconstructing this result."""

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

        try:
            safe = _validated_result(self)
            return (
                "BeamRedactionResult("
                f"records={len(safe.redacted_records)}, "
                f"counters={safe.counters.to_dict()!r}, "
                f"input_fingerprint={safe.input_fingerprint!r}, "
                f"output_fingerprint={safe.output_fingerprint!r}, "
                f"spec_fingerprint={safe.spec_fingerprint!r})"
            )
        except BaseException:
            return "BeamRedactionResult(<invalid>)"


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
        safe_spec = _validated_spec(spec)
        _require_optional_callable(deidentifier, "deidentifier")
        _require_optional_callable(loader_factory, "loader_factory")
        self._spec = safe_spec
        self._deidentifier = deidentifier
        self._loader_factory = loader_factory
        self._loader: Any = None
        self._state = BeamRedactionState(
            max_records=safe_spec.max_records,
            max_input_bytes=safe_spec.max_input_bytes,
            max_record_bytes=safe_spec.max_record_bytes,
        )
        self._counters = _CounterAccumulator()
        self._metrics: dict[str, Any] = {}

    def setup(self) -> None:
        """Initialize one local loader and value-free Beam metrics."""

        if self._loader is None and self._deidentifier is None:
            try:
                factory = (
                    _new_model_loader
                    if self._loader_factory is None
                    else self._loader_factory
                )
                loaded = factory()
                if loaded is None:
                    raise BeamRedactionError("worker-local model setup failed")
                self._loader = loaded
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException:
                raise BeamRedactionError("worker-local model setup failed") from None
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
            output_bytes = _serialize_output_record(redacted, self._spec)
            _validate_output_budget(
                output_bytes,
                spec=self._spec,
                current_output_bytes=self._counters.output_bytes,
            )
        except BeamRedactionError:
            self._counters.records_failed += 1
            self._inc("records_failed")
            raise
        self._counters.attempts += attempts
        self._counters.retries += retries
        self._counters.spans_redacted += spans
        self._counters.output_bytes += len(output_bytes)
        self._inc("attempts", attempts)
        self._inc("retries", retries)
        self._inc("spans_redacted", spans)
        if output_bytes != serialized:
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
        max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES,
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
            if _direct_transform_options_supplied(
                text_field=text_field,
                policy=policy,
                method=method,
                max_records=max_records,
                max_input_bytes=max_input_bytes,
                max_record_bytes=max_record_bytes,
                max_attempts=max_attempts,
                retry_backoff_seconds=retry_backoff_seconds,
                extra_kwargs=extra_kwargs,
            ):
                raise TypeError("spec cannot be combined with direct options")
            self.spec = _validated_spec(spec)
        else:
            self.spec = BeamRedactionSpec(
                text_field=text_field,
                policy=policy,
                method=method,
                max_records=max_records,
                max_input_bytes=max_input_bytes,
                max_output_bytes=max_output_bytes,
                max_record_bytes=max_record_bytes,
                max_attempts=max_attempts,
                retry_backoff_seconds=retry_backoff_seconds,
                extra_kwargs={} if extra_kwargs is None else extra_kwargs,
            )
        _require_optional_callable(deidentifier, "deidentifier")
        _require_optional_callable(loader_factory, "loader_factory")
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

    _require_optional_callable(deidentifier, "deidentifier")
    _require_optional_callable(loader_factory, "loader_factory")
    resolved_spec = BeamRedactionSpec() if spec is None else _validated_spec(spec)
    state = BeamRedactionState(
        max_records=resolved_spec.max_records,
        max_input_bytes=resolved_spec.max_input_bytes,
        max_output_bytes=resolved_spec.max_output_bytes,
        max_record_bytes=resolved_spec.max_record_bytes,
    )
    counters = _CounterAccumulator()
    input_serialized: list[bytes] = []
    output_serialized: list[bytes] = []
    redacted_records: list[Record] = []
    loader: Any = None

    for record in _iter_bounded_records(records, resolved_spec.max_records):
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
            try:
                factory = (
                    _new_model_loader if loader_factory is None else loader_factory
                )
                loader = factory()
                if loader is None:
                    raise BeamRedactionError("worker-local model setup failed")
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException:
                raise BeamRedactionError("worker-local model setup failed") from None

        redacted, spans, attempts, retries = _redact_with_retries(
            normalized,
            spec=resolved_spec,
            deidentifier=deidentifier,
            loader=loader,
        )
        output = _serialize_output_record(redacted, resolved_spec)
        _validate_output_budget(
            output,
            spec=resolved_spec,
            current_output_bytes=counters.output_bytes,
        )
        redacted_records.append(redacted)
        output_serialized.append(output)
        counters.attempts += attempts
        counters.retries += retries
        counters.spans_redacted += spans
        counters.output_bytes += len(output)
        counters.records_changed += int(output != serialized)

    serialized_input = _serialize_lines(
        input_serialized,
        max_records=resolved_spec.max_records,
        max_bytes=resolved_spec.max_input_bytes + resolved_spec.max_records,
    )
    serialized_output = _serialize_lines(
        output_serialized,
        max_records=resolved_spec.max_records,
        max_bytes=resolved_spec.max_output_bytes + resolved_spec.max_records,
    )
    return BeamRedactionResult(
        redacted_records=tuple(redacted_records),
        counters=counters.freeze(),
        input_fingerprint=_digest_bytes(serialized_input),
        output_fingerprint=_digest_bytes(serialized_output),
        spec_fingerprint=resolved_spec.fingerprint(),
        serialized_output=serialized_output,
    )


def serialize_record(record: Record) -> bytes:
    """Return one bounded record in deterministic canonical JSON bytes."""

    normalized = _copy_record(record, maximum_text_chars=_DEFAULT_MAX_RECORD_BYTES)
    serialized = _serialize_json_safely(normalized)
    if len(serialized) > _DEFAULT_MAX_RECORD_BYTES:
        raise BeamRedactionError("record exceeds the default byte limit")
    return serialized


def serialize_records(records: Iterable[Record]) -> bytes:
    """Return a bounded deterministic newline-delimited serialization."""

    serialized: list[bytes] = []
    total_bytes = 0
    for record in _iter_bounded_records(records, _DEFAULT_MAX_RECORDS):
        value = serialize_record(record)
        total_bytes += len(value) + 1
        if total_bytes > _DEFAULT_MAX_INPUT_BYTES:
            raise BeamRedactionError("record batch exceeds the default byte limit")
        serialized.append(value)
    return _serialize_lines(
        serialized,
        max_records=_DEFAULT_MAX_RECORDS,
        max_bytes=_DEFAULT_MAX_INPUT_BYTES,
    )


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

    try:
        kwargs = spec.to_deidentify_kwargs()
        if deidentifier is None:
            deidentifier = _default_deidentifier
            kwargs["config"] = _offline_config()
            kwargs["keep_mapping"] = False
            kwargs["audit"] = False
            kwargs["use_safety_sweep"] = True
        if loader is not None:
            kwargs["loader"] = loader
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise BeamRedactionError("redaction setup failed") from None

    attempts = 0
    retries = 0
    while attempts < spec.max_attempts:
        attempts += 1
        try:
            result = deidentifier(text, **kwargs)
            redacted, spans = _result_text_and_spans(
                result,
                text,
                maximum_output_chars=min(
                    spec.max_record_bytes * _MAX_OUTPUT_EXPANSION,
                    _MAX_OUTPUT_RECORD_BYTES,
                ),
            )
            break
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            if attempts >= spec.max_attempts:
                fingerprint = _digest_bytes(
                    text.encode("utf-8", errors="surrogatepass")
                )
                raise BeamRedactionError(
                    "redaction failed after the configured attempts; "
                    f"record_fingerprint={fingerprint}"
                ) from None
            retries += 1
            if spec.retry_backoff_seconds:
                time.sleep(float(spec.retry_backoff_seconds))
    else:  # pragma: no cover - loop is bounded by max_attempts
        raise BeamRedactionError("redaction attempts were exhausted") from None

    if type(record) is str:
        return redacted, spans, attempts, retries
    output = dict(cast(Mapping[str, Any], record))
    output[spec.text_field] = redacted
    return output, spans, attempts, retries


def _validate_and_serialize_record(
    record: Record,
    spec: BeamRedactionSpec,
) -> tuple[Record, bytes]:
    normalized = _copy_record(
        record,
        maximum_text_chars=spec.max_record_bytes,
    )
    if type(normalized) is dict:
        if spec.text_field not in normalized:
            raise BeamRedactionError("record is missing the configured text field")
        value = normalized[spec.text_field]
        if value is not None and type(value) is not str:
            raise BeamRedactionError(
                "configured record field must contain text or null"
            )
    serialized = _serialize_json_safely(normalized)
    if len(serialized) > spec.max_record_bytes:
        raise BeamRedactionError("record exceeds the configured byte limit")
    return normalized, serialized


def _record_text(record: Record, text_field: str) -> str | None:
    if type(record) is str:
        return record
    value = cast(Mapping[str, Any], record)[text_field]
    return value


def _result_text_and_spans(
    result: Any,
    original: str,
    *,
    maximum_output_chars: int,
) -> tuple[str, int]:
    redacted: Any
    entities: Any
    if type(result) is str:
        redacted = result
        entities = None
    elif isinstance(result, Mapping):
        redacted = result.get("deidentified_text")
        entities = _result_entities(result)
    else:
        redacted = getattr(result, "deidentified_text", None)
        entities = _result_entities(result)
    if type(redacted) is not str:
        raise TypeError("deidentifier must return text or deidentified_text")
    allowed_chars = min(
        maximum_output_chars,
        max(_MIN_OUTPUT_CHARS, len(original) * _MAX_OUTPUT_EXPANSION),
    )
    if len(redacted) > allowed_chars:
        raise ValueError("deidentifier output exceeds the bounded size")
    if redacted == original:
        return redacted, 0
    if isinstance(entities, (list, tuple)) and type(entities) in (list, tuple):
        spans = min(len(entities), _MAX_SPANS_PER_RECORD)
    else:
        spans = 0
    return redacted, spans or 1


def _result_entities(result: Any) -> Any:
    """Return optional entity metadata without trusting fallback accessors."""

    try:
        if isinstance(result, Mapping):
            entities = result.get("pii_entities", _MISSING)
            return result.get("entities") if entities is _MISSING else entities
        entities = getattr(result, "pii_entities", _MISSING)
        return getattr(result, "entities", None) if entities is _MISSING else entities
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        return None


def _iter_bounded_records(
    records: Iterable[Record],
    maximum_records: int,
) -> Iterator[Record]:
    """Yield a bounded record source while sanitizing iterator failures."""

    try:
        iterator = iter(records)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise BeamRedactionError("record source could not be read") from None
    for index in range(maximum_records + 1):
        try:
            record = next(iterator)
        except StopIteration:
            return
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise BeamRedactionError("record source could not be read") from None
        if index == maximum_records:
            raise BeamRedactionError("record batch exceeds the configured limit")
        yield record


def _bounded_tuple(value: Any, *, label: str, maximum: int) -> tuple[Any, ...]:
    """Snapshot at most ``maximum`` items with value-free boundary errors."""

    try:
        iterator = iter(value)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise BeamRedactionError(f"{label} must be a bounded iterable") from None
    collected: list[Any] = []
    for index in range(maximum + 1):
        try:
            item = next(iterator)
        except StopIteration:
            return tuple(collected)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise BeamRedactionError(f"{label} iteration failed") from None
        if index == maximum:
            raise BeamRedactionError(f"{label} exceed the configured limit")
        collected.append(item)
    raise AssertionError("unreachable")


def _copy_record(record: Any, *, maximum_text_chars: int) -> Record:
    """Return a bounded detached copy of one JSON-compatible record."""

    try:
        copied = _copy_record_value(
            record,
            maximum_text_chars=maximum_text_chars,
            depth=0,
            item_count=[0],
            active_containers=set(),
        )
        if type(copied) not in (str, dict):
            raise _BoundaryError
        return copied
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise BeamRedactionError("record could not be inspected") from None


def _copy_record_value(
    value: Any,
    *,
    maximum_text_chars: int,
    depth: int,
    item_count: list[int],
    active_containers: set[int],
) -> Any:
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        if value.bit_length() > _MAX_RECORD_INT_BITS:
            raise _BoundaryError
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise _BoundaryError
        return value
    if type(value) is str:
        if len(value) > maximum_text_chars:
            raise _BoundaryError
        return value
    if depth > _MAX_RECORD_DEPTH:
        raise _BoundaryError

    if isinstance(value, Mapping):
        marker = id(value)
        if marker in active_containers:
            raise _BoundaryError
        active_containers.add(marker)
        try:
            try:
                iterator = iter(value.items())
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as error:
                raise _BoundaryError from error
            copied_mapping: dict[str, Any] = {}
            while True:
                try:
                    entry = next(iterator)
                except StopIteration:
                    return copied_mapping
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException as error:
                    raise _BoundaryError from error
                if item_count[0] >= _MAX_RECORD_ITEMS:
                    raise _BoundaryError
                item_count[0] += 1
                try:
                    key, item = entry
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException as error:
                    raise _BoundaryError from error
                if (
                    type(key) is not str
                    or len(key) > _MAX_RECORD_KEY_CHARS
                    or key in copied_mapping
                ):
                    raise _BoundaryError
                copied_mapping[key] = _copy_record_value(
                    item,
                    maximum_text_chars=maximum_text_chars,
                    depth=depth + 1,
                    item_count=item_count,
                    active_containers=active_containers,
                )
        finally:
            active_containers.discard(marker)

    if type(value) in (list, tuple):
        marker = id(value)
        if marker in active_containers:
            raise _BoundaryError
        if len(value) > _MAX_RECORD_ITEMS - item_count[0]:
            raise _BoundaryError
        active_containers.add(marker)
        try:
            copied_values = []
            for item in value:
                item_count[0] += 1
                copied_values.append(
                    _copy_record_value(
                        item,
                        maximum_text_chars=maximum_text_chars,
                        depth=depth + 1,
                        item_count=item_count,
                        active_containers=active_containers,
                    )
                )
        finally:
            active_containers.discard(marker)
        return tuple(copied_values) if type(value) is tuple else copied_values

    raise _BoundaryError


def _serialize_json_safely(record: Record) -> bytes:
    try:
        return _canonical_json(record)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise BeamRedactionError("record is not JSON serializable") from None


def _serialize_output_record(record: Record, spec: BeamRedactionSpec) -> bytes:
    normalized = _copy_record(
        record,
        maximum_text_chars=min(
            spec.max_record_bytes * _MAX_OUTPUT_EXPANSION,
            _MAX_OUTPUT_RECORD_BYTES,
        ),
    )
    return _serialize_json_safely(normalized)


def _validate_output_budget(
    serialized_record: bytes,
    *,
    spec: BeamRedactionSpec,
    current_output_bytes: int,
) -> None:
    maximum_record_bytes = min(
        spec.max_record_bytes * _MAX_OUTPUT_EXPANSION,
        _MAX_OUTPUT_RECORD_BYTES,
    )
    maximum_total_bytes = min(
        spec.max_output_bytes,
        spec.max_input_bytes * _MAX_OUTPUT_EXPANSION,
        _MAX_OUTPUT_BYTES,
    )
    if len(serialized_record) > maximum_record_bytes:
        raise BeamRedactionError("redacted record exceeds the output byte limit")
    if current_output_bytes + len(serialized_record) > maximum_total_bytes:
        raise BeamRedactionError("redacted batch exceeds the output byte limit")


def _serialize_lines(
    records: Iterable[bytes],
    *,
    max_records: int,
    max_bytes: int,
) -> bytes:
    values = _bounded_tuple(records, label="serialized records", maximum=max_records)
    if any(type(value) is not bytes for value in values):
        raise TypeError("serialized records must contain bytes")
    output_size = sum(len(value) + 1 for value in values)
    if output_size > max_bytes:
        raise BeamRedactionError("serialized records exceed the configured byte limit")
    return b"\n".join(values) + (b"\n" if values else b"")


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except Exception:
        raise BeamRedactionError("value is not safely JSON serializable") from None


def _digest_bytes(value: bytes) -> str:
    if type(value) is not bytes:
        raise TypeError("digest input must be bytes")
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _fingerprint_value(value: Any) -> Any:
    if value is None or type(value) in (bool, int, float):
        return value
    if type(value) is str:
        return {"type": "str", "sha256": _digest_bytes(value.encode("utf-8"))}
    if type(value) is bytes:
        return {"type": "bytes", "sha256": _digest_bytes(value)}
    if isinstance(value, _FrozenOptions):
        return {key: _fingerprint_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (_FrozenList, tuple)):
        return [_fingerprint_value(item) for item in value]
    raise TypeError("unsupported fingerprint value")


def _snapshot_extra_kwargs(options: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a bounded detached snapshot of worker options."""

    try:
        iterator = iter(options.items())
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise ValueError("extra_kwargs could not be inspected") from None

    snapshot: dict[str, Any] = {}
    item_count = [0]
    active_containers: set[int] = set()
    for index in range(_MAX_EXTRA_KWARGS + 1):
        try:
            entry = next(iterator)
        except StopIteration:
            return _FrozenOptions(tuple(snapshot.items()))
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise ValueError("extra_kwargs could not be inspected") from None
        if index == _MAX_EXTRA_KWARGS:
            raise ValueError("extra_kwargs contain too many entries")
        try:
            key, value = entry
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise ValueError("extra_kwargs could not be inspected") from None
        if (
            type(key) is not str
            or not key
            or len(key) > _MAX_EXTRA_KEY_CHARS
            or not key.isprintable()
        ):
            raise ValueError("extra_kwargs must use bounded string keys")
        if key in _RESERVED_EXTRA_KEYS:
            raise ValueError("extra_kwargs cannot override reserved worker options")
        if key in snapshot:
            raise ValueError("extra_kwargs must not contain duplicate keys")
        try:
            snapshot[key] = _snapshot_extra_value(
                value,
                depth=0,
                item_count=item_count,
                active_containers=active_containers,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise ValueError(
                "extra_kwargs contain unsupported or unbounded values"
            ) from None
    raise AssertionError("unreachable")


def _snapshot_extra_value(
    value: Any,
    *,
    depth: int,
    item_count: list[int],
    active_containers: set[int],
) -> Any:
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        if value.bit_length() > _MAX_EXTRA_INT_BITS:
            raise _BoundaryError
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise _BoundaryError
        return value
    if type(value) is str:
        if len(value) > _MAX_EXTRA_STRING_CHARS:
            raise _BoundaryError
        return value
    if type(value) is bytes:
        if len(value) > _MAX_EXTRA_STRING_CHARS:
            raise _BoundaryError
        return value
    if depth > _MAX_EXTRA_DEPTH:
        raise _BoundaryError

    if isinstance(value, Mapping):
        marker = id(value)
        if marker in active_containers:
            raise _BoundaryError
        active_containers.add(marker)
        try:
            iterator = iter(value.items())
            copied: dict[str, Any] = {}
            while True:
                try:
                    entry = next(iterator)
                except StopIteration:
                    return _FrozenOptions(tuple(copied.items()))
                if item_count[0] >= _MAX_EXTRA_ITEMS:
                    raise _BoundaryError
                item_count[0] += 1
                key, item = entry
                if (
                    type(key) is not str
                    or not key
                    or len(key) > _MAX_EXTRA_KEY_CHARS
                    or not key.isprintable()
                    or key in copied
                ):
                    raise _BoundaryError
                copied[key] = _snapshot_extra_value(
                    item,
                    depth=depth + 1,
                    item_count=item_count,
                    active_containers=active_containers,
                )
        finally:
            active_containers.discard(marker)

    if type(value) in (list, tuple):
        marker = id(value)
        if marker in active_containers:
            raise _BoundaryError
        if len(value) > _MAX_EXTRA_ITEMS - item_count[0]:
            raise _BoundaryError
        active_containers.add(marker)
        try:
            copied_values = []
            for item in value:
                item_count[0] += 1
                copied_values.append(
                    _snapshot_extra_value(
                        item,
                        depth=depth + 1,
                        item_count=item_count,
                        active_containers=active_containers,
                    )
                )
        finally:
            active_containers.discard(marker)
        return (
            tuple(copied_values) if type(value) is tuple else _FrozenList(copied_values)
        )

    raise _BoundaryError


def _clone_extra_value(value: Any) -> Any:
    if isinstance(value, _FrozenOptions):
        return {key: _clone_extra_value(item) for key, item in value.items()}
    if isinstance(value, _FrozenList):
        return [_clone_extra_value(item) for item in value]
    if type(value) is tuple:
        return tuple(_clone_extra_value(item) for item in value)
    return value


def _validated_spec(value: Any) -> BeamRedactionSpec:
    """Reconstruct a specification so post-init checks cannot be bypassed."""

    if type(value) is not BeamRedactionSpec:
        raise TypeError("spec must be a BeamRedactionSpec")
    try:
        return BeamRedactionSpec(
            text_field=value.text_field,
            policy=value.policy,
            method=value.method,
            max_records=value.max_records,
            max_input_bytes=value.max_input_bytes,
            max_output_bytes=value.max_output_bytes,
            max_record_bytes=value.max_record_bytes,
            max_attempts=value.max_attempts,
            retry_backoff_seconds=value.retry_backoff_seconds,
            extra_kwargs={
                key: _clone_extra_value(item)
                for key, item in value.extra_kwargs.items()
            },
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except (BeamRedactionError, TypeError, ValueError):
        raise
    except BaseException:
        raise BeamRedactionError(
            "redaction specification could not be read safely"
        ) from None


def _validated_counters(value: Any) -> BeamRedactionCounters:
    """Reconstruct aggregate counters before they reach a public report."""

    if type(value) is not BeamRedactionCounters:
        raise TypeError("counters must be BeamRedactionCounters")
    try:
        return BeamRedactionCounters(
            records_processed=value.records_processed,
            records_changed=value.records_changed,
            records_failed=value.records_failed,
            attempts=value.attempts,
            retries=value.retries,
            spans_redacted=value.spans_redacted,
            input_bytes=value.input_bytes,
            output_bytes=value.output_bytes,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except (TypeError, ValueError):
        raise
    except BaseException:
        raise BeamRedactionError(
            "redaction counters could not be read safely"
        ) from None


def _validated_result(value: Any) -> BeamRedactionResult:
    """Reconstruct a result before rendering its public metadata."""

    if type(value) is not BeamRedactionResult:
        raise TypeError("result must be a BeamRedactionResult")
    try:
        return BeamRedactionResult(
            redacted_records=value.redacted_records,
            counters=value.counters,
            input_fingerprint=value.input_fingerprint,
            output_fingerprint=value.output_fingerprint,
            spec_fingerprint=value.spec_fingerprint,
            serialized_output=value.serialized_output,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except (BeamRedactionError, TypeError, ValueError):
        raise
    except BaseException:
        raise BeamRedactionError("redaction result could not be read safely") from None


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

    return OpenMedConfig(local_only=True, hf_token="")


def _normalize_config_text(value: Any, name: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be bounded text")
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > _MAX_CONFIG_TEXT_CHARS
        or not normalized.isprintable()
    ):
        raise ValueError(f"{name} must be bounded text")
    return normalized


def _normalize_text_field(value: Any) -> str:
    normalized = _normalize_config_text(value, "text_field")
    if (
        _SAFE_METADATA_RE.fullmatch(normalized) is None
        or _METADATA_IDENTIFIER_RE.search(normalized) is not None
    ):
        raise ValueError("text_field must be a safe field identifier")
    return normalized


def _normalize_method(value: Any) -> str:
    normalized = _normalize_config_text(value, "method").lower()
    if normalized not in _DEIDENTIFICATION_METHODS:
        raise ValueError("method is not supported")
    return normalized


def _normalize_policy(policy: Any) -> str:
    normalized = _normalize_config_text(policy, "policy")
    try:
        canonical = canonical_policy_name(normalized)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise ValueError("policy is invalid") from None
    if type(canonical) is not str or len(canonical) > _MAX_CONFIG_TEXT_CHARS:
        raise ValueError("policy is invalid")
    return canonical


def _require_optional_callable(value: Any, name: str) -> None:
    if value is not None and not callable(value):
        raise TypeError(f"{name} must be callable")


def _direct_transform_options_supplied(
    *,
    text_field: Any,
    policy: Any,
    method: Any,
    max_records: Any,
    max_input_bytes: Any,
    max_record_bytes: Any,
    max_attempts: Any,
    retry_backoff_seconds: Any,
    extra_kwargs: Any,
) -> bool:
    return (
        type(text_field) is not str
        or text_field != _DEFAULT_TEXT_FIELD
        or type(policy) is not str
        or policy != _DEFAULT_POLICY
        or type(method) is not str
        or method != _DEFAULT_METHOD
        or type(max_records) is not int
        or max_records != _DEFAULT_MAX_RECORDS
        or type(max_input_bytes) is not int
        or max_input_bytes != _DEFAULT_MAX_INPUT_BYTES
        or type(max_record_bytes) is not int
        or max_record_bytes != _DEFAULT_MAX_RECORD_BYTES
        or type(max_attempts) is not int
        or max_attempts != _DEFAULT_MAX_ATTEMPTS
        or type(retry_backoff_seconds) not in (int, float)
        or float(retry_backoff_seconds) != 0.0
        or extra_kwargs is not None
    )


def _require_digest(value: Any, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{name} must be a SHA-256 digest")


def _require_positive_int(
    value: Any,
    name: str,
    *,
    maximum: int | None = None,
) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} exceeds the bounded maximum")


def _require_non_negative_int(
    value: Any,
    name: str,
    *,
    maximum: int | None = None,
) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} exceeds the bounded maximum")


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
