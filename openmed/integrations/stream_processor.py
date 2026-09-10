"""Framework-neutral stream processor helpers for record de-identification.

The lifecycle methods intentionally mirror common stateful stream runtimes:
``open`` initializes one resident processing callable per map-function instance,
``map`` handles a single record, and ``invoke`` writes one result to a sink.
The helpers do not depend on a specific cluster runtime, so applications can
adapt them to their stream processor without pulling that runtime into OpenMed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Protocol

from openmed.core.policy import PolicyName, canonical_policy_name

DEFAULT_STREAM_BATCH_SIZE = 32
DEFAULT_STREAM_POLICY = "hipaa_safe_harbor"

ProcessBatchCallable = Callable[..., Any]
Record = Mapping[str, Any]


class RecordWriter(Protocol):
    """Object that accepts one de-identified record."""

    def write(self, record: Record) -> Any:
        """Write ``record`` to the downstream system."""


class StreamDeidentifyMapFunction:
    """De-identify one named field while preserving the rest of each record.

    One instance is intended to live in one parallel stream subtask. The first
    call to :meth:`open`, :meth:`map`, or :meth:`map_batch` resolves and retains
    the batch processor for that instance. Records are copied before their
    configured text field is replaced, keeping the transformation stateless and
    a pure function of that field.
    """

    def __init__(
        self,
        text_field: str = "text",
        *,
        policy: str | PolicyName = DEFAULT_STREAM_POLICY,
        method: str = "mask",
        model_name: str = "disease_detection_superclinical",
        batch_size: int = DEFAULT_STREAM_BATCH_SIZE,
        process_batch_fn: ProcessBatchCallable | None = None,
        **deidentify_kwargs: Any,
    ) -> None:
        """Create a record-level stream map function.

        Args:
            text_field: Record field containing free text to de-identify.
            policy: OpenMed de-identification policy profile.
            method: De-identification method forwarded to ``process_batch``.
            model_name: Model registry key or artifact identifier.
            batch_size: Maximum records passed to one ``process_batch`` call.
            process_batch_fn: Optional test or runtime-specific replacement for
                :func:`openmed.processing.process_batch`.
            **deidentify_kwargs: Additional de-identification options forwarded
                to ``process_batch``.
        """

        self.text_field = _non_empty_string(text_field, "text_field")
        self.policy = canonical_policy_name(policy)
        self.method = _non_empty_string(method, "method")
        self.model_name = _non_empty_string(model_name, "model_name")
        self.batch_size = _positive_int(batch_size, "batch_size")
        if process_batch_fn is not None and not callable(process_batch_fn):
            raise TypeError("process_batch_fn must be callable")

        kwargs = dict(deidentify_kwargs)
        for reserved in (
            "batch_size",
            "ids",
            "method",
            "model_name",
            "operation",
            "policy",
            "texts",
        ):
            if reserved in kwargs:
                raise ValueError(
                    f"deidentify kwargs must not include reserved key {reserved!r}"
                )

        self.deidentify_kwargs = kwargs
        self._configured_process_batch = process_batch_fn
        self._process_batch: ProcessBatchCallable | None = None
        self._is_open = False

    @property
    def is_open(self) -> bool:
        """Whether this subtask instance has initialized its batch processor."""

        return self._is_open

    def open(self, runtime_context: Any | None = None) -> None:
        """Initialize this subtask once and retain its batch processor.

        Args:
            runtime_context: Optional runtime-owned context. It is accepted for
                compatibility with stream processors and is not inspected.
        """

        del runtime_context
        if self._is_open:
            return
        if self._configured_process_batch is None:
            from openmed.processing import process_batch

            self._process_batch = process_batch
        else:
            self._process_batch = self._configured_process_batch
        self._is_open = True

    def map(self, record: Record) -> dict[str, Any]:
        """Return one copied record with its configured field de-identified."""

        return self.map_batch((record,))[0]

    def map_batch(self, records: Iterable[Record]) -> list[dict[str, Any]]:
        """De-identify records in bounded micro-batches.

        Args:
            records: Input record mappings. Missing target fields and non-string
                target values fail closed; ``None`` and empty strings pass
                through unchanged without invoking the model.

        Returns:
            Copied records in input order with only ``text_field`` replaced.
        """

        if not self._is_open:
            self.open()

        output: list[dict[str, Any]] = []
        pending_texts: list[str] = []
        pending_positions: list[int] = []

        for record in records:
            if not isinstance(record, Mapping):
                raise TypeError("stream records must be mappings")
            if self.text_field not in record:
                raise KeyError(f"record is missing text field {self.text_field!r}")

            copied = dict(record)
            value = copied[self.text_field]
            if value is not None and not isinstance(value, str):
                raise TypeError(
                    f"record field {self.text_field!r} must be a string or None"
                )
            output.append(copied)
            if value:
                pending_texts.append(value)
                pending_positions.append(len(output) - 1)

            if len(pending_texts) == self.batch_size:
                self._apply_batch(output, pending_positions, pending_texts)
                pending_texts = []
                pending_positions = []

        if pending_texts:
            self._apply_batch(output, pending_positions, pending_texts)
        return output

    def _apply_batch(
        self,
        output: list[dict[str, Any]],
        positions: Sequence[int],
        texts: Sequence[str],
    ) -> None:
        process_batch_fn = self._process_batch
        if process_batch_fn is None:  # pragma: no cover - defensive invariant
            raise RuntimeError("stream map function is not open")

        kwargs = {
            "operation": "deidentify",
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "continue_on_error": False,
            "method": self.method,
            "policy": self.policy,
            **self.deidentify_kwargs,
        }
        batch_result = process_batch_fn(list(texts), **kwargs)
        redacted = _deidentified_texts(batch_result, expected=len(texts))
        for position, text in zip(positions, redacted):
            output[position][self.text_field] = text


class StreamSink:
    """Thin sink adapter around a callable or object with ``write(record)``."""

    def __init__(self, writer: Callable[[Record], Any] | RecordWriter) -> None:
        """Create a sink for a downstream record writer.

        Args:
            writer: Callable receiving one record, or an object exposing
                ``write(record)``. If the object exposes ``flush()``, it is
                called when :meth:`flush` is invoked.
        """

        write_method = getattr(writer, "write", None)
        if not callable(writer) and not callable(write_method):
            raise TypeError("writer must be callable or expose write(record)")
        self._writer = writer

    def invoke(self, record: Record, context: Any | None = None) -> Any:
        """Write one record using a stream-runtime-compatible sink shape."""

        del context
        write_method = getattr(self._writer, "write", None)
        if callable(write_method):
            return write_method(record)
        return self._writer(record)  # type: ignore[operator]

    def write_batch(self, records: Iterable[Record]) -> int:
        """Write all records and return the number accepted by the sink."""

        written = 0
        for record in records:
            self.invoke(record)
            written += 1
        return written

    def flush(self) -> Any:
        """Flush the wrapped writer when it provides a ``flush`` method."""

        flush_method = getattr(self._writer, "flush", None)
        if callable(flush_method):
            return flush_method()
        return None


DeidentifyMapFunction = StreamDeidentifyMapFunction


def run_stream_job(
    source: Iterable[Record],
    map_function: StreamDeidentifyMapFunction,
    sink: StreamSink | Callable[[Record], Any] | RecordWriter,
) -> int:
    """Run an offline source -> map -> sink flow with bounded micro-batches.

    This helper is suitable for local harnesses and examples. Cluster runtimes
    can wire the same map and sink lifecycle methods into their native graph.

    Args:
        source: Iterable of input record mappings.
        map_function: Configured record de-identification map function.
        sink: A :class:`StreamSink`, callable, or object with ``write(record)``.

    Returns:
        Number of records written to the sink.
    """

    resolved_sink = sink if isinstance(sink, StreamSink) else StreamSink(sink)
    if not map_function.is_open:
        map_function.open()

    written = 0
    batch: list[Record] = []
    for record in source:
        batch.append(record)
        if len(batch) == map_function.batch_size:
            written += resolved_sink.write_batch(map_function.map_batch(batch))
            batch = []
    if batch:
        written += resolved_sink.write_batch(map_function.map_batch(batch))
    resolved_sink.flush()
    return written


def _deidentified_texts(batch_result: Any, *, expected: int) -> list[str]:
    raw_items = getattr(batch_result, "items", batch_result)
    try:
        items = list(raw_items)
    except TypeError as exc:
        raise TypeError("process_batch must return an iterable of results") from exc
    if len(items) != expected:
        raise ValueError(
            f"process_batch returned {len(items)} results for {expected} inputs"
        )
    return [_deidentified_text(item) for item in items]


def _deidentified_text(item: Any) -> str:
    if getattr(item, "success", True) is False:
        error = getattr(item, "error", None) or "batch de-identification failed"
        raise RuntimeError(str(error))

    result = getattr(item, "result", item)
    if isinstance(result, str):
        return result
    if isinstance(result, Mapping) and "deidentified_text" in result:
        return str(result["deidentified_text"])
    text = getattr(result, "deidentified_text", None)
    if text is not None:
        return str(text)
    raise TypeError("process_batch results must expose deidentified_text")


def _non_empty_string(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if integer < 1 or integer != value:
        raise ValueError(f"{name} must be a positive integer")
    return integer


__all__ = [
    "DEFAULT_STREAM_BATCH_SIZE",
    "DEFAULT_STREAM_POLICY",
    "DeidentifyMapFunction",
    "ProcessBatchCallable",
    "RecordWriter",
    "StreamDeidentifyMapFunction",
    "StreamSink",
    "run_stream_job",
]
