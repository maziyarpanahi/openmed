"""Arrow Flight service for streaming record-batch de-identification.

The service implements ``DoExchange`` so clients can send Arrow record batches
and receive one redacted batch for every input batch. Only the configured text
column is materialized for de-identification; the service never materializes
the complete input stream.

Request configuration is carried in a versioned JSON command descriptor. Raw
cell values are intentionally never logged or included in service errors.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

try:
    import pyarrow as pa
    import pyarrow.flight as flight
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "Arrow Flight de-identification requires pyarrow. Install "
        "openmed[columnar] or install pyarrow directly."
    ) from exc

from openmed.processing.batch import process_batch

DEFAULT_ARROW_FLIGHT_MODEL = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"
ARROW_FLIGHT_DESCRIPTOR_VERSION = 1

ProcessBatchCallable = Callable[..., Any]

_PROTECTED_PROCESS_BATCH_OPTIONS = {
    "batch_size",
    "ids",
    "model_name",
    "operation",
    "policy",
    "texts",
}


def make_deidentify_descriptor(
    text_column: str | None = None,
    *,
    policy: str | None = None,
) -> flight.FlightDescriptor:
    """Build a command descriptor for a de-identification exchange.

    Args:
        text_column: Optional string column to redact in every incoming record
            batch. It may be omitted when the server defines a default.
        policy: Optional OpenMed de-identification policy profile.

    Returns:
        A Flight command descriptor containing versioned JSON metadata.
    """
    payload: dict[str, Any] = {"version": ARROW_FLIGHT_DESCRIPTOR_VERSION}
    if text_column is not None:
        payload["text_column"] = _normalize_required_string(text_column, "text_column")
    if policy is not None:
        payload["policy"] = _normalize_required_string(policy, "policy")
    command = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return flight.FlightDescriptor.for_command(command)


class ArrowFlightDeidentificationServer(flight.FlightServerBase):
    """Stream Arrow record batches through OpenMed de-identification.

    Args:
        location: Flight listen location. Port ``0`` requests a free local port.
        text_column: Optional default text column when a descriptor omits it.
        policy: Optional default de-identification policy profile.
        model_name: PII model forwarded to :func:`process_batch`.
        batch_size: Internal OpenMed processing batch size for each Arrow batch.
        process_batch_options: Additional de-identification options, such as
            ``method``, ``lang``, or ``use_safety_sweep``.
        process_batch_fn: Test or embedding seam. Defaults to OpenMed's
            :func:`process_batch`.
        auth_handler: Optional Arrow Flight authentication handler.
        tls_certificates: Optional Flight TLS certificate/key pairs.
        verify_client: Whether Flight should require client certificates.
        root_certificates: Optional roots used to verify client certificates.
        middleware: Optional Arrow Flight server middleware mapping.

    The authentication and TLS arguments are configuration hooks only. This
    integration does not define an authentication policy or certificate
    lifecycle.
    """

    def __init__(
        self,
        location: str | tuple[str, int] | flight.Location = "grpc://127.0.0.1:0",
        *,
        text_column: str | None = None,
        policy: str | None = None,
        model_name: str = DEFAULT_ARROW_FLIGHT_MODEL,
        batch_size: int = 512,
        process_batch_options: Mapping[str, Any] | None = None,
        process_batch_fn: ProcessBatchCallable | None = None,
        auth_handler: Any = None,
        tls_certificates: Sequence[tuple[bytes, bytes]] | None = None,
        verify_client: bool = False,
        root_certificates: bytes | None = None,
        middleware: Mapping[str, Any] | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._text_column = (
            _normalize_required_string(text_column, "text_column")
            if text_column is not None
            else None
        )
        self._policy = (
            _normalize_required_string(policy, "policy") if policy is not None else None
        )
        self._model_name = _normalize_required_string(model_name, "model_name")
        self._batch_size = batch_size
        self._process_batch_options = dict(process_batch_options or {})
        protected = sorted(
            _PROTECTED_PROCESS_BATCH_OPTIONS.intersection(self._process_batch_options)
        )
        if protected:
            joined = ", ".join(protected)
            raise ValueError(
                "process_batch_options cannot override managed option(s): " + joined
            )
        self._process_batch = process_batch_fn or process_batch
        super().__init__(
            location,
            auth_handler=auth_handler,
            tls_certificates=tls_certificates,
            verify_client=verify_client,
            root_certificates=root_certificates,
            middleware=dict(middleware) if middleware is not None else None,
        )

    def do_exchange(
        self,
        context: flight.ServerCallContext,
        descriptor: flight.FlightDescriptor,
        reader: flight.MetadataRecordBatchReader,
        writer: flight.MetadataRecordBatchWriter,
    ) -> None:
        """Redact each incoming record batch and stream it back immediately."""
        request = _parse_descriptor(descriptor)
        text_column = request.get("text_column", self._text_column)
        if text_column is None:
            raise ValueError(
                "A text_column is required in the Flight descriptor or server config"
            )
        policy = request["policy"] if "policy" in request else self._policy

        schema = reader.schema
        column_index = _validate_text_column(schema, text_column)
        writer.begin(schema)

        for chunk in reader:
            if context.is_cancelled():
                break
            record_batch = chunk.data
            if record_batch is None:
                continue
            redacted = self._redact_record_batch(
                record_batch,
                column_index=column_index,
                text_column=text_column,
                policy=policy,
            )
            if chunk.app_metadata is None:
                writer.write_batch(redacted)
            else:
                writer.write_with_metadata(redacted, chunk.app_metadata)

    def _redact_record_batch(
        self,
        record_batch: pa.RecordBatch,
        *,
        column_index: int,
        text_column: str,
        policy: str | None,
    ) -> pa.RecordBatch:
        field = record_batch.schema.field(column_index)
        values = record_batch.column(column_index).to_pylist()
        row_indexes = [index for index, value in enumerate(values) if value is not None]
        if not row_indexes:
            return record_batch

        texts = [str(values[index]) for index in row_indexes]
        ids = [f"flight_row_{index}" for index in row_indexes]
        options = dict(self._process_batch_options)
        options.update(
            {
                "operation": "deidentify",
                "batch_size": self._batch_size,
                "policy": policy,
            }
        )
        try:
            result = self._process_batch(
                texts,
                model_name=self._model_name,
                ids=ids,
                **options,
            )
        except Exception:
            raise RuntimeError(
                f"De-identification failed for column {text_column!r}"
            ) from None
        items = list(getattr(result, "items", ()) or ())
        if len(items) != len(row_indexes):
            raise RuntimeError(
                "process_batch returned an unexpected number of results for "
                f"column {text_column!r}"
            )

        redacted_values = list(values)
        for row_index, item in zip(row_indexes, items):
            if getattr(item, "error", None):
                raise RuntimeError(
                    "De-identification failed for "
                    f"column {text_column!r} at row {row_index}"
                )
            redacted_values[row_index] = _deidentified_text(
                getattr(item, "result", item)
            )

        redacted_array = pa.array(redacted_values, type=field.type)
        return record_batch.set_column(column_index, field, redacted_array)


def _parse_descriptor(descriptor: flight.FlightDescriptor) -> dict[str, Any]:
    if descriptor.descriptor_type != flight.DescriptorType.CMD:
        raise ValueError("Arrow Flight de-identification requires a command descriptor")
    try:
        payload = json.loads(descriptor.command.decode("utf-8"))
    except (AttributeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Arrow Flight command descriptor must contain valid UTF-8 JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Arrow Flight command descriptor must contain a JSON object")
    if payload.get("version") != ARROW_FLIGHT_DESCRIPTOR_VERSION:
        raise ValueError(
            "Unsupported Arrow Flight descriptor version; "
            f"expected {ARROW_FLIGHT_DESCRIPTOR_VERSION}"
        )

    if set(payload).difference({"version", "text_column", "policy"}):
        raise ValueError("Arrow Flight command descriptor has unsupported fields")

    request: dict[str, Any] = {}
    if "text_column" in payload:
        request["text_column"] = _normalize_required_string(
            payload["text_column"], "text_column"
        )
    if "policy" in payload:
        request["policy"] = _normalize_required_string(payload["policy"], "policy")
    return request


def _validate_text_column(schema: pa.Schema, text_column: str) -> int:
    column_index = schema.get_field_index(text_column)
    if column_index < 0:
        raise ValueError(f"Text column is missing from the Arrow schema: {text_column}")
    field_type = schema.field(column_index).type
    if not (
        pa.types.is_string(field_type)
        or pa.types.is_large_string(field_type)
        or pa.types.is_null(field_type)
    ):
        raise ValueError(f"Arrow text column must be string-typed: {text_column}")
    return column_index


def _deidentified_text(result: Any) -> str:
    if hasattr(result, "deidentified_text"):
        return str(result.deidentified_text)
    if isinstance(result, Mapping) and "deidentified_text" in result:
        return str(result["deidentified_text"])
    if isinstance(result, str):
        return result
    raise TypeError("process_batch results must expose deidentified_text")


def _normalize_required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


__all__ = [
    "ARROW_FLIGHT_DESCRIPTOR_VERSION",
    "DEFAULT_ARROW_FLIGHT_MODEL",
    "ArrowFlightDeidentificationServer",
    "make_deidentify_descriptor",
]
