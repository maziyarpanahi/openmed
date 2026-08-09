"""Airflow-compatible, local-first redaction for bounded inputs.

The Airflow dependency is optional.  This module can be imported without
Airflow installed so the interop registry remains dependency-light; when the
extra is installed, :class:`OpenMedRedactionOperator` is a regular Airflow
``BaseOperator``.

The operator accepts either one bounded UTF-8 file or one bounded in-memory
record batch.  It writes output atomically and keeps a sidecar containing only
hashes, counts, and sizes.  A matching sidecar lets an Airflow retry reuse a
verified output without invoking the redactor again.  Raw input is never
included in logs, exceptions, XCom summaries, or the sidecar.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Final

try:
    from airflow.exceptions import AirflowException as _AirflowException
    from airflow.models import BaseOperator as _AirflowBaseOperator

    _AIRFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in core-only installs
    _AIRFLOW_AVAILABLE = False

    class _AirflowException(RuntimeError):
        """Fallback exception used when the optional Airflow extra is absent."""

    class _AirflowBaseOperator:
        """Fallback base so the local operator remains unit-testable."""


_LOGGER = logging.getLogger(__name__)
_DEFAULT_POLICY = "hipaa_safe_harbor"
_DEFAULT_TEXT_FIELD = "text"
_DEFAULT_MAX_INPUT_BYTES = 10 * 1024 * 1024
_DEFAULT_MAX_RECORDS = 10_000
_MANIFEST_SCHEMA_VERSION = 1
_MAX_MANIFEST_BYTES = 64 * 1024
_JSONL_SUFFIXES: Final[frozenset[str]] = frozenset({".jsonl", ".ndjson"})

Deidentifier = Callable[..., Any]
Record = Mapping[str, Any] | str


class RedactionOperatorError(_AirflowException):
    """Raised for a safe, deterministic Airflow redaction failure."""


@dataclass(frozen=True)
class _InputPayload:
    mode: str
    input_bytes: bytes
    input_fingerprint: str
    records: tuple[Record, ...] | None = None
    input_format: str = "text"


@dataclass(frozen=True)
class _RedactionStats:
    records_processed: int
    records_redacted: int
    spans_redacted: int


class OpenMedRedactionOperator(_AirflowBaseOperator):
    """Redact one bounded local file or one bounded record batch.

    Exactly one of ``input_path`` and ``records`` (or one of its aliases) is
    required.  File input requires ``output_path``.  Record batches may return
    redacted records directly when no output path is supplied; providing an
    output path writes JSON or JSON Lines according to its suffix.

    The default redactor is OpenMed's local de-identification function.  A
    caller may inject ``deidentifier`` for a preloaded model or an offline
    test.  The operator itself does not make network calls and uses a
    cache-only OpenMed configuration for the default redactor.

    Args:
        input_path: Bounded UTF-8 text, JSON, or JSON Lines input file.
        records: Bounded sequence of strings or mappings containing
            ``text_field``.  ``input_records`` and ``record_batch`` are
            accepted as explicit aliases.
        output_path: Destination for the redacted file or serialized record
            batch.  File input always requires this argument.
        fingerprint_path: Optional PHI-free sidecar location.  When omitted,
            file outputs use ``<output>.openmed-fingerprint.json``.
        text_field: Mapping field to redact in record batches.
        policy: OpenMed policy passed to the deidentifier.
        method: OpenMed redaction method, defaulting to deterministic masking.
        max_input_bytes: Maximum file size accepted by the operator.
        max_records: Maximum records accepted from a batch or JSON input.
        deidentifier: Optional callable returning a string or an object with
            ``deidentified_text``.
        deidentify_kwargs: Additional keyword arguments passed to the
            deidentifier.  ``method`` and ``policy`` remain explicit defaults
            unless supplied in this mapping.
        task_id: Airflow task id.  A default makes the fallback class useful
            in dependency-free tests.
        **operator_kwargs: Standard Airflow ``BaseOperator`` options.

    Raises:
        RedactionOperatorError: If the input, output, or deidentifier
            contract is invalid.  Messages contain only stable metadata and
            never include input values or exception text from a redactor.
    """

    template_fields: Sequence[str] = (
        "input_path",
        "output_path",
        "fingerprint_path",
        "text_field",
    )

    def __init__(
        self,
        *,
        input_path: str | Path | None = None,
        records: Sequence[Record] | None = None,
        input_records: Sequence[Record] | None = None,
        record_batch: Sequence[Record] | None = None,
        output_path: str | Path | None = None,
        fingerprint_path: str | Path | None = None,
        text_field: str = _DEFAULT_TEXT_FIELD,
        policy: str = _DEFAULT_POLICY,
        method: str = "mask",
        max_input_bytes: int = _DEFAULT_MAX_INPUT_BYTES,
        max_records: int = _DEFAULT_MAX_RECORDS,
        deidentifier: Deidentifier | None = None,
        deidentify_kwargs: Mapping[str, Any] | None = None,
        task_id: str = "openmed_redaction",
        **operator_kwargs: Any,
    ) -> None:
        sources = [
            source
            for source in (records, input_records, record_batch)
            if source is not None
        ]
        if input_path is not None and sources:
            raise ValueError("provide either input_path or one record batch")
        if input_path is None and len(sources) != 1:
            raise ValueError("provide exactly one input_path or record batch")
        if input_path is not None and output_path is None:
            raise ValueError("output_path is required for file input")
        if fingerprint_path is not None and output_path is None:
            raise ValueError("fingerprint_path requires output_path")
        if not isinstance(text_field, str) or not text_field:
            raise ValueError("text_field must be a non-empty string")
        if not isinstance(policy, str) or not policy:
            raise ValueError("policy must be a non-empty string")
        if not isinstance(method, str) or not method:
            raise ValueError("method must be a non-empty string")
        _require_positive_int(max_input_bytes, "max_input_bytes")
        _require_positive_int(max_records, "max_records")
        if deidentify_kwargs is not None and not isinstance(deidentify_kwargs, Mapping):
            raise TypeError("deidentify_kwargs must be a mapping")

        if _AIRFLOW_AVAILABLE:
            super().__init__(task_id=task_id, **operator_kwargs)
        else:
            self.task_id = task_id
            self.operator_kwargs = dict(operator_kwargs)

        self.input_path = input_path
        self._records_source = sources[0] if sources else None
        self.output_path = output_path
        self.fingerprint_path = fingerprint_path
        self.text_field = text_field
        self.policy = policy
        self.method = method
        self.max_input_bytes = max_input_bytes
        self.max_records = max_records
        self._deidentifier = deidentifier
        self._deidentify_kwargs = dict(deidentify_kwargs or {})

        if input_path is not None and output_path is not None:
            if _same_path(input_path, output_path):
                raise ValueError("input and output paths must differ")
        if output_path is not None and fingerprint_path is not None:
            if _same_path(output_path, fingerprint_path):
                raise ValueError("fingerprint path must differ from output path")

    def execute(self, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Run one bounded redaction and return a PHI-free summary.

        ``context`` is accepted for Airflow compatibility and intentionally
        ignored.  No context values are copied into logs, output, or reports.
        """

        del context
        payload = self._load_payload()
        configuration_fingerprint = self._configuration_fingerprint(
            payload.input_format
        )
        manifest_path = self._manifest_path()

        if manifest_path is not None:
            existing = _load_existing_manifest(manifest_path)
            if existing is not None:
                result = self._reuse_existing_output(
                    payload,
                    configuration_fingerprint,
                    existing,
                )
                self._log_result(result)
                return result

        if payload.records is None:
            output_bytes, stats = self._redact_text_file(payload)
        else:
            redacted_records, stats = self._redact_records(
                payload.records,
                input_fingerprint=payload.input_fingerprint,
            )
            output_bytes = _serialize_records(
                redacted_records,
                output_path=self.output_path,
            )

        output_fingerprint = _digest_bytes(output_bytes)
        if self.output_path is not None:
            _atomic_write(self.output_path, output_bytes)
            if manifest_path is not None:
                _atomic_write_json(
                    manifest_path,
                    _manifest_payload(
                        mode=payload.mode,
                        input_fingerprint=payload.input_fingerprint,
                        configuration_fingerprint=configuration_fingerprint,
                        output_fingerprint=output_fingerprint,
                        input_size=len(payload.input_bytes),
                        output_size=len(output_bytes),
                        stats=stats,
                    ),
                )

        result = self._result_payload(
            status="success",
            payload=payload,
            configuration_fingerprint=configuration_fingerprint,
            output_fingerprint=output_fingerprint,
            output_size=len(output_bytes),
            stats=stats,
        )
        if payload.records is not None and self.output_path is None:
            result["redacted_records"] = redacted_records
        self._log_result(result)
        return result

    def _load_payload(self) -> _InputPayload:
        if self._records_source is not None:
            records = _materialize_records(self._records_source, self.max_records)
            _validate_records(records, text_field=self.text_field)
            try:
                input_bytes = _serialize_canonical(records)
            except (TypeError, ValueError, OverflowError):
                raise RedactionOperatorError(
                    "record batch is not JSON serializable"
                ) from None
            if len(input_bytes) > self.max_input_bytes:
                raise RedactionOperatorError(
                    "record batch exceeds the configured byte bound"
                )
            return _InputPayload(
                mode="records",
                input_bytes=input_bytes,
                input_fingerprint=_digest_bytes(input_bytes),
                records=tuple(records),
                input_format="records",
            )

        path = _coerce_path(self.input_path, "input_path")
        try:
            input_bytes = path.read_bytes()
        except (OSError, ValueError):
            raise RedactionOperatorError("unable to read input file") from None
        if len(input_bytes) > self.max_input_bytes:
            raise RedactionOperatorError("input file exceeds the configured byte bound")

        suffix = path.suffix.lower()
        if suffix in _JSONL_SUFFIXES:
            records = _parse_json_lines(input_bytes, self.max_records)
            _validate_records(records, text_field=self.text_field)
            return _InputPayload(
                mode="file-records",
                input_bytes=input_bytes,
                input_fingerprint=_digest_bytes(input_bytes),
                records=tuple(records),
                input_format="jsonl",
            )
        if suffix == ".json":
            records = _parse_json_document(input_bytes, self.max_records)
            _validate_records(records, text_field=self.text_field)
            return _InputPayload(
                mode="file-records",
                input_bytes=input_bytes,
                input_fingerprint=_digest_bytes(input_bytes),
                records=tuple(records),
                input_format="json",
            )

        try:
            input_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raise RedactionOperatorError("input file must be valid UTF-8") from None
        return _InputPayload(
            mode="file",
            input_bytes=input_bytes,
            input_fingerprint=_digest_bytes(input_bytes),
            input_format="text",
        )

    def _redact_text_file(
        self,
        payload: _InputPayload,
    ) -> tuple[bytes, _RedactionStats]:
        try:
            text = payload.input_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raise RedactionOperatorError("input file must be valid UTF-8") from None
        redacted, spans = self._redact_text(
            text,
            input_fingerprint=payload.input_fingerprint,
        )
        return redacted.encode("utf-8"), _RedactionStats(
            records_processed=1,
            records_redacted=int(redacted != text),
            spans_redacted=spans,
        )

    def _redact_records(
        self,
        records: Sequence[Record],
        *,
        input_fingerprint: str,
    ) -> tuple[list[Record], _RedactionStats]:
        redacted_records: list[Record] = []
        records_redacted = 0
        spans_redacted = 0
        for record in records:
            if isinstance(record, str):
                redacted, spans = self._redact_text(
                    record,
                    input_fingerprint=input_fingerprint,
                )
                redacted_records.append(redacted)
                records_redacted += int(redacted != record)
                spans_redacted += spans
                continue

            value = record[self.text_field]
            if value is None:
                redacted_records.append(dict(record))
                continue
            redacted, spans = self._redact_text(
                value,
                input_fingerprint=input_fingerprint,
            )
            output_record = dict(record)
            output_record[self.text_field] = redacted
            redacted_records.append(output_record)
            records_redacted += int(redacted != value)
            spans_redacted += spans
        return redacted_records, _RedactionStats(
            records_processed=len(records),
            records_redacted=records_redacted,
            spans_redacted=spans_redacted,
        )

    def _redact_text(
        self,
        text: str,
        *,
        input_fingerprint: str,
    ) -> tuple[str, int]:
        deidentifier = self._deidentifier
        if deidentifier is None:
            deidentifier = _default_deidentifier
        try:
            result = deidentifier(text, **self._deidentifier_options())
            redacted = _result_text(result)
        except RedactionOperatorError:
            raise
        except Exception:
            raise RedactionOperatorError(
                f"redaction failed; input_fingerprint={input_fingerprint}"
            ) from None

        entities = getattr(result, "pii_entities", None)
        if isinstance(entities, Sequence) and not isinstance(entities, (str, bytes)):
            span_count = len(entities)
        else:
            span_count = int(redacted != text)
        return redacted, span_count

    def _deidentifier_options(self) -> dict[str, Any]:
        options = dict(self._deidentify_kwargs)
        options.setdefault("method", self.method)
        options.setdefault("policy", self.policy)
        if self._deidentifier is None and "config" not in options:
            options["config"] = _offline_config()
        return options

    def _configuration_fingerprint(self, input_format: str) -> str:
        payload = {
            "schema_version": _MANIFEST_SCHEMA_VERSION,
            "adapter": "openmed.interop.airflow",
            "mode": "records" if self._records_source is not None else "file",
            "input_format": input_format,
            "text_field": self.text_field,
            "options": _fingerprint_value(self._deidentifier_options()),
            "deidentifier": _callable_identity(self._deidentifier),
        }
        return _digest_bytes(_serialize_canonical(payload))

    def _manifest_path(self) -> Path | None:
        if self.output_path is None:
            return None
        if self.fingerprint_path is not None:
            return _coerce_path(self.fingerprint_path, "fingerprint_path")
        output_path = _coerce_path(self.output_path, "output_path")
        return output_path.with_name(output_path.name + ".openmed-fingerprint.json")

    def _reuse_existing_output(
        self,
        payload: _InputPayload,
        configuration_fingerprint: str,
        manifest: Mapping[str, Any],
    ) -> dict[str, Any]:
        output_path = _coerce_path(self.output_path, "output_path")
        expected = {
            "schema_version": _MANIFEST_SCHEMA_VERSION,
            "mode": payload.mode,
            "input_fingerprint": payload.input_fingerprint,
            "configuration_fingerprint": configuration_fingerprint,
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise RedactionOperatorError(
                "existing output fingerprint does not match this run"
            )
        try:
            output_fingerprint = _digest_file(output_path)
        except (OSError, ValueError):
            raise RedactionOperatorError(
                "fingerprint manifest exists but output is unavailable"
            ) from None
        if output_fingerprint != manifest.get("output_fingerprint"):
            raise RedactionOperatorError("existing output fingerprint is invalid")

        stats = _stats_from_manifest(manifest)
        result = self._result_payload(
            status="skipped",
            payload=payload,
            configuration_fingerprint=configuration_fingerprint,
            output_fingerprint=output_fingerprint,
            output_size=_manifest_size(manifest, "output_size"),
            stats=stats,
        )
        return result

    def _result_payload(
        self,
        *,
        status: str,
        payload: _InputPayload,
        configuration_fingerprint: str,
        output_fingerprint: str,
        output_size: int,
        stats: _RedactionStats,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": status,
            "mode": payload.mode,
            "input_fingerprint": payload.input_fingerprint,
            "configuration_fingerprint": configuration_fingerprint,
            "output_fingerprint": output_fingerprint,
            "input_size": len(payload.input_bytes),
            "output_size": output_size,
            "records_processed": stats.records_processed,
            "records_redacted": stats.records_redacted,
            "spans_redacted": stats.spans_redacted,
        }
        if self.output_path is not None:
            result["output_path"] = str(self.output_path)
            manifest_path = self._manifest_path()
            if manifest_path is not None:
                result["fingerprint_path"] = str(manifest_path)
        return result

    def _log_result(self, result: Mapping[str, Any]) -> None:
        logger = getattr(self, "log", _LOGGER)
        logger.info(
            "OpenMed redaction %s: mode=%s records=%d records_redacted=%d "
            "spans_redacted=%d input_fingerprint=%s output_fingerprint=%s",
            result["status"],
            result["mode"],
            result["records_processed"],
            result["records_redacted"],
            result["spans_redacted"],
            result["input_fingerprint"],
            result["output_fingerprint"],
        )


AirflowRedactionOperator = OpenMedRedactionOperator


def _default_deidentifier(text: str, **kwargs: Any) -> Any:
    deidentify = getattr(import_module("openmed.core.pii"), "deidentify")
    return deidentify(text, **kwargs)


def _offline_config() -> Any:
    config_type = getattr(import_module("openmed.core.config"), "OpenMedConfig")
    return config_type(local_only=True)


def _result_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    value = getattr(result, "deidentified_text", None)
    if not isinstance(value, str):
        raise RedactionOperatorError(
            "deidentifier must return text or an object with deidentified_text"
        )
    return value


def _materialize_records(
    records: Iterable[Record],
    max_records: int,
) -> list[Record]:
    if isinstance(records, (str, bytes, bytearray, Mapping)):
        raise TypeError("record batch must be an iterable of records")
    try:
        iterator = iter(records)
    except TypeError:
        raise TypeError("record batch must be iterable") from None

    materialized: list[Record] = []
    for index, record in enumerate(iterator):
        if index >= max_records:
            raise RedactionOperatorError("record batch exceeds the configured limit")
        materialized.append(record)
    return materialized


def _validate_records(records: Sequence[Record], *, text_field: str) -> None:
    for record in records:
        if isinstance(record, str):
            continue
        if not isinstance(record, Mapping):
            raise TypeError("records must contain strings or mappings")
        if text_field not in record:
            raise RedactionOperatorError("record is missing the configured text field")
        value = record[text_field]
        if value is not None and not isinstance(value, str):
            raise TypeError("configured record field must contain text or null")


def _parse_json_lines(payload: bytes, max_records: int) -> list[Record]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        raise RedactionOperatorError("input file must be valid UTF-8") from None
    records: list[Record] = []
    for line in lines:
        if not line.strip():
            continue
        if len(records) >= max_records:
            raise RedactionOperatorError("record batch exceeds the configured limit")
        try:
            record = json.loads(line)
        except (TypeError, ValueError, json.JSONDecodeError):
            raise RedactionOperatorError("input JSON Lines file is invalid") from None
        records.append(record)
    return records


def _parse_json_document(payload: bytes, max_records: int) -> list[Record]:
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError):
        raise RedactionOperatorError("input JSON file is invalid") from None
    if isinstance(document, list):
        if len(document) > max_records:
            raise RedactionOperatorError("record batch exceeds the configured limit")
        return document
    if isinstance(document, (str, Mapping)):
        return [document]
    raise RedactionOperatorError("input JSON must contain records")


def _serialize_records(
    records: Sequence[Record],
    *,
    output_path: str | Path | None,
) -> bytes:
    try:
        if output_path is not None and _coerce_path(output_path, "output_path").suffix:
            suffix = _coerce_path(output_path, "output_path").suffix.lower()
        else:
            suffix = ".jsonl"
        if suffix == ".json":
            return _serialize_canonical(records) + b"\n"
        if not records:
            return b""
        return b"".join(_serialize_canonical(record) + b"\n" for record in records)
    except (TypeError, ValueError, OverflowError):
        raise RedactionOperatorError(
            "redacted records are not JSON serializable"
        ) from None


def _serialize_canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _manifest_payload(
    *,
    mode: str,
    input_fingerprint: str,
    configuration_fingerprint: str,
    output_fingerprint: str,
    input_size: int,
    output_size: int,
    stats: _RedactionStats,
) -> dict[str, Any]:
    return {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "mode": mode,
        "input_fingerprint": input_fingerprint,
        "configuration_fingerprint": configuration_fingerprint,
        "output_fingerprint": output_fingerprint,
        "input_size": input_size,
        "output_size": output_size,
        "records_processed": stats.records_processed,
        "records_redacted": stats.records_redacted,
        "spans_redacted": stats.spans_redacted,
    }


def _load_existing_manifest(path: Path) -> Mapping[str, Any] | None:
    try:
        if not path.exists():
            return None
        payload = path.read_bytes()
    except (OSError, ValueError):
        raise RedactionOperatorError("unable to read fingerprint manifest") from None
    if len(payload) > _MAX_MANIFEST_BYTES:
        raise RedactionOperatorError("fingerprint manifest exceeds the size bound")
    try:
        manifest = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError):
        raise RedactionOperatorError("fingerprint manifest is invalid") from None
    if not isinstance(manifest, Mapping):
        raise RedactionOperatorError("fingerprint manifest is invalid")
    return manifest


def _stats_from_manifest(manifest: Mapping[str, Any]) -> _RedactionStats:
    return _RedactionStats(
        records_processed=_manifest_size(manifest, "records_processed"),
        records_redacted=_manifest_size(manifest, "records_redacted"),
        spans_redacted=_manifest_size(manifest, "spans_redacted"),
    )


def _manifest_size(manifest: Mapping[str, Any], key: str) -> int:
    value = manifest.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RedactionOperatorError("fingerprint manifest contains invalid counts")
    return value


def _atomic_write(path_value: str | Path, payload: bytes) -> None:
    path = _coerce_path(path_value, "output_path")
    temporary_path: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=".openmed-redaction-",
            suffix=".tmp",
            dir=str(path.parent),
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    except (OSError, ValueError):
        raise RedactionOperatorError("unable to write redaction output") from None
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        serialized = _serialize_canonical(payload) + b"\n"
    except (TypeError, ValueError, OverflowError):
        raise RedactionOperatorError(
            "fingerprint manifest could not be serialized"
        ) from None
    _atomic_write(path, serialized)


def _digest_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _fingerprint_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"type": "bytes", "sha256": _digest_bytes(value)}
    if isinstance(value, Path):
        return {"type": "path", "sha256": _digest_bytes(str(value).encode())}
    if isinstance(value, Mapping):
        return {
            str(key): _fingerprint_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_fingerprint_value(item) for item in value]
    if callable(value):
        return {"type": "callable", "identity": _callable_identity(value)}
    return {"type": type(value).__name__}


def _callable_identity(value: Any) -> str:
    if value is None:
        return "default"
    module = getattr(value, "__module__", type(value).__module__)
    qualname = getattr(value, "__qualname__", type(value).__qualname__)
    return f"{module}.{qualname}"


def _coerce_path(value: str | Path | None, field_name: str) -> Path:
    if value is None:
        raise RedactionOperatorError(f"{field_name} is required")
    if not isinstance(value, (str, Path)):
        raise RedactionOperatorError(f"{field_name} must be a path")
    try:
        return Path(value)
    except (TypeError, ValueError, OSError):
        raise RedactionOperatorError(f"{field_name} is invalid") from None


def _same_path(first: str | Path, second: str | Path) -> bool:
    try:
        return Path(first).expanduser().resolve() == Path(second).expanduser().resolve()
    except (TypeError, ValueError, OSError):
        return False


def _require_positive_int(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


__all__ = [
    "AirflowRedactionOperator",
    "OpenMedRedactionOperator",
    "RedactionOperatorError",
]
