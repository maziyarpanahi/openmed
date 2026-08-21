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

import copy
import hashlib
import json
import logging
import marshal
import os
import stat
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from importlib import import_module
from pathlib import Path
from types import BuiltinFunctionType, CodeType, FunctionType, MethodType, ModuleType
from typing import Any, Final

try:
    from airflow.exceptions import AirflowException as _AirflowException
    from airflow.models import BaseOperator as _AirflowBaseOperator

    _AIRFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in core-only installs
    _AIRFLOW_AVAILABLE = False

    class _AirflowException(RuntimeError):  # type: ignore[no-redef]
        """Fallback exception used when the optional Airflow extra is absent."""

    class _AirflowBaseOperator:  # type: ignore[no-redef]
        """Fallback base so the local operator remains unit-testable."""


_LOGGER = logging.getLogger(__name__)
_DEFAULT_POLICY = "hipaa_safe_harbor"
_DEFAULT_TEXT_FIELD = "text"
_DEFAULT_MAX_INPUT_BYTES = 10 * 1024 * 1024
_DEFAULT_MAX_RECORDS = 10_000
_MANIFEST_SCHEMA_VERSION = 1
_MAX_MANIFEST_BYTES = 64 * 1024
_MAX_OUTPUT_EXPANSION = 8
_MAX_FINGERPRINT_DEPTH = 16
_MAX_FINGERPRINT_ITEMS = 10_000
_MAX_FINGERPRINT_LEAF_BYTES = 1024 * 1024
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
    required.  Every run requires ``output_path`` so Airflow receives only a
    counts-and-fingerprints task result instead of redacted records through
    XCom. Record batches are written as JSON or JSON Lines according to the
    output suffix.

    The default redactor is OpenMed's local de-identification function.  A
    caller may inject ``deidentifier`` for a preloaded model or an offline
    test.  The operator itself does not make network calls and uses a
    cache-only OpenMed configuration for the default redactor.

    Args:
        input_path: Bounded UTF-8 text, JSON, or JSON Lines input file.
        records: Bounded sequence of strings or mappings containing
            ``text_field``.  ``input_records`` and ``record_batch`` are
            accepted as explicit aliases.
        output_path: Required destination for the redacted file or serialized
            record batch.
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
        if output_path is None:
            raise ValueError("output_path is required")
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
        try:
            self._deidentifier_identity = _callable_identity(deidentifier)
        except Exception:
            raise ValueError("deidentifier identity could not be captured") from None
        try:
            options = dict(deidentify_kwargs or {})
            if any(type(key) is not str or not key for key in options):
                raise TypeError
            self._deidentify_kwargs_snapshot = copy.deepcopy(options)
            self._deidentify_kwargs_fingerprint = _digest_bytes(
                _serialize_canonical(
                    _callable_state_value(self._deidentify_kwargs_snapshot)
                )
            )
        except Exception:
            raise ValueError(
                "deidentify_kwargs must contain copyable, fingerprintable "
                "string-keyed options"
            ) from None

        if input_path is not None and output_path is not None:
            if _same_path(input_path, output_path):
                raise ValueError("input and output paths must differ")
        if output_path is not None and fingerprint_path is not None:
            if _same_path(output_path, fingerprint_path):
                raise ValueError("fingerprint path must differ from output path")
        if input_path is not None and output_path is not None:
            manifest_candidate = fingerprint_path
            if manifest_candidate is None and isinstance(output_path, (str, Path)):
                candidate = Path(output_path)
                manifest_candidate = candidate.with_name(
                    candidate.name + ".openmed-fingerprint.json"
                )
            if manifest_candidate is not None and _same_path(
                input_path, manifest_candidate
            ):
                raise ValueError("input and fingerprint paths must differ")

    def execute(self, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Run one bounded redaction and return a PHI-free summary.

        ``context`` is accepted for Airflow compatibility and intentionally
        ignored.  No context values are copied into logs, output, or reports.
        """

        del context
        self._validate_runtime_paths()
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

        if len(output_bytes) > self.max_input_bytes * _MAX_OUTPUT_EXPANSION:
            raise RedactionOperatorError(
                "redaction output exceeds the configured expansion bound"
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
        self._log_result(result)
        return result

    def _validate_runtime_paths(self) -> None:
        """Reject path collisions after Airflow template rendering."""

        output_path = _coerce_path(self.output_path, "output_path")
        manifest_path = self._manifest_path()
        if manifest_path is None:
            raise RedactionOperatorError("fingerprint path is required")
        if _same_path(output_path, manifest_path):
            raise RedactionOperatorError(
                "fingerprint path must differ from output path"
            )
        if self.input_path is None:
            return
        input_path = _coerce_path(self.input_path, "input_path")
        if _same_path(input_path, output_path):
            raise RedactionOperatorError("input and output paths must differ")
        if _same_path(input_path, manifest_path):
            raise RedactionOperatorError("input and fingerprint paths must differ")

    def _load_payload(self) -> _InputPayload:
        if self._records_source is not None:
            records = _materialize_records(self._records_source, self.max_records)
            _validate_records(records, text_field=self.text_field)
            try:
                input_bytes = _serialize_canonical(records)
            except Exception:
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
        input_bytes = _read_bounded_file(path, self.max_input_bytes)

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

            try:
                value = record[self.text_field]
            except Exception:
                raise RedactionOperatorError("unable to read record metadata") from None
            if value is None:
                try:
                    redacted_records.append(dict(record))
                except Exception:
                    raise RedactionOperatorError(
                        "unable to copy record metadata"
                    ) from None
                continue
            redacted, spans = self._redact_text(
                value,
                input_fingerprint=input_fingerprint,
            )
            try:
                output_record = dict(record)
            except Exception:
                raise RedactionOperatorError("unable to copy record metadata") from None
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
            entities = getattr(result, "pii_entities", None)
            if isinstance(entities, Sequence) and not isinstance(
                entities, (str, bytes)
            ):
                span_count = len(entities)
            else:
                span_count = int(redacted != text)
        except Exception:
            raise RedactionOperatorError(
                f"redaction failed; input_fingerprint={input_fingerprint}"
            ) from None
        return redacted, span_count

    def _deidentifier_options(self) -> dict[str, Any]:
        try:
            options = copy.deepcopy(self._deidentify_kwargs_snapshot)
        except Exception:
            raise RedactionOperatorError(
                "redaction options could not be restored"
            ) from None
        if not isinstance(options, dict):
            raise RedactionOperatorError("redaction options are invalid")
        options.setdefault("method", self.method)
        options.setdefault("policy", self.policy)
        if self._deidentifier is None and "config" not in options:
            options["config"] = _offline_config()
        return options

    def _configuration_fingerprint(self, input_format: str) -> str:
        try:
            payload = {
                "schema_version": _MANIFEST_SCHEMA_VERSION,
                "adapter": "openmed.interop.airflow",
                "mode": "records" if self._records_source is not None else "file",
                "input_format": input_format,
                "text_field": self.text_field,
                "options_snapshot": self._deidentify_kwargs_fingerprint,
                "method": self.method,
                "policy": self.policy,
                "default_offline_config": self._deidentifier is None,
                "deidentifier": self._deidentifier_identity,
            }
            return _digest_bytes(_serialize_canonical(payload))
        except Exception:
            raise RedactionOperatorError(
                "redaction configuration could not be fingerprinted"
            ) from None

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
        output_size = _manifest_size(
            manifest,
            "output_size",
            maximum=self.max_input_bytes * _MAX_OUTPUT_EXPANSION,
        )
        output_bytes = _read_stable_regular_file(
            output_path,
            output_size,
            read_error="fingerprint manifest exists but output is unavailable",
            too_large_error="existing output exceeds the configured expansion bound",
        )
        if len(output_bytes) != output_size:
            raise RedactionOperatorError("existing output size is invalid")
        output_fingerprint = _digest_bytes(output_bytes)
        if output_fingerprint != manifest.get("output_fingerprint"):
            raise RedactionOperatorError("existing output fingerprint is invalid")

        stats = _stats_from_manifest(manifest)
        result = self._result_payload(
            status="skipped",
            payload=payload,
            configuration_fingerprint=configuration_fingerprint,
            output_fingerprint=output_fingerprint,
            output_size=output_size,
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
    while len(materialized) <= max_records:
        try:
            record = next(iterator)
        except StopIteration:
            return materialized
        except Exception:
            raise RedactionOperatorError("unable to read record batch") from None
        if len(materialized) == max_records:
            raise RedactionOperatorError("record batch exceeds the configured limit")
        materialized.append(record)
    return materialized


def _validate_records(records: Sequence[Record], *, text_field: str) -> None:
    for record in records:
        if isinstance(record, str):
            continue
        if not isinstance(record, Mapping):
            raise TypeError("records must contain strings or mappings")
        try:
            field_present = text_field in record
            value = record[text_field] if field_present else None
        except Exception:
            raise RedactionOperatorError("unable to read record metadata") from None
        if not field_present:
            raise RedactionOperatorError("record is missing the configured text field")
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
        except Exception:
            raise RedactionOperatorError("input JSON Lines file is invalid") from None
        records.append(record)
    return records


def _parse_json_document(payload: bytes, max_records: int) -> list[Record]:
    try:
        document = json.loads(payload.decode("utf-8"))
    except Exception:
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
    except Exception:
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
        os.lstat(path)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        raise RedactionOperatorError("unable to read fingerprint manifest") from None
    payload = _read_stable_regular_file(
        path,
        _MAX_MANIFEST_BYTES,
        read_error="unable to read fingerprint manifest",
        too_large_error="fingerprint manifest exceeds the size bound",
    )
    if not payload:
        raise RedactionOperatorError("fingerprint manifest is invalid")
    try:
        manifest = json.loads(payload.decode("utf-8"))
    except Exception:
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


def _manifest_size(
    manifest: Mapping[str, Any],
    key: str,
    *,
    maximum: int | None = None,
) -> int:
    value = manifest.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or (maximum is not None and value > maximum)
    ):
        raise RedactionOperatorError("fingerprint manifest contains invalid counts")
    return value


class _FileTooLargeError(Exception):
    """Signal a bounded file exceeded its accepted size."""


def _read_stable_regular_file(
    path: Path,
    max_bytes: int,
    *,
    read_error: str,
    too_large_error: str,
) -> bytes:
    """Read one stable regular file without following a final symlink."""

    descriptor: int | None = None
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(os.fspath(path), flags)
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise OSError
        if opened_stat.st_size > max_bytes:
            raise _FileTooLargeError
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            payload = handle.read(opened_stat.st_size + 1)
            final_stat = os.fstat(handle.fileno())
            path_stat = os.stat(path, follow_symlinks=False)
        if len(payload) > max_bytes:
            raise _FileTooLargeError
        if (
            not stat.S_ISREG(path_stat.st_mode)
            or not os.path.samestat(opened_stat, path_stat)
            or _stable_stat_state(opened_stat) != _stable_stat_state(final_stat)
            or len(payload) != opened_stat.st_size
        ):
            raise OSError
        return payload
    except _FileTooLargeError:
        raise RedactionOperatorError(too_large_error) from None
    except Exception:
        raise RedactionOperatorError(read_error) from None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _stable_stat_state(source_stat: os.stat_result) -> tuple[int, ...]:
    return (
        source_stat.st_dev,
        source_stat.st_ino,
        source_stat.st_size,
        source_stat.st_mtime_ns,
        source_stat.st_ctime_ns,
        source_stat.st_mode,
    )


def _read_bounded_file(path: Path, max_bytes: int) -> bytes:
    """Read a bounded stable regular input file."""

    return _read_stable_regular_file(
        path,
        max_bytes,
        read_error="unable to read input file",
        too_large_error="input file exceeds the configured byte bound",
    )


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


def _callable_identity(value: Any) -> str:
    if value is None:
        return "default"
    module = getattr(value, "__module__", type(value).__module__)
    qualname = getattr(value, "__qualname__", type(value).__qualname__)
    identity = f"{module}.{qualname}"
    _bounded_fingerprint_payload(identity.encode("utf-8"))
    target = getattr(value, "__func__", value)
    owner = getattr(value, "__self__", None)
    code = getattr(target, "__code__", None)
    if code is None:
        call = getattr(value, "__call__", None)
        target = getattr(call, "__func__", call)
        code = getattr(target, "__code__", None)
        if code is not None:
            owner = value
    if code is None:
        state = _callable_state_value(value)
        state_digest = _digest_bytes(_serialize_canonical(state))
        return f"{identity}:{state_digest}"

    digest = hashlib.sha256(_bounded_fingerprint_payload(marshal.dumps(code)))
    defaults = getattr(target, "__defaults__", None)
    keyword_defaults = getattr(target, "__kwdefaults__", None)
    closure = getattr(target, "__closure__", None) or ()
    closure_values: list[Any] = []
    for cell in closure:
        try:
            closure_values.append(_callable_state_value(cell.cell_contents))
        except ValueError:
            closure_values.append({"type": "empty-cell"})
    state = {
        "defaults": _callable_state_value(defaults),
        "keyword_defaults": _callable_state_value(keyword_defaults),
        "closure": closure_values,
        "owner": _callable_owner_state(owner),
    }
    digest.update(_serialize_canonical(state))
    return f"{identity}:{digest.hexdigest()}"


def _callable_owner_state(value: Any) -> Any:
    """Return stable state for a bound method or callable object."""

    if value is None:
        return None
    return _callable_state_value(value)


def _callable_state_value(value: Any) -> Any:
    """Return bounded, deterministic, value-free callback state."""

    try:
        return _normalize_fingerprint_value(
            value,
            depth=0,
            active=set(),
            item_count=[0],
        )
    except Exception:
        raise TypeError(
            "callback captured state must be bounded and deterministic"
        ) from None


def _normalize_fingerprint_value(
    value: Any,
    *,
    depth: int,
    active: set[int],
    item_count: list[int],
) -> Any:
    """Normalize supported state without serializing executable objects."""

    if depth > _MAX_FINGERPRINT_DEPTH:
        raise TypeError("callback captured state exceeds the depth limit")
    item_count[0] += 1
    if item_count[0] > _MAX_FINGERPRINT_ITEMS:
        raise TypeError("callback captured state exceeds the item limit")

    value_type = type(value)
    if value is None:
        return {"type": "none"}
    if value_type is bool:
        return {"type": "bool", "value": value}
    if value_type is int:
        return _fingerprint_leaf("int", str(value).encode("ascii"))
    if value_type is float:
        return _fingerprint_leaf("float", value.hex().encode("ascii"))
    if value_type is complex:
        payload = f"{value.real.hex()}:{value.imag.hex()}".encode("ascii")
        return _fingerprint_leaf("complex", payload)
    if value_type is str:
        return _fingerprint_leaf("str", value.encode("utf-8"))
    if value_type is bytes:
        return _fingerprint_leaf("bytes", value)
    if value_type is bytearray:
        return _fingerprint_leaf("bytearray", bytes(value))
    if value_type is memoryview:
        return _fingerprint_leaf("memoryview", value.tobytes())
    if isinstance(value, Path):
        return _fingerprint_leaf("path", str(value).encode("utf-8"))
    if isinstance(value, ModuleType):
        return _fingerprint_leaf("module", value.__name__.encode("utf-8"))
    if isinstance(value, CodeType):
        return _fingerprint_leaf("code", marshal.dumps(value))
    if isinstance(value, type):
        return _fingerprint_class(value, item_count=item_count)

    object_id = id(value)
    if object_id in active:
        raise TypeError("callback captured state must not contain cycles")
    active.add(object_id)
    try:
        if value_type in (list, tuple):
            return {
                "type": value_type.__name__,
                "items": [
                    _normalize_fingerprint_value(
                        item,
                        depth=depth + 1,
                        active=active,
                        item_count=item_count,
                    )
                    for item in value
                ],
            }
        if value_type in (set, frozenset):
            normalized_items = [
                _normalize_fingerprint_value(
                    item,
                    depth=depth + 1,
                    active=active,
                    item_count=item_count,
                )
                for item in value
            ]
            normalized_items.sort(key=_serialize_canonical)
            return {"type": value_type.__name__, "items": normalized_items}
        if isinstance(value, Mapping):
            normalized_pairs: list[tuple[bytes, list[Any]]] = []
            for key, item in value.items():
                normalized_key = _normalize_fingerprint_value(
                    key,
                    depth=depth + 1,
                    active=active,
                    item_count=item_count,
                )
                normalized_value = _normalize_fingerprint_value(
                    item,
                    depth=depth + 1,
                    active=active,
                    item_count=item_count,
                )
                pair = [normalized_key, normalized_value]
                normalized_pairs.append((_serialize_canonical(normalized_key), pair))
            normalized_pairs.sort(key=lambda pair: pair[0])
            return {"type": "mapping", "items": [pair[1] for pair in normalized_pairs]}
        if isinstance(value, (FunctionType, MethodType, BuiltinFunctionType)):
            return _fingerprint_function(
                value,
                depth=depth,
                active=active,
                item_count=item_count,
            )
        if is_dataclass(value) and not isinstance(value, type):
            attributes = {
                field.name: getattr(value, field.name) for field in fields(value)
            }
        else:
            attributes = vars(value)
        return {
            "type": "object",
            "identity": _type_identity(value_type),
            "attributes": _normalize_fingerprint_value(
                attributes,
                depth=depth + 1,
                active=active,
                item_count=item_count,
            ),
        }
    finally:
        active.remove(object_id)


def _fingerprint_leaf(kind: str, payload: bytes) -> dict[str, str]:
    return {
        "type": kind,
        "sha256": _digest_bytes(_bounded_fingerprint_payload(payload)),
    }


def _bounded_fingerprint_payload(payload: bytes) -> bytes:
    if len(payload) > _MAX_FINGERPRINT_LEAF_BYTES:
        raise TypeError("callback captured state exceeds the byte limit")
    return payload


def _type_identity(value: type[Any]) -> str:
    identity = f"{value.__module__}.{value.__qualname__}"
    _bounded_fingerprint_payload(identity.encode("utf-8"))
    return identity


def _fingerprint_class(
    value: type[Any],
    *,
    item_count: list[int],
) -> dict[str, str]:
    """Fingerprint class identity and executable members without descriptors."""

    identity = _type_identity(value)
    digest = hashlib.sha256(_bounded_fingerprint_payload(identity.encode("utf-8")))
    members: list[tuple[str, Any]] = []
    for name, member in vars(value).items():
        item_count[0] += 1
        if item_count[0] > _MAX_FINGERPRINT_ITEMS:
            raise TypeError("callback captured state exceeds the item limit")
        members.append((name, member))
    members.sort(key=lambda item: item[0])
    for name, member in members:
        target = member.fget if isinstance(member, property) else member
        if isinstance(target, (staticmethod, classmethod)):
            target = target.__func__
        code = getattr(target, "__code__", None)
        if code is not None:
            digest.update(_bounded_fingerprint_payload(name.encode("utf-8")))
            digest.update(_bounded_fingerprint_payload(marshal.dumps(code)))
        elif target is None or type(target) in (bool, int, float, str):
            digest.update(_bounded_fingerprint_payload(name.encode("utf-8")))
            digest.update(_bounded_fingerprint_payload(repr(target).encode("utf-8")))
    return {"type": "class", "identity": identity, "sha256": digest.hexdigest()}


def _fingerprint_function(
    value: FunctionType | MethodType | BuiltinFunctionType,
    *,
    depth: int,
    active: set[int],
    item_count: list[int],
) -> dict[str, Any]:
    """Fingerprint nested function state using code and captured values."""

    target = getattr(value, "__func__", value)
    code = getattr(target, "__code__", None)
    identity = (
        f"{getattr(value, '__module__', type(value).__module__)}."
        f"{getattr(value, '__qualname__', type(value).__qualname__)}"
    )
    state: dict[str, Any] = {
        "type": "function",
        "identity": identity,
    }
    if code is None:
        return state
    state["code"] = _digest_bytes(_bounded_fingerprint_payload(marshal.dumps(code)))
    state["defaults"] = _normalize_fingerprint_value(
        getattr(target, "__defaults__", None),
        depth=depth + 1,
        active=active,
        item_count=item_count,
    )
    state["keyword_defaults"] = _normalize_fingerprint_value(
        getattr(target, "__kwdefaults__", None),
        depth=depth + 1,
        active=active,
        item_count=item_count,
    )
    closure = getattr(target, "__closure__", None) or ()
    closure_values: list[Any] = []
    for cell in closure:
        try:
            captured = cell.cell_contents
        except ValueError:
            closure_values.append({"type": "empty-cell"})
            continue
        closure_values.append(
            _normalize_fingerprint_value(
                captured,
                depth=depth + 1,
                active=active,
                item_count=item_count,
            )
        )
    state["closure"] = closure_values
    owner = getattr(value, "__self__", None)
    if owner is not None:
        state["owner"] = _normalize_fingerprint_value(
            owner,
            depth=depth + 1,
            active=active,
            item_count=item_count,
        )
    return state


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
