"""Local-first OpenSearch ingest redaction processing.

This module implements the small document contract needed by a Python-side
OpenSearch ingest bridge without importing ``opensearch-py`` or creating a
client.  :class:`OpenSearchRedactionProcessor` accepts one mapping, returns a
redacted copy, and only processes explicitly configured text fields.

The processor never puts source text in exceptions or diagnostics.  Its
optional report contains configuration and aggregate counts only.  The
default deidentifier is imported lazily so importing this adapter does not
load a model or make a network call.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from openmed.core.policy import canonical_policy_name

Deidentifier = Callable[..., Any]

_DEFAULT_FIELDS = ("text",)
_DEFAULT_METHOD = "mask"
_DEFAULT_POLICY = "hipaa_safe_harbor"
_MISSING = object()


class OpenSearchRedactionError(ValueError):
    """Raised for a safe, deterministic OpenSearch processor failure."""


@dataclass(frozen=True)
class RedactionReport:
    """Value-free aggregate diagnostics for one processed document."""

    policy: str
    fields: tuple[str, ...]
    values_seen: int
    values_redacted: int
    spans_redacted: int

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible diagnostics without source values."""

        return {
            "adapter": "opensearch",
            "policy": self.policy,
            "fields": list(self.fields),
            "values_seen": self.values_seen,
            "values_redacted": self.values_redacted,
            "spans_redacted": self.spans_redacted,
        }


class OpenSearchRedactionProcessor:
    """Redact selected fields in one OpenSearch-style ingest document.

    ``process`` and ``execute`` accept a mapping and return a new ``dict``;
    the input document is never modified.  Field names may use dotted paths
    for nested objects.  A selected field must contain a string, ``None``, or
    a list/tuple of strings and ``None`` values.

    The adapter has no OpenSearch client dependency.  A caller can inject a
    local ``deidentifier`` for a preloaded model or an offline test.  The
    callable receives ``text``, ``policy``, ``method``, and any entries from
    ``deidentify_kwargs``.  It may return a string or an object/mapping with a
    string ``deidentified_text`` value.  If it also returns ``pii_entities``
    as a sequence, that sequence supplies the report's span count.

    Args:
        fields: One field name or a sequence of selected field names.  The
            default is ``("text",)``.
        field: Singular alias for ``fields``.  It cannot be combined with
            ``fields``.
        policy: Valid OpenMed policy profile name or alias.
        method: Redaction method passed to the deidentifier.
        ignore_missing: Leave absent selected fields unchanged instead of
            raising a processor error.
        deidentifier: Optional local callable used instead of OpenMed's
            default deidentifier.
        deidentify_kwargs: Additional local deidentifier options.
    """

    def __init__(
        self,
        *,
        fields: Sequence[str] | str | None = None,
        field: str | None = None,
        policy: str = _DEFAULT_POLICY,
        method: str = _DEFAULT_METHOD,
        ignore_missing: bool = False,
        deidentifier: Deidentifier | None = None,
        deidentify_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if fields is not None and field is not None:
            raise OpenSearchRedactionError("configure either fields or field, not both")

        selected_fields = field if field is not None else fields
        self.fields = _normalize_fields(selected_fields)
        self.policy = _normalize_policy(policy)
        self.method = _normalize_method(method)
        if not isinstance(ignore_missing, bool):
            raise OpenSearchRedactionError("ignore_missing must be a boolean")
        if deidentify_kwargs is not None and not isinstance(deidentify_kwargs, Mapping):
            raise OpenSearchRedactionError("deidentify_kwargs must be a mapping")

        self.ignore_missing = ignore_missing
        self._deidentifier = deidentifier
        self._deidentify_kwargs = dict(deidentify_kwargs or {})
        self._field_paths = tuple(
            (field_name, tuple(field_name.split("."))) for field_name in self.fields
        )

    def process(self, document: Mapping[str, Any]) -> dict[str, Any]:
        """Return a redacted copy of one ingest document."""

        redacted, _ = self._process_with_report(document)
        return redacted

    def execute(self, document: Mapping[str, Any]) -> dict[str, Any]:
        """Execute the processor using an ingest-bridge-compatible name."""

        return self.process(document)

    def process_with_report(
        self,
        document: Mapping[str, Any],
    ) -> tuple[dict[str, Any], RedactionReport]:
        """Return a redacted copy and aggregate, source-free diagnostics."""

        return self._process_with_report(document)

    def _process_with_report(
        self,
        document: Mapping[str, Any],
    ) -> tuple[dict[str, Any], RedactionReport]:
        if not isinstance(document, Mapping):
            raise OpenSearchRedactionError("document must be a mapping")

        try:
            redacted_document = deepcopy(dict(document))
        except Exception:
            raise OpenSearchRedactionError("document could not be copied") from None

        values_seen = 0
        values_redacted = 0
        spans_redacted = 0

        for field_name, path in self._field_paths:
            try:
                value = _get_path(redacted_document, path)
            except Exception:
                raise OpenSearchRedactionError(
                    "selected field could not be inspected"
                ) from None
            if value is _MISSING:
                if self.ignore_missing:
                    continue
                raise OpenSearchRedactionError("selected field is missing")

            try:
                replacement, seen, changed, spans = self._redact_value(value)
                _set_path(redacted_document, path, replacement)
            except OpenSearchRedactionError:
                raise
            except Exception:
                raise OpenSearchRedactionError(
                    "selected field could not be redacted"
                ) from None

            values_seen += seen
            values_redacted += changed
            spans_redacted += spans

        report = RedactionReport(
            policy=self.policy,
            fields=self.fields,
            values_seen=values_seen,
            values_redacted=values_redacted,
            spans_redacted=spans_redacted,
        )
        return redacted_document, report

    def _redact_value(self, value: Any) -> tuple[Any, int, int, int]:
        if value is None:
            return None, 0, 0, 0
        if isinstance(value, str):
            if not value:
                return value, 0, 0, 0
            redacted, spans = self._redact_text(value)
            return redacted, 1, int(redacted != value), spans
        if isinstance(value, (list, tuple)):
            redacted_values: list[Any] = []
            values_seen = 0
            values_redacted = 0
            spans_redacted = 0
            for item in value:
                replacement, seen, changed, spans = self._redact_value(item)
                redacted_values.append(replacement)
                values_seen += seen
                values_redacted += changed
                spans_redacted += spans
            if isinstance(value, tuple):
                return (
                    tuple(redacted_values),
                    values_seen,
                    values_redacted,
                    spans_redacted,
                )
            return redacted_values, values_seen, values_redacted, spans_redacted
        raise OpenSearchRedactionError("selected field must contain text")

    def _redact_text(self, text: str) -> tuple[str, int]:
        deidentifier = (
            self._deidentifier
            if self._deidentifier is not None
            else _default_deidentifier
        )
        try:
            result = deidentifier(text, **self._deidentify_options())
            redacted = _result_text(result)
            spans = _result_span_count(result, changed=redacted != text)
        except Exception:
            raise OpenSearchRedactionError("redaction failed") from None
        return redacted, spans

    def _deidentify_options(self) -> dict[str, Any]:
        options = dict(self._deidentify_kwargs)
        options["policy"] = self.policy
        options["method"] = self.method
        if self._deidentifier is None:
            options.setdefault("keep_mapping", False)
            options.setdefault("audit", False)
            options.setdefault("use_safety_sweep", True)
            options.setdefault("config", _offline_config())
        return options


def redact_document(
    document: Mapping[str, Any],
    *,
    fields: Sequence[str] | str | None = None,
    field: str | None = None,
    policy: str = _DEFAULT_POLICY,
    method: str = _DEFAULT_METHOD,
    ignore_missing: bool = False,
    deidentifier: Deidentifier | None = None,
    deidentify_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Redact one document using a one-shot processor configuration."""

    processor = OpenSearchRedactionProcessor(
        fields=fields,
        field=field,
        policy=policy,
        method=method,
        ignore_missing=ignore_missing,
        deidentifier=deidentifier,
        deidentify_kwargs=deidentify_kwargs,
    )
    return processor.process(document)


def _normalize_fields(fields: Sequence[str] | str | None) -> tuple[str, ...]:
    selected = _DEFAULT_FIELDS if fields is None else fields
    if isinstance(selected, str):
        selected = (selected,)
    elif not isinstance(selected, Sequence) or isinstance(selected, (bytes, bytearray)):
        raise OpenSearchRedactionError("fields must be a string or sequence")

    normalized: list[str] = []
    for field_name in selected:
        if not isinstance(field_name, str) or not field_name.strip():
            raise OpenSearchRedactionError("fields must contain field names")
        field_name = field_name.strip()
        if any(not part for part in field_name.split(".")):
            raise OpenSearchRedactionError("fields must contain valid field names")
        if field_name in normalized:
            raise OpenSearchRedactionError("fields must not contain duplicates")
        normalized.append(field_name)
    if not normalized:
        raise OpenSearchRedactionError("at least one field is required")
    return tuple(normalized)


def _normalize_policy(policy: str) -> str:
    try:
        return canonical_policy_name(policy)
    except Exception:
        raise OpenSearchRedactionError("policy is invalid") from None


def _normalize_method(method: str) -> str:
    if not isinstance(method, str) or not method.strip():
        raise OpenSearchRedactionError("method is invalid")
    return method.strip().lower()


def _get_path(document: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = document
    for segment in path:
        if not isinstance(current, Mapping) or segment not in current:
            return _MISSING
        current = current[segment]
    return current


def _set_path(document: dict[str, Any], path: Sequence[str], value: Any) -> None:
    current: Any = document
    for segment in path[:-1]:
        if not isinstance(current, dict) or segment not in current:
            raise OpenSearchRedactionError("selected field path is not an object")
        current = current[segment]
    if not isinstance(current, dict):
        raise OpenSearchRedactionError("selected field path is not an object")
    current[path[-1]] = value


def _default_deidentifier() -> Deidentifier:
    from openmed.core.pii import deidentify

    return deidentify


def _offline_config() -> Any:
    """Create a cache-only config without reading credential environment vars."""

    from openmed.core.config import OpenMedConfig

    return OpenMedConfig(local_only=True, hf_token="")


def _result_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, Mapping):
        text = result.get("deidentified_text")
    else:
        text = getattr(result, "deidentified_text", None)
    if not isinstance(text, str):
        raise TypeError("deidentifier must return deidentified_text as a string")
    return text


def _result_span_count(result: Any, *, changed: bool) -> int:
    if isinstance(result, Mapping):
        entities = result.get("pii_entities")
    else:
        entities = getattr(result, "pii_entities", None)
    if isinstance(entities, Sequence) and not isinstance(entities, (str, bytes)):
        return len(entities)
    return int(changed)


OpenMedRedactionProcessor = OpenSearchRedactionProcessor

__all__ = [
    "Deidentifier",
    "OpenMedRedactionProcessor",
    "OpenSearchRedactionError",
    "OpenSearchRedactionProcessor",
    "RedactionReport",
    "redact_document",
]
