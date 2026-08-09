"""Dependency-free Elasticsearch ingest redaction configuration.

The adapter builds an Elasticsearch ``redact`` ingest processor for an
explicit allow-list of fields.  It does not import the Elasticsearch client,
open a connection, or submit the pipeline.  A caller that wants to execute a
local redaction before indexing can inject a text redactor into
:meth:`ElasticsearchRedactionProcessor.process`.

Field selection is deliberately static.  Wildcards, templates, array
selectors, and mapping/list values are rejected instead of being expanded
implicitly.  Processor results expose only aggregate counters; source values
and redactor exception details never enter diagnostics or adapter exceptions.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

DEFAULT_ELASTICSEARCH_GROK_PATTERNS = (
    "%{EMAILADDRESS:openmed_email}",
    "%{IP:openmed_ip}",
    "%{URI:openmed_uri}",
)
"""Conservative built-in Grok patterns for the native ingest processor."""

DEFAULT_REDACTION_MARKER = "[REDACTED]"
DEFAULT_PIPELINE_ID = "openmed-redaction"
DEFAULT_PROCESSOR_TAG = "openmed-redaction"

TextRedactor: TypeAlias = Callable[[str], Any]
FieldSelection: TypeAlias = (
    Sequence[str]
    | Mapping[str, Sequence[str] | str]
    | Sequence["ElasticsearchFieldRule"]
)

_MISSING = object()
_NO_DEFAULT = object()
_DYNAMIC_PATH_MARKERS = frozenset("*?[]{}")


class ElasticsearchRedactionError(RuntimeError):
    """Raised when a configured ingest document cannot be redacted safely."""


class UnsupportedDynamicFieldError(ElasticsearchRedactionError, ValueError):
    """Raised when a dynamic field path or value is encountered."""


@dataclass(frozen=True)
class ElasticsearchFieldRule:
    """One explicit field and the Grok patterns applied to its text.

    ``field`` is a dotted path made only of literal field-name segments.
    ``patterns`` use Elasticsearch's Grok syntax and are passed through to the
    native ``redact`` processor after non-empty string validation.
    """

    field: str
    patterns: tuple[str, ...]
    ignore_missing: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "field", _normalize_field_path(self.field))
        object.__setattr__(self, "patterns", _normalize_patterns(self.patterns))
        _require_bool(self.ignore_missing, "ignore_missing")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible, source-value-free field rule."""

        return {
            "field": self.field,
            "patterns": list(self.patterns),
            "ignore_missing": self.ignore_missing,
        }


class ElasticsearchRedactionConfig:
    """Configuration for a deterministic Elasticsearch ingest pipeline.

    ``fields`` may be a sequence of field paths, in which case the shared
    ``patterns`` are used, or a mapping from each field path to its own Grok
    pattern sequence.  All selected fields must be explicit literal paths.

    The configuration only describes a pipeline; constructing it performs no
    network operation.  Use :meth:`to_ingest_pipeline` to obtain the body for
    the Elasticsearch ingest-pipeline API.
    """

    def __init__(
        self,
        fields: FieldSelection | None = None,
        *,
        selected_fields: FieldSelection | None = None,
        patterns: Sequence[str] | str = DEFAULT_ELASTICSEARCH_GROK_PATTERNS,
        ignore_missing: bool = True,
        pipeline_id: str = DEFAULT_PIPELINE_ID,
        processor_tag: str = DEFAULT_PROCESSOR_TAG,
        prefix: str = DEFAULT_REDACTION_MARKER,
        suffix: str = "",
    ) -> None:
        if fields is not None and selected_fields is not None:
            raise TypeError("provide fields or selected_fields, not both")
        raw_fields = fields if fields is not None else selected_fields
        if raw_fields is None:
            raise TypeError("fields are required")

        _require_bool(ignore_missing, "ignore_missing")
        normalized_patterns = _normalize_patterns(patterns)
        rules = _normalize_field_rules(
            raw_fields,
            default_patterns=normalized_patterns,
            ignore_missing=ignore_missing,
        )
        object.__setattr__(self, "_rules", rules)
        object.__setattr__(self, "_fields", tuple(rule.field for rule in rules))
        object.__setattr__(self, "_patterns", normalized_patterns)
        object.__setattr__(self, "ignore_missing", ignore_missing)
        object.__setattr__(self, "pipeline_id", _normalize_name(pipeline_id))
        object.__setattr__(self, "processor_tag", _normalize_name(processor_tag))
        object.__setattr__(self, "prefix", _normalize_marker(prefix, "prefix"))
        object.__setattr__(self, "suffix", _normalize_marker(suffix, "suffix"))

    @property
    def fields(self) -> tuple[str, ...]:
        """Return selected field paths in stable configuration order."""

        return self._fields

    @property
    def selected_fields(self) -> tuple[str, ...]:
        """Alias for :attr:`fields` used by configuration-oriented callers."""

        return self._fields

    @property
    def field_rules(self) -> tuple[ElasticsearchFieldRule, ...]:
        """Return immutable field-specific processor rules."""

        return self._rules

    @property
    def patterns(self) -> tuple[str, ...]:
        """Return the shared default pattern set."""

        return self._patterns

    def to_ingest_pipeline(self) -> dict[str, Any]:
        """Return the native Elasticsearch ingest-pipeline request body.

        Each selected field becomes one ``redact`` processor.  The output is
        composed solely from configuration metadata and is stable across
        calls, making it suitable for checked-in deployment manifests.
        """

        processors = []
        for index, rule in enumerate(self._rules):
            redact_options: dict[str, Any] = {
                "field": rule.field,
                "patterns": list(rule.patterns),
                "prefix": self.prefix,
                "suffix": self.suffix,
                "ignore_missing": rule.ignore_missing,
                "tag": f"{self.processor_tag}-{index}",
            }
            processors.append({"redact": redact_options})

        return {
            "description": (
                "OpenMed redaction for explicitly configured Elasticsearch fields"
            ),
            "processors": processors,
        }

    to_pipeline = to_ingest_pipeline

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible configuration summary."""

        return {
            "pipeline_id": self.pipeline_id,
            "processor_tag": self.processor_tag,
            "fields": [rule.to_dict() for rule in self._rules],
            "prefix": self.prefix,
            "suffix": self.suffix,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize the ingest pipeline deterministically as JSON."""

        return json.dumps(
            self.to_ingest_pipeline(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
        )


@dataclass(frozen=True)
class ElasticsearchProcessorDiagnostics:
    """Counts-only diagnostics for one local processor invocation."""

    documents_processed: int = 0
    fields_configured: int = 0
    fields_processed: int = 0
    fields_redacted: int = 0
    fields_skipped: int = 0
    spans_redacted: int = 0
    dynamic_fields_rejected: int = 0

    def __post_init__(self) -> None:
        for name in (
            "documents_processed",
            "fields_configured",
            "fields_processed",
            "fields_redacted",
            "fields_skipped",
            "spans_redacted",
            "dynamic_fields_rejected",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must not be negative")

    def to_dict(self) -> dict[str, int]:
        """Return only numeric diagnostics; no source values are retained."""

        return {
            "documents_processed": self.documents_processed,
            "fields_configured": self.fields_configured,
            "fields_processed": self.fields_processed,
            "fields_redacted": self.fields_redacted,
            "fields_skipped": self.fields_skipped,
            "spans_redacted": self.spans_redacted,
            "dynamic_fields_rejected": self.dynamic_fields_rejected,
        }

    def __getitem__(self, key: str) -> int:
        """Allow count access using the stable diagnostic field names."""

        return self.to_dict()[key]


@dataclass(frozen=True)
class ElasticsearchRedactionResult:
    """Redacted document plus a count-only diagnostic summary."""

    document: dict[str, Any]
    diagnostics: ElasticsearchProcessorDiagnostics

    def to_dict(self) -> dict[str, int]:
        """Return the PHI-safe summary, not the document payload."""

        return self.diagnostics.to_dict()


class ElasticsearchRedactionProcessor:
    """Build and optionally execute an explicit Elasticsearch redaction plan.

    The native execution path is :meth:`to_ingest_pipeline`; Elasticsearch
    applies that body inside its ingest pipeline.  ``process`` is a local,
    dependency-free path for applications that redact before indexing.  It
    requires a caller-provided ``redactor`` so model loading and network
    policy remain explicit and testable.
    """

    def __init__(
        self,
        config: ElasticsearchRedactionConfig | FieldSelection | None = None,
        *,
        fields: FieldSelection | None = None,
        selected_fields: FieldSelection | None = None,
        patterns: Sequence[str] | str = DEFAULT_ELASTICSEARCH_GROK_PATTERNS,
        ignore_missing: bool = True,
        pipeline_id: str = DEFAULT_PIPELINE_ID,
        processor_tag: str = DEFAULT_PROCESSOR_TAG,
        prefix: str = DEFAULT_REDACTION_MARKER,
        suffix: str = "",
        redactor: TextRedactor | None = None,
    ) -> None:
        supplied_fields = [item is not None for item in (fields, selected_fields)]
        if sum(supplied_fields) > 1:
            raise TypeError("provide fields or selected_fields, not both")

        if isinstance(config, ElasticsearchRedactionConfig):
            if any(supplied_fields):
                raise TypeError("config cannot be combined with fields")
            resolved_config = config
        else:
            if config is not None and any(supplied_fields):
                raise TypeError("provide positional fields or keyword fields, not both")
            resolved_fields = (
                fields
                if fields is not None
                else selected_fields
                if selected_fields is not None
                else config
            )
            if resolved_fields is None:
                raise TypeError("config or fields are required")
            resolved_config = ElasticsearchRedactionConfig(
                resolved_fields,
                patterns=patterns,
                ignore_missing=ignore_missing,
                pipeline_id=pipeline_id,
                processor_tag=processor_tag,
                prefix=prefix,
                suffix=suffix,
            )

        if redactor is not None and not callable(redactor):
            raise TypeError("redactor must be callable")
        self.config = resolved_config
        self._redactor = redactor

    @property
    def fields(self) -> tuple[str, ...]:
        """Return the explicit field allow-list."""

        return self.config.fields

    def to_ingest_pipeline(self) -> dict[str, Any]:
        """Return the native Elasticsearch ingest-pipeline request body."""

        return self.config.to_ingest_pipeline()

    to_pipeline = to_ingest_pipeline

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize the native ingest-pipeline request body deterministically."""

        return self.config.to_json(indent=indent)

    def diagnose(
        self,
        document: Mapping[str, Any],
    ) -> ElasticsearchProcessorDiagnostics:
        """Validate one document and return aggregate field diagnostics.

        Missing, null, and empty selected fields are counted as skipped.  A
        selected mapping, sequence, or other non-string value is rejected so
        dynamic data cannot be traversed or coerced implicitly.
        """

        if not isinstance(document, Mapping):
            raise TypeError("ingest document must be a mapping")

        counts = self._inspect_document(document)
        return ElasticsearchProcessorDiagnostics(
            documents_processed=1,
            fields_configured=len(self.config.field_rules),
            fields_processed=counts["fields_processed"],
            fields_redacted=0,
            fields_skipped=counts["fields_skipped"],
            spans_redacted=0,
            dynamic_fields_rejected=counts["dynamic_fields_rejected"],
        )

    validate = diagnose

    def process(
        self,
        document: Mapping[str, Any],
        *,
        redactor: TextRedactor | None = None,
    ) -> ElasticsearchRedactionResult:
        """Redact selected text fields using an injected local callable.

        The input mapping is copied before processing.  The callable receives
        one source string and may return a string or an object/mapping with a
        ``deidentified_text`` value.  Exceptions from that callable are
        replaced by a stable value-free :class:`ElasticsearchRedactionError`.
        """

        if not isinstance(document, Mapping):
            raise TypeError("ingest document must be a mapping")
        callback = redactor if redactor is not None else self._redactor
        if callback is None:
            raise ElasticsearchRedactionError(
                "a local redactor callback is required for process()"
            )

        try:
            output = copy.deepcopy(dict(document))
        except Exception:
            raise ElasticsearchRedactionError(
                "failed to copy the ingest document"
            ) from None

        counts = self._inspect_document(output)
        fields_redacted = 0
        spans_redacted = 0

        for rule in self.config.field_rules:
            path = _resolve_document_path(output, rule.field)
            if path is None:
                if not rule.ignore_missing:
                    raise ElasticsearchRedactionError(
                        "a configured ingest field is missing"
                    )
                continue

            value = _get_path(output, path)
            if value is None or value == "":
                continue
            if not isinstance(value, str):
                raise UnsupportedDynamicFieldError(
                    "dynamic or non-string configured fields are unsupported"
                )

            try:
                redaction_result = callback(value)
                redacted_text, span_count = _coerce_redacted_result(
                    redaction_result,
                    changed=value,
                )
            except ElasticsearchRedactionError:
                raise
            except Exception:
                raise ElasticsearchRedactionError(
                    "failed to redact a configured ingest field"
                ) from None

            _set_path(output, path, redacted_text)
            if redacted_text != value:
                fields_redacted += 1
            spans_redacted += span_count

        diagnostics = ElasticsearchProcessorDiagnostics(
            documents_processed=1,
            fields_configured=len(self.config.field_rules),
            fields_processed=counts["fields_processed"],
            fields_redacted=fields_redacted,
            fields_skipped=counts["fields_skipped"],
            spans_redacted=spans_redacted,
            dynamic_fields_rejected=counts["dynamic_fields_rejected"],
        )
        return ElasticsearchRedactionResult(output, diagnostics)

    run = process
    redact = process

    def _inspect_document(self, document: Mapping[str, Any]) -> dict[str, int]:
        fields_processed = 0
        fields_skipped = 0
        dynamic_fields_rejected = 0

        for rule in self.config.field_rules:
            path = _resolve_document_path(document, rule.field)
            if path is None:
                if not rule.ignore_missing:
                    raise ElasticsearchRedactionError(
                        "a configured ingest field is missing"
                    )
                fields_skipped += 1
                continue

            value = _get_path(document, path)
            if value is None or value == "":
                fields_skipped += 1
                continue
            if not isinstance(value, str):
                dynamic_fields_rejected += 1
                raise UnsupportedDynamicFieldError(
                    "dynamic or non-string configured fields are unsupported"
                )
            fields_processed += 1

        return {
            "fields_processed": fields_processed,
            "fields_skipped": fields_skipped,
            "dynamic_fields_rejected": dynamic_fields_rejected,
        }


def build_ingest_pipeline(
    fields: FieldSelection,
    *,
    patterns: Sequence[str] | str = DEFAULT_ELASTICSEARCH_GROK_PATTERNS,
    ignore_missing: bool = True,
    pipeline_id: str = DEFAULT_PIPELINE_ID,
    processor_tag: str = DEFAULT_PROCESSOR_TAG,
    prefix: str = DEFAULT_REDACTION_MARKER,
    suffix: str = "",
) -> dict[str, Any]:
    """Build a native Elasticsearch ingest-pipeline request body offline."""

    return ElasticsearchRedactionConfig(
        fields,
        patterns=patterns,
        ignore_missing=ignore_missing,
        pipeline_id=pipeline_id,
        processor_tag=processor_tag,
        prefix=prefix,
        suffix=suffix,
    ).to_ingest_pipeline()


create_ingest_pipeline = build_ingest_pipeline


def redact_ingest_document(
    document: Mapping[str, Any],
    *,
    fields: FieldSelection,
    redactor: TextRedactor,
    patterns: Sequence[str] | str = DEFAULT_ELASTICSEARCH_GROK_PATTERNS,
    ignore_missing: bool = True,
    pipeline_id: str = DEFAULT_PIPELINE_ID,
    processor_tag: str = DEFAULT_PROCESSOR_TAG,
    prefix: str = DEFAULT_REDACTION_MARKER,
    suffix: str = "",
) -> ElasticsearchRedactionResult:
    """Redact one ingest document with an injected local text redactor."""

    return ElasticsearchRedactionProcessor(
        fields,
        patterns=patterns,
        ignore_missing=ignore_missing,
        pipeline_id=pipeline_id,
        processor_tag=processor_tag,
        prefix=prefix,
        suffix=suffix,
        redactor=redactor,
    ).process(document)


def _normalize_field_rules(
    fields: FieldSelection,
    *,
    default_patterns: tuple[str, ...],
    ignore_missing: bool,
) -> tuple[ElasticsearchFieldRule, ...]:
    if isinstance(fields, ElasticsearchFieldRule):
        raw_items = ((fields, default_patterns),)
    elif isinstance(fields, Mapping):
        raw_items = tuple(fields.items())
    else:
        if isinstance(fields, (str, bytes)):
            raise TypeError("fields must be a sequence or mapping of explicit paths")
        raw_items = tuple((item, default_patterns) for item in fields)

    rules: list[ElasticsearchFieldRule] = []
    seen: set[str] = set()
    for raw_field, raw_patterns in raw_items:
        if isinstance(raw_field, ElasticsearchFieldRule):
            rule = raw_field
        else:
            field_path = _normalize_field_path(raw_field)
            if isinstance(raw_patterns, ElasticsearchFieldRule):
                rule = raw_patterns
                if rule.field != field_path:
                    raise ValueError("field rule path does not match its mapping key")
            else:
                rule = ElasticsearchFieldRule(
                    field=field_path,
                    patterns=(
                        default_patterns
                        if raw_patterns is None
                        else _normalize_patterns(raw_patterns)
                    ),
                    ignore_missing=ignore_missing,
                )

        if rule.field in seen:
            continue
        seen.add(rule.field)
        rules.append(rule)

    if not rules:
        raise ValueError("fields must include at least one explicit field path")
    return tuple(rules)


def _normalize_field_path(field: Any) -> str:
    if not isinstance(field, str):
        raise TypeError("selected fields must contain strings")
    normalized = field.strip()
    if not normalized:
        raise ValueError("selected field paths must not be empty")
    if any(not part for part in normalized.split(".")):
        raise ValueError("selected field paths must use non-empty segments")
    if any(marker in normalized for marker in _DYNAMIC_PATH_MARKERS):
        raise UnsupportedDynamicFieldError(
            "dynamic field paths are unsupported; use explicit field paths"
        )
    return normalized


def _normalize_patterns(patterns: Sequence[str] | str) -> tuple[str, ...]:
    raw_patterns = (patterns,) if isinstance(patterns, str) else tuple(patterns)
    normalized: list[str] = []
    for pattern in raw_patterns:
        if not isinstance(pattern, str):
            raise TypeError("redaction patterns must contain strings")
        pattern = pattern.strip()
        if not pattern:
            raise ValueError("redaction patterns must not be empty")
        if pattern not in normalized:
            normalized.append(pattern)
    if not normalized:
        raise ValueError("redaction patterns must include at least one pattern")
    return tuple(normalized)


def _normalize_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("pipeline names and processor tags must be non-empty strings")
    return value.strip()


def _normalize_marker(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _require_bool(value: Any, name: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")


def _resolve_document_path(
    document: Mapping[str, Any],
    field: str,
) -> tuple[str, ...] | None:
    parts = tuple(field.split("."))
    if parts[0] == "_source":
        return parts if _path_exists(document, parts) else None

    if field in document:
        return (field,)

    source = document.get("_source")
    if isinstance(source, Mapping):
        source_path = ("_source", *parts)
        if _path_exists(document, source_path):
            return source_path

    return parts if _path_exists(document, parts) else None


def _path_exists(document: Mapping[str, Any], path: Sequence[str]) -> bool:
    return _get_path(document, path, default=_MISSING) is not _MISSING


def _get_path(
    document: Mapping[str, Any],
    path: Sequence[str],
    *,
    default: Any = _NO_DEFAULT,
) -> Any:
    current: Any = document
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            if default is not _NO_DEFAULT:
                return default
            raise KeyError(part)
        current = current[part]
    return current


def _set_path(document: dict[str, Any], path: Sequence[str], value: str) -> None:
    current: dict[str, Any] = document
    for part in path[:-1]:
        nested = current.get(part)
        if not isinstance(nested, dict):
            raise UnsupportedDynamicFieldError(
                "configured ingest field path is not an object"
            )
        current = nested
    current[path[-1]] = value


def _coerce_redacted_result(result: Any, *, changed: str) -> tuple[str, int]:
    if isinstance(result, str):
        redacted_text = result
        entities: Any = None
    elif isinstance(result, Mapping):
        redacted_text = result.get("deidentified_text")
        entities = result.get("pii_entities", result.get("entities"))
    else:
        redacted_text = getattr(result, "deidentified_text", None)
        entities = getattr(result, "pii_entities", getattr(result, "entities", None))

    if not isinstance(redacted_text, str):
        raise ElasticsearchRedactionError(
            "redactor must return text or deidentified_text"
        )

    try:
        span_count = len(entities) if entities is not None else 0
    except TypeError:
        span_count = 0
    if span_count == 0 and redacted_text != changed:
        span_count = 1
    return redacted_text, span_count


__all__ = [
    "DEFAULT_ELASTICSEARCH_GROK_PATTERNS",
    "DEFAULT_PIPELINE_ID",
    "DEFAULT_PROCESSOR_TAG",
    "DEFAULT_REDACTION_MARKER",
    "ElasticsearchFieldRule",
    "ElasticsearchProcessorDiagnostics",
    "ElasticsearchRedactionConfig",
    "ElasticsearchRedactionError",
    "ElasticsearchRedactionProcessor",
    "ElasticsearchRedactionResult",
    "UnsupportedDynamicFieldError",
    "build_ingest_pipeline",
    "create_ingest_pipeline",
    "redact_ingest_document",
]
