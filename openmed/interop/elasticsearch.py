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

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias, cast

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

_MISSING = object()
_NO_DEFAULT = object()
_DYNAMIC_PATH_MARKERS = frozenset("*?[]{}")
_MAX_DOCUMENT_DEPTH = 32
_MAX_DOCUMENT_ITEMS = 10_000
_MAX_DOCUMENT_KEY_CHARS = 4_096
_MAX_DOCUMENT_TOTAL_BYTES = 32 * 1024 * 1024
_MAX_FIELDS = 64
_MAX_FIELD_PATH_CHARS = 512
_MAX_FIELD_SEGMENTS = 32
_MAX_FIELD_SEGMENT_CHARS = 128
_MAX_JSON_INDENT = 16
_MAX_MARKER_CHARS = 1_024
_MAX_NAME_CHARS = 255
_MAX_OUTPUT_EXPANSION = 8
_MIN_OUTPUT_CHARS = 4_096
_MAX_PATTERNS_PER_FIELD = 64
_MAX_PATTERN_CHARS = 4_096
_MAX_SPANS_PER_FIELD = 10_000
_MAX_TEXT_CHARS = 10 * 1024 * 1024
_MIN_INTEGER = -(1 << 63)
_MAX_INTEGER = (1 << 63) - 1


class ElasticsearchRedactionError(RuntimeError):
    """Raised when a configured ingest document cannot be redacted safely."""


class UnsupportedDynamicFieldError(ElasticsearchRedactionError, ValueError):
    """Raised when a dynamic field path or value is encountered."""


class _BoundaryError(Exception):
    """Internal marker for a rejected untrusted input boundary."""


@dataclass(frozen=True, slots=True)
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

        safe = _validated_field_rule(self)
        return {
            "field": safe.field,
            "patterns": list(safe.patterns),
            "ignore_missing": safe.ignore_missing,
        }


FieldSelection: TypeAlias = (
    ElasticsearchFieldRule
    | Sequence[str]
    | Mapping[str, Sequence[str] | str | ElasticsearchFieldRule]
    | Sequence[ElasticsearchFieldRule]
)


@dataclass(frozen=True, init=False, slots=True)
class ElasticsearchRedactionConfig:
    """Configuration for a deterministic Elasticsearch ingest pipeline.

    ``fields`` may be a sequence of field paths, in which case the shared
    ``patterns`` are used, or a mapping from each field path to its own Grok
    pattern sequence.  All selected fields must be explicit literal paths.

    The configuration only describes a pipeline; constructing it performs no
    network operation.  Use :meth:`to_ingest_pipeline` to obtain the body for
    the Elasticsearch ingest-pipeline API.
    """

    _fields: tuple[str, ...]
    _patterns: tuple[str, ...]
    _rules: tuple[ElasticsearchFieldRule, ...]
    ignore_missing: bool
    pipeline_id: str
    prefix: str
    processor_tag: str
    suffix: str

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

        return _validated_config(self)._fields

    @property
    def selected_fields(self) -> tuple[str, ...]:
        """Alias for :attr:`fields` used by configuration-oriented callers."""

        return _validated_config(self)._fields

    @property
    def field_rules(self) -> tuple[ElasticsearchFieldRule, ...]:
        """Return immutable field-specific processor rules."""

        return _validated_config(self)._rules

    @property
    def patterns(self) -> tuple[str, ...]:
        """Return the shared default pattern set."""

        return _validated_config(self)._patterns

    def to_ingest_pipeline(self) -> dict[str, Any]:
        """Return the native Elasticsearch ingest-pipeline request body.

        Each selected field becomes one ``redact`` processor.  The output is
        composed solely from configuration metadata and is stable across
        calls, making it suitable for checked-in deployment manifests.
        """

        return _validated_config(self)._to_ingest_pipeline_unchecked()

    def _to_ingest_pipeline_unchecked(self) -> dict[str, Any]:
        """Build a pipeline after this configuration has been reconstructed."""

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

        safe = _validated_config(self)
        return {
            "pipeline_id": safe.pipeline_id,
            "processor_tag": safe.processor_tag,
            "fields": [rule.to_dict() for rule in safe._rules],
            "prefix": safe.prefix,
            "suffix": safe.suffix,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize the ingest pipeline deterministically as JSON."""

        if indent is not None and (
            type(indent) is not int or indent < 0 or indent > _MAX_JSON_INDENT
        ):
            raise ValueError("indent must be bounded and non-negative")

        safe = _validated_config(self)
        return json.dumps(
            safe._to_ingest_pipeline_unchecked(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
        )


@dataclass(frozen=True, slots=True)
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
        values: dict[str, int] = {}
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
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must not be negative")
            values[name] = value
        if values["documents_processed"] > 1:
            raise ValueError("documents_processed exceeds the per-call limit")
        for name in (
            "fields_configured",
            "fields_processed",
            "fields_redacted",
            "fields_skipped",
            "dynamic_fields_rejected",
        ):
            if values[name] > _MAX_FIELDS:
                raise ValueError(f"{name} exceeds the configured field limit")
        if values["spans_redacted"] > _MAX_FIELDS * _MAX_SPANS_PER_FIELD:
            raise ValueError("spans_redacted exceeds the per-call limit")
        if values["fields_redacted"] > values["fields_processed"]:
            raise ValueError("fields_redacted exceeds fields_processed")
        accounted = (
            values["fields_processed"]
            + values["fields_skipped"]
            + values["dynamic_fields_rejected"]
        )
        if accounted > values["fields_configured"]:
            raise ValueError("diagnostic field counts are inconsistent")
        if values["documents_processed"] == 0 and any(
            value for name, value in values.items() if name != "documents_processed"
        ):
            raise ValueError("empty diagnostics must not contain field counts")

    def to_dict(self) -> dict[str, int]:
        """Return only numeric diagnostics; no source values are retained."""

        safe = _validated_diagnostics(self)
        return {
            "documents_processed": safe.documents_processed,
            "fields_configured": safe.fields_configured,
            "fields_processed": safe.fields_processed,
            "fields_redacted": safe.fields_redacted,
            "fields_skipped": safe.fields_skipped,
            "spans_redacted": safe.spans_redacted,
            "dynamic_fields_rejected": safe.dynamic_fields_rejected,
        }

    def __getitem__(self, key: str) -> int:
        """Allow count access using the stable diagnostic field names."""

        return self.to_dict()[key]


@dataclass(frozen=True, slots=True)
class ElasticsearchRedactionResult:
    """Redacted document plus a count-only diagnostic summary."""

    document: dict[str, Any] = field(repr=False)
    diagnostics: ElasticsearchProcessorDiagnostics

    def __post_init__(self) -> None:
        if type(self.document) is not dict:
            raise TypeError("document must be a dictionary")
        object.__setattr__(
            self, "diagnostics", _validated_diagnostics(self.diagnostics)
        )

    def to_dict(self) -> dict[str, int]:
        """Return the PHI-safe summary, not the document payload."""

        return _validated_diagnostics(self.diagnostics).to_dict()


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

        if type(config) is ElasticsearchRedactionConfig:
            if any(supplied_fields):
                raise TypeError("config cannot be combined with fields")
            resolved_config = _validated_config(config)
        else:
            if config is not None and any(supplied_fields):
                raise TypeError("provide positional fields or keyword fields, not both")
            resolved_fields = (
                fields
                if fields is not None
                else selected_fields
                if selected_fields is not None
                else cast(FieldSelection | None, config)
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
        self.config = _validated_config(resolved_config)
        self._redactor = redactor

    @property
    def fields(self) -> tuple[str, ...]:
        """Return the explicit field allow-list."""

        return _processor_config(self)._fields

    def to_ingest_pipeline(self) -> dict[str, Any]:
        """Return the native Elasticsearch ingest-pipeline request body."""

        return _processor_config(self)._to_ingest_pipeline_unchecked()

    to_pipeline = to_ingest_pipeline

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize the native ingest-pipeline request body deterministically."""

        return _processor_config(self).to_json(indent=indent)

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

        config = _processor_config(self)
        rules = config._rules
        inspected_document = _copy_ingest_document(document)
        counts = self._inspect_document(inspected_document, rules)
        return ElasticsearchProcessorDiagnostics(
            documents_processed=1,
            fields_configured=len(rules),
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
        config = _processor_config(self)
        rules = config._rules
        callback = redactor if redactor is not None else self._redactor
        if callback is None:
            raise ElasticsearchRedactionError(
                "a local redactor callback is required for process()"
            )
        if not callable(callback):
            raise TypeError("redactor must be callable")

        output = _copy_ingest_document(document)

        counts = self._inspect_document(output, rules)
        fields_redacted = 0
        spans_redacted = 0

        for rule in rules:
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
            if type(value) is not str:
                raise UnsupportedDynamicFieldError(
                    "dynamic or non-string configured fields are unsupported"
                )

            try:
                redaction_result = callback(value)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException:
                raise ElasticsearchRedactionError(
                    "failed to redact a configured ingest field"
                ) from None
            try:
                redacted_text, span_count = _coerce_redacted_result(
                    redaction_result,
                    original=value,
                )
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException:
                raise ElasticsearchRedactionError(
                    "failed to redact a configured ingest field"
                ) from None

            _set_path(output, path, redacted_text)
            if redacted_text != value:
                fields_redacted += 1
            spans_redacted += span_count

        diagnostics = ElasticsearchProcessorDiagnostics(
            documents_processed=1,
            fields_configured=len(rules),
            fields_processed=counts["fields_processed"],
            fields_redacted=fields_redacted,
            fields_skipped=counts["fields_skipped"],
            spans_redacted=spans_redacted,
            dynamic_fields_rejected=counts["dynamic_fields_rejected"],
        )
        return ElasticsearchRedactionResult(_copy_ingest_document(output), diagnostics)

    run = process
    redact = process

    def _inspect_document(
        self,
        document: Mapping[str, Any],
        rules: tuple[ElasticsearchFieldRule, ...],
    ) -> dict[str, int]:
        fields_processed = 0
        fields_skipped = 0
        dynamic_fields_rejected = 0

        for rule in rules:
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
            if type(value) is not str:
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


def _validated_field_rule(rule: Any) -> ElasticsearchFieldRule:
    if type(rule) is not ElasticsearchFieldRule:
        raise ValueError("Elasticsearch field rule is invalid")
    try:
        return ElasticsearchFieldRule(
            field=object.__getattribute__(rule, "field"),
            patterns=object.__getattribute__(rule, "patterns"),
            ignore_missing=object.__getattribute__(rule, "ignore_missing"),
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise ValueError("Elasticsearch field rule is invalid") from None


def _validated_config(config: Any) -> ElasticsearchRedactionConfig:
    if type(config) is not ElasticsearchRedactionConfig:
        raise ValueError("Elasticsearch configuration is invalid")
    try:
        raw_rules = object.__getattribute__(config, "_rules")
        raw_fields = object.__getattribute__(config, "_fields")
        raw_patterns = object.__getattribute__(config, "_patterns")
        if (
            type(raw_rules) is not tuple
            or not raw_rules
            or len(raw_rules) > _MAX_FIELDS
            or type(raw_fields) is not tuple
            or type(raw_patterns) is not tuple
        ):
            raise ValueError
        rules = tuple(_validated_field_rule(rule) for rule in raw_rules)
        fields = tuple(rule.field for rule in rules)
        patterns = _normalize_patterns(raw_patterns)
        if raw_fields != fields or raw_patterns != patterns:
            raise ValueError

        ignore_missing = object.__getattribute__(config, "ignore_missing")
        _require_bool(ignore_missing, "ignore_missing")
        pipeline_id = _normalize_name(object.__getattribute__(config, "pipeline_id"))
        processor_tag = _normalize_name(
            object.__getattribute__(config, "processor_tag")
        )
        prefix = _normalize_marker(object.__getattribute__(config, "prefix"), "prefix")
        suffix = _normalize_marker(object.__getattribute__(config, "suffix"), "suffix")
        return ElasticsearchRedactionConfig(
            rules,
            patterns=patterns,
            ignore_missing=ignore_missing,
            pipeline_id=pipeline_id,
            processor_tag=processor_tag,
            prefix=prefix,
            suffix=suffix,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise ValueError("Elasticsearch configuration is invalid") from None


def _validated_diagnostics(value: Any) -> ElasticsearchProcessorDiagnostics:
    if type(value) is not ElasticsearchProcessorDiagnostics:
        raise ValueError("processor diagnostics are invalid")
    try:
        return ElasticsearchProcessorDiagnostics(
            documents_processed=object.__getattribute__(value, "documents_processed"),
            fields_configured=object.__getattribute__(value, "fields_configured"),
            fields_processed=object.__getattribute__(value, "fields_processed"),
            fields_redacted=object.__getattribute__(value, "fields_redacted"),
            fields_skipped=object.__getattribute__(value, "fields_skipped"),
            spans_redacted=object.__getattribute__(value, "spans_redacted"),
            dynamic_fields_rejected=object.__getattribute__(
                value, "dynamic_fields_rejected"
            ),
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise ValueError("processor diagnostics are invalid") from None


def _processor_config(processor: Any) -> ElasticsearchRedactionConfig:
    if not isinstance(processor, ElasticsearchRedactionProcessor):
        raise ElasticsearchRedactionError("processor configuration is invalid")
    try:
        return _validated_config(processor.config)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise ElasticsearchRedactionError(
            "processor configuration is invalid"
        ) from None


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
    iterator: Any
    if type(fields) is ElasticsearchFieldRule:
        iterator = iter(((fields, default_patterns),))
        mapping_entries = True
    elif isinstance(fields, Mapping):
        try:
            iterator = iter(fields.items())
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise ValueError("fields could not be inspected") from None
        mapping_entries = True
    else:
        if isinstance(fields, (str, bytes, bytearray)) or not isinstance(
            fields, Sequence
        ):
            raise TypeError("fields must be a sequence or mapping of explicit paths")
        try:
            iterator = iter(fields)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise ValueError("fields could not be inspected") from None
        mapping_entries = False

    rules: list[ElasticsearchFieldRule] = []
    seen: set[str] = set()
    for index in range(_MAX_FIELDS + 1):
        try:
            entry = next(iterator)
        except StopIteration:
            break
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise ValueError("fields could not be inspected") from None
        if index == _MAX_FIELDS:
            raise ValueError("fields contain too many entries")
        raw_field: Any
        raw_patterns: Any
        if mapping_entries:
            try:
                raw_field, raw_patterns = entry
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException:
                raise ValueError("fields could not be inspected") from None
        else:
            raw_field = entry
            raw_patterns = default_patterns

        if type(raw_field) is ElasticsearchFieldRule:
            rule = _validated_field_rule(raw_field)
        else:
            field_path = _normalize_field_path(raw_field)
            if type(raw_patterns) is ElasticsearchFieldRule:
                rule = _validated_field_rule(raw_patterns)
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
            raise ValueError("fields must not contain duplicate paths")
        seen.add(rule.field)
        rules.append(rule)

    if not rules:
        raise ValueError("fields must include at least one explicit field path")
    return tuple(rules)


def _normalize_field_path(field: Any) -> str:
    if type(field) is not str:
        raise TypeError("selected fields must contain strings")
    normalized = field.strip()
    if not normalized:
        raise ValueError("selected field paths must not be empty")
    parts = normalized.split(".")
    if (
        len(normalized) > _MAX_FIELD_PATH_CHARS
        or len(parts) > _MAX_FIELD_SEGMENTS
        or any(
            not part or len(part) > _MAX_FIELD_SEGMENT_CHARS or not part.isprintable()
            for part in parts
        )
    ):
        raise ValueError("selected field paths must use bounded literal segments")
    if any(marker in normalized for marker in _DYNAMIC_PATH_MARKERS):
        raise UnsupportedDynamicFieldError(
            "dynamic field paths are unsupported; use explicit field paths"
        )
    return normalized


def _normalize_patterns(patterns: Sequence[str] | str) -> tuple[str, ...]:
    if type(patterns) is str:
        iterator = iter((patterns,))
    elif isinstance(patterns, (bytes, bytearray)) or not isinstance(patterns, Sequence):
        raise TypeError("redaction patterns must be a string or sequence")
    else:
        try:
            iterator = iter(patterns)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise ValueError("redaction patterns could not be inspected") from None

    normalized: list[str] = []
    seen: set[str] = set()
    for index in range(_MAX_PATTERNS_PER_FIELD + 1):
        try:
            pattern = next(iterator)
        except StopIteration:
            break
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise ValueError("redaction patterns could not be inspected") from None
        if index == _MAX_PATTERNS_PER_FIELD:
            raise ValueError("redaction patterns contain too many entries")
        if type(pattern) is not str:
            raise TypeError("redaction patterns must contain strings")
        pattern = pattern.strip()
        if (
            not pattern
            or len(pattern) > _MAX_PATTERN_CHARS
            or not pattern.isprintable()
        ):
            raise ValueError("redaction patterns must contain bounded text")
        if pattern not in seen:
            normalized.append(pattern)
            seen.add(pattern)
    if not normalized:
        raise ValueError("redaction patterns must include at least one pattern")
    return tuple(normalized)


def _normalize_name(value: Any) -> str:
    if type(value) is not str:
        raise ValueError("pipeline names and processor tags must be bounded strings")
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > _MAX_NAME_CHARS
        or not normalized.isprintable()
    ):
        raise ValueError("pipeline names and processor tags must be bounded strings")
    return normalized


def _normalize_marker(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or len(value) > _MAX_MARKER_CHARS
        or (value and not value.isprintable())
    ):
        raise TypeError(f"{name} must be a bounded string")
    return value


def _require_bool(value: Any, name: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")


def _copy_ingest_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return a bounded, detached copy of one JSON-compatible document."""

    try:
        copied = _copy_document_value(
            document,
            depth=0,
            item_count=[0],
            byte_budget=[0],
            active_containers=set(),
        )
        if type(copied) is not dict:
            raise _BoundaryError
        return copied
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        raise ElasticsearchRedactionError(
            "failed to copy the ingest document"
        ) from None


def _copy_document_value(
    value: Any,
    *,
    depth: int,
    item_count: list[int],
    byte_budget: list[int],
    active_containers: set[int],
) -> Any:
    if depth > _MAX_DOCUMENT_DEPTH:
        raise _BoundaryError
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        if value < _MIN_INTEGER or value > _MAX_INTEGER:
            raise _BoundaryError
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise _BoundaryError
        return value
    if type(value) is str:
        if len(value) > _MAX_TEXT_CHARS:
            raise _BoundaryError
        _consume_document_text(value, byte_budget)
        return value

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
                if item_count[0] >= _MAX_DOCUMENT_ITEMS:
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
                    or len(key) > _MAX_DOCUMENT_KEY_CHARS
                    or key in copied_mapping
                ):
                    raise _BoundaryError
                _consume_document_text(key, byte_budget)
                copied_mapping[key] = _copy_document_value(
                    item,
                    depth=depth + 1,
                    item_count=item_count,
                    byte_budget=byte_budget,
                    active_containers=active_containers,
                )
        finally:
            active_containers.discard(marker)

    if type(value) in (list, tuple):
        marker = id(value)
        if marker in active_containers:
            raise _BoundaryError
        if len(value) > _MAX_DOCUMENT_ITEMS - item_count[0]:
            raise _BoundaryError
        active_containers.add(marker)
        try:
            copied_values = []
            for item in value:
                item_count[0] += 1
                copied_values.append(
                    _copy_document_value(
                        item,
                        depth=depth + 1,
                        item_count=item_count,
                        byte_budget=byte_budget,
                        active_containers=active_containers,
                    )
                )
        finally:
            active_containers.discard(marker)
        return tuple(copied_values) if type(value) is tuple else copied_values

    raise _BoundaryError


def _consume_document_text(value: str, byte_budget: list[int]) -> None:
    try:
        byte_budget[0] += len(value.encode("utf-8"))
    except UnicodeError:
        raise _BoundaryError from None
    if byte_budget[0] > _MAX_DOCUMENT_TOTAL_BYTES:
        raise _BoundaryError


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


def _coerce_redacted_result(result: Any, *, original: str) -> tuple[str, int]:
    redacted_text: Any
    entities: Any
    if type(result) is str:
        redacted_text = result
        entities = None
    elif isinstance(result, Mapping):
        redacted_text = result.get("deidentified_text")
        entities = _result_entities(result)
    else:
        redacted_text = getattr(result, "deidentified_text", None)
        entities = _result_entities(result)

    if type(redacted_text) is not str:
        raise ElasticsearchRedactionError(
            "redactor must return text or deidentified_text"
        )
    maximum_output_chars = min(
        _MAX_TEXT_CHARS,
        max(_MIN_OUTPUT_CHARS, len(original) * _MAX_OUTPUT_EXPANSION),
    )
    if len(redacted_text) > maximum_output_chars:
        raise ElasticsearchRedactionError("redactor output exceeds the size limit")

    if redacted_text == original:
        return redacted_text, 0
    if isinstance(entities, (list, tuple)) and type(entities) in (list, tuple):
        span_count = min(len(entities), _MAX_SPANS_PER_FIELD)
    else:
        span_count = 0
    if span_count == 0:
        span_count = 1
    return redacted_text, span_count


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
