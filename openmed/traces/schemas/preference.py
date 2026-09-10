"""Local-first redaction for preference-pair training records.

Preference records keep a prompt beside a chosen and a rejected response. A
redactor that processes those branches independently can assign different
surrogates to the same person, phone number, or identifier, creating a
training signal that was not present in the source pair. This module copies a
JSON-ready record, walks the three content branches, and gives them one
request-scoped :class:`PreferenceRedactionState`.

The default detector is deliberately small and local. It covers common
structured identifiers plus conservative Latin-script names. Applications
with a richer local detector can provide ``span_detector`` or a
``text_redactor``. Neither the default adapter nor its report loads a model,
contacts a service, or writes an artifact.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

CONTENT_FIELDS = ("prompt", "chosen", "rejected")
PREFERENCE_SCHEMA_VERSION = "openmed.traces.preference_pair.v1"

PreferenceContent: TypeAlias = str | Mapping[str, Any] | Sequence[Any] | None
TextRedactor: TypeAlias = Callable[..., str]
SpanDetector: TypeAlias = Callable[[str], Iterable[Any]]

_CONTENT_NODE_FIELDS = frozenset(
    {
        "arguments",
        "chosen",
        "content",
        "input",
        "messages",
        "message",
        "output",
        "parts",
        "prompt",
        "rejected",
        "response",
        "text",
        "tool_calls",
        "value",
    }
)
_NON_CONTENT_FIELDS = frozenset(
    {
        "attributes",
        "channel",
        "created_at",
        "format",
        "id",
        "index",
        "label",
        "metadata",
        "meta",
        "mime_type",
        "name",
        "pair_id",
        "preference",
        "role",
        "score",
        "scores",
        "source",
        "speaker",
        "timestamp",
        "type",
        "updated_at",
        "weight",
    }
)


class PreferenceSchemaError(ValueError):
    """Raised when a preference record is not schema-compatible."""


class PreferenceRedactionError(PreferenceSchemaError):
    """Raised when a local detector or redactor cannot process a record."""


def _plain_text(value: object) -> str | None:
    """Copy a string into a base ``str`` without calling subclass hooks."""

    if not isinstance(value, str):
        return None
    try:
        return str.encode(value, "utf-8").decode("utf-8")
    except Exception:
        return None


def _mapping_entries(value: Mapping[Any, Any]) -> tuple[tuple[Any, Any], ...]:
    """Materialize mapping entries without exposing custom iterator errors."""

    try:
        raw_entries = tuple(value.items())
    except Exception:
        raise PreferenceSchemaError("preference mapping could not be read") from None
    entries: list[tuple[Any, Any]] = []
    for raw_entry in raw_entries:
        try:
            key, item = raw_entry
        except Exception:
            raise PreferenceSchemaError(
                "preference mapping could not be read"
            ) from None
        entries.append((key, item))
    return tuple(entries)


@dataclass(frozen=True, slots=True)
class SensitiveSpan:
    """A PHI span supplied to a preference text redactor.

    The span stores offsets and a label only. It intentionally does not retain
    the matched surface so its representation and reports are PHI-safe.
    """

    start: int
    end: int
    label: str = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.start) is not int:
            raise ValueError("span.start must be an integer")
        if type(self.end) is not int:
            raise ValueError("span.end must be an integer")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("span offsets must be non-empty and ordered")
        normalized_label = _plain_text(self.label)
        if normalized_label is None or not normalized_label.strip():
            raise ValueError("span.label must be a non-empty string")
        object.__setattr__(self, "label", normalized_label.strip())


PreferenceSpan = SensitiveSpan


@dataclass(slots=True)
class PreferenceRedactionState:
    """Shared, in-memory pseudonym state for one or more preference pairs.

    Raw source values are used only as keys while the current process is
    running. They are excluded from ``repr`` and from the public summary.
    Reusing this state for ``prompt``, ``chosen``, and ``rejected`` makes an
    identical ``(label, value)`` pair resolve to one replacement.
    """

    seed: int = 0
    lang: str = "en"
    locale: str | None = None
    anonymizer: Any | None = None
    _pseudonyms: dict[tuple[str, str], str] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _text_nodes_seen: int = field(default=0, init=False, repr=False)
    _text_nodes_changed: int = field(default=0, init=False, repr=False)
    _replacement_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise TypeError("seed must be an integer")
        normalized_lang = _plain_text(self.lang)
        if normalized_lang is None or not normalized_lang.strip():
            raise ValueError("lang must be a non-empty string")
        self.lang = normalized_lang.strip()
        if self.locale is not None:
            normalized_locale = _plain_text(self.locale)
            if normalized_locale is None or not normalized_locale.strip():
                raise ValueError("locale must be a non-empty string when provided")
            self.locale = normalized_locale.strip()
        if self.anonymizer is None:
            from openmed.core.anonymizer import Anonymizer

            self.anonymizer = Anonymizer(
                lang=self.lang,
                locale=self.locale,
                consistent=True,
                seed=self.seed,
            )
        try:
            surrogate = getattr(self.anonymizer, "surrogate", None)
        except Exception:
            raise PreferenceRedactionError(
                "anonymizer could not be inspected safely"
            ) from None
        if not callable(surrogate):
            raise TypeError("anonymizer must provide a surrogate method")

    def pseudonym(self, value: str, label: str) -> str:
        """Return a deterministic, non-leaking surrogate for ``value``."""

        source_value = _plain_text(value)
        if source_value is None:
            raise TypeError("pseudonym values must be strings")
        if not source_value:
            return source_value
        canonical_label = _canonical_label(label, lang=self.lang)
        key = (canonical_label, source_value)
        existing = self._pseudonyms.get(key)
        if existing is not None:
            return existing

        anonymizer = self.anonymizer
        if anonymizer is None:  # pragma: no cover - guarded by __post_init__
            raise PreferenceRedactionError("local pseudonym generation failed")
        try:
            candidate = anonymizer.surrogate(
                source_value,
                canonical_label,
                lang=self.lang,
                locale=self.locale,
            )
        except Exception:
            raise PreferenceRedactionError(
                "local pseudonym generation failed"
            ) from None

        normalized_candidate = _plain_text(candidate)
        if normalized_candidate is None or not normalized_candidate:
            raise PreferenceRedactionError("local pseudonym generation failed")
        if normalized_candidate == source_value or _contains_source_fragment(
            source_value, normalized_candidate
        ):
            normalized_candidate = _digest_surrogate(
                source_value,
                canonical_label,
                seed=self.seed,
            )
        self._pseudonyms[key] = normalized_candidate
        return normalized_candidate

    def redact_spans(self, text: str, spans: Iterable[Any]) -> str:
        """Replace validated spans in ``text`` while preserving offsets."""

        source_text = _plain_text(text)
        if source_text is None:
            raise PreferenceRedactionError("text input must be a string")
        normalized = _normalize_spans(spans, source_text)
        if not normalized:
            return source_text
        pieces: list[str] = []
        cursor = 0
        for span in normalized:
            pieces.append(source_text[cursor : span.start])
            pieces.append(
                self.pseudonym(source_text[span.start : span.end], span.label)
            )
            cursor = span.end
            self._replacement_count += 1
        pieces.append(source_text[cursor:])
        return "".join(pieces)

    def note_text_result(self, original: str, redacted: str) -> None:
        """Record PHI-safe counters for one visited string leaf."""

        source_text = _plain_text(original)
        redacted_text = _plain_text(redacted)
        if source_text is None or redacted_text is None:
            raise PreferenceRedactionError("text redaction result is invalid")
        self._text_nodes_seen += 1
        if source_text != redacted_text:
            self._text_nodes_changed += 1

    def _snapshot(self) -> tuple[int, int, int]:
        return (
            self._text_nodes_seen,
            self._text_nodes_changed,
            self._replacement_count,
        )

    def report(self, before: tuple[int, int, int]) -> "PreferenceRedactionReport":
        """Return counters accumulated since ``before`` without source text."""

        seen, changed, replacements = before
        return PreferenceRedactionReport(
            text_nodes_seen=self._text_nodes_seen - seen,
            text_nodes_changed=self._text_nodes_changed - changed,
            replacement_count=self._replacement_count - replacements,
        )


@dataclass(frozen=True, slots=True)
class PreferenceRedactionReport:
    """PHI-safe aggregate counters for one adapter operation."""

    text_nodes_seen: int = 0
    text_nodes_changed: int = 0
    replacement_count: int = 0
    branches_visited: int = len(CONTENT_FIELDS)
    schema_version: str = PREFERENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "text_nodes_seen",
            "text_nodes_changed",
            "replacement_count",
            "branches_visited",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise PreferenceSchemaError(
                    "preference redaction report counts must be non-negative integers"
                )
        schema_version = _plain_text(self.schema_version)
        if schema_version != PREFERENCE_SCHEMA_VERSION:
            raise PreferenceSchemaError(
                "preference redaction report version is invalid"
            )
        object.__setattr__(self, "schema_version", schema_version)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready report containing no source surfaces."""

        return {
            "branches_visited": self.branches_visited,
            "replacement_count": self.replacement_count,
            "schema_version": self.schema_version,
            "text_nodes_changed": self.text_nodes_changed,
            "text_nodes_seen": self.text_nodes_seen,
        }


@dataclass(frozen=True, slots=True)
class PreferenceRedactionResult:
    """A redacted record together with its PHI-safe aggregate report."""

    record: Mapping[str, Any]
    report: PreferenceRedactionReport

    def to_mapping(self) -> dict[str, Any]:
        """Return a defensive copy of the redacted record."""

        return _copy_mapping(self.record)


@dataclass(frozen=True, slots=True, repr=False)
class PreferencePair:
    """Typed view of one preference pair while preserving extra fields.

    ``extra_fields`` contains scores, IDs, and all non-content metadata from
    the source mapping. It is intentionally hidden from ``repr`` because this
    class can also represent an unredacted source record in memory.
    """

    prompt: PreferenceContent = field(repr=False)
    chosen: PreferenceContent = field(repr=False)
    rejected: PreferenceContent = field(repr=False)
    extra_fields: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __repr__(self) -> str:
        return "PreferencePair(<content omitted>)"

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any]) -> "PreferencePair":
        """Build a pair view and retain every non-content source field."""

        _validate_record(record)
        extras: dict[str, Any] = {}
        for raw_key, value in _mapping_entries(record):
            key = _plain_text(raw_key)
            if key is None:
                raise PreferenceSchemaError("preference keys must be strings")
            if key not in CONTENT_FIELDS:
                extras[key] = _copy_value(value)
        return cls(
            prompt=_copy_value(record["prompt"]),
            chosen=_copy_value(record["chosen"]),
            rejected=_copy_value(record["rejected"]),
            extra_fields=extras,
        )

    @property
    def metadata(self) -> Any:
        """Return the preserved metadata field, if present."""

        return self.extra_fields.get("metadata")

    @property
    def scores(self) -> Any:
        """Return the preserved score field or score mapping, if present."""

        if "scores" in self.extra_fields:
            return self.extra_fields["scores"]
        return self.extra_fields.get("score")

    def to_mapping(self) -> dict[str, Any]:
        """Return a JSON-ready mapping with all source fields preserved."""

        result = {key: _copy_value(value) for key, value in self.extra_fields.items()}
        result.update(
            {
                "prompt": _copy_value(self.prompt),
                "chosen": _copy_value(self.chosen),
                "rejected": _copy_value(self.rejected),
            }
        )
        return result

    to_dict = to_mapping

    def redact(
        self, adapter: "PreferencePairAdapter" | None = None
    ) -> "PreferencePair":
        """Return this pair after local redaction with one shared state."""

        redactor = adapter if adapter is not None else PreferencePairAdapter()
        return PreferencePair.from_mapping(redactor.redact(self.to_mapping()))


class PreferencePairAdapter:
    """Adapt and redact standard ``prompt``/``chosen``/``rejected`` records.

    Args:
        text_redactor: Optional callable. It may accept ``(text)`` or
            ``(text, state)`` and must return redacted text. Supplying one is
            useful when a local model already provides span-aware redaction.
        span_detector: Optional callable returning ``SensitiveSpan`` objects,
            ``{"start", "end", "label"}`` mappings, or three-item tuples.
            It is used by the local default text redactor.
        sensitive_values: Optional exact source-value to label mapping for
            deterministic synthetic fixtures or application-specific IDs.
        seed: Stable seed for the default pseudonym generator.
        anonymizer: Optional existing local anonymizer. One adapter state is
            still shared across every branch of a pair.
    """

    def __init__(
        self,
        text_redactor: TextRedactor | None = None,
        *,
        redactor: TextRedactor | None = None,
        span_detector: SpanDetector | None = None,
        detector: SpanDetector | None = None,
        sensitive_values: Mapping[str, str] | None = None,
        seed: int = 0,
        lang: str = "en",
        locale: str | None = None,
        anonymizer: Any | None = None,
    ) -> None:
        if text_redactor is not None and redactor is not None:
            raise TypeError("provide only one of text_redactor or redactor")
        if span_detector is not None and detector is not None:
            raise TypeError("provide only one of span_detector or detector")
        self.text_redactor = text_redactor if text_redactor is not None else redactor
        self.span_detector = span_detector if span_detector is not None else detector
        if sensitive_values is None:
            self.sensitive_values: dict[str, str] = {}
        elif not isinstance(sensitive_values, Mapping):
            raise TypeError("sensitive_values must be a mapping")
        else:
            self.sensitive_values = {}
            for raw_value, raw_label in _mapping_entries(sensitive_values):
                value = _plain_text(raw_value)
                if value is None or not value:
                    raise ValueError("sensitive_values keys must be non-empty strings")
                label = _plain_text(raw_label)
                if label is None or not label.strip():
                    raise ValueError(
                        "sensitive_values labels must be non-empty strings"
                    )
                self.sensitive_values[value] = label.strip()
        self.seed = seed
        self.lang = lang
        self.locale = locale
        self.anonymizer = anonymizer
        # Validate constructor settings without retaining a second generator.
        self.new_state()

    def new_state(self) -> PreferenceRedactionState:
        """Create a fresh state for a pair or a dataset operation."""

        return PreferenceRedactionState(
            seed=self.seed,
            lang=self.lang,
            locale=self.locale,
            anonymizer=self.anonymizer,
        )

    def redact(
        self,
        record: Mapping[str, Any] | PreferencePair,
        *,
        state: PreferenceRedactionState | None = None,
    ) -> dict[str, Any]:
        """Return a redacted copy of one preference record.

        The same ``state`` is used for all three required branches. The input
        mapping and its metadata are never mutated.
        """

        source = record.to_mapping() if isinstance(record, PreferencePair) else record
        _validate_record(source)
        active_state = state if state is not None else self.new_state()
        output = _copy_mapping(source)
        for field_name in CONTENT_FIELDS:
            output[field_name] = self._redact_content(
                source[field_name],
                active_state,
            )
        return output

    adapt = redact

    def redact_with_report(
        self,
        record: Mapping[str, Any] | PreferencePair,
        *,
        state: PreferenceRedactionState | None = None,
    ) -> PreferenceRedactionResult:
        """Redact one pair and return a source-text-free processing report."""

        active_state = state if state is not None else self.new_state()
        before = active_state._snapshot()
        output = self.redact(record, state=active_state)
        return PreferenceRedactionResult(
            record=output,
            report=active_state.report(before),
        )

    def redact_dataset(
        self,
        records: Iterable[Mapping[str, Any] | PreferencePair],
        *,
        state: PreferenceRedactionState | None = None,
    ) -> list[dict[str, Any]]:
        """Redact records with one shared state and preserve record order."""

        active_state = state if state is not None else self.new_state()
        try:
            iterator = iter(records)
        except Exception:
            raise PreferenceSchemaError("preference dataset is not iterable") from None

        output: list[dict[str, Any]] = []
        while True:
            try:
                record = next(iterator)
            except StopIteration:
                break
            except Exception:
                raise PreferenceSchemaError(
                    "preference dataset could not be read"
                ) from None
            output.append(self.redact(record, state=active_state))
        return output

    def redact_dataset_with_report(
        self,
        records: Iterable[Mapping[str, Any] | PreferencePair],
        *,
        state: PreferenceRedactionState | None = None,
    ) -> tuple[list[dict[str, Any]], PreferenceRedactionReport]:
        """Redact records and return one PHI-safe aggregate report."""

        active_state = state if state is not None else self.new_state()
        before = active_state._snapshot()
        output = self.redact_dataset(records, state=active_state)
        return output, active_state.report(before)

    def _redact_content(
        self,
        value: Any,
        state: PreferenceRedactionState,
    ) -> Any:
        if isinstance(value, str):
            text = _plain_text(value)
            if text is None:
                raise PreferenceRedactionError("preference text is invalid")
            return self._redact_text(text, state)
        if isinstance(value, Mapping):
            return self._redact_mapping(value, state)
        if isinstance(value, list):
            return [self._redact_content(item, state) for item in value]
        if isinstance(value, tuple):
            return tuple(self._redact_content(item, state) for item in value)
        return _copy_value(value)

    def _redact_mapping(
        self,
        value: Mapping[Any, Any],
        state: PreferenceRedactionState,
    ) -> dict[Any, Any]:
        entries = _mapping_entries(value)
        has_content_node = any(
            (_plain_text(key) or "").casefold() in _CONTENT_NODE_FIELDS
            for key, _item in entries
        )
        result: dict[Any, Any] = {}
        for raw_key, item in entries:
            key = _plain_text(raw_key)
            if key is None:
                raise PreferenceSchemaError("preference mapping keys must be strings")
            key_name = key.casefold()
            if key in result:
                raise PreferenceSchemaError("preference mapping keys must be unique")
            if key_name in _CONTENT_NODE_FIELDS:
                result[key] = self._redact_content(item, state)
            elif key_name in _NON_CONTENT_FIELDS:
                result[key] = _copy_value(item)
            elif has_content_node:
                result[key] = _copy_value(item)
            else:
                # A branch-specific mapping without a known message shape is
                # treated as content, while explicit metadata remains opaque.
                result[key] = self._redact_content(item, state)
        return result

    def _redact_text(self, text: str, state: PreferenceRedactionState) -> str:
        source_text = _plain_text(text)
        if source_text is None:
            raise PreferenceRedactionError("preference text is invalid")
        if self.text_redactor is None:
            detector = (
                self.span_detector
                if self.span_detector is not None
                else self._default_detector
            )
            try:
                spans = detector(source_text)
            except Exception:
                raise PreferenceRedactionError(
                    "local sensitive-value detection failed"
                ) from None
            try:
                redacted = state.redact_spans(source_text, spans)
            except PreferenceRedactionError:
                raise
            except Exception:
                raise PreferenceRedactionError("local text redaction failed") from None
        else:
            replacements_before = state._replacement_count
            redacted = _call_text_redactor(self.text_redactor, source_text, state)
            if (
                state._replacement_count == replacements_before
                and redacted != source_text
            ):
                state._replacement_count += 1
        redacted_text = _plain_text(redacted)
        if redacted_text is None:
            raise PreferenceRedactionError("text redactor must return a string")
        state.note_text_result(source_text, redacted_text)
        return redacted_text

    def _default_detector(self, text: str) -> tuple[SensitiveSpan, ...]:
        return _detect_sensitive_spans(text, self.sensitive_values)


PreferenceSchemaAdapter = PreferencePairAdapter


def redact_preference_pair(
    record: Mapping[str, Any] | PreferencePair,
    *,
    adapter: PreferencePairAdapter | None = None,
    **adapter_kwargs: Any,
) -> dict[str, Any]:
    """Redact one preference pair using a local adapter."""

    active_adapter = (
        adapter if adapter is not None else PreferencePairAdapter(**adapter_kwargs)
    )
    return active_adapter.redact(record)


def adapt_preference_pair(
    record: Mapping[str, Any] | PreferencePair,
    *,
    adapter: PreferencePairAdapter | None = None,
    **adapter_kwargs: Any,
) -> dict[str, Any]:
    """Alias for :func:`redact_preference_pair`."""

    return redact_preference_pair(record, adapter=adapter, **adapter_kwargs)


def redact_preference_dataset(
    records: Iterable[Mapping[str, Any] | PreferencePair],
    *,
    adapter: PreferencePairAdapter | None = None,
    **adapter_kwargs: Any,
) -> list[dict[str, Any]]:
    """Redact an iterable of pairs while preserving order and membership."""

    active_adapter = (
        adapter if adapter is not None else PreferencePairAdapter(**adapter_kwargs)
    )
    return active_adapter.redact_dataset(records)


def _validate_record(record: Any) -> None:
    if not isinstance(record, Mapping):
        raise PreferenceSchemaError("preference pair must be a mapping")
    try:
        missing = [
            field_name for field_name in CONTENT_FIELDS if field_name not in record
        ]
        if missing:
            joined = ", ".join(missing)
            raise PreferenceSchemaError(
                f"preference pair missing required field(s): {joined}"
            )
        for field_name in CONTENT_FIELDS:
            if not _is_content_value(record[field_name]):
                raise PreferenceSchemaError(
                    f"preference field {field_name!r} has an unsupported value type"
                )
    except PreferenceSchemaError:
        raise
    except Exception:
        raise PreferenceSchemaError("preference pair could not be read") from None


def _is_content_value(value: Any) -> bool:
    if value is None or isinstance(value, (str, Mapping)):
        return True
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return True
    return False


def _copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return {key: copy.deepcopy(item) for key, item in value.items()}
    except Exception:
        raise PreferenceSchemaError(
            "preference record contains an unsupported value"
        ) from None


def _copy_value(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception:
        raise PreferenceSchemaError(
            "preference record contains an unsupported value"
        ) from None


def _canonical_label(label: str, *, lang: str) -> str:
    normalized_label = _plain_text(label)
    normalized_lang = _plain_text(lang)
    if normalized_label is None or not normalized_label.strip():
        raise PreferenceRedactionError("sensitive span label must be non-empty")
    if normalized_lang is None or not normalized_lang.strip():
        raise PreferenceRedactionError("sensitive span language must be non-empty")
    try:
        from openmed.core.labels import normalize_label

        result = normalize_label(normalized_label, normalized_lang)
    except Exception:
        return "OTHER"
    normalized_result = _plain_text(result)
    return normalized_result if normalized_result is not None else "OTHER"


def _normalize_spans(spans: Iterable[Any], text: str) -> tuple[SensitiveSpan, ...]:
    if spans is None:
        return ()
    try:
        candidates = list(spans)
    except Exception:
        raise PreferenceRedactionError(
            "span detector must return an iterable"
        ) from None

    normalized: list[SensitiveSpan] = []
    seen: set[tuple[int, int, str]] = set()
    for item in candidates:
        try:
            span = _coerce_span(item)
        except PreferenceRedactionError:
            raise
        except Exception:
            raise PreferenceRedactionError(
                "span detector returned an invalid span"
            ) from None
        if span.end > len(text):
            raise PreferenceRedactionError("span offsets exceed text length")
        key = (span.start, span.end, span.label)
        if key not in seen:
            seen.add(key)
            normalized.append(span)
    normalized.sort(key=lambda span: (span.start, span.end, span.label))
    previous: SensitiveSpan | None = None
    for span in normalized:
        if previous is not None and span.start < previous.end:
            raise PreferenceRedactionError("span detector returned overlapping spans")
        previous = span
    return tuple(normalized)


def _coerce_span(item: Any) -> SensitiveSpan:
    if isinstance(item, SensitiveSpan):
        return item
    if isinstance(item, Mapping):
        return SensitiveSpan(
            start=item["start"],
            end=item["end"],
            label=item.get("label", item.get("entity_type", "OTHER")),
        )
    if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
        if len(item) != 3:
            raise ValueError("span tuples must contain start, end, and label")
        return SensitiveSpan(start=item[0], end=item[1], label=item[2])
    raise ValueError("span must be a mapping, tuple, or SensitiveSpan")


def _call_text_redactor(
    redactor: TextRedactor,
    text: str,
    state: PreferenceRedactionState,
) -> str:
    try:
        signature = inspect.signature(redactor)
        parameters = tuple(signature.parameters.values())
        state_parameter = next(
            (
                parameter
                for parameter in parameters
                if parameter.name in {"state", "context"}
            ),
            None,
        )
        accepts_state = state_parameter is not None or any(
            parameter.kind is inspect.Parameter.VAR_POSITIONAL
            for parameter in parameters
        )
        required_positional = sum(
            parameter.default is inspect.Parameter.empty
            and parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
            for parameter in parameters
        )
        accepts_state = accepts_state or required_positional >= 2
        if state_parameter is not None and (
            state_parameter.kind is inspect.Parameter.KEYWORD_ONLY
        ):
            result = redactor(text, **{state_parameter.name: state})
        else:
            result = redactor(text, state) if accepts_state else redactor(text)
    except PreferenceRedactionError:
        raise
    except Exception:
        raise PreferenceRedactionError("text redactor failed") from None
    normalized_result = _plain_text(result)
    if normalized_result is None:
        raise PreferenceRedactionError("text redactor must return a string")
    return normalized_result


def _contains_source_fragment(source: str, candidate: str) -> bool:
    fragments = re.findall(
        r"[A-Za-z]{3,}|\d{3,}|[\u3400-\u4dbf\u4e00-\u9fff]{2,}",
        source,
    )
    folded_candidate = candidate.casefold()
    return any(fragment.casefold() in folded_candidate for fragment in fragments)


def _digest_surrogate(value: str, label: str, *, seed: int) -> str:
    material = f"{seed}|{label}|{value}".encode("utf-8")
    digest = hashlib.blake2b(material, digest_size=8).hexdigest()
    return f"[{label.casefold()}-{digest}]"


_EMAIL_RE = re.compile(
    r"(?<![\w.+-])[\w.!#$%&'*+/=?^`{|}~-]+@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+(?![\w-])"
)
_CREDIT_CARD_RE = re.compile(r"(?<!\w)(?:\d[ -]?){13,19}(?!\w)")
_SSN_RE = re.compile(r"(?<!\w)\d{3}-\d{2}-\d{4}(?!\w)")
_IP_ADDRESS_RE = re.compile(r"(?<!\w)(?:\d{1,3}\.){3}\d{1,3}(?!\w)")
_DATE_RE = re.compile(
    r"(?<!\w)(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|"
    r"\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})(?!\w)"
)
_PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d(). -]{5,}\d(?!\w)")
_ID_RE = re.compile(
    r"(?i:\b(?:mrn|medical record(?: number)?|patient id|member id|"
    r"account(?: number)?)\s*[:#-]?\s*)"
    r"(?P<identifier>[A-Z0-9][A-Z0-9-]{3,})\b"
)
_CUE_NAME_RE = re.compile(
    r"(?i:\b(?:patient|name|mr|mrs|ms|dr)\s*[:#-]?\s+)"
    r"(?P<name>[A-Z][a-z]{1,31}(?:['-][A-Z][a-z]{1,31})?"
    r"(?:\s+[A-Z][a-z]{1,31}(?:['-][A-Z][a-z]{1,31})?){1,2})"
)
_COMMON_NAME_FALSE_POSITIVES = frozenset(
    {
        "A Good",
        "An Example",
        "Assistant Response",
        "Chosen Response",
        "Medical Record",
        "Patient Name",
        "Rejected Response",
        "The Patient",
        "This Example",
        "User Message",
    }
)
_NON_NAME_LEADS = frozenset(
    {
        "Arrange",
        "Assistant",
        "At",
        "Call",
        "Chosen",
        "Contact",
        "Confirm",
        "Email",
        "Follow",
        "Ignore",
        "Medical",
        "Patient",
        "Please",
        "Provide",
        "Record",
        "Rejected",
        "Request",
        "Response",
        "Same",
        "Send",
        "The",
        "This",
        "That",
        "Use",
        "User",
    }
)
_NAME_LEAD_PATTERN = "|".join(sorted(_NON_NAME_LEADS))
_NAME_RE = re.compile(
    rf"\b(?P<name>(?!(?:{_NAME_LEAD_PATTERN})\b)"
    r"[A-Z][a-z]{1,31}(?:['-][A-Z][a-z]{1,31})?"
    r"\s+[A-Z][a-z]{1,31}(?:['-][A-Z][a-z]{1,31})?)\b"
)


def _detect_sensitive_spans(
    text: str,
    sensitive_values: Mapping[str, str],
) -> tuple[SensitiveSpan, ...]:
    candidates: list[tuple[int, int, str]] = []
    for value, label in sorted(
        sensitive_values.items(),
        key=lambda item: (-len(item[0]), item[0]),
    ):
        start = text.find(value)
        while start >= 0:
            candidates.append((start, start + len(value), label))
            start = text.find(value, start + len(value))

    pattern_specs: tuple[tuple[re.Pattern[str], str, str | None], ...] = (
        (_EMAIL_RE, "EMAIL", None),
        (_CREDIT_CARD_RE, "CREDIT_CARD", None),
        (_SSN_RE, "SSN", None),
        (_IP_ADDRESS_RE, "IP_ADDRESS", None),
        (_DATE_RE, "DATE", None),
        (_PHONE_RE, "PHONE", None),
        (_ID_RE, "ID_NUM", "identifier"),
        (_CUE_NAME_RE, "PERSON", "name"),
        (_NAME_RE, "PERSON", "name"),
    )
    for pattern, label, group_name in pattern_specs:
        for match in pattern.finditer(text):
            if group_name is None:
                start, end = match.span()
                matched_text = match.group(0)
            else:
                start, end = match.span(group_name)
                matched_text = match.group(group_name)
            if label == "PERSON" and matched_text in _COMMON_NAME_FALSE_POSITIVES:
                continue
            candidates.append((start, end, label))

    candidates.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))
    accepted: list[SensitiveSpan] = []
    accepted_end = -1
    for start, end, label in candidates:
        if start < accepted_end:
            continue
        accepted.append(SensitiveSpan(start=start, end=end, label=label))
        accepted_end = end
    accepted.sort(key=lambda span: span.start)
    return tuple(accepted)


__all__ = [
    "CONTENT_FIELDS",
    "PREFERENCE_SCHEMA_VERSION",
    "PreferencePair",
    "PreferencePairAdapter",
    "PreferenceRedactionError",
    "PreferenceRedactionReport",
    "PreferenceRedactionResult",
    "PreferenceRedactionState",
    "PreferenceSchemaAdapter",
    "PreferenceSchemaError",
    "PreferenceSpan",
    "SensitiveSpan",
    "adapt_preference_pair",
    "redact_preference_dataset",
    "redact_preference_pair",
]
