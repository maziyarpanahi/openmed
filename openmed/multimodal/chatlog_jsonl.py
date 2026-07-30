"""Streaming JSONL chat-log de-identification.

Clinical chat and LLM fine-tuning corpora are often stored as JSONL records
where each line is either a single chat turn or a conversation object with a
``messages`` list. This module redacts string content in those structures while
leaving roles, turn indexes, timestamps, and unconfigured structural metadata
unchanged.
"""

from __future__ import annotations

import io
import json
import os
from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TextIO, runtime_checkable

from openmed.core.labels import USERNAME

from .base import ExtractedDocument, register_handler

TextRedactor = Callable[[str], Any]
FieldPath = tuple[str, ...]

DEFAULT_CONTENT_FIELDS = ("content",)
DEFAULT_MESSAGES_FIELD = "messages"
DEFAULT_SPEAKER_FIELDS = (
    "speaker",
    "speaker_id",
    "participant",
    "participant_id",
    "author",
    "author_id",
    "user",
    "user_id",
    "name",
)


@dataclass(frozen=True)
class ChatLogRedactionSummary:
    """PHI-safe counts collected while streaming a JSONL chat log."""

    line_count: int = 0
    message_count: int = 0
    redacted_field_count: int = 0
    pseudonymized_speaker_count: int = 0
    schema_counts: Mapping[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the summary as a JSON-serializable mapping."""
        return {
            "line_count": self.line_count,
            "message_count": self.message_count,
            "redacted_field_count": self.redacted_field_count,
            "pseudonymized_speaker_count": self.pseudonymized_speaker_count,
            "schema_counts": dict(self.schema_counts),
        }


@dataclass(frozen=True)
class RedactedChatLog:
    """Materialized JSONL output plus PHI-safe processing metadata."""

    text: str
    summary: ChatLogRedactionSummary

    def to_document(self) -> ExtractedDocument:
        """Bridge materialized JSONL output into the multimodal contract."""
        return ExtractedDocument(
            text=self.text,
            metadata={
                "format": "jsonl_chatlog",
                "redaction_summary": self.summary.to_dict(),
            },
        )


@runtime_checkable
class ChatSchemaAdapter(Protocol):
    """Pluggable adapter for a JSONL chat record schema."""

    name: str

    def matches(self, record: Any) -> bool:
        """Return whether this adapter can redact ``record``."""
        ...

    def redact(self, record: Any, context: "_RedactionContext") -> None:
        """Redact supported fields in ``record`` in place."""
        ...


@dataclass(frozen=True)
class TurnRecordAdapter:
    """Adapter for one-chat-turn-per-line records."""

    content_fields: Sequence[str] = DEFAULT_CONTENT_FIELDS
    name: str = "turn"

    def matches(self, record: Any) -> bool:
        return isinstance(record, MutableMapping) and any(
            field in record for field in self.content_fields
        )

    def redact(self, record: Any, context: "_RedactionContext") -> None:
        if not isinstance(record, MutableMapping):
            return
        context.note_message()
        _redact_content_fields(record, self.content_fields, context)
        _pseudonymize_speaker_fields(record, context)
        _redact_configured_paths(record, context)


@dataclass(frozen=True)
class MessagesListAdapter:
    """Adapter for records containing a role/content ``messages`` list."""

    messages_field: str = DEFAULT_MESSAGES_FIELD
    content_fields: Sequence[str] = DEFAULT_CONTENT_FIELDS
    name: str = "messages"

    def matches(self, record: Any) -> bool:
        return isinstance(record, MutableMapping) and isinstance(
            record.get(self.messages_field), list
        )

    def redact(self, record: Any, context: "_RedactionContext") -> None:
        if not isinstance(record, MutableMapping):
            return
        messages = record.get(self.messages_field)
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, MutableMapping):
                    context.note_message()
                    _redact_content_fields(message, self.content_fields, context)
                    _pseudonymize_speaker_fields(message, context)
                    _redact_configured_paths(message, context)
        _pseudonymize_speaker_fields(record, context)
        _redact_configured_paths(record, context)


@dataclass
class _RedactionStats:
    line_count: int = 0
    message_count: int = 0
    redacted_field_count: int = 0
    schema_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class _RedactionContext:
    redactor: TextRedactor
    nested_paths: tuple[FieldPath, ...]
    pseudonymize_speakers: bool
    speaker_fields: frozenset[str]
    speaker_label: str
    speaker_map: dict[str, str] = field(default_factory=dict)
    stats: _RedactionStats = field(default_factory=_RedactionStats)

    def redact_text(self, value: str) -> str:
        """Redact one string value and count changed fields."""
        if not value:
            return value
        redacted = _coerce_redacted_text(self.redactor(value))
        if redacted != value:
            self.stats.redacted_field_count += 1
        return redacted

    def pseudonymize_speaker(self, value: Any) -> Any:
        """Return a stable pseudonym for one speaker identifier."""
        if not self.pseudonymize_speakers or not isinstance(value, str) or not value:
            return value
        pseudonym = self.speaker_map.get(value)
        if pseudonym is None:
            pseudonym = _hash_speaker_identifier(value, self.speaker_label)
            self.speaker_map[value] = pseudonym
        return pseudonym

    def note_message(self) -> None:
        self.stats.message_count += 1

    def note_schema(self, name: str) -> None:
        self.stats.schema_counts[name] = self.stats.schema_counts.get(name, 0) + 1

    def summary(self) -> ChatLogRedactionSummary:
        """Return immutable public counts."""
        return ChatLogRedactionSummary(
            line_count=self.stats.line_count,
            message_count=self.stats.message_count,
            redacted_field_count=self.stats.redacted_field_count,
            pseudonymized_speaker_count=len(self.speaker_map),
            schema_counts=dict(self.stats.schema_counts),
        )


def iter_redacted_chatlog_jsonl(
    source: str | os.PathLike[str] | Iterable[str] | TextIO,
    *,
    policy: Any | None = None,
    models: Any | None = None,
    lang: str = "en",
    nested_fields: Sequence[str] | None = None,
    pseudonymize_speakers: bool = False,
    speaker_fields: Sequence[str] = DEFAULT_SPEAKER_FIELDS,
    speaker_label: str = USERNAME,
    schema_adapters: Sequence[ChatSchemaAdapter] | None = None,
    text_redactor: TextRedactor | None = None,
) -> Iterator[str]:
    """Yield redacted JSONL lines without materializing the source file.

    Args:
        source: Filesystem path, text stream, iterable of JSONL lines, or JSONL
            text content.
        policy: Optional de-identification policy name, or a mapping with
            chat-log options such as ``nested_fields`` and
            ``pseudonymize_speakers``.
        models: Optional redactor source. A callable, ``{"text_redactor": fn}``,
            or object with ``text_redactor`` can be used to keep tests/offline
            workflows from loading a model.
        lang: Language hint passed to the default OpenMed de-identifier.
        nested_fields: Dot paths whose string leaves should be redacted, for
            example ``("metadata.patient", "messages.tool_call.arguments")``.
        pseudonymize_speakers: Whether to hash speaker identifiers consistently
            within this file.
        speaker_fields: Field names considered speaker identifiers. ``role`` is
            intentionally not included so role labels remain unchanged.
        speaker_label: Label prefix used by the existing hash redaction helper.
        schema_adapters: Optional custom schema adapters. Defaults support
            message-list and one-turn-per-line chat records.
        text_redactor: Explicit callable that redacts a string.

    Yields:
        Redacted JSONL lines, preserving each source line ending.
    """
    yield from iter_redacted_chatlog_jsonl_with_summary(
        source,
        policy=policy,
        models=models,
        lang=lang,
        nested_fields=nested_fields,
        pseudonymize_speakers=pseudonymize_speakers,
        speaker_fields=speaker_fields,
        speaker_label=speaker_label,
        schema_adapters=schema_adapters,
        text_redactor=text_redactor,
    )


def write_redacted_chatlog_jsonl(
    source: str | os.PathLike[str] | Iterable[str] | TextIO,
    output: str | os.PathLike[str] | TextIO,
    **kwargs: Any,
) -> ChatLogRedactionSummary:
    """Stream redacted JSONL from ``source`` into ``output``.

    This is the large-file path: it processes and writes one JSONL record at a
    time and returns only PHI-safe counts.
    """
    redacted_lines = iter_redacted_chatlog_jsonl_with_summary(source, **kwargs)
    if hasattr(output, "write"):
        for line in redacted_lines:
            output.write(line)
    else:
        with Path(output).open("w", encoding="utf-8", newline="") as handle:
            for line in redacted_lines:
                handle.write(line)
    return redacted_lines.summary


def iter_redacted_chatlog_jsonl_with_summary(
    source: str | os.PathLike[str] | Iterable[str] | TextIO,
    *,
    policy: Any | None = None,
    models: Any | None = None,
    lang: str = "en",
    nested_fields: Sequence[str] | None = None,
    pseudonymize_speakers: bool = False,
    speaker_fields: Sequence[str] = DEFAULT_SPEAKER_FIELDS,
    speaker_label: str = USERNAME,
    schema_adapters: Sequence[ChatSchemaAdapter] | None = None,
    text_redactor: TextRedactor | None = None,
) -> "_SummaryIterator":
    """Return a streaming iterator exposing its summary after exhaustion."""
    options = _resolve_policy_options(
        policy,
        nested_fields=nested_fields,
        pseudonymize_speakers=pseudonymize_speakers,
        speaker_fields=speaker_fields,
        speaker_label=speaker_label,
        schema_adapters=schema_adapters,
    )
    context = _RedactionContext(
        redactor=text_redactor
        or _text_redactor_from_models(models)
        or _default_text_redactor(
            policy=options.deidentify_policy,
            lang=lang,
        ),
        nested_paths=_parse_field_paths(options.nested_fields),
        pseudonymize_speakers=options.pseudonymize_speakers,
        speaker_fields=frozenset(options.speaker_fields),
        speaker_label=options.speaker_label,
    )
    return _SummaryIterator(
        _iter_redacted_lines_with_context(
            source,
            context=context,
            schema_adapters=options.schema_adapters,
        ),
        context,
    )


def redact_chatlog_jsonl(
    source: str | os.PathLike[str] | Iterable[str] | TextIO,
    **kwargs: Any,
) -> RedactedChatLog:
    """Return materialized redacted JSONL text for small files or dispatchers.

    Use :func:`write_redacted_chatlog_jsonl` for large files to avoid retaining
    the redacted output in memory.
    """
    iterator = iter_redacted_chatlog_jsonl_with_summary(source, **kwargs)
    text = "".join(iterator)
    return RedactedChatLog(text=text, summary=iterator.summary)


@dataclass(frozen=True)
class _ResolvedOptions:
    nested_fields: tuple[str, ...]
    pseudonymize_speakers: bool
    speaker_fields: tuple[str, ...]
    speaker_label: str
    schema_adapters: tuple[ChatSchemaAdapter, ...]
    deidentify_policy: str | None


class _SummaryIterator:
    def __init__(
        self,
        iterator: Iterator[str],
        context: _RedactionContext,
    ) -> None:
        self._iterator = iterator
        self._context = context

    def __iter__(self) -> "_SummaryIterator":
        return self

    def __next__(self) -> str:
        return next(self._iterator)

    @property
    def summary(self) -> ChatLogRedactionSummary:
        return self._context.summary()


def _iter_redacted_lines_with_context(
    source: str | os.PathLike[str] | Iterable[str] | TextIO,
    *,
    context: _RedactionContext,
    schema_adapters: Sequence[ChatSchemaAdapter],
) -> Iterator[str]:
    for line_number, line in enumerate(_iter_source_lines(source), start=1):
        context.stats.line_count += 1
        payload, ending = _split_line_ending(line)
        if not payload.strip():
            yield line
            continue
        try:
            record = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {line_number}: {exc.msg}") from exc

        adapter = _select_adapter(record, schema_adapters)
        if adapter is None:
            raise ValueError(
                f"unsupported chat JSONL schema at line {line_number}: "
                "expected a messages list or one-turn content field"
            )
        adapter.redact(record, context)
        context.note_schema(adapter.name)
        yield _dump_record(record) + ending


def _resolve_policy_options(
    policy: Any | None,
    *,
    nested_fields: Sequence[str] | None,
    pseudonymize_speakers: bool,
    speaker_fields: Sequence[str],
    speaker_label: str,
    schema_adapters: Sequence[ChatSchemaAdapter] | None,
) -> _ResolvedOptions:
    policy_mapping = policy if isinstance(policy, Mapping) else {}
    chatlog_policy = policy_mapping.get("chatlog")
    if isinstance(chatlog_policy, Mapping):
        policy_mapping = {**policy_mapping, **chatlog_policy}

    resolved_nested = nested_fields
    if resolved_nested is None:
        policy_nested = policy_mapping.get("nested_fields")
        if isinstance(policy_nested, str):
            resolved_nested = (policy_nested,)
        elif isinstance(policy_nested, Sequence):
            resolved_nested = tuple(str(path) for path in policy_nested)
    resolved_nested = tuple(resolved_nested or ())

    resolved_speaker_fields = tuple(str(field) for field in speaker_fields)
    policy_speaker_fields = policy_mapping.get("speaker_fields")
    if isinstance(policy_speaker_fields, str):
        resolved_speaker_fields = (policy_speaker_fields,)
    elif isinstance(policy_speaker_fields, Sequence):
        resolved_speaker_fields = tuple(str(field) for field in policy_speaker_fields)

    policy_pseudonymize = policy_mapping.get("pseudonymize_speakers")
    if policy_pseudonymize is not None:
        pseudonymize_speakers = bool(policy_pseudonymize)

    policy_speaker_label = policy_mapping.get("speaker_label")
    if policy_speaker_label is not None:
        speaker_label = str(policy_speaker_label)

    deidentify_policy = policy if isinstance(policy, str) else None
    mapping_deidentify_policy = policy_mapping.get("deidentify_policy")
    if mapping_deidentify_policy is not None:
        deidentify_policy = str(mapping_deidentify_policy)

    return _ResolvedOptions(
        nested_fields=tuple(str(path) for path in resolved_nested),
        pseudonymize_speakers=pseudonymize_speakers,
        speaker_fields=resolved_speaker_fields,
        speaker_label=speaker_label,
        schema_adapters=tuple(schema_adapters or _default_schema_adapters()),
        deidentify_policy=deidentify_policy,
    )


def _default_schema_adapters() -> tuple[ChatSchemaAdapter, ...]:
    return (MessagesListAdapter(), TurnRecordAdapter())


def _select_adapter(
    record: Any, adapters: Sequence[ChatSchemaAdapter]
) -> ChatSchemaAdapter | None:
    for adapter in adapters:
        if adapter.matches(record):
            return adapter
    return None


def _redact_content_fields(
    record: MutableMapping[str, Any],
    content_fields: Sequence[str],
    context: _RedactionContext,
) -> None:
    for field_name in content_fields:
        if field_name in record:
            record[field_name] = _redact_string_leaves(record[field_name], context)


def _redact_configured_paths(
    record: MutableMapping[str, Any],
    context: _RedactionContext,
) -> None:
    for path in context.nested_paths:
        _redact_path(record, path, context)


def _redact_path(value: Any, path: FieldPath, context: _RedactionContext) -> Any:
    if not path:
        return _redact_string_leaves(value, context)

    head, *tail = path
    rest = tuple(tail)
    if isinstance(value, MutableMapping):
        if head == "*":
            for key in list(value):
                value[key] = _redact_path(value[key], rest, context)
        elif head in value:
            value[head] = _redact_path(value[head], rest, context)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _redact_path(item, path, context)
    return value


def _redact_string_leaves(value: Any, context: _RedactionContext) -> Any:
    if isinstance(value, str):
        return context.redact_text(value)
    if isinstance(value, MutableMapping):
        for key in list(value):
            value[key] = _redact_string_leaves(value[key], context)
        return value
    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _redact_string_leaves(item, context)
        return value
    return value


def _pseudonymize_speaker_fields(
    record: MutableMapping[str, Any],
    context: _RedactionContext,
) -> None:
    if not context.pseudonymize_speakers:
        return
    for field_name in context.speaker_fields:
        if field_name in record:
            record[field_name] = context.pseudonymize_speaker(record[field_name])


def _parse_field_paths(paths: Sequence[str]) -> tuple[FieldPath, ...]:
    parsed: list[FieldPath] = []
    for path in paths:
        parts = tuple(part for part in str(path).split(".") if part)
        if not parts:
            raise ValueError("nested field paths must not be empty")
        parsed.append(parts)
    return tuple(parsed)


def _text_redactor_from_models(models: Any | None) -> TextRedactor | None:
    if callable(models):
        return models
    if isinstance(models, Mapping):
        for key in ("text_redactor", "deidentifier", "redactor"):
            candidate = models.get(key)
            if callable(candidate):
                return candidate
    for attribute in ("text_redactor", "deidentifier", "redactor"):
        candidate = getattr(models, attribute, None)
        if callable(candidate):
            return candidate
    return None


def _default_text_redactor(*, policy: str | None, lang: str) -> TextRedactor:
    def redactor(text: str) -> str:
        from openmed.core.pii import deidentify

        result = deidentify(text, method="mask", lang=lang, policy=policy)
        return _coerce_redacted_text(result)

    return redactor


def _coerce_redacted_text(result: Any) -> str:
    if hasattr(result, "deidentified_text"):
        return str(result.deidentified_text)
    return str(result)


def _hash_speaker_identifier(value: str, label: str) -> str:
    from openmed.core.pii import PIIEntity, _redact_entity

    entity = PIIEntity(
        text=value,
        label=label,
        entity_type=label,
        start=0,
        end=len(value),
        confidence=1.0,
        original_text=value,
    )
    return _redact_entity(entity, "hash")


def _iter_source_lines(
    source: str | os.PathLike[str] | Iterable[str] | TextIO,
) -> Iterator[str]:
    if isinstance(source, os.PathLike):
        with Path(source).open("r", encoding="utf-8", newline="") as handle:
            yield from handle
        return

    if isinstance(source, str):
        path = Path(source)
        if "\n" not in source and "\r" not in source and path.exists():
            with path.open("r", encoding="utf-8", newline="") as handle:
                yield from handle
            return
        yield from io.StringIO(source, newline="")
        return

    if isinstance(source, Iterable):
        for line in source:
            yield str(line)
        return

    if hasattr(source, "read"):
        text = source.read()
        yield from io.StringIO(str(text), newline="")
        return

    raise TypeError("source must be a path, JSONL text, text stream, or line iterable")


def _split_line_ending(line: str) -> tuple[str, str]:
    stripped = line.rstrip("\r\n")
    return stripped, line[len(stripped) :]


def _dump_record(record: Any) -> str:
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"))


def _chatlog_jsonl_handler(
    path: str | os.PathLike[str],
    *,
    policy: Any = None,
    models: Any = None,
    lang: str | None = None,
) -> ExtractedDocument:
    return redact_chatlog_jsonl(
        path,
        policy=policy,
        models=models,
        lang=lang or "en",
    ).to_document()


register_handler(".jsonl", _chatlog_jsonl_handler, requires_multimodal=False)


__all__ = [
    "ChatLogRedactionSummary",
    "ChatSchemaAdapter",
    "DEFAULT_CONTENT_FIELDS",
    "DEFAULT_MESSAGES_FIELD",
    "DEFAULT_SPEAKER_FIELDS",
    "MessagesListAdapter",
    "RedactedChatLog",
    "TurnRecordAdapter",
    "iter_redacted_chatlog_jsonl",
    "iter_redacted_chatlog_jsonl_with_summary",
    "redact_chatlog_jsonl",
    "write_redacted_chatlog_jsonl",
]
