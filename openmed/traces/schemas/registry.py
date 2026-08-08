"""Offline registry for nested training-conversation schemas.

Training exports commonly represent the same conversation with different
message and preference layouts.  This module keeps those layouts intact while
giving redaction code three small operations: detect a schema, walk its text
content, and reconstruct a copy with replacement text.

The registry deliberately uses structural checks only.  It does not load a
model, read a dataset, or make a network request during detection or
reconstruction.  Error messages identify schema names and paths, never record
values.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, cast, runtime_checkable

PathPart: TypeAlias = str | int
ContentPath: TypeAlias = tuple[PathPart, ...]
ContentReplacementMap: TypeAlias = Mapping[ContentPath | str, str]
ContentItem: TypeAlias = tuple[ContentPath, str]


class SchemaRegistryError(ValueError):
    """Base class for deterministic training-schema registry failures."""


class InvalidSchemaError(TypeError, SchemaRegistryError):
    """Raised when a schema does not implement the registry contract."""


class UnknownSchemaError(SchemaRegistryError):
    """Raised when an explicit schema name is not registered."""


class SchemaMismatchError(SchemaRegistryError):
    """Raised when an explicitly selected schema does not match a record."""


class AmbiguousSchemaError(SchemaRegistryError):
    """Raised when more than one schema can safely handle a record."""


class SchemaDetectionError(SchemaRegistryError):
    """Raised when a schema cannot provide a valid detection or walk result."""


class SchemaReconstructionError(SchemaRegistryError):
    """Raised when content replacements cannot be applied safely."""


class SchemaTransformError(SchemaRegistryError):
    """Raised when a content transformation fails or returns invalid text."""


# These aliases keep the vocabulary discoverable for callers that describe the
# failure as ambiguity or absence rather than using the registry's class names.
SchemaAmbiguityError = AmbiguousSchemaError
SchemaNotFoundError = UnknownSchemaError


@runtime_checkable
class TrainingConversationSchema(Protocol):
    """Protocol implemented by one training-conversation schema.

    ``walk`` yields ``(path, text)`` pairs.  A path is made only of mapping
    keys and sequence indexes, which lets a caller redact text without
    flattening or rebuilding the surrounding record shape.
    """

    name: str

    def detect(self, record: Any) -> bool:
        """Return whether this schema can handle ``record``."""
        ...

    def walk(self, record: Any) -> Iterable[ContentItem]:
        """Yield every redaction-safe text path and its current value."""
        ...

    def reconstruct(
        self,
        record: Any,
        replacements: Mapping[ContentPath, str],
    ) -> Any:
        """Return a record with replacements applied at known content paths."""
        ...


# Short aliases are useful when the protocol is used in type annotations.
ConversationSchema = TrainingConversationSchema
SchemaProtocol = TrainingConversationSchema


_MESSAGE_ROOTS: tuple[ContentPath, ...] = (
    ("messages",),
    ("conversation", "messages"),
    ("data", "messages"),
)
_SHAREGPT_ROOTS: tuple[ContentPath, ...] = (
    ("conversations",),
    ("conversation", "conversations"),
    ("data", "conversations"),
)
_PREFERENCE_ROOTS: tuple[ContentPath, ...] = (
    (),
    ("preference",),
    ("preferences",),
    ("data",),
)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _lookup(record: Any, path: ContentPath) -> Any:
    current = record
    for part in path:
        if isinstance(current, Mapping):
            if part not in current:
                return None
            current = current[part]
        elif isinstance(current, Sequence) and not isinstance(
            current, (str, bytes, bytearray)
        ):
            if not isinstance(part, int) or isinstance(part, bool):
                return None
            if part < 0 or part >= len(current):
                return None
            current = current[part]
        else:
            return None
    return current


def _walk_text_value(value: Any, path: ContentPath) -> tuple[ContentItem, ...]:
    """Walk a known text slot without descending into arbitrary metadata."""

    if isinstance(value, str):
        return ((path, value),)

    if isinstance(value, Mapping):
        text = value.get("text")
        if isinstance(text, str):
            return ((path + ("text",), text),)
        content = value.get("content")
        if isinstance(content, str):
            return ((path + ("content",), content),)
        return ()

    if not _is_sequence(value):
        return ()

    items: list[ContentItem] = []
    for index, part in enumerate(value):
        part_path = path + (index,)
        if isinstance(part, str):
            items.append((part_path, part))
        elif isinstance(part, Mapping):
            text = part.get("text")
            if isinstance(text, str):
                items.append((part_path + ("text",), text))
            else:
                content = part.get("content")
                if isinstance(content, str):
                    items.append((part_path + ("content",), content))
    return tuple(items)


def _walk_message_list(
    messages: Any,
    path: ContentPath,
    *,
    content_key: str,
) -> tuple[ContentItem, ...]:
    if not _is_sequence(messages):
        return ()

    items: list[ContentItem] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping) or content_key not in message:
            continue
        items.extend(
            _walk_text_value(message[content_key], path + (index, content_key))
        )
    return tuple(items)


def _candidate_roots(
    record: Any,
    roots: Sequence[ContentPath],
    walker: Any,
) -> tuple[ContentPath, ...]:
    candidates: list[ContentPath] = []
    for root in roots:
        value = _lookup(record, root)
        if walker(value, root):
            candidates.append(root)
    return tuple(candidates)


def _single_root(
    record: Any,
    roots: Sequence[ContentPath],
    walker: Any,
    *,
    schema_name: str,
) -> ContentPath:
    candidates = _candidate_roots(record, roots, walker)
    if not candidates:
        raise SchemaMismatchError(
            f"record does not match the {schema_name!r} training schema"
        )
    if len(candidates) > 1:
        raise AmbiguousSchemaError(
            f"the {schema_name!r} training schema has multiple content roots; "
            "select one schema layout explicitly"
        )
    return candidates[0]


def _message_root_walker(value: Any, path: ContentPath) -> tuple[ContentItem, ...]:
    return _walk_message_list(value, path, content_key="content")


def _sharegpt_root_walker(value: Any, path: ContentPath) -> tuple[ContentItem, ...]:
    return _walk_message_list(value, path, content_key="value")


def _preference_value_items(value: Any, path: ContentPath) -> tuple[ContentItem, ...]:
    if isinstance(value, str):
        return ((path, value),)

    if isinstance(value, Mapping):
        messages = value.get("messages")
        if _is_sequence(messages):
            return _walk_message_list(
                messages, path + ("messages",), content_key="content"
            )
        conversations = value.get("conversations")
        if _is_sequence(conversations):
            return _walk_message_list(
                conversations,
                path + ("conversations",),
                content_key="value",
            )
        return _walk_text_value(value, path)

    return _walk_text_value(value, path)


def _preference_root_walker(
    value: Any,
    path: ContentPath,
) -> tuple[ContentItem, ...]:
    if not isinstance(value, Mapping):
        return ()

    required = ("chosen", "rejected")
    if not all(key in value for key in required):
        return ()

    items: list[ContentItem] = []
    for key in ("prompt", "chosen", "rejected"):
        if key not in value:
            continue
        field_items = _preference_value_items(value[key], path + (key,))
        if key != "prompt" and not field_items:
            return ()
        if key == "prompt" and not field_items and value[key] is not None:
            return ()
        items.extend(field_items)
    return tuple(items)


def _reconstruct_record(
    record: Any,
    walked: Iterable[ContentItem],
    replacements: Mapping[ContentPath | str, str],
) -> Any:
    """Apply validated replacements to a deep copy of a JSON-like record."""

    if not isinstance(replacements, Mapping):
        raise SchemaReconstructionError("replacements must be a mapping")
    known_paths = {path for path, _ in walked}
    normalized: dict[ContentPath, str] = {}
    for raw_path, replacement in replacements.items():
        path = _normalize_path(raw_path)
        if path not in known_paths:
            raise SchemaReconstructionError(
                "replacement path is not a discovered content path"
            )
        if not isinstance(replacement, str):
            raise SchemaReconstructionError("replacement content must be text")
        normalized[path] = replacement

    result = copy.deepcopy(record)
    for path in sorted(normalized, key=_path_sort_key):
        result = _replace_at_path(result, path, normalized[path])
    return result


def _normalize_path(raw_path: ContentPath | str | Sequence[PathPart]) -> ContentPath:
    if isinstance(raw_path, str):
        parts: tuple[PathPart, ...] = tuple(
            int(part) if part.isdecimal() else part
            for part in raw_path.split(".")
            if part
        )
        if not parts:
            raise SchemaReconstructionError("content paths must not be empty")
        return parts

    if not isinstance(raw_path, Sequence) or isinstance(raw_path, (bytes, bytearray)):
        raise SchemaReconstructionError("content paths must be sequences")

    parts: list[PathPart] = []
    for part in raw_path:
        if isinstance(part, bool) or not isinstance(part, (str, int)):
            raise SchemaReconstructionError(
                "content paths may contain only string keys and indexes"
            )
        if isinstance(part, str) and not part:
            raise SchemaReconstructionError("content path keys must not be empty")
        if isinstance(part, int) and part < 0:
            raise SchemaReconstructionError("content path indexes must be positive")
        parts.append(part)
    if not parts:
        raise SchemaReconstructionError("content paths must not be empty")
    return tuple(parts)


def _path_sort_key(path: ContentPath) -> tuple[tuple[int, str], ...]:
    return tuple(
        (0, str(part)) if isinstance(part, int) else (1, part) for part in path
    )


def _replace_at_path(value: Any, path: ContentPath, replacement: str) -> Any:
    if not path:
        return replacement

    part = path[0]
    remainder = path[1:]
    if isinstance(value, MutableMapping):
        if part not in value:
            raise SchemaReconstructionError("content path no longer exists")
        value[part] = _replace_at_path(value[part], remainder, replacement)
        return value

    if isinstance(value, list):
        if not isinstance(part, int) or isinstance(part, bool):
            raise SchemaReconstructionError("content path does not address a list")
        if part < 0 or part >= len(value):
            raise SchemaReconstructionError("content path index is out of range")
        value[part] = _replace_at_path(value[part], remainder, replacement)
        return value

    if isinstance(value, tuple):
        if not isinstance(part, int) or isinstance(part, bool):
            raise SchemaReconstructionError("content path does not address a tuple")
        if part < 0 or part >= len(value):
            raise SchemaReconstructionError("content path index is out of range")
        items = list(value)
        items[part] = _replace_at_path(items[part], remainder, replacement)
        return tuple(items)

    if isinstance(value, Mapping):
        if part not in value:
            raise SchemaReconstructionError("content path no longer exists")
        copied = dict(value)
        copied[part] = _replace_at_path(copied[part], remainder, replacement)
        return copied

    raise SchemaReconstructionError("content path enters a non-container value")


def _schema_method(schema: object, *names: str) -> Any:
    for name in names:
        method = getattr(schema, name, None)
        if callable(method):
            return method
    return None


def _validated_schema(schema: object) -> TrainingConversationSchema:
    name = getattr(schema, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise InvalidSchemaError("training schemas must declare a non-empty name")
    if _schema_method(schema, "detect", "matches") is None:
        raise InvalidSchemaError(
            f"training schema {name.strip()!r} must define detect()"
        )
    if _schema_method(schema, "walk", "iter_content") is None:
        raise InvalidSchemaError(f"training schema {name.strip()!r} must define walk()")
    if _schema_method(schema, "reconstruct") is None:
        raise InvalidSchemaError(
            f"training schema {name.strip()!r} must define reconstruct()"
        )
    return cast(TrainingConversationSchema, schema)


def _schema_name(schema: object) -> str:
    name = getattr(schema, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise InvalidSchemaError("training schemas must declare a non-empty name")
    return name.strip()


def _normalize_walk_items(
    schema: object, raw_items: Iterable[Any]
) -> tuple[ContentItem, ...]:
    items: list[ContentItem] = []
    seen: set[ContentPath] = set()
    try:
        for raw_item in raw_items:
            if (
                not isinstance(raw_item, Sequence)
                or isinstance(raw_item, (str, bytes, bytearray))
                or len(raw_item) != 2
            ):
                raise SchemaDetectionError(
                    f"training schema {_schema_name(schema)!r} returned an invalid "
                    "content item"
                )
            path = _normalize_path(raw_item[0])
            text = raw_item[1]
            if not isinstance(text, str):
                raise SchemaDetectionError(
                    f"training schema {_schema_name(schema)!r} returned non-text "
                    "content"
                )
            if path in seen:
                raise SchemaDetectionError(
                    f"training schema {_schema_name(schema)!r} returned a duplicate "
                    "content path"
                )
            seen.add(path)
            items.append((path, text))
    except SchemaRegistryError:
        raise
    except Exception:
        raise SchemaDetectionError(
            f"training schema {_schema_name(schema)!r} could not walk the record"
        ) from None
    return tuple(items)


def _call_detect(schema: object, record: Any) -> bool:
    method = _schema_method(schema, "detect", "matches")
    if method is None:
        raise InvalidSchemaError("training schema does not define detect()")
    try:
        result = method(record)
    except Exception:
        raise SchemaDetectionError(
            f"training schema {_schema_name(schema)!r} could not inspect the record"
        ) from None
    if not isinstance(result, bool):
        raise SchemaDetectionError(
            f"training schema {_schema_name(schema)!r} must return a boolean from "
            "detect()"
        )
    return result


def _call_walk(schema: object, record: Any) -> tuple[ContentItem, ...]:
    method = _schema_method(schema, "walk", "iter_content")
    if method is None:
        raise InvalidSchemaError("training schema does not define walk()")
    try:
        raw_items = method(record)
        if isinstance(raw_items, (str, bytes, bytearray)):
            raise SchemaDetectionError(
                f"training schema {_schema_name(schema)!r} returned invalid walk output"
            )
        return _normalize_walk_items(schema, raw_items)
    except SchemaRegistryError:
        raise
    except Exception:
        raise SchemaDetectionError(
            f"training schema {_schema_name(schema)!r} could not walk the record"
        ) from None


@dataclass(frozen=True)
class MessagesSchema:
    """Schema for records containing role/content message lists."""

    name: str = "messages"
    roots: tuple[ContentPath, ...] = _MESSAGE_ROOTS

    def detect(self, record: Any) -> bool:
        """Return true when one known message root contains text content."""

        return bool(_candidate_roots(record, self.roots, _message_root_walker))

    matches = detect

    def walk(self, record: Any) -> tuple[ContentItem, ...]:
        """Yield content strings from the one unambiguous message root."""

        root = _single_root(
            record,
            self.roots,
            _message_root_walker,
            schema_name=self.name,
        )
        return _message_root_walker(_lookup(record, root), root)

    iter_content = walk

    def reconstruct(
        self,
        record: Any,
        replacements: Mapping[ContentPath, str],
    ) -> Any:
        """Rebuild a copy of a message record with path replacements."""

        return _reconstruct_record(record, self.walk(record), replacements)


@dataclass(frozen=True)
class ShareGPTSchema:
    """Schema for ShareGPT-style ``conversations[].value`` records."""

    name: str = "sharegpt"
    roots: tuple[ContentPath, ...] = _SHAREGPT_ROOTS

    def detect(self, record: Any) -> bool:
        """Return true when a known conversations root contains text values."""

        return bool(_candidate_roots(record, self.roots, _sharegpt_root_walker))

    matches = detect

    def walk(self, record: Any) -> tuple[ContentItem, ...]:
        """Yield ShareGPT turn values without touching speaker metadata."""

        root = _single_root(
            record,
            self.roots,
            _sharegpt_root_walker,
            schema_name=self.name,
        )
        return _sharegpt_root_walker(_lookup(record, root), root)

    iter_content = walk

    def reconstruct(
        self,
        record: Any,
        replacements: Mapping[ContentPath, str],
    ) -> Any:
        """Rebuild a copy of a ShareGPT record with path replacements."""

        return _reconstruct_record(record, self.walk(record), replacements)


@dataclass(frozen=True)
class PreferenceSchema:
    """Schema for prompt/chosen/rejected preference records."""

    name: str = "preference"
    roots: tuple[ContentPath, ...] = _PREFERENCE_ROOTS

    def detect(self, record: Any) -> bool:
        """Return true when one known preference root has both responses."""

        return bool(_candidate_roots(record, self.roots, _preference_root_walker))

    matches = detect

    def walk(self, record: Any) -> tuple[ContentItem, ...]:
        """Yield prompt and response text from one preference root."""

        root = _single_root(
            record,
            self.roots,
            _preference_root_walker,
            schema_name=self.name,
        )
        return _preference_root_walker(_lookup(record, root), root)

    iter_content = walk

    def reconstruct(
        self,
        record: Any,
        replacements: Mapping[ContentPath, str],
    ) -> Any:
        """Rebuild a copy of a preference record with path replacements."""

        return _reconstruct_record(record, self.walk(record), replacements)


class TrainingSchemaRegistry:
    """Deterministic registry for local training-conversation schemas.

    The default registry contains the built-in ``messages``, ``sharegpt``, and
    ``preference`` schemas.  Custom schemas can be registered without any
    discovery or network operation.  Auto-detection is fail-closed: zero
    matches and multiple matches both require caller action, while explicit
    selection names one schema and still validates its structure before a
    reconstruction is attempted.
    """

    def __init__(
        self,
        schemas: Iterable[TrainingConversationSchema] | None = None,
        *,
        include_defaults: bool = True,
    ) -> None:
        self._schemas: dict[str, TrainingConversationSchema] = {}
        self._aliases: dict[str, str] = {}
        if include_defaults:
            self.register(MessagesSchema(), aliases=("chat", "chatml"))
            self.register(ShareGPTSchema(), aliases=("conversations",))
            self.register(
                PreferenceSchema(), aliases=("dpo", "preference_pair", "preferences")
            )
        if schemas is not None:
            for schema in schemas:
                self.register(schema)

    def register(
        self,
        schema: TrainingConversationSchema,
        *,
        replace: bool = False,
        aliases: Iterable[str] = (),
    ) -> None:
        """Register one schema under its deterministic name.

        Args:
            schema: Object implementing :class:`TrainingConversationSchema`.
            replace: Replace a schema with the same canonical name.
            aliases: Optional explicit names accepted by :meth:`get`.

        Raises:
            InvalidSchemaError: If the object does not satisfy the contract.
            SchemaRegistryError: If a name is already registered.
        """

        validated = _validated_schema(schema)
        name = _schema_name(validated)
        if not replace and name in self._schemas:
            raise SchemaRegistryError(f"training schema {name!r} is already registered")
        normalized_aliases: list[str] = []
        for raw_alias in aliases:
            alias = _schema_name_value(raw_alias, "schema aliases")
            if alias == name:
                continue
            if alias in normalized_aliases:
                continue
            owner = self._aliases.get(alias) or (
                alias if alias in self._schemas and alias != name else None
            )
            if owner is not None and owner != name:
                raise SchemaRegistryError(
                    f"training schema name {alias!r} is already registered"
                )
            normalized_aliases.append(alias)

        self._schemas[name] = validated
        for alias in normalized_aliases:
            self._aliases[alias] = name

    def get(self, name: str) -> TrainingConversationSchema:
        """Return a registered schema by canonical name or alias."""

        normalized = _schema_name_value(name, "schema names")
        canonical = self._aliases.get(normalized, normalized)
        try:
            return self._schemas[canonical]
        except KeyError:
            raise UnknownSchemaError(
                f"training schema {normalized!r} is not registered"
            ) from None

    def available(self) -> tuple[str, ...]:
        """Return canonical schema names in deterministic order."""

        return tuple(sorted(self._schemas))

    def matching_schemas(self, record: Any) -> tuple[str, ...]:
        """Return all schema names that structurally match ``record``."""

        matches: list[str] = []
        for name in self.available():
            schema = self._schemas[name]
            if _call_detect(schema, record):
                matches.append(name)
        return tuple(matches)

    detect = matching_schemas

    def resolve(
        self,
        record: Any,
        schema: str | TrainingConversationSchema | None = None,
        *,
        schema_name: str | None = None,
    ) -> TrainingConversationSchema:
        """Resolve one schema, requiring explicit choice when auto-detection is ambiguous."""

        if schema is not None and schema_name is not None:
            raise SchemaRegistryError("provide schema or schema_name, not both")
        explicit = schema if schema is not None else schema_name
        if explicit is not None:
            selected = (
                self.get(explicit)
                if isinstance(explicit, str)
                else _validated_schema(explicit)
            )
            if not _call_detect(selected, record):
                raise SchemaMismatchError(
                    f"selected training schema {_schema_name(selected)!r} does not "
                    "match the record"
                )
            return selected

        matches = self.matching_schemas(record)
        if not matches:
            raise UnknownSchemaError("no training schema matched the record")
        if len(matches) > 1:
            names = ", ".join(matches)
            raise AmbiguousSchemaError(
                "multiple training schemas matched the record; select one "
                f"explicitly: {names}"
            )
        return self._schemas[matches[0]]

    select = resolve

    def walk(
        self,
        record: Any,
        schema: str | TrainingConversationSchema | None = None,
        *,
        schema_name: str | None = None,
    ) -> tuple[ContentItem, ...]:
        """Walk content after resolving one unambiguous schema."""

        selected = self.resolve(record, schema, schema_name=schema_name)
        return _call_walk(selected, record)

    walk_content = walk

    def reconstruct(
        self,
        record: Any,
        replacements: ContentReplacementMap,
        schema: str | TrainingConversationSchema | None = None,
        *,
        schema_name: str | None = None,
    ) -> Any:
        """Reconstruct a copy after validating schema and replacement paths."""

        selected = self.resolve(record, schema, schema_name=schema_name)
        walked = _call_walk(selected, record)
        normalized = _validated_replacements(walked, replacements)
        return self._reconstruct_selected(record, selected, normalized)

    reconstruct_record = reconstruct

    def transform(
        self,
        record: Any,
        transform: Any,
        schema: str | TrainingConversationSchema | None = None,
        *,
        schema_name: str | None = None,
    ) -> Any:
        """Transform each discovered text value and reconstruct a new record."""

        if not callable(transform):
            raise SchemaTransformError("content transform must be callable")
        selected = self.resolve(record, schema, schema_name=schema_name)
        walked = _call_walk(selected, record)
        replacements: dict[ContentPath, str] = {}
        for path, text in walked:
            try:
                replacement = transform(text)
            except Exception:
                raise SchemaTransformError(
                    "content transform failed for a discovered path"
                ) from None
            if not isinstance(replacement, str):
                raise SchemaTransformError("content transform must return text")
            replacements[path] = replacement
        return self._reconstruct_selected(record, selected, replacements)

    map_content = transform
    redact = transform

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and (
            name in self._schemas or name in self._aliases
        )

    def __len__(self) -> int:
        return len(self._schemas)

    def _reconstruct_selected(
        self,
        record: Any,
        schema: TrainingConversationSchema,
        replacements: Mapping[ContentPath, str],
    ) -> Any:
        try:
            result = schema.reconstruct(copy.deepcopy(record), replacements)
        except SchemaRegistryError:
            raise
        except Exception:
            raise SchemaReconstructionError(
                f"training schema {_schema_name(schema)!r} could not reconstruct "
                "the record"
            ) from None
        if result is None:
            raise SchemaReconstructionError(
                f"training schema {_schema_name(schema)!r} returned no record"
            )
        return result


def _schema_name_value(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidSchemaError(f"{description} must be non-empty strings")
    normalized = value.strip()
    if any(character.isspace() for character in normalized):
        raise InvalidSchemaError(f"{description} must not contain whitespace")
    return normalized


def _validated_replacements(
    walked: Iterable[ContentItem],
    replacements: ContentReplacementMap,
) -> dict[ContentPath, str]:
    if not isinstance(replacements, Mapping):
        raise SchemaReconstructionError("replacements must be a mapping")
    known_paths = {path for path, _ in walked}
    normalized: dict[ContentPath, str] = {}
    for raw_path, replacement in replacements.items():
        path = _normalize_path(raw_path)
        if path not in known_paths:
            raise SchemaReconstructionError(
                "replacement path is not a discovered content path"
            )
        if not isinstance(replacement, str):
            raise SchemaReconstructionError("replacement content must be text")
        normalized[path] = replacement
    return normalized


def create_default_registry() -> TrainingSchemaRegistry:
    """Create a fresh registry containing only built-in offline schemas."""

    return TrainingSchemaRegistry()


DEFAULT_SCHEMA_REGISTRY = create_default_registry()
default_registry = DEFAULT_SCHEMA_REGISTRY


def resolve_schema(
    record: Any,
    schema: str | TrainingConversationSchema | None = None,
    *,
    registry: TrainingSchemaRegistry = DEFAULT_SCHEMA_REGISTRY,
) -> TrainingConversationSchema:
    """Resolve a schema using the process-local default registry."""

    return registry.resolve(record, schema)


def walk_content(
    record: Any,
    schema: str | TrainingConversationSchema | None = None,
    *,
    registry: TrainingSchemaRegistry = DEFAULT_SCHEMA_REGISTRY,
) -> tuple[ContentItem, ...]:
    """Return content paths and values using a local schema registry."""

    return registry.walk(record, schema)


def reconstruct_record(
    record: Any,
    replacements: ContentReplacementMap,
    schema: str | TrainingConversationSchema | None = None,
    *,
    registry: TrainingSchemaRegistry = DEFAULT_SCHEMA_REGISTRY,
) -> Any:
    """Return a reconstructed copy of a training record."""

    return registry.reconstruct(record, replacements, schema)


def transform_record(
    record: Any,
    transform: Any,
    schema: str | TrainingConversationSchema | None = None,
    *,
    registry: TrainingSchemaRegistry = DEFAULT_SCHEMA_REGISTRY,
) -> Any:
    """Transform discovered content and return a reconstructed copy."""

    return registry.transform(record, transform, schema)


__all__ = [
    "AmbiguousSchemaError",
    "ContentItem",
    "ContentPath",
    "ContentReplacementMap",
    "ConversationSchema",
    "DEFAULT_SCHEMA_REGISTRY",
    "InvalidSchemaError",
    "MessagesSchema",
    "PreferenceSchema",
    "SchemaAmbiguityError",
    "SchemaDetectionError",
    "SchemaMismatchError",
    "SchemaNotFoundError",
    "SchemaProtocol",
    "SchemaReconstructionError",
    "SchemaRegistry",
    "SchemaRegistryError",
    "SchemaTransformError",
    "ShareGPTSchema",
    "TrainingConversationSchema",
    "TrainingSchemaRegistry",
    "create_default_registry",
    "default_registry",
    "reconstruct_record",
    "resolve_schema",
    "transform_record",
    "walk_content",
]


SchemaRegistry = TrainingSchemaRegistry
