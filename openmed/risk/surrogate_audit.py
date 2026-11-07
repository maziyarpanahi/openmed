"""Offline referential-integrity checks for surrogate maps.

The auditor consumes privacy-safe map material only: each binding is a
``key_hash`` paired with a surrogate, and optional metadata declares expected
cardinality.  Parent/child relationships describe which maps participate in
orphan and cross-table checks.  The returned report contains aggregate counts
and stable category names; it never returns a key hash, surrogate, map name,
or offending entry.

The implementation deliberately does not read files, call a service, or
attempt to resolve a hash back to an original identifier.  Callers should
construct keyed hashes before invoking it and retain any reversible mapping
outside the audit report's privacy boundary.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final, cast

__all__ = [
    "CARDINALITY_FAILURE",
    "COLLISION_FAILURE",
    "CROSS_TABLE_CONSISTENCY_FAILURE",
    "FAILURE_CATEGORIES",
    "SURROGATE_AUDIT_FAILURE_CATEGORIES",
    "ORPHAN_FAILURE",
    "SurrogateAuditInputError",
    "SurrogateMapAuditFailure",
    "SurrogateMapAuditReport",
    "audit_surrogate_map",
    "audit_surrogate_maps",
    "check_surrogate_map_integrity",
]

SCHEMA_VERSION: Final = 1

_MAX_MAPS: Final = 64
_MAX_ENTRIES_PER_MAP: Final = 100_000
_MAX_RELATIONSHIPS: Final = 4_096
_MAX_TEXT_CHARS: Final = 16_384
_MAX_METADATA_FIELDS: Final = 32
_MAX_COUNT: Final = 2**63 - 1

CARDINALITY_FAILURE: Final = "cardinality"
COLLISION_FAILURE: Final = "collision"
ORPHAN_FAILURE: Final = "orphan"
CROSS_TABLE_CONSISTENCY_FAILURE: Final = "cross_table_consistency"
SURROGATE_AUDIT_FAILURE_CATEGORIES: Final[tuple[str, ...]] = (
    CARDINALITY_FAILURE,
    COLLISION_FAILURE,
    ORPHAN_FAILURE,
    CROSS_TABLE_CONSISTENCY_FAILURE,
)
# Keep the original contributor-facing name as a compatibility alias.
FAILURE_CATEGORIES: Final = SURROGATE_AUDIT_FAILURE_CATEGORIES

_MISSING = object()
_DEFAULT_MAP_NAME = "default"

_KEY_FIELD_ALIASES: Final[tuple[str, ...]] = (
    "key_hash",
    "hashed_key",
    "source_hash",
    "text_hash",
    "original_hash",
    "surrogate_key",
    "foreign_key_hash",
    "hash",
)
_SURROGATE_FIELD_ALIASES: Final[tuple[str, ...]] = (
    "surrogate",
    "surrogate_value",
    "surrogate_id",
    "replacement",
    "value",
)
_MAP_NAME_ALIASES: Final[tuple[str, ...]] = (
    "name",
    "map_name",
    "table",
    "table_name",
    "id",
)
_ENTRY_CONTAINER_ALIASES: Final[tuple[str, ...]] = (
    "entries",
    "bindings",
    "items",
    "mapping",
    "surrogate_map",
)
_KEY_LIST_ALIASES: Final[tuple[str, ...]] = (
    "key_hashes",
    "hashed_keys",
    "source_hashes",
)
_SURROGATE_LIST_ALIASES: Final[tuple[str, ...]] = (
    "surrogates",
    "surrogate_values",
    "replacement_values",
)
_ROW_CONTAINER_ALIASES: Final[tuple[str, ...]] = ("rows", "records")
_KEY_COLUMN_ALIASES: Final[tuple[str, ...]] = (
    "key_field",
    "key_column",
    "hash_field",
    "hash_column",
)
_SURROGATE_COLUMN_ALIASES: Final[tuple[str, ...]] = (
    "surrogate_field",
    "surrogate_column",
    "replacement_field",
)
_CARDINALITY_ALIASES: Final[tuple[str, ...]] = (
    "cardinality",
    "expected_cardinality",
    "expected_key_count",
    "unique_key_count",
)
_ENTRY_COUNT_ALIASES: Final[tuple[str, ...]] = (
    "entry_count",
    "expected_entry_count",
)
_PARENT_ALIASES: Final[tuple[str, ...]] = (
    "parent",
    "parent_map",
    "parent_table",
    "referenced",
    "referenced_map",
)
_CHILD_ALIASES: Final[tuple[str, ...]] = (
    "child",
    "child_map",
    "child_table",
    "referencing",
    "referencing_map",
)

_MAP_STRUCTURAL_FIELDS: Final[frozenset[str]] = frozenset(
    {
        *_MAP_NAME_ALIASES,
        *_ENTRY_CONTAINER_ALIASES,
        *_KEY_LIST_ALIASES,
        *_SURROGATE_LIST_ALIASES,
        *_ROW_CONTAINER_ALIASES,
        *_KEY_COLUMN_ALIASES,
        *_SURROGATE_COLUMN_ALIASES,
        "metadata",
    }
)
_MAP_METADATA_FIELDS: Final[frozenset[str]] = frozenset(
    {*_CARDINALITY_ALIASES, *_ENTRY_COUNT_ALIASES, *_KEY_LIST_ALIASES}
)
_MAP_DEFINITION_FIELDS: Final = _MAP_STRUCTURAL_FIELDS | _MAP_METADATA_FIELDS
_MAP_DEFINITION_MARKERS: Final = _MAP_DEFINITION_FIELDS - frozenset(_MAP_NAME_ALIASES)
_ENTRY_FIELDS: Final = frozenset({*_KEY_FIELD_ALIASES, *_SURROGATE_FIELD_ALIASES})
_RELATIONSHIP_FIELDS: Final = frozenset({*_PARENT_ALIASES, *_CHILD_ALIASES})


class SurrogateAuditInputError(ValueError):
    """Raised when a surrogate-audit input cannot be interpreted safely."""


def _empty_failure_counts() -> dict[str, int]:
    return {category: 0 for category in SURROGATE_AUDIT_FAILURE_CATEGORIES}


@dataclass(frozen=True)
class SurrogateMapAuditFailure:
    """Aggregate count for one referential-integrity failure category."""

    category: str
    count: int

    def __post_init__(self) -> None:
        if self.category not in SURROGATE_AUDIT_FAILURE_CATEGORIES:
            raise ValueError("unknown surrogate-audit failure category")
        if isinstance(self.count, bool) or not isinstance(self.count, int):
            raise TypeError("failure count must be an integer")
        if self.count < 0 or self.count > _MAX_COUNT:
            raise ValueError("failure count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Return the category and count without sensitive evidence."""

        return {"category": self.category, "count": self.count}


@dataclass(frozen=True)
class SurrogateMapAuditReport:
    """Counts-only result from :func:`audit_surrogate_maps`.

    ``failure_categories`` always contains all four stable category names,
    including categories whose count is zero.  A count represents an affected
    map, key group, or relationship depending on the check; it is never a
    list of offending values.
    """

    checked_maps: int
    checked_entries: int
    checked_keys: int
    relationships_checked: int
    failure_categories: Mapping[str, int] = field(default_factory=_empty_failure_counts)

    def __post_init__(self) -> None:
        for field_name in (
            "checked_maps",
            "checked_entries",
            "checked_keys",
            "relationships_checked",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if value < 0 or value > _MAX_COUNT:
                raise ValueError(f"{field_name} must be non-negative")

        if self.checked_maps > _MAX_MAPS:
            raise ValueError("checked_maps exceeds the map limit")
        if self.checked_entries > self.checked_maps * _MAX_ENTRIES_PER_MAP:
            raise ValueError("checked_entries exceeds the entry limit")
        if self.relationships_checked > _MAX_RELATIONSHIPS:
            raise ValueError("relationships_checked exceeds the relationship limit")

        if not isinstance(self.failure_categories, Mapping):
            raise TypeError("failure_categories must be a mapping")
        counts = _empty_failure_counts()
        items = _mapping_items(
            self.failure_categories,
            "failure categories",
            limit=len(SURROGATE_AUDIT_FAILURE_CATEGORIES),
        )
        seen: set[str] = set()
        for category, value in items:
            if category not in counts:
                raise ValueError("failure_categories contains an unknown category")
            if category in seen:
                raise ValueError("failure_categories contains duplicate categories")
            seen.add(category)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("failure category counts must be integers")
            if value < 0 or value > _MAX_COUNT:
                raise ValueError("failure category counts must be non-negative")
            counts[category] = value

        if self.checked_keys > self.checked_entries:
            raise ValueError("checked_keys cannot exceed checked_entries")
        if self.checked_maps < 2 and self.relationships_checked:
            raise ValueError("relationships require at least two checked maps")
        if not self.relationships_checked and (
            counts[ORPHAN_FAILURE] or counts[CROSS_TABLE_CONSISTENCY_FAILURE]
        ):
            raise ValueError("relationship failures require a checked relationship")
        if not self.checked_maps and (self.checked_entries or any(counts.values())):
            raise ValueError("an empty audit cannot contain entries or failures")
        object.__setattr__(self, "failure_categories", MappingProxyType(counts))

    @property
    def failure_counts(self) -> Mapping[str, int]:
        """Alias for the stable aggregate category counts."""

        return self.failure_categories

    @property
    def total_failures(self) -> int:
        """Return the total number of counted integrity findings."""

        return sum(self.failure_categories.values())

    @property
    def passed(self) -> bool:
        """Return whether every checked integrity category is clean."""

        return self.total_failures == 0

    @property
    def failures(self) -> tuple[SurrogateMapAuditFailure, ...]:
        """Return non-zero categories in deterministic category order."""

        return tuple(
            SurrogateMapAuditFailure(category, self.failure_categories[category])
            for category in SURROGATE_AUDIT_FAILURE_CATEGORIES
            if self.failure_categories[category]
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible, counts-only report."""

        counts = dict(self.failure_categories)
        return {
            "schema_version": SCHEMA_VERSION,
            "deterministic": True,
            "passed": self.passed,
            "checked_maps": self.checked_maps,
            "checked_entries": self.checked_entries,
            "checked_keys": self.checked_keys,
            "relationships_checked": self.relationships_checked,
            "failure_categories": counts,
            "failure_counts": dict(counts),
            "failures": [failure.to_dict() for failure in self.failures],
            "total_failures": self.total_failures,
        }

    def as_dict(self) -> dict[str, Any]:
        """Return :meth:`to_dict` using the common report naming."""

        return self.to_dict()


@dataclass(frozen=True)
class _Binding:
    """Internal binding whose values are never placed in a report."""

    key_hash: str = field(repr=False)
    surrogate: str = field(repr=False)


@dataclass(frozen=True)
class _MapData:
    """Normalized map data held only during the in-memory audit."""

    name: str = field(repr=False)
    bindings: tuple[_Binding, ...] = field(repr=False)
    expected_cardinality: int | None = None
    expected_entry_count: int | None = None
    declared_key_hashes: frozenset[str] | None = field(default=None, repr=False)
    declared_key_count: int | None = field(default=None, repr=False)
    invalid_entry_count: int = 0


def _audit_surrogate_maps(
    surrogate_maps: Mapping[str, Any] | Iterable[Any],
    relationships: Mapping[str, str] | Iterable[Any] | None = None,
    *,
    map_metadata: Mapping[str, Any] | None = None,
    expected_cardinality: Mapping[str, int] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SurrogateMapAuditReport:
    """Audit hashed surrogate maps without returning sensitive values.

    Args:
        surrogate_maps: A mapping of map/table names to map definitions, or a
            sequence of definitions.  A definition may contain ``entries``
            with ``key_hash`` and ``surrogate`` fields, a direct
            ``key_hash -> surrogate`` mapping, or parallel ``key_hashes`` and
            ``surrogates`` sequences.  A single map may be supplied directly
            and is assigned an internal default name.
        relationships: Optional parent/child map relationships.  Each item
            may be ``(parent, child)`` or a mapping using names such as
            ``parent_map`` and ``child_map``.  If omitted and multiple maps
            are present, shared hashes are compared across every map pair;
            orphan checks require an explicit relationship.
        map_metadata: Optional mapping of map name to metadata.  Supported
            metadata includes ``cardinality`` (distinct key count),
            ``entry_count``, and ``key_hashes``.
        expected_cardinality: Convenience mapping of map name to expected
            distinct key count.
        metadata: Optional bundle metadata.  ``relationships`` and
            ``map_metadata`` keys are recognized; other values are ignored.

    Returns:
        A deterministic :class:`SurrogateMapAuditReport` containing aggregate
        counts for cardinality, collision, orphan, and cross-table
        consistency failures.

    Raises:
        SurrogateAuditInputError: If the input shape is not auditable.  Error
            messages contain only fixed schema descriptions, never input
            values.
    """

    map_source, embedded_relationships, embedded_metadata = _unwrap_bundle(
        surrogate_maps
    )
    global_metadata = _merge_metadata_sources(embedded_metadata, metadata)
    effective_map_metadata = _merge_map_metadata(
        global_metadata.get("metadata"),
        global_metadata.get("map_metadata"),
    )
    effective_map_metadata = _merge_map_metadata(
        effective_map_metadata,
        map_metadata,
    )
    effective_map_metadata = _apply_expected_cardinality(
        effective_map_metadata,
        expected_cardinality,
    )

    maps = _coerce_maps(map_source, effective_map_metadata)
    unknown_metadata = set(effective_map_metadata) - {item.name for item in maps}
    if unknown_metadata:
        raise SurrogateAuditInputError("map metadata references an unknown map")
    relationship_source = relationships
    if relationship_source is None:
        relationship_source = embedded_relationships
    if relationship_source is None:
        relationship_source = global_metadata.get("relationships")
    explicit_relationships = relationship_source is not None
    normalized_relationships = _coerce_relationships(relationship_source, maps)

    if not explicit_relationships and len(maps) > 1:
        normalized_relationships = tuple(
            (left.name, right.name)
            for index, left in enumerate(maps)
            for right in maps[index + 1 :]
        )

    counts = _empty_failure_counts()
    checked_entries = 0
    checked_keys = 0
    bindings_by_map: dict[str, tuple[_Binding, ...]] = {}

    for map_data in maps:
        checked_entries += len(map_data.bindings) + map_data.invalid_entry_count
        by_key: dict[str, set[str]] = {}
        by_surrogate: dict[str, set[str]] = {}
        for binding in map_data.bindings:
            by_key.setdefault(binding.key_hash, set()).add(binding.surrogate)
            by_surrogate.setdefault(binding.surrogate, set()).add(binding.key_hash)
        checked_keys += len(by_key)
        bindings_by_map[map_data.name] = map_data.bindings

        counts[CARDINALITY_FAILURE] += map_data.invalid_entry_count
        counts[CARDINALITY_FAILURE] += _cardinality_failures(map_data, by_key)
        counts[COLLISION_FAILURE] += sum(len(values) > 1 for values in by_key.values())
        counts[COLLISION_FAILURE] += sum(
            len(values) > 1 for values in by_surrogate.values()
        )

    for parent_name, child_name in normalized_relationships:
        parent_by_key = _bindings_by_key(bindings_by_map[parent_name])
        child_by_key = _bindings_by_key(bindings_by_map[child_name])
        if explicit_relationships:
            counts[ORPHAN_FAILURE] += len(set(child_by_key) - set(parent_by_key))
        counts[CROSS_TABLE_CONSISTENCY_FAILURE] += sum(
            parent_by_key[key] != child_by_key[key]
            for key in set(parent_by_key).intersection(child_by_key)
        )

    return SurrogateMapAuditReport(
        checked_maps=len(maps),
        checked_entries=checked_entries,
        checked_keys=checked_keys,
        relationships_checked=len(normalized_relationships),
        failure_categories=counts,
    )


def _cardinality_failures(
    map_data: _MapData,
    by_key: Mapping[str, set[str]],
) -> int:
    failures = 0
    observed_cardinality = len(by_key)
    if (
        map_data.expected_cardinality is not None
        and observed_cardinality != map_data.expected_cardinality
    ):
        failures += 1
    observed_entry_count = len(map_data.bindings) + map_data.invalid_entry_count
    if (
        map_data.expected_entry_count is not None
        and observed_entry_count != map_data.expected_entry_count
    ):
        failures += 1
    if map_data.declared_key_hashes is not None:
        if map_data.declared_key_count != len(map_data.declared_key_hashes) or set(
            by_key
        ) != set(map_data.declared_key_hashes):
            failures += 1
    return failures


def _bindings_by_key(bindings: Sequence[_Binding]) -> dict[str, set[str]]:
    by_key: dict[str, set[str]] = {}
    for binding in bindings:
        by_key.setdefault(binding.key_hash, set()).add(binding.surrogate)
    return by_key


def _unwrap_bundle(
    value: Mapping[str, Any] | Iterable[Any],
) -> tuple[Any, Any, dict[str, Any]]:
    if not isinstance(value, Mapping) or "maps" not in value:
        return value, None, {}
    map_value = value.get("maps")
    if not isinstance(map_value, Mapping) and not _is_iterable(map_value):
        return value, None, {}
    embedded_metadata: dict[str, Any] = {}
    raw_metadata = value.get("metadata")
    if raw_metadata is not None:
        if not isinstance(raw_metadata, Mapping):
            raise SurrogateAuditInputError("metadata must be a mapping")
        for key, item in _mapping_items(
            raw_metadata,
            "metadata",
            limit=_MAX_METADATA_FIELDS,
        ):
            if isinstance(key, str):
                embedded_metadata[key] = item
    if "map_metadata" in value:
        embedded_metadata["map_metadata"] = value.get("map_metadata")
    if "relationships" in value:
        embedded_metadata["relationships"] = value.get("relationships")
    return map_value, value.get("relationships"), embedded_metadata


def _merge_metadata_sources(
    first: Mapping[str, Any] | None,
    second: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for source in (first, second):
        if source is None:
            continue
        if not isinstance(source, Mapping):
            raise SurrogateAuditInputError("metadata must be a mapping")
        for key, value in _mapping_items(
            source, "metadata", limit=_MAX_METADATA_FIELDS
        ):
            if isinstance(key, str):
                merged[key] = value
    return merged


def _merge_map_metadata(
    first: Mapping[str, Any] | None,
    second: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for source in (first, second):
        if source is None:
            continue
        if not isinstance(source, Mapping):
            raise SurrogateAuditInputError("map_metadata must be a mapping")
        for name, value in _mapping_items(source, "map metadata", limit=_MAX_MAPS):
            normalized_name = _map_name(name)
            if value is None:
                continue
            if not isinstance(value, Mapping):
                raise SurrogateAuditInputError("map metadata entries must be mappings")
            current = merged.setdefault(normalized_name, {})
            for key, item in _mapping_items(
                value,
                "map metadata entry",
                limit=_MAX_METADATA_FIELDS,
            ):
                if not isinstance(key, str) or key not in _MAP_METADATA_FIELDS:
                    raise SurrogateAuditInputError(
                        "map metadata contains unsupported fields"
                    )
                current[key] = item
    return merged


def _apply_expected_cardinality(
    metadata: Mapping[str, Mapping[str, Any]],
    expected: Mapping[str, int] | None,
) -> dict[str, dict[str, Any]]:
    result = {name: dict(value) for name, value in metadata.items()}
    if expected is None:
        return result
    if not isinstance(expected, Mapping):
        raise SurrogateAuditInputError("expected_cardinality must be a mapping")
    seen: set[str] = set()
    for name, value in _mapping_items(
        expected,
        "expected cardinality",
        limit=_MAX_MAPS,
    ):
        normalized_name = _map_name(name)
        if normalized_name in seen:
            raise SurrogateAuditInputError(
                "expected_cardinality contains duplicate map names"
            )
        seen.add(normalized_name)
        current = result.setdefault(normalized_name, {})
        current["expected_cardinality"] = value
    return result


def _coerce_maps(
    source: Any,
    map_metadata: Mapping[str, Mapping[str, Any]],
) -> tuple[_MapData, ...]:
    specs = _map_specs(source)
    normalized: list[_MapData] = []
    seen_names: set[str] = set()
    for name, definition in specs:
        normalized_name = _map_name(name)
        if normalized_name in seen_names:
            raise SurrogateAuditInputError("map names must be unique")
        seen_names.add(normalized_name)
        metadata = map_metadata.get(normalized_name, {})
        normalized.append(_coerce_map_definition(normalized_name, definition, metadata))
    return tuple(sorted(normalized, key=lambda item: item.name))


def _map_specs(source: Any) -> list[tuple[str, Any]]:
    if isinstance(source, Mapping):
        if (
            _looks_like_entry(source)
            or _looks_like_direct_map(source)
            or _looks_like_map_definition(source)
        ):
            name = _field_value(source, _MAP_NAME_ALIASES, default=_DEFAULT_MAP_NAME)
            return [(_map_name(name), source)]
        return [
            (name, definition)
            for name, definition in _mapping_items(
                source,
                "surrogate maps",
                limit=_MAX_MAPS,
            )
        ]
    items = _materialize(source, "surrogate maps", limit=_MAX_MAPS)
    if not items:
        return []
    if all(_looks_like_entry(item) or _looks_like_pair(item) for item in items):
        return [(_DEFAULT_MAP_NAME, items)]
    specs: list[tuple[str, Any]] = []
    for index, item in enumerate(items):
        if isinstance(item, Mapping):
            name = _field_value(item, _MAP_NAME_ALIASES, default=None)
            definition = item
            if name is None:
                name = f"map-{index + 1}"
        elif _looks_like_named_definition(item):
            name, definition = item
        else:
            raise SurrogateAuditInputError("map definitions must be mappings")
        specs.append((_map_name(name), definition))
    return specs


def _coerce_map_definition(
    name: str,
    definition: Any,
    external_metadata: Mapping[str, Any],
) -> _MapData:
    local_metadata: dict[str, Any] = dict(external_metadata)
    source: Any = definition

    if isinstance(definition, Mapping):
        if _looks_like_map_definition(definition):
            definition = _copy_closed_mapping(
                definition,
                allowed=_MAP_DEFINITION_FIELDS,
                label="map definition",
            )
            source = definition
        nested_metadata = definition.get("metadata")
        if nested_metadata is not None:
            if not isinstance(nested_metadata, Mapping):
                raise SurrogateAuditInputError("map metadata must be a mapping")
            for key, item in _mapping_items(
                nested_metadata,
                "map metadata",
                limit=_MAX_METADATA_FIELDS,
            ):
                if not isinstance(key, str) or key not in _MAP_METADATA_FIELDS:
                    raise SurrogateAuditInputError(
                        "map metadata contains unsupported fields"
                    )
                local_metadata[key] = item
        for field_name in _MAP_METADATA_FIELDS:
            if field_name in definition:
                local_metadata[field_name] = definition[field_name]

        container_name = _first_present_name(definition, _ENTRY_CONTAINER_ALIASES)
        if container_name is not None:
            source = definition[container_name]
        elif _looks_like_entry(definition):
            source = (definition,)
        elif _has_parallel_lists(definition):
            source = _parallel_entries(definition)
        else:
            row_container = _first_present_name(definition, _ROW_CONTAINER_ALIASES)
            if row_container is not None:
                source = _row_entries(definition, definition[row_container])
            else:
                source = {
                    key: value
                    for key, value in definition.items()
                    if key not in _MAP_STRUCTURAL_FIELDS
                    and key not in _MAP_METADATA_FIELDS
                }
                if source and not all(
                    isinstance(key, str) and isinstance(value, str)
                    for key, value in source.items()
                ):
                    raise SurrogateAuditInputError(
                        "map definitions must contain surrogate bindings"
                    )

    bindings, invalid_entries = _coerce_entries(source)
    expected_key_count = _optional_nonnegative_int(
        _field_value(local_metadata, _CARDINALITY_ALIASES, default=None),
        "cardinality",
    )
    expected_entry_count = _optional_nonnegative_int(
        _field_value(local_metadata, _ENTRY_COUNT_ALIASES, default=None),
        "entry count",
    )
    declared_hashes, declared_count = _declared_hashes(local_metadata)
    return _MapData(
        name=name,
        bindings=tuple(
            sorted(bindings, key=lambda item: (item.key_hash, item.surrogate))
        ),
        expected_cardinality=expected_key_count,
        expected_entry_count=expected_entry_count,
        declared_key_hashes=declared_hashes,
        declared_key_count=declared_count,
        invalid_entry_count=invalid_entries,
    )


def _coerce_entries(source: Any) -> tuple[list[_Binding], int]:
    if source is None:
        return [], 0
    if isinstance(source, Mapping):
        if _looks_like_entry(source):
            raw_items: Iterable[Any] = (source,)
        else:
            raw_items = _mapping_items(
                source,
                "map entries",
                limit=_MAX_ENTRIES_PER_MAP,
            )
    else:
        raw_items = _materialize(
            source,
            "map entries",
            limit=_MAX_ENTRIES_PER_MAP,
        )

    bindings: list[_Binding] = []
    invalid_entries = 0
    for item in raw_items:
        binding = _coerce_binding(item)
        if binding is None:
            invalid_entries += 1
        else:
            bindings.append(binding)
    return bindings, invalid_entries


def _coerce_binding(item: Any) -> _Binding | None:
    key_hash: Any = _MISSING
    surrogate: Any = _MISSING
    if isinstance(item, Mapping):
        item = _copy_closed_mapping(
            item,
            allowed=_ENTRY_FIELDS,
            label="surrogate binding",
        )
        key_hash = _field_value(item, _KEY_FIELD_ALIASES, default=_MISSING)
        surrogate = _field_value(
            item,
            _SURROGATE_FIELD_ALIASES,
            default=_MISSING,
        )
    elif _looks_like_pair(item):
        key_hash, surrogate = item

    if not _valid_text(key_hash) or not _valid_text(surrogate):
        return None
    return _Binding(key_hash=key_hash, surrogate=surrogate)


def _row_entries(definition: Mapping[str, Any], rows: Any) -> tuple[Any, ...]:
    row_items = _materialize(rows, "map rows", limit=_MAX_ENTRIES_PER_MAP)
    key_field = _field_value(definition, _KEY_COLUMN_ALIASES, default="key_hash")
    surrogate_field = _field_value(
        definition,
        _SURROGATE_COLUMN_ALIASES,
        default="surrogate",
    )
    if not _valid_text(key_field) or not _valid_text(surrogate_field):
        raise SurrogateAuditInputError("map key and surrogate fields must be strings")
    entries: list[dict[str, Any]] = []
    for row in row_items:
        if not isinstance(row, Mapping):
            entries.append({})
            continue
        entries.append(
            {
                "key_hash": row.get(key_field),
                "surrogate": row.get(surrogate_field),
            }
        )
    return tuple(entries)


def _has_parallel_lists(definition: Mapping[str, Any]) -> bool:
    return (
        _first_present_name(definition, _KEY_LIST_ALIASES) is not None
        or _first_present_name(
            definition,
            _SURROGATE_LIST_ALIASES,
        )
        is not None
    )


def _parallel_entries(definition: Mapping[str, Any]) -> tuple[tuple[Any, Any], ...]:
    key_name = _first_present_name(definition, _KEY_LIST_ALIASES)
    surrogate_name = _first_present_name(definition, _SURROGATE_LIST_ALIASES)
    if key_name is None or surrogate_name is None:
        raise SurrogateAuditInputError(
            "parallel key and surrogate sequences are both required"
        )
    keys = _materialize(
        definition[key_name],
        "key hashes",
        limit=_MAX_ENTRIES_PER_MAP,
    )
    surrogates = _materialize(
        definition[surrogate_name],
        "surrogates",
        limit=_MAX_ENTRIES_PER_MAP,
    )
    if len(keys) != len(surrogates):
        raise SurrogateAuditInputError("parallel key and surrogate sequences differ")
    return tuple(zip(keys, surrogates))


def _declared_hashes(
    metadata: Mapping[str, Any],
) -> tuple[frozenset[str] | None, int | None]:
    name = _first_present_name(metadata, _KEY_LIST_ALIASES)
    if name is None:
        return None, None
    values = _materialize(
        metadata[name],
        "declared key hashes",
        limit=_MAX_ENTRIES_PER_MAP,
    )
    if not all(_valid_text(value) for value in values):
        raise SurrogateAuditInputError("declared key hashes must be strings")
    declared = tuple(values)
    return frozenset(declared), len(declared)


def _coerce_relationships(
    source: Any,
    maps: Sequence[_MapData],
) -> tuple[tuple[str, str], ...]:
    if source is None:
        return ()
    if isinstance(source, Mapping):
        if _first_present_name(source, _PARENT_ALIASES) is not None:
            items: Iterable[Any] = (source,)
        elif "relationships" in source:
            items = _materialize(
                source["relationships"],
                "relationships",
                limit=_MAX_RELATIONSHIPS,
            )
        else:
            items = tuple(
                (parent, child)
                for child, parent in _mapping_items(
                    source,
                    "relationships",
                    limit=_MAX_RELATIONSHIPS,
                )
            )
    elif _looks_like_pair(source):
        items = (source,)
    else:
        items = _materialize(
            source,
            "relationships",
            limit=_MAX_RELATIONSHIPS,
        )

    known = {item.name for item in maps}
    normalized: set[tuple[str, str]] = set()
    for item in items:
        if isinstance(item, Mapping):
            item = _copy_closed_mapping(
                item,
                allowed=_RELATIONSHIP_FIELDS,
                label="relationship",
            )
            parent = _field_value(item, _PARENT_ALIASES, default=_MISSING)
            child = _field_value(item, _CHILD_ALIASES, default=_MISSING)
        elif _looks_like_pair(item):
            parent, child = item
        else:
            raise SurrogateAuditInputError(
                "relationships must contain parent/child pairs"
            )
        parent_name = _map_name(parent)
        child_name = _map_name(child)
        if parent_name not in known or child_name not in known:
            raise SurrogateAuditInputError("relationship references an unknown map")
        if parent_name == child_name:
            raise SurrogateAuditInputError("relationships must reference distinct maps")
        if (parent_name, child_name) in normalized:
            raise SurrogateAuditInputError("relationships must be unique")
        normalized.add((parent_name, child_name))
    return tuple(sorted(normalized))


def _field_value(
    mapping: Mapping[str, Any],
    aliases: Sequence[str],
    *,
    default: Any = _MISSING,
) -> Any:
    present = [name for name in aliases if name in mapping]
    if not present:
        return default
    if len(present) > 1:
        raise SurrogateAuditInputError("input contains ambiguous aliases")
    return mapping[present[0]]


def _first_present_name(
    mapping: Mapping[str, Any],
    aliases: Sequence[str],
) -> str | None:
    present = [name for name in aliases if name in mapping]
    if len(present) > 1:
        raise SurrogateAuditInputError("input contains ambiguous aliases")
    return present[0] if present else None


def _optional_nonnegative_int(value: Any, field_name: str) -> int | None:
    if value is None or value is _MISSING:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > _MAX_COUNT
    ):
        raise SurrogateAuditInputError(f"{field_name} must be a non-negative integer")
    return value


def _map_name(value: Any) -> str:
    if not _valid_text(value):
        raise SurrogateAuditInputError("map names must be non-empty strings")
    normalized = unicodedata.normalize("NFC", value.strip())
    if not normalized or len(normalized) > _MAX_TEXT_CHARS:
        raise SurrogateAuditInputError("map names must be non-empty strings")
    return normalized


def _valid_text(value: Any) -> bool:
    return isinstance(value, str) and 0 < len(value) <= _MAX_TEXT_CHARS


def _materialize(value: Any, field_name: str, *, limit: int) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not _is_iterable(value):
        raise SurrogateAuditInputError(f"{field_name} must be an iterable")
    try:
        iterator = iter(value)
    except MemoryError:
        raise
    except Exception:
        raise SurrogateAuditInputError(f"{field_name} must be an iterable") from None
    items: list[Any] = []
    for _ in range(limit + 1):
        try:
            items.append(next(iterator))
        except StopIteration:
            return tuple(items)
        except MemoryError:
            raise
        except Exception:
            raise SurrogateAuditInputError(
                f"{field_name} must be an iterable"
            ) from None
    raise SurrogateAuditInputError(f"{field_name} exceeds the item limit")


def _mapping_items(
    value: Mapping[Any, Any],
    field_name: str,
    *,
    limit: int,
) -> list[tuple[Any, Any]]:
    try:
        raw_items = value.items()
    except MemoryError:
        raise
    except Exception:
        raise SurrogateAuditInputError(f"{field_name} must be a mapping") from None
    items = _materialize(raw_items, field_name, limit=limit)
    if not all(isinstance(item, tuple) and len(item) == 2 for item in items):
        raise SurrogateAuditInputError(f"{field_name} must be a mapping")
    return cast(list[tuple[Any, Any]], list(items))


def _copy_closed_mapping(
    value: Mapping[Any, Any],
    *,
    allowed: frozenset[str],
    label: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in _mapping_items(value, label, limit=len(allowed)):
        if not isinstance(key, str) or key not in allowed:
            raise SurrogateAuditInputError(f"{label} contains unsupported fields")
        if key in result:
            raise SurrogateAuditInputError(f"{label} contains duplicate fields")
        result[key] = item
    return result


def _is_iterable(value: Any) -> bool:
    if isinstance(value, (str, bytes, bytearray)):
        return False
    try:
        iter(value)
    except TypeError:
        return False
    return True


def _looks_like_entry(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    return (
        _first_present_name(value, _KEY_FIELD_ALIASES) is not None
        and _first_present_name(value, _SURROGATE_FIELD_ALIASES) is not None
    )


def _looks_like_pair(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and len(value) == 2
        and all(isinstance(item, str) for item in value)
    )


def _looks_like_direct_map(value: Mapping[str, Any]) -> bool:
    items = _mapping_items(
        value,
        "surrogate map",
        limit=_MAX_ENTRIES_PER_MAP,
    )
    if not items:
        return False
    if any(
        key in _MAP_STRUCTURAL_FIELDS or key in _MAP_METADATA_FIELDS for key, _ in items
    ):
        return False
    return all(isinstance(key, str) and isinstance(item, str) for key, item in items)


def _looks_like_map_definition(value: Mapping[str, Any]) -> bool:
    return any(
        key in _MAP_DEFINITION_MARKERS
        for key, _ in _mapping_items(
            value,
            "surrogate map",
            limit=_MAX_ENTRIES_PER_MAP,
        )
    )


def _looks_like_named_definition(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and len(value) == 2
        and isinstance(value[0], str)
        and not isinstance(value[1], str)
    )


def audit_surrogate_maps(
    surrogate_maps: Mapping[str, Any] | Iterable[Any],
    relationships: Mapping[str, str] | Iterable[Any] | None = None,
    *,
    map_metadata: Mapping[str, Any] | None = None,
    expected_cardinality: Mapping[str, int] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SurrogateMapAuditReport:
    """Audit hashed surrogate maps and return deterministic aggregate counts.

    Args:
        surrogate_maps: Named map definitions or an iterable of definitions.
        relationships: Optional explicit parent/child map relationships.
        map_metadata: Optional map-specific cardinality metadata.
        expected_cardinality: Optional expected distinct-key counts by map.
        metadata: Optional bundle metadata containing relationships or map metadata.

    Returns:
        A counts-only report with stable failure categories.

    Raises:
        SurrogateAuditInputError: If an input is invalid, ambiguous, or exceeds
            a resource limit.
    """

    try:
        return _audit_surrogate_maps(
            surrogate_maps,
            relationships,
            map_metadata=map_metadata,
            expected_cardinality=expected_cardinality,
            metadata=metadata,
        )
    except MemoryError:
        raise
    except SurrogateAuditInputError:
        raise
    except Exception:
        raise SurrogateAuditInputError("surrogate audit input is invalid") from None


audit_surrogate_map = audit_surrogate_maps
check_surrogate_map_integrity = audit_surrogate_maps
