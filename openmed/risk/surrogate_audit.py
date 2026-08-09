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

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

__all__ = [
    "CARDINALITY_FAILURE",
    "COLLISION_FAILURE",
    "CROSS_TABLE_CONSISTENCY_FAILURE",
    "FAILURE_CATEGORIES",
    "ORPHAN_FAILURE",
    "SurrogateAuditInputError",
    "SurrogateMapAuditFailure",
    "SurrogateMapAuditReport",
    "audit_surrogate_map",
    "audit_surrogate_maps",
    "check_surrogate_map_integrity",
]

SCHEMA_VERSION: Final = 1

CARDINALITY_FAILURE: Final = "cardinality"
COLLISION_FAILURE: Final = "collision"
ORPHAN_FAILURE: Final = "orphan"
CROSS_TABLE_CONSISTENCY_FAILURE: Final = "cross_table_consistency"
FAILURE_CATEGORIES: Final[tuple[str, ...]] = (
    CARDINALITY_FAILURE,
    COLLISION_FAILURE,
    ORPHAN_FAILURE,
    CROSS_TABLE_CONSISTENCY_FAILURE,
)

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


class SurrogateAuditInputError(ValueError):
    """Raised when a surrogate-audit input cannot be interpreted safely."""


def _empty_failure_counts() -> dict[str, int]:
    return {category: 0 for category in FAILURE_CATEGORIES}


@dataclass(frozen=True)
class SurrogateMapAuditFailure:
    """Aggregate count for one referential-integrity failure category."""

    category: str
    count: int

    def __post_init__(self) -> None:
        if self.category not in FAILURE_CATEGORIES:
            raise ValueError("unknown surrogate-audit failure category")
        if isinstance(self.count, bool) or not isinstance(self.count, int):
            raise TypeError("failure count must be an integer")
        if self.count < 0:
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
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")

        if not isinstance(self.failure_categories, Mapping):
            raise TypeError("failure_categories must be a mapping")
        unknown = set(self.failure_categories) - set(FAILURE_CATEGORIES)
        if unknown:
            raise ValueError("failure_categories contains an unknown category")
        counts = _empty_failure_counts()
        for category in FAILURE_CATEGORIES:
            value = self.failure_categories.get(category, 0)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("failure category counts must be integers")
            if value < 0:
                raise ValueError("failure category counts must be non-negative")
            counts[category] = value
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
            for category in FAILURE_CATEGORIES
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


def audit_surrogate_maps(
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
    global_metadata = _merge_metadata_sources(metadata, embedded_metadata)
    effective_map_metadata = _merge_map_metadata(
        global_metadata.get("map_metadata"),
        map_metadata,
    )
    effective_map_metadata = _merge_map_metadata(
        effective_map_metadata,
        global_metadata.get("metadata"),
    )
    effective_map_metadata = _apply_expected_cardinality(
        effective_map_metadata,
        expected_cardinality,
    )

    maps = _coerce_maps(map_source, effective_map_metadata)
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
    if isinstance(value.get("metadata"), Mapping):
        embedded_metadata.update(value["metadata"])
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
        merged.update(source)
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
        for name, value in source.items():
            normalized_name = _map_name(name)
            if value is None:
                continue
            if not isinstance(value, Mapping):
                raise SurrogateAuditInputError("map metadata entries must be mappings")
            current = merged.setdefault(normalized_name, {})
            current.update(value)
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
    for name, value in expected.items():
        normalized_name = _map_name(name)
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
        return [(str(name), definition) for name, definition in source.items()]
    items = _materialize(source, "surrogate_maps")
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
        nested_metadata = definition.get("metadata")
        if nested_metadata is not None:
            if not isinstance(nested_metadata, Mapping):
                raise SurrogateAuditInputError("map metadata must be a mapping")
            local_metadata.update(nested_metadata)
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
            raw_items = source.items()
    else:
        raw_items = _materialize(source, "map entries")

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
        key_hash = _field_value(item, _KEY_FIELD_ALIASES, default=_MISSING)
        surrogate = _field_value(
            item,
            _SURROGATE_FIELD_ALIASES,
            default=_MISSING,
        )
    elif _looks_like_pair(item):
        key_hash, surrogate = item
    else:
        key_hash = _object_field(item, _KEY_FIELD_ALIASES)
        surrogate = _object_field(item, _SURROGATE_FIELD_ALIASES)
        if key_hash is _MISSING:
            key = _object_field(item, ("key",))
            key_hash = _object_field(key, ("text_hash", "key_hash", "hash"))
        if surrogate is _MISSING:
            surrogate = _object_field(item, ("surrogate", "replacement"))

    if not _valid_text(key_hash) or not _valid_text(surrogate):
        return None
    return _Binding(key_hash=key_hash, surrogate=surrogate)


def _row_entries(definition: Mapping[str, Any], rows: Any) -> tuple[Any, ...]:
    row_items = _materialize(rows, "map rows")
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
    keys = _materialize(definition[key_name], "key hashes")
    surrogates = _materialize(definition[surrogate_name], "surrogates")
    if len(keys) != len(surrogates):
        raise SurrogateAuditInputError("parallel key and surrogate sequences differ")
    return tuple(zip(keys, surrogates))


def _declared_hashes(
    metadata: Mapping[str, Any],
) -> tuple[frozenset[str] | None, int | None]:
    name = _first_present_name(metadata, _KEY_LIST_ALIASES)
    if name is None:
        return None, None
    values = _materialize(metadata[name], "declared key hashes")
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
            items = _materialize(source["relationships"], "relationships")
        else:
            items = tuple((parent, child) for child, parent in source.items())
    elif _looks_like_pair(source):
        items = (source,)
    else:
        items = _materialize(source, "relationships")

    known = {item.name for item in maps}
    normalized: set[tuple[str, str]] = set()
    for item in items:
        if isinstance(item, Mapping):
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
    value = mapping[present[0]]
    for name in present[1:]:
        if mapping[name] != value:
            raise SurrogateAuditInputError("conflicting input fields")
    return value


def _first_present_name(
    mapping: Mapping[str, Any],
    aliases: Sequence[str],
) -> str | None:
    for name in aliases:
        if name in mapping:
            return name
    return None


def _object_field(value: Any, aliases: Sequence[str]) -> Any:
    if value is None:
        return _MISSING
    for name in aliases:
        try:
            result = getattr(value, name)
        except AttributeError:
            continue
        if result is not None:
            return result
    return _MISSING


def _optional_nonnegative_int(value: Any, field_name: str) -> int | None:
    if value is None or value is _MISSING:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SurrogateAuditInputError(f"{field_name} must be a non-negative integer")
    return value


def _map_name(value: Any) -> str:
    if not _valid_text(value):
        raise SurrogateAuditInputError("map names must be non-empty strings")
    return value


def _valid_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _materialize(value: Any, field_name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not _is_iterable(value):
        raise SurrogateAuditInputError(f"{field_name} must be an iterable")
    try:
        return tuple(value)
    except (TypeError, ValueError) as exc:
        raise SurrogateAuditInputError(f"{field_name} must be an iterable") from exc


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
    if not value:
        return False
    if any(
        key in _MAP_STRUCTURAL_FIELDS or key in _MAP_METADATA_FIELDS for key in value
    ):
        return False
    return all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    )


def _looks_like_map_definition(value: Mapping[str, Any]) -> bool:
    return any(
        key in _MAP_STRUCTURAL_FIELDS or key in _MAP_METADATA_FIELDS for key in value
    )


def _looks_like_named_definition(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and len(value) == 2
        and isinstance(value[0], str)
        and not isinstance(value[1], str)
    )


audit_surrogate_map = audit_surrogate_maps
check_surrogate_map_integrity = audit_surrogate_maps
