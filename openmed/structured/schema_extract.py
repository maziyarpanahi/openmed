"""Schema-guided JSON extraction (roadmap section 4.2).

Consumers frequently need a document reduced to a fixed JSON shape: a target
schema declares the fields and their types, and the note must be projected onto
those typed slots deterministically. This module binds already-detected material
-- named entities, inline ``key: value`` pairs, and reconstructed table cells --
to the slots of a caller-supplied JSON Schema, coercing each raw value to the
declared type and recording the character offsets it came from.

The schema is a small, standard subset of JSON Schema: a top-level object with
``properties`` (each carrying a scalar ``type`` of ``string``/``integer``/
``number``/``boolean``) and an optional ``required`` list. Extraction hints ride
along as extension keywords that a standard validator ignores: ``aliases`` lists
alternative slot labels, ``entity`` names an entity label to bind, ``pattern`` is
a full-match constraint, and ``enum`` restricts the allowed values. A malformed
*schema* raises ``SchemaDefinitionError``; a malformed *document* never raises --
unfilled required slots and per-field coercion failures are reported instead, so
partial extraction always yields a result. Extraction is deterministic and
offline: no network, no model inference, and a fixed source-priority and
source-order tie-break so the same inputs always produce the same object.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any, Literal, TypedDict

SCHEMA_EXTRACT_ADVISORY = (
    "Schema-guided extraction binds already-detected entities, key-value pairs "
    "and table cells to declared JSON slots deterministically, coercing to the "
    "declared type and keeping source offsets. It does not perform free-form or "
    "model-based extraction and never invents values for unfilled slots."
)

ScalarType = Literal["string", "integer", "number", "boolean"]
FieldSource = Literal["entity", "table", "key_value"]

_SCALAR_TYPES: frozenset[str] = frozenset({"string", "integer", "number", "boolean"})
# Sources are consulted in this fixed order; the first source that yields a
# candidate for a slot wins, and within a source the earliest source offset wins.
_SOURCE_PRIORITY: tuple[FieldSource, ...] = ("entity", "table", "key_value")

# One inline ``Key: Value`` line. The key is a short label of letters, digits and
# a few separators; the value is the remainder of the line.
_KEY_VALUE_RE = re.compile(
    r"^[ \t]*(?P<key>[A-Za-z][A-Za-z0-9 /_.-]*?)[ \t]*[:：][ \t]*(?P<value>\S.*?)[ \t]*$"
)
_INTEGER_RE = re.compile(r"-?\d+")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_TRUE_TOKENS = frozenset({"true", "yes", "y", "positive", "present", "1"})
_FALSE_TOKENS = frozenset({"false", "no", "n", "negative", "absent", "0"})


class SchemaDefinitionError(ValueError):
    """Raised when the supplied target schema is itself malformed."""


class FieldBinding(TypedDict):
    """A filled slot with its coerced value and source provenance."""

    field: str
    value: Any
    raw: str
    start: int
    end: int
    source: FieldSource


class SchemaValidationIssue(TypedDict):
    """A candidate value that was found but failed to satisfy the slot."""

    field: str
    reason: str
    raw: str
    start: int
    end: int
    source: FieldSource


class SchemaExtraction(TypedDict):
    """The projection of a note onto a target schema.

    ``data`` holds only slots that filled and validated, so its values conform to
    the declared types and constraints. ``bindings`` carries per-field source
    provenance for those same slots. ``missing_required`` lists required slots
    that no source filled, and ``errors`` lists candidate values that were found
    but rejected -- neither is silently dropped.
    """

    data: dict[str, Any]
    bindings: dict[str, FieldBinding]
    missing_required: list[str]
    errors: list[SchemaValidationIssue]


class _Candidate(TypedDict):
    raw: str
    start: int
    end: int
    source: FieldSource


class _FieldSpec(TypedDict):
    name: str
    type: ScalarType
    keys: frozenset[str]
    entity_labels: frozenset[str]
    enum: tuple[str, ...] | None
    pattern: re.Pattern[str] | None


def normalize_field_key(label: str) -> str:
    """Normalize a slot name or source label to a comparable key.

    Casing, surrounding whitespace and the separators used between words
    (spaces, underscores, hyphens, dots and slashes) are collapsed so that, for
    example, ``"Patient_Age"``, ``"patient age"`` and ``"PATIENT-AGE"`` all
    compare equal.
    """

    return re.sub(r"[^a-z0-9]+", " ", label.casefold()).strip()


def extract_to_schema(
    text: str,
    schema: Mapping[str, Any],
    *,
    entities: Iterable[Any] = (),
    tables: Iterable[Mapping[str, Any]] = (),
) -> SchemaExtraction:
    """Project ``text`` onto ``schema`` and return a typed, validated object.

    Each schema slot is filled from, in priority order, a matching detected
    entity, a matching table cell, then an inline ``key: value`` line; the first
    source to yield a candidate wins and, within a source, the earliest offset in
    the document wins. The raw value is coerced to the slot's declared type and
    checked against its optional ``enum`` and ``pattern``. A slot that coerces and
    validates lands in ``data`` with source offsets recorded in ``bindings``; a
    candidate that fails coercion or a constraint is recorded in ``errors`` and
    the slot is left unfilled. Required slots that no source fills are listed in
    ``missing_required``. The document is never a cause for raising.

    Args:
        text: The source note. All reported offsets index into it.
        schema: A JSON-Schema-subset object describing the target slots.
        entities: Detected entities, each a mapping or attribute-bearing object
            exposing a ``label`` and ``start``/``end`` offsets (and optionally
            ``text``). Bound to slots that declare a matching ``entity`` label.
        tables: Reconstructed tables (``openmed.structured.Table`` shape). Each
            row's first cell is read as a slot label and its second cell as the
            value.

    Returns:
        A ``SchemaExtraction`` with the validated object, per-field provenance,
        the unfilled required slots and the rejected candidates.

    Raises:
        SchemaDefinitionError: If ``schema`` is not a well-formed object schema.
    """

    specs, required = _compile_schema(schema)

    candidates: dict[FieldSource, dict[str, list[_Candidate]]] = {
        "entity": _entity_candidates(entities),
        "table": _table_candidates(tables),
        "key_value": _key_value_candidates(text),
    }

    data: dict[str, Any] = {}
    bindings: dict[str, FieldBinding] = {}
    errors: list[SchemaValidationIssue] = []
    missing_required: list[str] = []

    for spec in specs:
        candidate = _select_candidate(spec, candidates)
        if candidate is None:
            if spec["name"] in required:
                missing_required.append(spec["name"])
            continue

        value, reason = _coerce(spec, candidate["raw"])
        if reason is not None:
            errors.append(
                SchemaValidationIssue(
                    field=spec["name"],
                    reason=reason,
                    raw=candidate["raw"],
                    start=candidate["start"],
                    end=candidate["end"],
                    source=candidate["source"],
                )
            )
            if spec["name"] in required:
                missing_required.append(spec["name"])
            continue

        data[spec["name"]] = value
        bindings[spec["name"]] = FieldBinding(
            field=spec["name"],
            value=value,
            raw=candidate["raw"],
            start=candidate["start"],
            end=candidate["end"],
            source=candidate["source"],
        )

    return SchemaExtraction(
        data=data,
        bindings=bindings,
        missing_required=missing_required,
        errors=errors,
    )


def _compile_schema(
    schema: Mapping[str, Any],
) -> tuple[list[_FieldSpec], frozenset[str]]:
    """Validate the target schema and pre-compile per-slot matching state."""

    if not isinstance(schema, Mapping):
        raise SchemaDefinitionError("schema must be a JSON object mapping")
    declared_type = schema.get("type", "object")
    if declared_type != "object":
        raise SchemaDefinitionError("schema type must be 'object'")

    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        raise SchemaDefinitionError("schema 'properties' must be a mapping")

    specs: list[_FieldSpec] = []
    for name, definition in properties.items():
        if not isinstance(name, str) or not name:
            raise SchemaDefinitionError(
                "schema property names must be non-empty strings"
            )
        if not isinstance(definition, Mapping):
            raise SchemaDefinitionError(
                f"property {name!r} definition must be a mapping"
            )

        field_type = definition.get("type")
        if field_type not in _SCALAR_TYPES:
            raise SchemaDefinitionError(
                f"property {name!r} must declare a scalar type "
                f"({', '.join(sorted(_SCALAR_TYPES))})"
            )

        specs.append(_compile_field(name, definition))

    required_raw = schema.get("required", [])
    if not isinstance(required_raw, list) or any(
        not isinstance(item, str) for item in required_raw
    ):
        raise SchemaDefinitionError("schema 'required' must be a list of strings")
    known = {spec["name"] for spec in specs}
    unknown = [item for item in required_raw if item not in known]
    if unknown:
        raise SchemaDefinitionError(
            f"schema 'required' names unknown properties: {', '.join(unknown)}"
        )

    return specs, frozenset(required_raw)


def _compile_field(name: str, definition: Mapping[str, Any]) -> _FieldSpec:
    keys = {normalize_field_key(name)}
    aliases = definition.get("aliases", [])
    if aliases:
        if not isinstance(aliases, list) or any(
            not isinstance(alias, str) for alias in aliases
        ):
            raise SchemaDefinitionError(
                f"property {name!r} 'aliases' must be a list of strings"
            )
        keys.update(normalize_field_key(alias) for alias in aliases)
    keys.discard("")

    entity = definition.get("entity")
    entity_labels: set[str] = set()
    if entity is not None:
        if isinstance(entity, str):
            entity_labels = {entity.casefold()}
        elif isinstance(entity, list) and all(
            isinstance(label, str) for label in entity
        ):
            entity_labels = {label.casefold() for label in entity}
        else:
            raise SchemaDefinitionError(
                f"property {name!r} 'entity' must be a string or list of strings"
            )

    enum = definition.get("enum")
    enum_values: tuple[str, ...] | None = None
    if enum is not None:
        if not isinstance(enum, list) or not enum:
            raise SchemaDefinitionError(
                f"property {name!r} 'enum' must be a non-empty list"
            )
        enum_values = tuple(str(value) for value in enum)

    pattern_src = definition.get("pattern")
    pattern: re.Pattern[str] | None = None
    if pattern_src is not None:
        if not isinstance(pattern_src, str):
            raise SchemaDefinitionError(f"property {name!r} 'pattern' must be a string")
        try:
            pattern = re.compile(pattern_src)
        except re.error as exc:  # pragma: no cover - defensive
            raise SchemaDefinitionError(
                f"property {name!r} 'pattern' is not a valid regex: {exc}"
            ) from exc

    return _FieldSpec(
        name=name,
        type=definition["type"],
        keys=frozenset(keys),
        entity_labels=frozenset(entity_labels),
        enum=enum_values,
        pattern=pattern,
    )


def _select_candidate(
    spec: _FieldSpec,
    candidates: Mapping[FieldSource, Mapping[str, list[_Candidate]]],
) -> _Candidate | None:
    """Pick the winning candidate for a slot by source priority then offset."""

    for source in _SOURCE_PRIORITY:
        if source == "entity":
            matches = [
                candidate
                for label in spec["entity_labels"]
                for candidate in candidates["entity"].get(label, ())
            ]
        else:
            matches = [
                candidate
                for key in spec["keys"]
                for candidate in candidates[source].get(key, ())
            ]
        if matches:
            return min(matches, key=lambda item: (item["start"], item["end"]))
    return None


def _coerce(spec: _FieldSpec, raw: str) -> tuple[Any, str | None]:
    """Coerce ``raw`` to the slot type and apply enum/pattern constraints."""

    if spec["pattern"] is not None and spec["pattern"].fullmatch(raw) is None:
        return None, "value does not match required pattern"

    field_type = spec["type"]
    if field_type == "string":
        value: Any = raw
    elif field_type == "integer":
        match = _INTEGER_RE.search(raw)
        if match is None:
            return None, "expected an integer value"
        value = int(match.group())
    elif field_type == "number":
        match = _NUMBER_RE.search(raw)
        if match is None:
            return None, "expected a numeric value"
        value = float(match.group())
    else:  # boolean
        token = normalize_field_key(raw)
        if token in _TRUE_TOKENS:
            value = True
        elif token in _FALSE_TOKENS:
            value = False
        else:
            return None, "expected a boolean value"

    if spec["enum"] is not None:
        allowed = {normalize_field_key(option): option for option in spec["enum"]}
        canonical = allowed.get(normalize_field_key(str(value)))
        if canonical is None:
            return None, "value is not one of the permitted enum options"
        # ``enum`` options are stored as strings; only a string slot adopts the
        # canonical spelling. For numeric/boolean slots the coerced value is
        # already canonical, so keep it to preserve the declared type.
        if field_type == "string":
            value = canonical

    return value, None


def _entity_candidates(entities: Iterable[Any]) -> dict[str, list[_Candidate]]:
    by_label: dict[str, list[_Candidate]] = {}
    for entity in entities:
        label = _field(entity, "label")
        start = _field(entity, "start")
        end = _field(entity, "end")
        if not isinstance(label, str) or not _is_offset(start) or not _is_offset(end):
            continue
        raw = _field(entity, "text")
        raw = str(raw).strip() if raw is not None else ""
        if not raw:
            continue
        by_label.setdefault(label.casefold(), []).append(
            _Candidate(raw=raw, start=int(start), end=int(end), source="entity")
        )
    return by_label


def _table_candidates(
    tables: Iterable[Mapping[str, Any]],
) -> dict[str, list[_Candidate]]:
    by_key: dict[str, list[_Candidate]] = {}
    for table in tables:
        cells = table.get("cells") if isinstance(table, Mapping) else None
        if not cells:
            continue
        rows: dict[int, dict[int, Mapping[str, Any]]] = {}
        for cell in cells:
            if isinstance(cell, Mapping):
                rows.setdefault(cell["row"], {})[cell["column"]] = cell
        for _, columns in sorted(rows.items()):
            ordered = [columns[col] for col in sorted(columns)]
            if len(ordered) < 2:
                continue
            key_cell, value_cell = ordered[0], ordered[1]
            key = normalize_field_key(str(key_cell.get("text", "")))
            raw = str(value_cell.get("text", "")).strip()
            if not key or not raw:
                continue
            start = value_cell.get("start")
            end = value_cell.get("end")
            if not (
                isinstance(start, int)
                and not isinstance(start, bool)
                and start >= 0
                and isinstance(end, int)
                and not isinstance(end, bool)
                and end >= 0
            ):
                continue
            by_key.setdefault(key, []).append(
                _Candidate(raw=raw, start=start, end=end, source="table")
            )
    return by_key


def _key_value_candidates(text: str) -> dict[str, list[_Candidate]]:
    by_key: dict[str, list[_Candidate]] = {}
    offset = 0
    for line in text.split("\n"):
        match = _KEY_VALUE_RE.match(line)
        if match is not None:
            key = normalize_field_key(match.group("key"))
            raw = match.group("value")
            if key and raw:
                start = offset + match.start("value")
                by_key.setdefault(key, []).append(
                    _Candidate(
                        raw=raw,
                        start=start,
                        end=start + len(raw),
                        source="key_value",
                    )
                )
        offset += len(line) + 1  # account for the newline separator
    return by_key


def _field(source: Any, name: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


def _is_offset(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


__all__ = [
    "SCHEMA_EXTRACT_ADVISORY",
    "FieldBinding",
    "FieldSource",
    "ScalarType",
    "SchemaDefinitionError",
    "SchemaExtraction",
    "SchemaValidationIssue",
    "extract_to_schema",
    "normalize_field_key",
]
