"""Shared JSON Schema model for the generated OpenMed contract types.

The canonical record contracts live in ``openmed/core/schemas/json/``. Every
emitter in ``scripts/codegen/`` lowers the same eight core schemas (span,
entity, action, audit event, attestation, relation, code and provenance)
through :func:`load_contract_model` into one small, language-neutral model and
renders it for one platform. The model only reads JSON, so the emitters run
without importing :mod:`openmed` or any third-party package.

Lowering is strict. A schema construct an emitter cannot represent exactly
(``$ref``, ``oneOf``, open records, multi-type unions, non-string enums, and so
on) raises :class:`ContractSchemaError` with a controlled reason code and the
property path, never a schema value, so a schema change can never silently
produce a lossy type.

Every emitter shares :func:`run_emitter`: without flags it rewrites its target
file; ``--check`` regenerates in memory and exits non-zero when the committed
file has drifted from the schemas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / "openmed" / "core" / "schemas" / "json"

#: The core record schemas every emitter generates, in output order.
CORE_SCHEMAS: tuple[str, ...] = (
    "span",
    "entity",
    "action",
    "audit",
    "attestation",
    "relation",
    "code",
    "provenance",
)

#: Prefix removed from schema titles before emitters add their own prefix.
TITLE_PREFIX = "OpenMed"

#: Every keyword the lowering understands. Anything else fails closed.
_ANNOTATION_KEYWORDS = frozenset({"$id", "$schema", "title", "schema_version"})
_SUPPORTED_KEYWORDS = frozenset(
    {
        "additionalProperties",
        "const",
        "description",
        "enum",
        "format",
        "items",
        "maximum",
        "maxItems",
        "maxLength",
        "minimum",
        "minItems",
        "minLength",
        "pattern",
        "properties",
        "required",
        "type",
    }
)
_CONSTRAINT_KEYWORDS = (
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
    "pattern",
    "format",
)
_SCALAR_KINDS = frozenset({"string", "integer", "number", "boolean"})
_REGEX_META = frozenset("\\^$.|?*+()[]{}")


@dataclass(frozen=True)
class NamedEnumSpec:
    """A shared string enum that emitters expose under a stable name."""

    type_name: str
    constant_name: str
    schema: str
    property: str


#: Enums exposed by name. Any other enum with exactly the same ordered values
#: (for example ``attestation.policy.default_action``) reuses the named type.
NAMED_ENUMS: tuple[NamedEnumSpec, ...] = (
    NamedEnumSpec("CanonicalLabel", "CANONICAL_LABELS", "span", "canonical_label"),
    NamedEnumSpec("PolicyLabel", "POLICY_LABELS", "span", "policy_label"),
    NamedEnumSpec("SpanAction", "SPAN_ACTIONS", "span", "action"),
)


class ContractSchemaError(Exception):
    """A value-free failure while lowering the contract schemas.

    Attributes:
        code: Controlled reason code, for example ``unsupported_keyword``.
        location: ``<schema>#/<property path>`` of the offending node.
    """

    def __init__(self, code: str, location: str) -> None:
        self.code = code
        self.location = location
        super().__init__(f"{code} at {location}")


@dataclass(frozen=True)
class EnumType:
    """String enum values, with the shared name when one applies."""

    values: tuple[str, ...]
    name: NamedEnumSpec | None = None


@dataclass(frozen=True)
class TypeRef:
    """The lowered type of one JSON value.

    Attributes:
        kind: ``string``, ``integer``, ``number``, ``boolean``, ``enum``,
            ``record``, ``array`` or ``json_object``.
        nullable: Whether JSON ``null`` is also accepted.
        const: Required literal value for ``const`` scalars, else ``None``.
        enum: Enum values for ``kind == "enum"``.
        record: Record name for ``kind == "record"``.
        items: Element type for ``kind == "array"``.
        string_prefix: Literal prefix every value must start with, taken from
            an anchored ``pattern`` (``^hmac-sha256:...`` gives
            ``hmac-sha256:``).
        constraints: Validation keywords kept for documentation, in a fixed
            order.
    """

    kind: str
    nullable: bool = False
    const: int | str | bool | None = None
    enum: EnumType | None = None
    record: str | None = None
    items: TypeRef | None = None
    string_prefix: str | None = None
    constraints: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class Field:
    """One record property."""

    name: str
    type: TypeRef
    required: bool


@dataclass(frozen=True)
class Record:
    """One closed JSON object type.

    Attributes:
        name: Language-neutral name without prefix, for example ``Span`` or
            ``DataResidencyAttestationTemplate``.
        schema: Core schema stem the record comes from, for example ``span``.
        pointer: JSON pointer of the record inside its schema file.
        title: Schema ``title`` for top-level records, else ``None``.
        schema_version: Declared ``schema_version`` for top-level records.
        fields: Properties, required ones first in ``required`` order.
    """

    name: str
    schema: str
    pointer: str
    title: str | None
    schema_version: int | None
    fields: tuple[Field, ...]

    @property
    def source(self) -> str:
        """Return ``<file>`` or ``<file> <pointer>`` for documentation."""
        file_name = schema_file_name(self.schema)
        return file_name if self.pointer == "#" else f"{file_name} {self.pointer}"


@dataclass(frozen=True)
class NamedEnum:
    """A resolved named enum."""

    spec: NamedEnumSpec
    values: tuple[str, ...]


@dataclass(frozen=True)
class ContractModel:
    """Everything an emitter needs, in a deterministic order."""

    enums: tuple[NamedEnum, ...]
    records: tuple[Record, ...]
    roots: tuple[Record, ...]
    digest: str


def schema_file_name(schema: str) -> str:
    """Return the file name of a core schema stem."""
    return f"{schema}.schema.json"


def load_contract_model(schema_dir: Path = SCHEMA_DIR) -> ContractModel:
    """Lower the core schemas in ``schema_dir`` into a :class:`ContractModel`.

    Args:
        schema_dir: Directory holding ``<name>.schema.json`` for every name in
            :data:`CORE_SCHEMAS`.

    Returns:
        The lowered model. Equal inputs always give an equal model.

    Raises:
        ContractSchemaError: A schema is missing, malformed, or uses a
            construct the emitters cannot represent exactly.
    """
    documents: dict[str, Mapping[str, Any]] = {}
    for name in CORE_SCHEMAS:
        path = schema_dir / schema_file_name(name)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise ContractSchemaError("schema_missing", f"{name}#") from None
        except json.JSONDecodeError:
            raise ContractSchemaError("schema_not_json", f"{name}#") from None
        if not isinstance(document, Mapping):
            raise ContractSchemaError("schema_not_object", f"{name}#")
        documents[name] = document

    enums = tuple(_resolve_named_enum(spec, documents) for spec in NAMED_ENUMS)
    lowering = _Lowering(enums)
    roots: list[Record] = []
    for name in CORE_SCHEMAS:
        roots.append(lowering.root(name, documents[name]))
    return ContractModel(
        enums=enums,
        records=tuple(lowering.records),
        roots=tuple(roots),
        digest=_digest(documents),
    )


def pascal_case(name: str) -> str:
    """Return ``snake_case`` or ``UPPER_SNAKE`` ``name`` in ``PascalCase``."""
    return "".join(part[:1].upper() + part[1:].lower() for part in _parts(name))


def camel_case(name: str) -> str:
    """Return ``snake_case`` or ``UPPER_SNAKE`` ``name`` in ``camelCase``."""
    pascal = pascal_case(name)
    return pascal[:1].lower() + pascal[1:]


def _parts(name: str) -> list[str]:
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", name) if part]
    if not parts:
        raise ContractSchemaError("property_name_unsupported", "name")
    return parts


def _resolve_named_enum(
    spec: NamedEnumSpec, documents: Mapping[str, Mapping[str, Any]]
) -> NamedEnum:
    location = f"{spec.schema}#/properties/{spec.property}"
    properties = documents[spec.schema].get("properties")
    node = properties.get(spec.property) if isinstance(properties, Mapping) else None
    if not isinstance(node, Mapping) or "enum" not in node:
        raise ContractSchemaError("named_enum_missing", location)
    return NamedEnum(spec=spec, values=_enum_values(node["enum"], location))


def _enum_values(raw: object, location: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise ContractSchemaError("enum_empty", location)
    if not all(isinstance(value, str) for value in raw):
        raise ContractSchemaError("enum_not_string", location)
    if len(set(raw)) != len(raw):
        raise ContractSchemaError("enum_duplicate_value", location)
    return tuple(raw)


def _digest(documents: Mapping[str, Mapping[str, Any]]) -> str:
    sha = hashlib.sha256()
    for name in CORE_SCHEMAS:
        canonical = json.dumps(
            documents[name],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        sha.update(name.encode("utf-8"))
        sha.update(b"\n")
        sha.update(canonical.encode("utf-8"))
        sha.update(b"\n")
    return f"sha256:{sha.hexdigest()}"


class _Lowering:
    def __init__(self, enums: Sequence[NamedEnum]) -> None:
        self._enums = tuple(enums)
        self.records: list[Record] = []
        self._names: set[str] = set()

    def root(self, schema: str, document: Mapping[str, Any]) -> Record:
        location = f"{schema}#"
        title = document.get("title")
        if not isinstance(title, str) or not title.startswith(TITLE_PREFIX):
            raise ContractSchemaError("schema_title_unsupported", location)
        base = title[len(TITLE_PREFIX) :]
        if not base or not re.fullmatch(r"[A-Z][A-Za-z0-9]*", base):
            raise ContractSchemaError("schema_title_unsupported", location)
        version = document.get("schema_version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise ContractSchemaError("schema_version_missing", location)
        if document.get("type") != "object":
            raise ContractSchemaError("schema_not_record", location)
        return self._record(
            node=document,
            name=base,
            schema=schema,
            pointer="#",
            title=title,
            schema_version=version,
        )

    def _record(
        self,
        *,
        node: Mapping[str, Any],
        name: str,
        schema: str,
        pointer: str,
        title: str | None,
        schema_version: int | None,
    ) -> Record:
        location = f"{schema}{pointer}"
        allowed = _SUPPORTED_KEYWORDS | (
            _ANNOTATION_KEYWORDS if pointer == "#" else set()
        )
        _reject_unknown_keywords(node, allowed, location)
        if node.get("additionalProperties") is not False:
            raise ContractSchemaError("open_record_unsupported", location)
        properties = node.get("properties")
        if not isinstance(properties, Mapping) or not properties:
            raise ContractSchemaError("record_without_properties", location)
        required_raw = node.get("required", [])
        if not isinstance(required_raw, list) or not all(
            isinstance(item, str) for item in required_raw
        ):
            raise ContractSchemaError("required_not_string_list", location)
        if len(set(required_raw)) != len(required_raw):
            raise ContractSchemaError("required_duplicate", location)
        unknown_required = [item for item in required_raw if item not in properties]
        if unknown_required:
            raise ContractSchemaError("required_property_undeclared", location)
        if name in self._names:
            raise ContractSchemaError("record_name_collision", location)
        self._names.add(name)

        # Nested records are appended while the fields are lowered; inserting
        # the parent at its starting index keeps a parent-first order.
        slot = len(self.records)
        ordered = list(required_raw) + [
            key for key in properties if key not in set(required_raw)
        ]
        fields: list[Field] = []
        for key in ordered:
            if not isinstance(key, str) or not re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*", key
            ):
                raise ContractSchemaError("property_name_unsupported", location)
            child = properties[key]
            child_pointer = f"{pointer}/properties/{key}"
            fields.append(
                Field(
                    name=key,
                    type=self._type(
                        child,
                        schema=schema,
                        pointer=child_pointer,
                        record_name=name + pascal_case(key),
                    ),
                    required=key in required_raw,
                )
            )
        record = Record(
            name=name,
            schema=schema,
            pointer=pointer,
            title=title,
            schema_version=schema_version,
            fields=tuple(fields),
        )
        self.records.insert(slot, record)
        return record

    def _type(
        self,
        node: object,
        *,
        schema: str,
        pointer: str,
        record_name: str,
    ) -> TypeRef:
        location = f"{schema}{pointer}"
        if not isinstance(node, Mapping):
            raise ContractSchemaError("property_schema_not_object", location)
        _reject_unknown_keywords(node, _SUPPORTED_KEYWORDS, location)
        kind, nullable = _json_type(node.get("type"), location)
        if kind == "object":
            if nullable:
                raise ContractSchemaError("nullable_object_unsupported", location)
            if "properties" in node:
                record = self._record(
                    node=node,
                    name=record_name,
                    schema=schema,
                    pointer=pointer,
                    title=None,
                    schema_version=None,
                )
                return TypeRef(kind="record", record=record.name)
            _reject_unknown_keywords(node, {"type", "description"}, location)
            return TypeRef(kind="json_object")

        for keyword in ("properties", "required", "additionalProperties"):
            if keyword in node:
                raise ContractSchemaError("record_keyword_on_scalar", location)
        constraints = tuple(
            (keyword, node[keyword])
            for keyword in _CONSTRAINT_KEYWORDS
            if keyword in node
        )
        if kind == "array":
            if nullable:
                raise ContractSchemaError("nullable_array_unsupported", location)
            if "items" not in node:
                raise ContractSchemaError("array_items_missing", location)
            items = self._type(
                node["items"],
                schema=schema,
                pointer=f"{pointer}/items",
                record_name=record_name + "Item",
            )
            return TypeRef(kind="array", items=items, constraints=constraints)
        if "items" in node:
            raise ContractSchemaError("items_on_scalar", location)
        if "enum" in node:
            if kind != "string":
                raise ContractSchemaError("enum_not_string", location)
            values = _enum_values(node["enum"], location)
            if nullable:
                raise ContractSchemaError("nullable_enum_unsupported", location)
            named = next(
                (item.spec for item in self._enums if item.values == values), None
            )
            return TypeRef(kind="enum", enum=EnumType(values=values, name=named))
        const = node.get("const")
        if "const" in node:
            if nullable:
                raise ContractSchemaError("nullable_const_unsupported", location)
            if not _const_matches(kind, const):
                raise ContractSchemaError("const_type_mismatch", location)
        prefix = None
        if kind == "string" and "pattern" in node:
            pattern = node["pattern"]
            if not isinstance(pattern, str):
                raise ContractSchemaError("pattern_not_string", location)
            prefix = _literal_prefix(pattern)
        return TypeRef(
            kind=kind,
            nullable=nullable,
            const=const,
            string_prefix=prefix,
            constraints=constraints,
        )


def _json_type(raw: object, location: str) -> tuple[str, bool]:
    if isinstance(raw, str):
        types = [raw]
    elif isinstance(raw, list) and all(isinstance(item, str) for item in raw):
        types = list(raw)
    else:
        raise ContractSchemaError("type_missing", location)
    nullable = "null" in types
    concrete = [item for item in types if item != "null"]
    if len(concrete) != 1 or len(set(types)) != len(types):
        raise ContractSchemaError("type_union_unsupported", location)
    kind = concrete[0]
    if kind not in _SCALAR_KINDS | {"object", "array"}:
        raise ContractSchemaError("type_unsupported", location)
    return kind, nullable


def _reject_unknown_keywords(
    node: Mapping[str, Any], allowed: frozenset[str] | set[str], location: str
) -> None:
    unknown = sorted(key for key in node if key not in allowed)
    if unknown:
        raise ContractSchemaError("unsupported_keyword", location)


def _const_matches(kind: str, value: object) -> bool:
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "string":
        return isinstance(value, str)
    return False


def _literal_prefix(pattern: str) -> str | None:
    # An alternation anywhere means no single literal prefix is required.
    if not pattern.startswith("^") or "|" in pattern:
        return None
    prefix: list[str] = []
    for char in pattern[1:]:
        if char in _REGEX_META:
            # ``?``, ``*`` and ``{m,n}`` may make the previous literal optional.
            if char in "?*{" and prefix:
                prefix.pop()
            break
        prefix.append(char)
    return "".join(prefix) or None


# ---------------------------------------------------------------------------
# Shared command line
# ---------------------------------------------------------------------------


def display_path(path: Path) -> str:
    """Return ``path`` relative to the repository root when possible."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def run_emitter(
    *,
    description: str,
    script: str,
    default_output: Path,
    render: Callable[[ContractModel], str],
    argv: Sequence[str] | None = None,
) -> int:
    """Run the shared ``generate`` / ``--check`` command line of an emitter.

    Args:
        description: Help text for the command.
        script: Repository-relative script path used in drift messages.
        default_output: Committed file the emitter owns.
        render: Function rendering the full file text from a model.
        argv: Arguments, defaulting to ``sys.argv[1:]``.

    Returns:
        ``0`` on success, ``1`` when ``--check`` finds drift, ``2`` when the
        schemas cannot be lowered.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the committed file differs from the schemas",
    )
    parser.add_argument(
        "--schema-dir",
        type=Path,
        default=SCHEMA_DIR,
        help="directory holding the core *.schema.json files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="generated file to write or check",
    )
    args = parser.parse_args(argv)
    try:
        expected = render(load_contract_model(args.schema_dir))
    except ContractSchemaError as exc:
        print(f"contract schema lowering failed: {exc}", file=sys.stderr)
        return 2
    output: Path = args.output
    if args.check:
        try:
            actual = output.read_text(encoding="utf-8")
        except FileNotFoundError:
            actual = None
        if actual != expected:
            print(
                f"generated contract types are stale: {display_path(output)}; "
                f"run `python {script}` and commit the result",
                file=sys.stderr,
            )
            return 1
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(expected, encoding="utf-8")
    return 0
