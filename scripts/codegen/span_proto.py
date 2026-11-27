"""Render the protobuf ``OpenMedSpan`` message from the canonical span schema.

``openmed/core/schemas/json/span.schema.json`` is the source of truth for the
span contract. This script renders the ``OpenMedSpan`` message block of
``openmed/service/proto/openmed.proto`` from it:

* each schema property becomes one field with the same name;
* ``string`` maps to ``string``, ``boolean`` to ``bool``, ``number`` to
  ``double``, an ``integer`` bounded to the int32 range (by ``const``, ``enum``
  or ``minimum``/``maximum``) to ``int32`` and any other ``integer`` to
  ``int64``;
* ``object`` maps to ``google.protobuf.Struct`` and ``array`` to a
  ``repeated`` field of its item type;
* a scalar gets ``optional`` when the property is not required, when it is
  nullable, or when the field map keeps explicit presence for compatibility.

Wire numbers come from the committed field map
``scripts/codegen/openmed_span_fields.json``. A new property takes the next free
number, above every number ever used, and a removed property becomes a
``reserved`` number and name, so a number is never reused.

Usage::

    python scripts/codegen/span_proto.py          # update the map and the block
    python scripts/codegen/span_proto.py --check  # fail when either is stale

After updating, run ``make grpc-proto`` to regenerate the committed stubs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_PATH = REPO_ROOT / "openmed/core/schemas/json/span.schema.json"
DEFAULT_PROTO_PATH = REPO_ROOT / "openmed/service/proto/openmed.proto"
DEFAULT_FIELD_MAP_PATH = Path(__file__).resolve().with_name("openmed_span_fields.json")
MESSAGE_NAME = "OpenMedSpan"
FIELD_MAP_VERSION = 1
STRUCT_TYPE = "google.protobuf.Struct"
HEADER = (
    "// Generated from openmed/core/schemas/json/span.schema.json by\n"
    "// scripts/codegen/span_proto.py; edit the schema, not this block.\n"
)
_INT32_MIN = -(2**31)
_INT32_MAX = 2**31 - 1
_MAX_FIELD_NUMBER = 2**29 - 1
_IMPLEMENTATION_RESERVED = range(19000, 20000)
_SCALARS = {"string": "string", "boolean": "bool", "number": "double"}
_FIELD_NAME_RE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_BLOCK_RE = re.compile(
    r"^(?:" + re.escape(HEADER) + r")?message " + MESSAGE_NAME + r" \{\n(?:.*\n)*?\}\n",
    re.MULTILINE,
)


class SpanCodegenError(ValueError):
    """Raised when the schema, field map or proto cannot be rendered."""


@dataclass(frozen=True)
class ReservedField:
    """A wire number and name retired from the message.

    Attributes:
        number: Retired field number.
        name: Retired field name.
    """

    number: int
    name: str


@dataclass(frozen=True)
class FieldMap:
    """Committed wire numbers of the ``OpenMedSpan`` message.

    Attributes:
        fields: Property name to field number.
        reserved: Numbers and names of removed properties.
        explicit_presence: Required, non-nullable properties that keep proto3
            ``optional`` presence because they shipped with it.
    """

    fields: Mapping[str, int]
    reserved: tuple[ReservedField, ...] = ()
    explicit_presence: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        numbers = [*self.fields.values(), *(item.number for item in self.reserved)]
        if len(numbers) != len(set(numbers)):
            raise SpanCodegenError("field map reuses a field number")
        for number in numbers:
            if (
                isinstance(number, bool)
                or not isinstance(number, int)
                or not 1 <= number <= _MAX_FIELD_NUMBER
                or number in _IMPLEMENTATION_RESERVED
            ):
                raise SpanCodegenError("field map has an invalid field number")
        names = [*self.fields, *(item.name for item in self.reserved)]
        if len(names) != len(set(names)):
            raise SpanCodegenError("field map reuses a field name")
        if any(not _FIELD_NAME_RE.match(name) for name in names):
            raise SpanCodegenError("field map has an invalid field name")
        if not self.explicit_presence <= set(self.fields):
            raise SpanCodegenError("explicit presence names an unmapped field")

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON form written to the committed field map."""
        return {
            "version": FIELD_MAP_VERSION,
            "message": MESSAGE_NAME,
            "fields": dict(sorted(self.fields.items(), key=lambda item: item[1])),
            "reserved": [
                {"number": item.number, "name": item.name}
                for item in sorted(self.reserved, key=lambda item: item.number)
            ],
            "explicit_presence": sorted(self.explicit_presence),
        }


def load_field_map(path: Path = DEFAULT_FIELD_MAP_PATH) -> FieldMap:
    """Load and validate a committed field map.

    Args:
        path: The field-map JSON file.

    Returns:
        The validated field map.

    Raises:
        SpanCodegenError: If the file is not a valid field map.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SpanCodegenError("field map is unreadable") from exc
    if (
        not isinstance(document, dict)
        or document.get("version") != FIELD_MAP_VERSION
        or document.get("message") != MESSAGE_NAME
        or not isinstance(document.get("fields"), dict)
    ):
        raise SpanCodegenError("field map has an unexpected shape")
    try:
        reserved = tuple(
            ReservedField(number=item["number"], name=str(item["name"]))
            for item in document.get("reserved", ())
        )
        return FieldMap(
            fields=dict(document["fields"]),
            reserved=reserved,
            explicit_presence=frozenset(document.get("explicit_presence", ())),
        )
    except (KeyError, TypeError) as exc:
        raise SpanCodegenError("field map has an unexpected shape") from exc


def render_field_map(field_map: FieldMap) -> str:
    """Render a field map as deterministic JSON text."""
    return json.dumps(field_map.to_dict(), indent=2) + "\n"


def update_field_map(schema: Mapping[str, Any], field_map: FieldMap) -> FieldMap:
    """Number new schema properties and reserve removed ones.

    Args:
        schema: The span JSON Schema.
        field_map: The committed field map.

    Returns:
        A field map covering exactly the schema's properties. New properties
        take the next free numbers in sorted name order; removed properties
        become reserved and leave explicit presence.
    """
    properties = _properties(schema)
    fields = dict(field_map.fields)
    reserved = list(field_map.reserved)
    for name in sorted(set(fields) - set(properties)):
        reserved.append(ReservedField(number=fields.pop(name), name=name))
    reserved_names = {item.name for item in reserved}
    used = [*fields.values(), *(item.number for item in reserved)]
    next_number = max(used, default=0) + 1
    for name in sorted(set(properties) - set(fields)):
        if name in reserved_names:
            raise SpanCodegenError("schema property reuses a reserved field name")
        if next_number in _IMPLEMENTATION_RESERVED:
            next_number = _IMPLEMENTATION_RESERVED.stop
        fields[name] = next_number
        next_number += 1
    return FieldMap(
        fields=fields,
        reserved=tuple(reserved),
        explicit_presence=field_map.explicit_presence & set(fields),
    )


def proto_field(
    name: str,
    prop: Mapping[str, Any],
    *,
    required: bool,
    explicit_presence: bool = False,
) -> tuple[str, str]:
    """Return the ``(label, type)`` of one schema property.

    Args:
        name: The property name, used only in error messages.
        prop: The property's JSON Schema.
        required: Whether the schema lists the property as required.
        explicit_presence: Whether the field map keeps ``optional`` presence.

    Returns:
        The proto label (``""``, ``"optional "`` or ``"repeated "``) and type.

    Raises:
        SpanCodegenError: If the property has no single supported type.
    """
    base, nullable = _base_type(name, prop)
    if base == "object":
        return "", STRUCT_TYPE
    if base == "array":
        items = prop.get("items")
        if not isinstance(items, Mapping):
            raise SpanCodegenError(f"array property {name!r} has no item schema")
        item_base, item_nullable = _base_type(name, items)
        if item_nullable or item_base == "array":
            raise SpanCodegenError(f"array property {name!r} has unsupported items")
        return "repeated ", (
            STRUCT_TYPE if item_base == "object" else _scalar(item_base, items)
        )
    label = "optional " if (not required or nullable or explicit_presence) else ""
    return label, _scalar(base, prop)


def render_message(schema: Mapping[str, Any], field_map: FieldMap) -> str:
    """Render the ``OpenMedSpan`` block, header comment included.

    Args:
        schema: The span JSON Schema.
        field_map: A field map covering exactly the schema's properties.

    Returns:
        The message text, ending with a newline.

    Raises:
        SpanCodegenError: If the map does not cover the schema's properties.
    """
    properties = _properties(schema)
    if set(properties) != set(field_map.fields):
        raise SpanCodegenError("field map does not match the schema properties")
    required = set(schema.get("required", ()))
    lines = [f"message {MESSAGE_NAME} {{"]
    reserved = sorted(field_map.reserved, key=lambda item: item.number)
    if reserved:
        numbers = ", ".join(str(item.number) for item in reserved)
        names = ", ".join(f'"{item.name}"' for item in reserved)
        lines.extend([f"  reserved {numbers};", f"  reserved {names};"])
    for name, number in sorted(field_map.fields.items(), key=lambda item: item[1]):
        label, proto_type = proto_field(
            name,
            properties[name],
            required=name in required,
            explicit_presence=name in field_map.explicit_presence,
        )
        lines.append(f"  {label}{proto_type} {name} = {number};")
    lines.append("}")
    return HEADER + "\n".join(lines) + "\n"


def extract_message(proto_text: str) -> str:
    """Return the committed ``OpenMedSpan`` block of a proto file.

    Raises:
        SpanCodegenError: If the file does not hold exactly one block.
    """
    matches = list(_BLOCK_RE.finditer(proto_text))
    if len(matches) != 1:
        raise SpanCodegenError("proto must declare exactly one OpenMedSpan message")
    return matches[0].group(0)


def splice_message(proto_text: str, block: str) -> str:
    """Return ``proto_text`` with its ``OpenMedSpan`` block replaced."""
    extract_message(proto_text)
    return _BLOCK_RE.sub(lambda _: block, proto_text, count=1)


def check(
    *,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    proto_path: Path = DEFAULT_PROTO_PATH,
    field_map_path: Path = DEFAULT_FIELD_MAP_PATH,
) -> list[str]:
    """Return the drift between the schema, the field map and the proto.

    Returns:
        Value-free problem descriptions; empty when everything is current.
    """
    schema = _load_schema(schema_path)
    field_map = load_field_map(field_map_path)
    problems: list[str] = []
    expected_map = update_field_map(schema, field_map)
    if expected_map != field_map:
        added = sorted(set(expected_map.fields) - set(field_map.fields))
        removed = sorted(set(field_map.fields) - set(expected_map.fields))
        problems.append(
            "field map is stale "
            f"(unnumbered properties: {added}; removed properties: {removed})"
        )
        return problems
    committed = extract_message(proto_path.read_text(encoding="utf-8"))
    if committed != render_message(schema, field_map):
        problems.append("openmed.proto OpenMedSpan block differs from the schema")
    return problems


def generate(
    *,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    proto_path: Path = DEFAULT_PROTO_PATH,
    field_map_path: Path = DEFAULT_FIELD_MAP_PATH,
) -> bool:
    """Update the field map and the proto block in place.

    Returns:
        Whether either file changed.
    """
    schema = _load_schema(schema_path)
    field_map = update_field_map(schema, load_field_map(field_map_path))
    map_text = render_field_map(field_map)
    proto_text = proto_path.read_text(encoding="utf-8")
    updated_proto = splice_message(proto_text, render_message(schema, field_map))
    changed = False
    if field_map_path.read_text(encoding="utf-8") != map_text:
        field_map_path.write_text(map_text, encoding="utf-8")
        changed = True
    if updated_proto != proto_text:
        proto_path.write_text(updated_proto, encoding="utf-8")
        changed = True
    return changed


def main(argv: Sequence[str] | None = None) -> int:
    """Run the generator; ``--check`` only reports drift."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when the field map or the OpenMedSpan block is stale",
    )
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--proto", type=Path, default=DEFAULT_PROTO_PATH)
    parser.add_argument("--field-map", type=Path, default=DEFAULT_FIELD_MAP_PATH)
    args = parser.parse_args(argv)
    paths = {
        "schema_path": args.schema,
        "proto_path": args.proto,
        "field_map_path": args.field_map,
    }
    try:
        if args.check:
            problems = check(**paths)
            for problem in problems:
                print(problem, file=sys.stderr)
            if problems:
                print(
                    "Run `python scripts/codegen/span_proto.py` and then "
                    "`make grpc-proto`, and commit the result.",
                    file=sys.stderr,
                )
            return 1 if problems else 0
        if generate(**paths):
            print("Updated OpenMedSpan; now run `make grpc-proto`.")
        return 0
    except SpanCodegenError as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _load_schema(path: Path) -> dict[str, Any]:
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SpanCodegenError("span schema is unreadable") from exc
    if not isinstance(schema, dict):
        raise SpanCodegenError("span schema must be a JSON object")
    return schema


def _properties(schema: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    properties = schema.get("properties")
    if not isinstance(properties, Mapping) or not properties:
        raise SpanCodegenError("span schema has no properties")
    for name, prop in properties.items():
        if not _FIELD_NAME_RE.match(str(name)) or not isinstance(prop, Mapping):
            raise SpanCodegenError("span schema has an invalid property")
    return dict(properties)


def _base_type(name: str, prop: Mapping[str, Any]) -> tuple[str, bool]:
    declared = prop.get("type")
    types = [declared] if isinstance(declared, str) else list(declared or ())
    nullable = "null" in types
    base = [item for item in types if item != "null"]
    if len(base) != 1 or base[0] not in {*_SCALARS, "integer", "object", "array"}:
        raise SpanCodegenError(f"property {name!r} has no single supported type")
    return base[0], nullable


def _scalar(base: str, prop: Mapping[str, Any]) -> str:
    if base == "integer":
        return "int32" if _fits_int32(prop) else "int64"
    if base not in _SCALARS:
        raise SpanCodegenError("unsupported scalar type")
    return _SCALARS[base]


def _fits_int32(prop: Mapping[str, Any]) -> bool:
    if "const" in prop:
        values = [prop["const"]]
    elif "enum" in prop:
        values = list(prop["enum"])
    elif "minimum" in prop and "maximum" in prop:
        values = [prop["minimum"], prop["maximum"]]
    else:
        return False
    return all(
        isinstance(value, int) and _INT32_MIN <= value <= _INT32_MAX for value in values
    )


if __name__ == "__main__":
    raise SystemExit(main())
