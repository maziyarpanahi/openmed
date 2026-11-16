#!/usr/bin/env python3
"""Generate Swift Codable contract types from the canonical JSON Schemas.

Writes ``swift/OpenMedKit/Sources/OpenMedKit/Generated/ContractTypes.swift``
from the eight core schemas in ``openmed/core/schemas/json/``. Every record
becomes a ``Codable, Equatable, Sendable`` struct prefixed ``OpenMedContract``
so it cannot clash with existing OpenMedKit types; string enums become
``String``-backed enums; free-form objects use ``OpenMedContractJSONValue``.

Decoding is exact: a required key that is missing fails even when its value
may be ``null``, an explicit ``null`` is encoded back as ``null``, a property
the schema does not declare fails, and ``const`` values are checked. Error
messages name the record or schema key, never a payload value.

``python scripts/codegen/emit_swift_types.py`` rewrites the file;
``--check`` fails without writing when the committed file has drifted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from contract_schema import (  # noqa: E402
    CORE_SCHEMAS,
    ROOT,
    ContractModel,
    ContractSchemaError,
    Field,
    NamedEnum,
    Record,
    TypeRef,
    camel_case,
    pascal_case,
    run_emitter,
)

SCRIPT = "scripts/codegen/emit_swift_types.py"
OUTPUT = (
    ROOT
    / "swift"
    / "OpenMedKit"
    / "Sources"
    / "OpenMedKit"
    / "Generated"
    / "ContractTypes.swift"
)
TYPE_PREFIX = "OpenMedContract"
JSON_VALUE = f"{TYPE_PREFIX}JSONValue"
INDENT = "    "

#: Swift keywords that must be escaped with backticks when used as names.
_SWIFT_KEYWORDS = frozenset(
    {
        "associatedtype",
        "break",
        "case",
        "catch",
        "class",
        "continue",
        "default",
        "defer",
        "deinit",
        "do",
        "else",
        "enum",
        "extension",
        "fallthrough",
        "false",
        "fileprivate",
        "for",
        "func",
        "guard",
        "if",
        "import",
        "in",
        "init",
        "inout",
        "internal",
        "is",
        "let",
        "nil",
        "operator",
        "private",
        "protocol",
        "public",
        "repeat",
        "rethrows",
        "return",
        "self",
        "Self",
        "static",
        "struct",
        "subscript",
        "super",
        "switch",
        "throw",
        "throws",
        "true",
        "try",
        "typealias",
        "var",
        "where",
        "while",
    }
)

_SUPPORT = """\
/// A JSON value inside a free-form contract object such as `evidence` or
/// `metadata`.
///
/// Integers and other numbers stay distinct, so decoding and encoding again
/// keeps every value.
public enum OpenMedContractJSONValue: Codable, Equatable, Sendable {
    case null
    case bool(Bool)
    case integer(Int)
    case number(Double)
    case string(String)
    case array([OpenMedContractJSONValue])
    case object([String: OpenMedContractJSONValue])

    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int.self) {
            self = .integer(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([OpenMedContractJSONValue].self) {
            self = .array(value)
        } else if let value = try? container.decode([String: OpenMedContractJSONValue].self) {
            self = .object(value)
        } else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Unsupported JSON value"
            )
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null:
            try container.encodeNil()
        case .bool(let value):
            try container.encode(value)
        case .integer(let value):
            try container.encode(value)
        case .number(let value):
            try container.encode(value)
        case .string(let value):
            try container.encode(value)
        case .array(let value):
            try container.encode(value)
        case .object(let value):
            try container.encode(value)
        }
    }
}

/// Any JSON object key, used to find properties a contract does not declare.
struct OpenMedContractAnyKey: CodingKey {
    let stringValue: String
    var intValue: Int? { nil }

    init(stringValue: String) {
        self.stringValue = stringValue
    }

    init?(intValue: Int) {
        nil
    }
}

/// Shared decoding checks for the generated contract types.
enum OpenMedContractDecoding {
    /// Throws when the decoded object has a key that `Keys` does not declare.
    ///
    /// The error names the record type only, never the undeclared key or a
    /// value.
    static func rejectUndeclaredKeys<Keys: CodingKey>(
        in decoder: any Decoder,
        keys: Keys.Type,
        record: String
    ) throws {
        let container = try decoder.container(keyedBy: OpenMedContractAnyKey.self)
        if container.allKeys.contains(where: { Keys(stringValue: $0.stringValue) == nil }) {
            throw DecodingError.dataCorrupted(
                DecodingError.Context(
                    codingPath: decoder.codingPath,
                    debugDescription: "\\(record) has an undeclared property"
                )
            )
        }
    }
}
"""


def render(model: ContractModel) -> str:
    """Return the full text of the generated Swift source file.

    Args:
        model: Lowered core contract schemas.

    Returns:
        Swift source ending with a newline, formatted the way
        ``swift format`` with the repository configuration leaves it.

    Raises:
        ContractSchemaError: Two enum values map to the same Swift case name.
    """
    sources = ",".join(CORE_SCHEMAS)
    blocks = [
        "\n".join(
            [
                f"// Generated by {SCRIPT}. Do not edit by hand.",
                f"// Source: openmed/core/schemas/json/{{{sources}}}.schema.json",
                f"// Schema digest: {model.digest}",
                f"// Regenerate with `python {SCRIPT}`; CI runs it with `--check`.",
            ]
        ),
        "import Foundation",
        _SUPPORT.rstrip("\n"),
    ]
    for named in model.enums:
        spec = named.spec
        source = f"{spec.schema}.schema.json #/properties/{spec.property}"
        blocks.append(
            _enum_block(
                _named_enum_name(named),
                named.values,
                f"Values of `{source}`, in schema order.",
                location=f"{spec.schema}#/properties/{spec.property}",
            )
        )
    for record in model.records:
        blocks.append(_record_block(record))
        for field in record.fields:
            anonymous = _anonymous_enum(field.type)
            if anonymous is not None:
                blocks.append(
                    _enum_block(
                        _anonymous_enum_name(record, field),
                        anonymous,
                        f"Values of `{record.source}` `{field.name}`.",
                        location=f"{record.schema}{record.pointer}/{field.name}",
                    )
                )
    return "\n\n".join(blocks) + "\n"


def type_name(record_name: str) -> str:
    """Return the Swift name of a lowered record."""
    return TYPE_PREFIX + record_name


def _named_enum_name(named: NamedEnum) -> str:
    return TYPE_PREFIX + named.spec.type_name


def _anonymous_enum(ref: TypeRef) -> tuple[str, ...] | None:
    while ref.kind == "array" and ref.items is not None:
        ref = ref.items
    if ref.kind == "enum" and ref.enum is not None and ref.enum.name is None:
        return ref.enum.values
    return None


def _anonymous_enum_name(record: Record, field: Field) -> str:
    return type_name(record.name) + pascal_case(field.name)


def _identifier(name: str) -> str:
    return f"`{name}`" if name in _SWIFT_KEYWORDS else name


def _enum_block(
    name: str, values: tuple[str, ...], summary: str, *, location: str
) -> str:
    cases: dict[str, str] = {}
    for value in values:
        case = camel_case(value)
        if case in cases:
            raise ContractSchemaError("enum_case_name_collision", location)
        cases[case] = value
    lines = [
        f"/// {summary}",
        f"public enum {name}: String, Codable, CaseIterable, Sendable {{",
    ]
    lines.extend(
        f"{INDENT}case {_identifier(case)} = {json.dumps(value)}"
        for case, value in cases.items()
    )
    # A custom decoder keeps the rejected value out of the error message.
    i1, i2, i3 = INDENT, INDENT * 2, INDENT * 3
    lines.extend(
        [
            "",
            f"{i1}public init(from decoder: any Decoder) throws {{",
            f"{i2}let container = try decoder.singleValueContainer()",
            f"{i2}guard let value = Self(rawValue: try container.decode(String.self))"
            " else {",
            f"{i3}throw DecodingError.dataCorruptedError(",
            f"{i3}{INDENT}in: container,",
            f'{i3}{INDENT}debugDescription: "Unknown {name} value"',
            f"{i3})",
            f"{i2}}}",
            f"{i2}self = value",
            f"{i1}}}",
            "}",
        ]
    )
    return "\n".join(lines)


def _field_type(field: Field, record: Record) -> str:
    """Return the property type; keys outside ``required`` are optional."""
    base = _swift_base(field.type, record, field)
    return f"{base}?" if field.type.nullable or not field.required else base


def _swift_type(ref: TypeRef, record: Record, field: Field) -> str:
    base = _swift_base(ref, record, field)
    return f"{base}?" if ref.nullable else base


def _swift_base(ref: TypeRef, record: Record, field: Field) -> str:
    if ref.kind == "string":
        return "String"
    if ref.kind == "integer":
        return "Int"
    if ref.kind == "number":
        return "Double"
    if ref.kind == "boolean":
        return "Bool"
    if ref.kind == "enum":
        assert ref.enum is not None
        if ref.enum.name is not None:
            return TYPE_PREFIX + ref.enum.name.type_name
        return _anonymous_enum_name(record, field)
    if ref.kind == "record":
        assert ref.record is not None
        return type_name(ref.record)
    if ref.kind == "json_object":
        return f"[String: {JSON_VALUE}]"
    if ref.kind == "array":
        assert ref.items is not None
        return f"[{_swift_type(ref.items, record, field)}]"
    raise AssertionError(f"unhandled kind {ref.kind}")


def _swift_literal(value: int | str | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return json.dumps(value)


def _doc(field: Field) -> str | None:
    ref = field.type
    if ref.const is not None:
        return f"Always `{_swift_literal(ref.const)}`."
    parts = [
        f"{keyword} {json.dumps(value, ensure_ascii=False)}"
        for keyword, value in ref.constraints
    ]
    if not parts:
        return None
    return "JSON Schema: " + "; ".join(parts) + "."


def _record_block(record: Record) -> str:
    name = type_name(record.name)
    if record.title is not None:
        summary = (
            f"`{record.title}` from `{record.source}`, schema version "
            f"{record.schema_version}."
        )
    else:
        summary = f"Nested object `{record.source}`."
    properties = [(field, camel_case(field.name)) for field in record.fields]
    if len({swift for _, swift in properties}) != len(properties):
        raise ContractSchemaError(
            "property_name_collision", f"{record.schema}{record.pointer}"
        )
    i1, i2, i3 = INDENT, INDENT * 2, INDENT * 3
    lines = [f"/// {summary}", f"public struct {name}: Codable, Equatable, Sendable {{"]
    for field, swift in properties:
        doc = _doc(field)
        if doc:
            lines.append(f"{i1}/// {doc}")
        lines.append(
            f"{i1}public let {_identifier(swift)}: "
            f"{_field_type(field, record)}"
        )

    # Memberwise initializer; constants are fixed by the schema.
    parameters = [
        (field, swift) for field, swift in properties if field.type.const is None
    ]
    lines.append("")
    lines.append(f"{i1}/// Creates the record; schema constants are filled in.")
    if parameters:
        lines.append(f"{i1}public init(")
        for index, (field, swift) in enumerate(parameters):
            default = " = nil" if not field.required else ""
            comma = "," if index < len(parameters) - 1 else ""
            lines.append(
                f"{i2}{swift}: {_field_type(field, record)}{default}{comma}"
            )
        lines.append(f"{i1}) {{")
    else:
        lines.append(f"{i1}public init() {{")
    for field, swift in properties:
        if field.type.const is not None:
            lines.append(f"{i2}self.{swift} = {_swift_literal(field.type.const)}")
        else:
            lines.append(f"{i2}self.{swift} = {_identifier(swift)}")
    lines.append(f"{i1}}}")

    lines.append("")
    lines.append(f"{i1}private enum CodingKeys: String, CodingKey {{")
    for field, swift in properties:
        lines.append(f"{i2}case {_identifier(swift)} = {json.dumps(field.name)}")
    lines.append(f"{i1}}}")

    lines.append("")
    lines.append(f"{i1}public init(from decoder: any Decoder) throws {{")
    lines.append(f"{i2}try OpenMedContractDecoding.rejectUndeclaredKeys(")
    lines.append(f"{i3}in: decoder,")
    lines.append(f"{i3}keys: CodingKeys.self,")
    lines.append(f'{i3}record: "{name}"')
    lines.append(f"{i2})")
    lines.append(f"{i2}let container = try decoder.container(keyedBy: CodingKeys.self)")
    for field, swift in properties:
        lines.extend(_decode_lines(field, swift, record, i2))
    lines.append(f"{i1}}}")

    lines.append("")
    lines.append(f"{i1}public func encode(to encoder: any Encoder) throws {{")
    lines.append(f"{i2}var container = encoder.container(keyedBy: CodingKeys.self)")
    for field, swift in properties:
        lines.extend(_encode_lines(field, swift, i2))
    lines.append(f"{i1}}}")
    lines.append("}")
    return "\n".join(lines)


def _decode_lines(field: Field, swift: str, record: Record, indent: str) -> list[str]:
    ref = field.type
    key = f".{swift}"
    base = _swift_base(ref, record, field)
    inner = indent + INDENT
    if not field.required:
        return [
            f"{indent}self.{swift} = try container.decodeIfPresent("
            f"{base}.self, forKey: {key})"
        ]
    if ref.nullable:
        return [
            f"{indent}if try container.decodeNil(forKey: {key}) {{",
            f"{inner}self.{swift} = nil",
            f"{indent}}} else {{",
            f"{inner}self.{swift} = try container.decode({base}.self, forKey: {key})",
            f"{indent}}}",
        ]
    lines = [f"{indent}self.{swift} = try container.decode({base}.self, forKey: {key})"]
    if ref.const is not None:
        literal = _swift_literal(ref.const)
        message = f"{field.name} must be {literal}".replace('"', '\\"')
        lines.extend(
            [
                f"{indent}guard self.{swift} == {literal} else {{",
                f"{inner}throw DecodingError.dataCorruptedError(",
                f"{inner}{INDENT}forKey: {key},",
                f"{inner}{INDENT}in: container,",
                f'{inner}{INDENT}debugDescription: "{message}"',
                f"{inner})",
                f"{indent}}}",
            ]
        )
    return lines


def _encode_lines(field: Field, swift: str, indent: str) -> list[str]:
    key = f".{swift}"
    inner = indent + INDENT
    if not field.required:
        return [f"{indent}try container.encodeIfPresent(self.{swift}, forKey: {key})"]
    if field.type.nullable:
        return [
            f"{indent}if let {_identifier(swift)} = self.{swift} {{",
            f"{inner}try container.encode({_identifier(swift)}, forKey: {key})",
            f"{indent}}} else {{",
            f"{inner}try container.encodeNil(forKey: {key})",
            f"{indent}}}",
        ]
    return [f"{indent}try container.encode(self.{swift}, forKey: {key})"]


def main(argv: list[str] | None = None) -> int:
    """Generate or ``--check`` the Swift contract types."""
    return run_emitter(
        description=__doc__ or "",
        script=SCRIPT,
        default_output=OUTPUT,
        render=render,
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
