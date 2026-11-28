#!/usr/bin/env python3
"""Generate Kotlin contract types from the canonical JSON Schemas.

Writes ``android/openmedkit/src/main/kotlin/com/openmed/openmedkit/contract/
ContractTypes.kt`` from the eight core schemas in ``openmed/core/schemas/json/``.
Every record becomes a ``data class`` prefixed ``OpenMedContract`` in the
``com.openmed.openmedkit.contract`` package with ``fromJson`` and ``toJson``
built on kotlinx ``JsonObject``, in the manual-parsing style of
``ModelCatalog.kt``. The build applies no serialization compiler plugin, so the
generated code needs none.

Decoding is exact: a missing required key fails even when its value may be
``null``, an explicit ``null`` is encoded back as ``null``, an undeclared key,
an unknown enum value, a mistyped value or a wrong ``const`` fails. Every
failure is an ``OpenMedContractException`` whose message holds the record, the
schema key and a reason code, never a payload value.

``python scripts/codegen/emit_kotlin_types.py`` rewrites the file;
``--check`` fails without writing when the committed file has drifted.
"""

from __future__ import annotations

import json
import re
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

SCRIPT = "scripts/codegen/emit_kotlin_types.py"
PACKAGE = "com.openmed.openmedkit.contract"
OUTPUT = (
    ROOT
    / "android"
    / "openmedkit"
    / "src"
    / "main"
    / "kotlin"
    / Path(*PACKAGE.split("."))
    / "ContractTypes.kt"
)
TYPE_PREFIX = "OpenMedContract"
INDENT = "    "

#: Kotlin hard keywords, escaped with backticks when used as names.
_KOTLIN_KEYWORDS = frozenset(
    {
        "as",
        "break",
        "class",
        "continue",
        "do",
        "else",
        "false",
        "for",
        "fun",
        "if",
        "in",
        "interface",
        "is",
        "null",
        "object",
        "package",
        "return",
        "super",
        "this",
        "throw",
        "true",
        "try",
        "typealias",
        "typeof",
        "val",
        "var",
        "when",
        "while",
    }
)

_SUPPORT = """\
/**
 * Thrown when JSON does not match a generated OpenMed contract type.
 *
 * The message names the record, the schema key and a reason code such as
 * `missing_required_key`, `expected_integer`, `unknown_enum_value` or
 * `constant_mismatch`. It never contains a payload value or an undeclared key.
 */
class OpenMedContractException(message: String) : IllegalArgumentException(message)

internal fun contractFailure(type: String, key: String, code: String): OpenMedContractException =
    OpenMedContractException("$type.$key: $code")

internal fun contractParseObject(text: String, type: String): JsonObject {
    val element = try {
        Json.parseToJsonElement(text)
    } catch (error: SerializationException) {
        throw OpenMedContractException("$type: malformed_json")
    }
    return element as? JsonObject ?: throw OpenMedContractException("$type: expected_object")
}

internal fun contractRequireDeclaredKeys(json: JsonObject, keys: Set<String>, type: String) {
    if (json.keys.any { it !in keys }) {
        throw OpenMedContractException("$type has an undeclared property")
    }
}

internal fun contractRequire(json: JsonObject, type: String, key: String): JsonElement =
    json[key] ?: throw contractFailure(type, key, "missing_required_key")

internal fun contractString(element: JsonElement, type: String, key: String): String {
    val primitive = element as? JsonPrimitive
    if (primitive == null || primitive is JsonNull || !primitive.isString) {
        throw contractFailure(type, key, "expected_string")
    }
    return primitive.content
}

internal fun contractInt(element: JsonElement, type: String, key: String): Int {
    val number = contractNumber(element, type, key, "expected_integer")
    return try {
        number.stripTrailingZeros().intValueExact()
    } catch (error: ArithmeticException) {
        throw contractFailure(type, key, "expected_integer")
    }
}

internal fun contractDouble(element: JsonElement, type: String, key: String): Double {
    val value = contractNumber(element, type, key, "expected_number").toDouble()
    if (!value.isFinite()) {
        throw contractFailure(type, key, "expected_number")
    }
    return value
}

internal fun contractBoolean(element: JsonElement, type: String, key: String): Boolean {
    val primitive = element as? JsonPrimitive
    if (primitive == null || primitive is JsonNull || primitive.isString) {
        throw contractFailure(type, key, "expected_boolean")
    }
    return when (primitive.content) {
        "true" -> true
        "false" -> false
        else -> throw contractFailure(type, key, "expected_boolean")
    }
}

internal fun contractObject(element: JsonElement, type: String, key: String): JsonObject =
    element as? JsonObject ?: throw contractFailure(type, key, "expected_object")

internal fun contractArray(element: JsonElement, type: String, key: String): JsonArray =
    element as? JsonArray ?: throw contractFailure(type, key, "expected_array")

private fun contractNumber(
    element: JsonElement,
    type: String,
    key: String,
    code: String,
): BigDecimal {
    val primitive = element as? JsonPrimitive
    if (primitive == null || primitive is JsonNull || primitive.isString) {
        throw contractFailure(type, key, code)
    }
    return primitive.content.toBigDecimalOrNull() ?: throw contractFailure(type, key, code)
}
"""

_IMPORTS = (
    "import kotlinx.serialization.SerializationException",
    "import kotlinx.serialization.json.Json",
    "import kotlinx.serialization.json.JsonArray",
    "import kotlinx.serialization.json.JsonElement",
    "import kotlinx.serialization.json.JsonNull",
    "import kotlinx.serialization.json.JsonObject",
    "import kotlinx.serialization.json.JsonPrimitive",
    "import java.math.BigDecimal",
)


def render(model: ContractModel) -> str:
    """Return the full text of the generated Kotlin source file.

    Args:
        model: Lowered core contract schemas.

    Returns:
        Kotlin source ending with a newline.

    Raises:
        ContractSchemaError: Two enum values or two properties map to the same
            Kotlin name.
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
        f"package {PACKAGE}",
        "\n".join(_IMPORTS),
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
    """Return the Kotlin name of a lowered record."""
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
    return f"`{name}`" if name in _KOTLIN_KEYWORDS else name


def _constant_name(value: str, location: str) -> str:
    name = "_".join(part.upper() for part in re.split(r"[^A-Za-z0-9]+", value) if part)
    if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", name):
        raise ContractSchemaError("enum_constant_name_unsupported", location)
    return name


def _enum_block(
    name: str, values: tuple[str, ...], summary: str, *, location: str
) -> str:
    constants: dict[str, str] = {}
    for value in values:
        constant = _constant_name(value, location)
        if constant in constants:
            raise ContractSchemaError("enum_case_name_collision", location)
        constants[constant] = value
    i1, i2 = INDENT, INDENT * 2
    lines = [f"/** {summary} */", f"enum class {name}(val value: String) {{"]
    entries = list(constants.items())
    for index, (constant, value) in enumerate(entries):
        end = ";" if index == len(entries) - 1 else ","
        lines.append(f"{i1}{constant}({json.dumps(value)}){end}")
    lines.extend(
        [
            "",
            f"{i1}companion object {{",
            f"{i2}/** Returns the constant for [value], or `null` when the schema"
            " does not list it. */",
            f"{i2}fun fromValue(value: String): {name}? = entries.firstOrNull "
            "{ it.value == value }",
            f"{i1}}}",
            "}",
        ]
    )
    return "\n".join(lines)


def _kotlin_base(ref: TypeRef, record: Record, field: Field) -> str:
    if ref.kind == "string":
        return "String"
    if ref.kind == "integer":
        return "Int"
    if ref.kind == "number":
        return "Double"
    if ref.kind == "boolean":
        return "Boolean"
    if ref.kind == "enum":
        assert ref.enum is not None
        if ref.enum.name is not None:
            return TYPE_PREFIX + ref.enum.name.type_name
        return _anonymous_enum_name(record, field)
    if ref.kind == "record":
        assert ref.record is not None
        return type_name(ref.record)
    if ref.kind == "json_object":
        return "JsonObject"
    if ref.kind == "array":
        assert ref.items is not None
        return f"List<{_kotlin_type(ref.items, record, field)}>"
    raise AssertionError(f"unhandled kind {ref.kind}")


def _kotlin_type(ref: TypeRef, record: Record, field: Field) -> str:
    base = _kotlin_base(ref, record, field)
    return f"{base}?" if ref.nullable else base


def _field_type(field: Field, record: Record) -> str:
    base = _kotlin_base(field.type, record, field)
    return f"{base}?" if field.type.nullable or not field.required else base


def _literal(value: int | str | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return json.dumps(value)


def _doc(field: Field) -> str | None:
    parts = [
        f"{keyword} {json.dumps(value, ensure_ascii=False)}"
        for keyword, value in field.type.constraints
    ]
    if not parts:
        return None
    return "JSON Schema: " + "; ".join(parts).replace("*/", "*\\/") + "."


def _decode(ref: TypeRef, element: str, key: str, depth: int, ctx: tuple) -> str:
    """Return an expression converting non-null JSON ``element`` to ``ref``."""
    record, field = ctx
    args = f"{element}, TYPE, {json.dumps(key)}"
    if ref.kind == "string":
        return f"contractString({args})"
    if ref.kind == "integer":
        return f"contractInt({args})"
    if ref.kind == "number":
        return f"contractDouble({args})"
    if ref.kind == "boolean":
        return f"contractBoolean({args})"
    if ref.kind == "enum":
        name = _kotlin_base(ref, record, field)
        return (
            f"{name}.fromValue(contractString({args}))\n"
            f"?: throw contractFailure(TYPE, {json.dumps(key)}, "
            '"unknown_enum_value")'
        )
    if ref.kind == "record":
        return f"{_kotlin_base(ref, record, field)}.fromJson(contractObject({args}))"
    if ref.kind == "json_object":
        return f"contractObject({args})"
    if ref.kind == "array":
        assert ref.items is not None
        item = f"item{depth}" if depth else "item"
        inner = _nullable_decode(ref.items, item, key, depth + 1, ctx)
        return f"contractArray({args}).map {{ {item} -> {inner} }}"
    raise AssertionError(f"unhandled kind {ref.kind}")


def _nullable_decode(
    ref: TypeRef, element: str, key: str, depth: int, ctx: tuple
) -> str:
    expression = _decode(ref, element, key, depth, ctx)
    if not ref.nullable:
        return expression
    return f"if ({element} is JsonNull) null else {expression}"


def _encode(ref: TypeRef, value: str, depth: int) -> str:
    """Return an expression converting non-null Kotlin ``value`` to JSON."""
    if ref.kind in {"string", "integer", "number", "boolean"}:
        return f"JsonPrimitive({value})"
    if ref.kind == "enum":
        return f"JsonPrimitive({value}.value)"
    if ref.kind == "record":
        return f"{value}.toJson()"
    if ref.kind == "json_object":
        return value
    if ref.kind == "array":
        assert ref.items is not None
        item = f"item{depth}" if depth else "item"
        inner = _nullable_encode(ref.items, item, depth + 1)
        return f"JsonArray({value}.map {{ {item} -> {inner} }})"
    raise AssertionError(f"unhandled kind {ref.kind}")


def _nullable_encode(ref: TypeRef, value: str, depth: int) -> str:
    expression = _encode(ref, value, depth)
    if not ref.nullable:
        return expression
    return f"{value}?.let {{ {_encode(ref, 'it', depth)} }} ?: JsonNull"


def _record_block(record: Record) -> str:
    name = type_name(record.name)
    location = f"{record.schema}{record.pointer}"
    if record.title is not None:
        summary = (
            f"`{record.title}` from `{record.source}`, schema version "
            f"{record.schema_version}."
        )
    else:
        summary = f"Nested object `{record.source}`."
    properties = [(field, camel_case(field.name)) for field in record.fields]
    if len({kotlin for _, kotlin in properties}) != len(properties):
        raise ContractSchemaError("property_name_collision", location)
    i1, i2, i3, i4 = INDENT, INDENT * 2, INDENT * 3, INDENT * 4
    parameters = [
        (field, kotlin) for field, kotlin in properties if field.type.const is None
    ]
    constants = [
        (field, kotlin) for field, kotlin in properties if field.type.const is not None
    ]

    lines = [
        "/**",
        f" * {summary}",
        " *",
        " * Decode with [fromJson] and encode with [toJson]; schema constants are",
        " * fixed properties rather than constructor parameters.",
        " */",
    ]
    if parameters:
        lines.append(f"data class {name}(")
        for field, kotlin in parameters:
            doc = _doc(field)
            if doc:
                lines.append(f"{i1}/** {doc} */")
            default = " = null" if not field.required else ""
            lines.append(
                f"{i1}val {_identifier(kotlin)}: {_field_type(field, record)}{default},"
            )
        lines.append(") {")
    else:
        raise ContractSchemaError("record_without_parameters", location)
    for field, kotlin in constants:
        assert field.type.const is not None
        kind = _kotlin_base(field.type, record, field)
        lines.append(f"{i1}/** Always `{_literal(field.type.const)}`. */")
        lines.append(
            f"{i1}val {_identifier(kotlin)}: {kind} get() = "
            f"{_literal(field.type.const)}"
        )
        lines.append("")

    # Encoder.
    lines.append(
        f"{i1}/** Encodes this record as a JSON object with the schema keys. */"
    )
    lines.append(f"{i1}fun toJson(): JsonObject {{")
    lines.append(f"{i2}val fields = LinkedHashMap<String, JsonElement>()")
    for field, kotlin in properties:
        key = json.dumps(field.name)
        ref = field.type
        if ref.const is not None:
            lines.append(f"{i2}fields[{key}] = JsonPrimitive({_literal(ref.const)})")
            continue
        value = _identifier(kotlin)
        if not field.required:
            lines.append(
                f"{i2}{value}?.let {{ fields[{key}] = "
                f"{_nullable_encode(_non_null(ref), 'it', 0)} }}"
            )
        else:
            lines.append(f"{i2}fields[{key}] = {_nullable_encode(ref, value, 0)}")
    lines.append(f"{i2}return JsonObject(fields)")
    lines.append(f"{i1}}}")

    # Decoder.
    keys = ", ".join(json.dumps(field.name) for field, _ in properties)
    lines.extend(
        [
            "",
            f"{i1}companion object {{",
            f'{i2}private const val TYPE = "{name}"',
            f"{i2}private val KEYS = setOf({keys})",
            "",
            f"{i2}/**",
            f"{i2} * Decodes [json], failing closed with [OpenMedContractException] on a",
            f"{i2} * missing, undeclared or mistyped key.",
            f"{i2} */",
            f"{i2}fun fromJson(json: JsonObject): {name} {{",
            f"{i3}contractRequireDeclaredKeys(json, KEYS, TYPE)",
        ]
    )
    ctx_base = record
    for field, _kotlin in constants:
        ref = field.type
        assert ref.const is not None
        key = json.dumps(field.name)
        expression = _decode(
            _non_const(ref),
            f"contractRequire(json, TYPE, {key})",
            field.name,
            0,
            (ctx_base, field),
        )
        lines.append(f"{i3}if ({expression} != {_literal(ref.const)}) {{")
        lines.append(f'{i4}throw contractFailure(TYPE, {key}, "constant_mismatch")')
        lines.append(f"{i3}}}")
    lines.append(f"{i3}return {name}(")
    for field, kotlin in parameters:
        key = json.dumps(field.name)
        ctx = (ctx_base, field)
        if field.required:
            element = f"contractRequire(json, TYPE, {key})"
            if field.type.nullable:
                inner = _decode(field.type, "it", field.name, 0, ctx)
                expression = (
                    f"{element}.let {{ if (it is JsonNull) null else {inner} }}"
                )
            else:
                expression = _decode(field.type, element, field.name, 0, ctx)
        else:
            inner = _decode(field.type, "it", field.name, 0, ctx)
            expression = f"json[{key}]?.let {{ if (it is JsonNull) null else {inner} }}"
        expression = expression.replace("\n", f"\n{i4}{INDENT}")
        lines.append(f"{i4}{_identifier(kotlin)} = {expression},")
    lines.append(f"{i3})")
    lines.append(f"{i2}}}")
    lines.extend(
        [
            "",
            f"{i2}/** Parses [text] and decodes it like the [JsonObject] overload. */",
            f"{i2}fun fromJson(text: String): {name} = fromJson(contractParseObject("
            "text, TYPE))",
            f"{i1}}}",
            "}",
        ]
    )
    return "\n".join(lines)


def _non_null(ref: TypeRef) -> TypeRef:
    return TypeRef(
        kind=ref.kind,
        nullable=False,
        const=ref.const,
        enum=ref.enum,
        record=ref.record,
        items=ref.items,
        string_prefix=ref.string_prefix,
        constraints=ref.constraints,
    )


def _non_const(ref: TypeRef) -> TypeRef:
    return TypeRef(
        kind=ref.kind,
        nullable=ref.nullable,
        enum=ref.enum,
        record=ref.record,
        items=ref.items,
        string_prefix=ref.string_prefix,
        constraints=ref.constraints,
    )


def main(argv: list[str] | None = None) -> int:
    """Generate or ``--check`` the Kotlin contract types."""
    return run_emitter(
        description=__doc__ or "",
        script=SCRIPT,
        default_output=OUTPUT,
        render=render,
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
