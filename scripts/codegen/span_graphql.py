"""Generate the canonical GraphQL ``OpenMedSpan`` fields from the span schema.

``openmed/core/schemas/json/span.schema.json`` is the source of truth for the
span contract. This script renders its properties, in canonical order, into
``openmed/service/graphql_span_fields.py``, which the Strawberry
``OpenMedSpan`` type is built from:

* ``string`` maps to ``String``, ``integer`` to ``Int``, ``number`` to
  ``Float`` and ``boolean`` to ``Boolean``; scalars stay nullable, as they were
  before generation, so a partial span or entity mapping still resolves;
* ``object`` maps to ``JSON!`` and ``array`` to a non-null list of non-null
  items, such as ``[String!]!``.

The canonical order is the schema's ``required`` order, followed by any other
property in name order. ``label`` and ``text`` are compatibility fields that
the schema does not define; they stay on the type, deprecated in favour of
``canonicalLabel`` and ``textHash``, until a 5.0 decision removes them.

Usage::

    python scripts/codegen/span_graphql.py          # rewrite the module
    python scripts/codegen/span_graphql.py --check  # fail on any drift

``--check`` fails when the committed module is stale, or when the live
Strawberry type and the schema disagree on field names or types.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
# The live-type check must import this checkout's ``openmed``, not an installed
# copy, or ``--check`` would validate a different tree.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_SCHEMA_PATH = REPO_ROOT / "openmed/core/schemas/json/span.schema.json"
DEFAULT_MODULE_PATH = REPO_ROOT / "openmed/service/graphql_span_fields.py"
TYPE_NAME = "OpenMedSpan"
#: Compatibility fields outside the canonical span, with their GraphQL types.
LEGACY_FIELDS: Mapping[str, str] = {"label": "String!", "text": "String"}
_SCALARS = {
    "string": "String",
    "integer": "Int",
    "number": "Float",
    "boolean": "Boolean",
}
_MODULE_HEADER = '''"""Canonical ``OpenMedSpan`` GraphQL fields, generated from the span schema.

Generated from ``openmed/core/schemas/json/span.schema.json`` by
``scripts/codegen/span_graphql.py``. Do not edit by hand: change the schema,
then run ``python scripts/codegen/span_graphql.py``.

Each entry is ``(property, GraphQL type)`` in the schema's canonical order.
Scalars are nullable so a partial span or entity mapping still resolves; arrays
and objects are non-null.
"""

from __future__ import annotations

#: Canonical span properties and their GraphQL types, in canonical order.
CANONICAL_SPAN_FIELDS: tuple[tuple[str, str], ...] = (
'''
_MODULE_FOOTER = """)

__all__ = ["CANONICAL_SPAN_FIELDS"]
"""


class SpanGraphQLCodegenError(ValueError):
    """Raised when the span schema cannot be mapped to GraphQL types."""


def graphql_type(name: str, prop: Mapping[str, Any]) -> str:
    """Return the GraphQL type of one span schema property.

    Args:
        name: The property name, used only in error messages.
        prop: The property's JSON Schema.

    Returns:
        A GraphQL type reference such as ``String`` or ``[String!]!``.

    Raises:
        SpanGraphQLCodegenError: If the property has no single supported type.
    """
    base = _base_type(name, prop)
    if base == "object":
        return "JSON!"
    if base == "array":
        items = prop.get("items")
        if not isinstance(items, Mapping):
            raise SpanGraphQLCodegenError(f"array property {name!r} has no items")
        item = _base_type(name, items, nullable_allowed=False)
        if item == "array":
            raise SpanGraphQLCodegenError(f"array property {name!r} nests arrays")
        return f"[{'JSON' if item == 'object' else _SCALARS[item]}!]!"
    return _SCALARS[base]


def canonical_fields(schema: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """Return ``(property, GraphQL type)`` pairs in canonical order.

    Raises:
        SpanGraphQLCodegenError: If the schema has no usable properties.
    """
    properties = schema.get("properties")
    if not isinstance(properties, Mapping) or not properties:
        raise SpanGraphQLCodegenError("span schema has no properties")
    required = [str(name) for name in schema.get("required", ())]
    if not set(required) <= set(properties):
        raise SpanGraphQLCodegenError("span schema requires an undefined property")
    order = [*required, *sorted(set(properties) - set(required))]
    collisions = {camel_case(name) for name in order} & set(LEGACY_FIELDS)
    if collisions:
        raise SpanGraphQLCodegenError("span property collides with a legacy field")
    return tuple((name, graphql_type(name, properties[name])) for name in order)


def render_module(schema: Mapping[str, Any]) -> str:
    """Render ``openmed/service/graphql_span_fields.py`` for ``schema``."""
    rows = "".join(
        f"    ({json.dumps(name)}, {json.dumps(kind)}),\n"
        for name, kind in canonical_fields(schema)
    )
    return _MODULE_HEADER + rows + _MODULE_FOOTER


def camel_case(name: str) -> str:
    """Return the GraphQL field name Strawberry derives from ``name``."""
    head, *rest = name.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in rest)


def live_type_fields() -> dict[str, tuple[str, str | None]]:
    """Return the live ``OpenMedSpan`` fields as ``name -> (type, deprecation)``.

    Raises:
        SpanGraphQLCodegenError: If the live schema has no ``OpenMedSpan`` type.
    """
    from graphql import GraphQLObjectType, build_schema

    from openmed.service.graphql_schema import schema

    span = build_schema(str(schema)).get_type(TYPE_NAME)
    if not isinstance(span, GraphQLObjectType):
        raise SpanGraphQLCodegenError("live schema has no OpenMedSpan object type")
    return {
        name: (str(field.type), field.deprecation_reason)
        for name, field in span.fields.items()
    }


def type_drift(
    fields: Sequence[tuple[str, str]],
    live: Mapping[str, tuple[str, str | None]],
) -> list[str]:
    """Return where the live type and the canonical fields disagree.

    Args:
        fields: Canonical ``(property, GraphQL type)`` pairs.
        live: Live fields from :func:`live_type_fields`.

    Returns:
        Value-free problem descriptions naming only field identifiers.
    """
    problems: list[str] = []
    expected = {camel_case(name): kind for name, kind in fields}
    for name, kind in expected.items():
        if name not in live:
            problems.append(f"OpenMedSpan is missing canonical field {name}")
        elif live[name][0] != kind:
            problems.append(f"OpenMedSpan.{name} is {live[name][0]}, expected {kind}")
    for name in sorted(set(live) - set(expected)):
        if name not in LEGACY_FIELDS:
            problems.append(f"OpenMedSpan.{name} is not in the span schema")
        elif live[name][0] != LEGACY_FIELDS[name]:
            problems.append(f"OpenMedSpan.{name} changed its compatibility type")
        elif not live[name][1]:
            problems.append(f"OpenMedSpan.{name} must stay deprecated")
    return problems


def check(
    *,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    module_path: Path = DEFAULT_MODULE_PATH,
    live: Mapping[str, tuple[str, str | None]] | None = None,
) -> list[str]:
    """Return the drift between the schema, the module and the live type.

    Args:
        schema_path: The span JSON Schema.
        module_path: The committed generated module.
        live: Live type fields; read from the service schema when omitted.

    Returns:
        Value-free problem descriptions; empty when everything agrees.
    """
    schema = _load_schema(schema_path)
    problems: list[str] = []
    committed = module_path.read_text(encoding="utf-8") if module_path.exists() else ""
    if committed != render_module(schema):
        problems.append("graphql_span_fields.py differs from the span schema")
    live_fields = live_type_fields() if live is None else live
    problems.extend(type_drift(canonical_fields(schema), live_fields))
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    """Rewrite the generated module, or report drift with ``--check``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when the module or the live OpenMedSpan type is stale",
    )
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--module", type=Path, default=DEFAULT_MODULE_PATH)
    args = parser.parse_args(argv)
    try:
        if args.check:
            problems = check(schema_path=args.schema, module_path=args.module)
            for problem in problems:
                print(problem, file=sys.stderr)
            if problems:
                print(
                    "Run `python scripts/codegen/span_graphql.py` and "
                    "`python scripts/export_graphql_schema.py`, and commit the "
                    "result.",
                    file=sys.stderr,
                )
            return 1 if problems else 0
        rendered = render_module(_load_schema(args.schema))
        current = (
            args.module.read_text(encoding="utf-8") if args.module.exists() else ""
        )
        if current != rendered:
            args.module.write_text(rendered, encoding="utf-8")
            print("Updated graphql_span_fields.py; re-export the GraphQL SDL.")
        return 0
    except SpanGraphQLCodegenError as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _load_schema(path: Path) -> dict[str, Any]:
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SpanGraphQLCodegenError("span schema is unreadable") from exc
    if not isinstance(schema, dict):
        raise SpanGraphQLCodegenError("span schema must be a JSON object")
    return schema


def _base_type(
    name: str, prop: Mapping[str, Any], *, nullable_allowed: bool = True
) -> str:
    declared = prop.get("type")
    types = [declared] if isinstance(declared, str) else list(declared or ())
    if "null" in types and not nullable_allowed:
        raise SpanGraphQLCodegenError(f"property {name!r} has nullable items")
    base = [item for item in types if item != "null"]
    if len(base) != 1 or base[0] not in {*_SCALARS, "object", "array"}:
        raise SpanGraphQLCodegenError(f"property {name!r} has no single type")
    return base[0]


if __name__ == "__main__":
    raise SystemExit(main())
