"""Snowflake Python UDF helpers for in-warehouse de-identification.

Snowpark is optional and is imported only when :func:`register_udf` is
called. The handler itself is deliberately a plain Python callable so it can
be serialized by Snowpark or tested without a Snowflake account.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from importlib import import_module as _import_module
from typing import Any

DEFAULT_HANDLER = "openmed.interop.snowflake_udf.deidentify_udf"
DEFAULT_NAME = "OPENMED_DEIDENTIFY"
DEFAULT_RUNTIME_VERSION = "3.10"


def deidentify_udf(
    text: str | None,
    method: str = "mask",
    policy: str | None = None,
) -> str | None:
    """Return de-identified text for one Snowflake UDF value.

    The function has no Snowpark-specific types or decorators, which keeps it
    usable as a direct Snowpark handler and as a local unit-test target. The
    OpenMed import is local so loading this adapter never initializes the
    de-identification pipeline until a row is actually processed.

    Args:
        text: Text value to de-identify. ``None`` is preserved for SQL NULL.
        method: OpenMed de-identification method, defaulting to ``"mask"``.
        policy: Optional OpenMed policy profile. When omitted, OpenMed applies
            its default policy.
    """

    if text is None:
        return None

    from openmed import deidentify

    result = deidentify(text, method=method, policy=policy)
    if isinstance(result, str):
        return result
    try:
        return str(result.deidentified_text)
    except AttributeError as exc:
        raise TypeError(
            "deidentify must return a string or an object with deidentified_text"
        ) from exc


def register_udf(
    session: Any,
    name: str = DEFAULT_NAME,
    *,
    imports: Sequence[str] | None = None,
    packages: Sequence[str] | None = None,
    replace: bool = True,
    is_permanent: bool = False,
    stage_location: str | None = None,
) -> Any:
    """Register the OpenMed handler on a Snowpark session.

    ``snowflake.snowpark`` is imported only inside this function. The default
    ``packages`` value tells Snowpark to make the published ``openmed``
    package available to the handler. Pass additional package names when the
    UDF needs extra Snowflake Anaconda dependencies.

    Args:
        session: A configured Snowpark ``Session``.
        name: SQL function name to create.
        imports: Optional stage paths for imported Python files or packages,
            forwarded to ``session.udf.register(imports=...)``.
        packages: Optional Snowflake package names. If omitted, ``openmed``
            is registered as the only package.
        replace: Replace an existing UDF with the same name when true.
        is_permanent: Ask Snowpark to create a permanent UDF.
        stage_location: Stage used for a permanent UDF's generated code.

    Returns:
        The object returned by ``session.udf.register``.

    Raises:
        ImportError: If the optional Snowpark dependency is not installed.
    """

    string_type = _load_string_type()
    register_kwargs: dict[str, Any] = {
        "name": name,
        "return_type": string_type(),
        # Register the SQL surface as one argument. Python defaults keep the
        # handler directly callable with optional method and policy values.
        "input_types": [string_type()],
        "packages": list(packages) if packages is not None else ["openmed"],
        "replace": replace,
        "is_permanent": is_permanent,
    }
    if imports is not None:
        register_kwargs["imports"] = list(imports)
    if stage_location is not None:
        register_kwargs["stage_location"] = stage_location

    return session.udf.register(deidentify_udf, **register_kwargs)


def generate_create_function_sql(
    name: str = DEFAULT_NAME,
    *,
    runtime_version: str = DEFAULT_RUNTIME_VERSION,
    packages: Sequence[str] = ("openmed",),
    imports: Sequence[str] = (),
    handler: str = DEFAULT_HANDLER,
    replace: bool = False,
) -> str:
    """Generate a Snowflake ``CREATE FUNCTION`` statement.

    The generated function accepts one ``STRING`` argument, so callers can
    use ``OPENMED_DEIDENTIFY(note)`` and rely on the handler's ``mask`` and
    default-policy values. ``packages`` must include the ``openmed`` package;
    ``imports`` can contain stage paths when deployment also needs local
    Python modules.

    Args:
        name: SQL function name, optionally qualified by database and schema.
        runtime_version: Snowflake Python runtime version, such as ``"3.10"``.
        packages: Snowflake package references for ``PACKAGES``.
        imports: Optional stage paths for the SQL ``IMPORTS`` clause.
        handler: Fully qualified Python handler path.
        replace: Emit ``CREATE OR REPLACE FUNCTION`` when true.

    Returns:
        A complete SQL statement ending in a semicolon.
    """

    function_name = _validate_qualified_identifier(name, field="name")
    runtime = str(runtime_version).strip()
    if not re.fullmatch(r"\d+\.\d+", runtime):
        raise ValueError("runtime_version must look like '<major>.<minor>'")
    handler_value = str(handler).strip()
    if not handler_value:
        raise ValueError("handler must not be empty")

    package_values = tuple(str(value) for value in packages)
    if not package_values:
        raise ValueError("packages must contain at least one package")
    if not any(
        value.split("==", 1)[0].strip().lower() == "openmed" for value in package_values
    ):
        raise ValueError("packages must include the openmed package")

    statement = "CREATE OR REPLACE FUNCTION" if replace else "CREATE FUNCTION"
    lines = [
        f"{statement} {function_name}(TEXT STRING)",
        "RETURNS STRING",
        "LANGUAGE PYTHON",
        f"RUNTIME_VERSION = '{_escape_sql_literal(runtime)}'",
        f"PACKAGES = ({_sql_literals(package_values)})",
    ]
    import_values = tuple(str(value) for value in imports)
    if import_values:
        lines.append(f"IMPORTS = ({_sql_literals(import_values)})")
    lines.append(f"HANDLER = '{_escape_sql_literal(handler_value)}';")
    return "\n".join(lines)


def _load_string_type() -> Any:
    """Load Snowpark's ``StringType`` without an import-time dependency."""

    try:
        types_module = _import_module("snowflake.snowpark.types")
    except ImportError as exc:
        raise ImportError(
            "Snowflake support requires the optional dependency; install "
            "openmed[snowflake] to use openmed.interop.snowflake_udf"
        ) from exc
    return types_module.StringType


def _validate_qualified_identifier(value: str, *, field: str) -> str:
    identifier = str(value).strip()
    component = r"[A-Za-z_][A-Za-z0-9_$]*"
    if not identifier or not re.fullmatch(
        rf"{component}(?:\.{component})*", identifier
    ):
        raise ValueError(f"{field} must be a simple or qualified SQL identifier")
    return identifier


def _escape_sql_literal(value: str) -> str:
    return value.replace("'", "''")


def _sql_literals(values: Sequence[str]) -> str:
    return ", ".join(f"'{_escape_sql_literal(value)}'" for value in values)


__all__ = [
    "DEFAULT_HANDLER",
    "DEFAULT_NAME",
    "DEFAULT_RUNTIME_VERSION",
    "deidentify_udf",
    "generate_create_function_sql",
    "register_udf",
]
