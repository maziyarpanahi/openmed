from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import openmed
from openmed.interop import adapter_spec, available_adapters, get_adapter, snowflake_udf

try:
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as _toml  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[3]


def test_deidentify_udf_returns_redacted_text_without_snowflake(monkeypatch):
    calls: list[tuple[str, str, str | None]] = []

    def fake_deidentify(text: str, *, method: str, policy: str | None):
        calls.append((text, method, policy))
        return SimpleNamespace(deidentified_text=text.replace("Jane Roe", "[NAME]"))

    monkeypatch.setattr(openmed, "deidentify", fake_deidentify, raising=False)

    assert (
        snowflake_udf.deidentify_udf(
            "Patient Jane Roe", method="mask", policy="hipaa_safe_harbor"
        )
        == "Patient [NAME]"
    )
    assert calls == [("Patient Jane Roe", "mask", "hipaa_safe_harbor")]


def test_deidentify_udf_preserves_sql_null():
    assert snowflake_udf.deidentify_udf(None) is None


def test_register_udf_forwards_handler_and_name(monkeypatch):
    class FakeStringType:
        pass

    registered: dict[str, object] = {}

    class FakeUDF:
        def register(self, function, **kwargs):
            registered["function"] = function
            registered.update(kwargs)
            return "registered-udf"

    session = SimpleNamespace(udf=FakeUDF())
    monkeypatch.setattr(snowflake_udf, "_load_string_type", lambda: FakeStringType)

    result = snowflake_udf.register_udf(
        session,
        name="OPENMED_REDACT",
        imports=["@stage/helper.py"],
        packages=["openmed", "pandas"],
    )

    assert result == "registered-udf"
    assert registered["function"] is snowflake_udf.deidentify_udf
    assert registered["name"] == "OPENMED_REDACT"
    assert registered["packages"] == ["openmed", "pandas"]
    assert registered["imports"] == ["@stage/helper.py"]
    assert isinstance(registered["return_type"], FakeStringType)
    assert isinstance(registered["input_types"][0], FakeStringType)  # type: ignore[index]


def test_generate_create_function_sql_contains_python_handler_and_package():
    sql = snowflake_udf.generate_create_function_sql()

    assert "CREATE FUNCTION OPENMED_DEIDENTIFY(TEXT STRING)" in sql
    assert "LANGUAGE PYTHON" in sql
    assert "PACKAGES = ('openmed')" in sql
    assert "HANDLER = 'openmed.interop.snowflake_udf.deidentify_udf'" in sql


def test_generate_create_function_sql_supports_permanent_style_options():
    sql = snowflake_udf.generate_create_function_sql(
        name="ANALYTICS.OPENMED_DEIDENTIFY",
        runtime_version="3.11",
        packages=["openmed", "pandas==2.2.3"],
        imports=["@stage/helpers.py"],
        replace=True,
    )

    assert "CREATE OR REPLACE FUNCTION ANALYTICS.OPENMED_DEIDENTIFY" in sql
    assert "RUNTIME_VERSION = '3.11'" in sql
    assert "PACKAGES = ('openmed', 'pandas==2.2.3')" in sql
    assert "IMPORTS = ('@stage/helpers.py')" in sql


def test_registry_and_extra_expose_snowflake_lazily():
    adapter = get_adapter("snowflake")

    assert adapter is snowflake_udf
    assert "snowflake" in available_adapters()
    assert adapter_spec("snowflake").extra == "snowflake"

    with (ROOT / "pyproject.toml").open("rb") as handle:
        dependencies = _toml.load(handle)["project"]["optional-dependencies"][
            "snowflake"
        ]

    assert any(
        requirement.startswith("snowflake-snowpark-python")
        for requirement in dependencies
    )


def test_register_udf_requires_optional_snowflake_extra(monkeypatch):
    def missing_dependency(name: str):
        raise ImportError("snowflake")

    monkeypatch.setattr(snowflake_udf, "_import_module", missing_dependency)

    with pytest.raises(ImportError, match=r"openmed\[snowflake\]"):
        snowflake_udf.register_udf(object())
