"""Focused offline tests for the standalone OpenMed dbt redaction macros."""

from __future__ import annotations

from pathlib import Path

import pytest

_MACRO_PATH = (
    Path(__file__).resolve().parents[3]
    / "integrations"
    / "dbt"
    / "macros"
    / "redact.sql"
)

_CANONICAL_POLICIES = (
    "hipaa_safe_harbor",
    "hipaa_expert_review_assist",
    "gdpr_pseudonymization",
    "gdpr_art9_health",
    "research_limited_dataset",
    "strict_no_leak",
    "clinical_minimal_redaction",
    "canada_pipeda",
    "uk_ico_anonymisation",
    "australia_privacy_act",
    "china_pipl",
    "india_dpdp_act",
    "africa_malabo_baseline",
    "za_popia",
    "ng_ndpa",
    "ke_dpa",
    "india_health_id",
    "eg_pdpl",
    "ma_law_09_08",
)


class _CompilerError(RuntimeError):
    """Small dbt exceptions stand-in used by the offline Jinja tests."""


class _Exceptions:
    @staticmethod
    def raise_compiler_error(message: str) -> None:
        raise _CompilerError(message)


def _render(call: str, **context: object) -> str:
    """Render one macro call without dbt, a warehouse, or network access."""
    jinja2 = pytest.importorskip("jinja2")
    source = _MACRO_PATH.read_text(encoding="utf-8")
    template = jinja2.Environment().from_string(source + "\n{{ " + call + " }}")
    return template.render(**context).strip()


def _render_compiler_error(call: str, **context: object) -> str:
    with pytest.raises(_CompilerError) as excinfo:
        _render(call, exceptions=_Exceptions(), **context)
    return str(excinfo.value)


def test_redact_renders_deterministic_parameterized_default_call() -> None:
    call = "redact('clinical_note')"
    expected = "openmed_deidentify(clinical_note, 'hipaa_safe_harbor')"

    assert _render(call) == expected
    assert _render(call) == expected


def test_redact_accepts_a_qualified_explicit_identifier() -> None:
    assert _render("redact('source.clinical_note')") == (
        "openmed_deidentify(source.clinical_note, 'hipaa_safe_harbor')"
    )


@pytest.mark.parametrize("policy", _CANONICAL_POLICIES)
def test_redact_accepts_each_canonical_policy(policy: str) -> None:
    rendered = _render(f"redact('clinical_note', '{policy}')")

    assert rendered == f"openmed_deidentify(clinical_note, '{policy}')"


def test_redact_uses_adapter_dispatch_when_available() -> None:
    class Adapter:
        def dispatch(self, name: str):
            assert name == "redact"

            def adapter_redact(column: str, policy: str) -> str:
                return f"warehouse_redact({column}, '{policy}')"

            return adapter_redact

    assert (
        _render("redact('clinical_note', 'strict_no_leak')", adapter=Adapter())
        == "warehouse_redact(clinical_note, 'strict_no_leak')"
    )


def test_redact_columns_requires_and_preserves_explicit_columns() -> None:
    rendered = _render("redact_columns(['clinical_note', 'free_text'])")

    assert rendered == (
        "openmed_deidentify(clinical_note, 'hipaa_safe_harbor') as clinical_note, "
        "openmed_deidentify(free_text, 'hipaa_safe_harbor') as free_text"
    )
    assert "*" not in rendered
    assert "select" not in rendered.lower()


def test_invalid_policy_fails_without_echoing_the_supplied_value() -> None:
    error = _render_compiler_error("redact('clinical_note', 'not_a_policy')")

    assert "policy" in error.lower()
    assert "not_a_policy" not in error


@pytest.mark.parametrize(
    "column",
    (
        "clinical note",
        "clinical-note",
        "clinical_note || 'synthetic-sensitive-value'",
        "coalesce(clinical_note, 'synthetic-sensitive-value')",
        "'synthetic-sensitive-value'",
        "schema.table.extra",
        ".clinical_note",
        "clinical_note.",
        "1clinical_note",
    ),
)
def test_sql_fragments_fail_closed_without_echoing_the_supplied_value(
    column: str,
) -> None:
    error = _render_compiler_error("redact(candidate)", candidate=column)

    assert "explicit column" in error
    assert column not in error


@pytest.mark.parametrize(
    "call",
    (
        "redact('*')",
        "redact('clinical_note.*')",
        "redact_columns('*')",
        "redact_columns([])",
        "redact_columns(['clinical_note', '*'])",
    ),
)
def test_wildcards_and_implicit_full_table_projections_fail_closed(call: str) -> None:
    error = _render_compiler_error(call)

    assert "full-table" in error or "wildcard" in error


def test_macro_source_contains_no_sensitive_fixture_values() -> None:
    source = _MACRO_PATH.read_text(encoding="utf-8")

    assert "clinical_note" not in source
    assert "free_text" not in source
