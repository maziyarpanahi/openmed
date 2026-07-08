"""Tests for the dbt de-identification package (examples/dbt-deidentify).

The package expresses PHI redaction as a dbt macro wrapping the
``openmed_deidentify(text, policy)`` warehouse UDF. These tests compile the
macro with Jinja and execute the generated SQL against DuckDB (the local
adapter's engine) with a stubbed UDF, so they need no warehouse and no network.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

_PACKAGE = Path(__file__).resolve().parents[3] / "examples" / "dbt-deidentify"


def _render(call: str) -> str:
    """Render a call to a package macro, returning the generated SQL."""
    jinja2 = pytest.importorskip("jinja2")
    source = (_PACKAGE / "macros" / "redact.sql").read_text(encoding="utf-8")
    template = jinja2.Environment().from_string(source + "{{ " + call + " }}")
    return template.render().strip()


def _seed_connection():
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE patients AS SELECT * FROM (VALUES "
        "(1, 'John Doe reports chest pain; MRN 12345'), "
        "(2, 'Jane Roe seen on 2024-03-02; phone 617-555-0134')) "
        "AS t(patient_id, notes)"
    )
    return con


def _stub_deidentifier():
    def _deidentify(text, **kwargs):  # noqa: ARG001 - policy/kwargs unused in the stub
        return SimpleNamespace(deidentified_text="[REDACTED]")

    return _deidentify


def test_redact_macro_renders_udf_call():
    assert (
        _render("redact('notes')") == "openmed_deidentify(notes, 'hipaa_safe_harbor')"
    )
    assert (
        _render("redact('reason', 'gdpr_pseudonymization')")
        == "openmed_deidentify(reason, 'gdpr_pseudonymization')"
    )


def test_redact_columns_lists_every_configured_column():
    sql = _render("redact_columns(['notes', 'reason_for_visit'])")
    assert "openmed_deidentify(notes, 'hipaa_safe_harbor') as notes" in sql
    assert (
        "openmed_deidentify(reason_for_visit, 'hipaa_safe_harbor') as reason_for_visit"
        in sql
    )


def test_generated_sql_redacts_columns_and_removes_phi():
    from openmed.interop.duckdb_udf import register_openmed_udfs

    con = _seed_connection()
    register_openmed_udfs(con, deidentifier=_stub_deidentifier())

    redacted = _render("redact('notes')")
    rows = con.execute(
        f"SELECT patient_id, {redacted} AS notes FROM patients ORDER BY patient_id"
    ).fetchall()

    # The stub UDF stands in for the real de-identifier (which is out of scope
    # here and covered by test_duckdb_udf.py): this proves the macro-generated
    # SQL routes each configured column through openmed_deidentify and replaces
    # its value, so no source text survives in the materialized output.
    assert [row[1] for row in rows] == ["[REDACTED]", "[REDACTED]"]
    materialized = " ".join(str(row) for row in rows)
    for leaked in ("John", "Jane", "12345", "617-555"):
        assert leaked not in materialized


def test_macro_errors_clearly_when_udf_not_registered():
    con = _seed_connection()  # UDF intentionally not registered
    redacted = _render("redact('notes')")

    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - DuckDB error type varies
        con.execute(f"SELECT {redacted} AS notes FROM patients").fetchall()

    assert "openmed_deidentify" in str(excinfo.value)


def test_seed_and_docs_are_present_and_synthetic():
    seed = (_PACKAGE / "seeds" / "synthetic_patients.csv").read_text(encoding="utf-8")
    assert seed.splitlines()[0] == "patient_id,notes,reason_for_visit"
    readme = (_PACKAGE / "README.md").read_text(encoding="utf-8").lower()
    assert "synthetic" in readme
    assert (_PACKAGE / "models" / "stg_redacted.sql").exists()
    assert (_PACKAGE / "dbt_project.yml").exists()
