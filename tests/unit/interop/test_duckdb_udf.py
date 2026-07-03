from __future__ import annotations

from types import SimpleNamespace

import duckdb
import pytest

from openmed.interop import adapter_spec, available_adapters, duckdb_udf, get_adapter
from openmed.interop.duckdb_udf import DuckDBUDFConfig, register_openmed_udfs


def fake_deidentifier(text: str, **kwargs):
    assert kwargs["method"] == "mask"
    replacements = {
        "Jane Roe": "[PERSON]",
        "jane.roe@example.com": "[EMAIL]",
        "555-0100": "[PHONE]",
    }
    redacted = (
        text.replace("Jane Roe", "[PERSON]")
        .replace("jane.roe@example.com", "[EMAIL]")
        .replace("555-0100", "[PHONE]")
    )
    entities = [
        SimpleNamespace(label=placeholder.strip("[]"))
        for surface, placeholder in replacements.items()
        if surface in text
    ]
    return SimpleNamespace(deidentified_text=redacted, pii_entities=entities)


def test_registry_loads_duckdb_adapter_lazily():
    adapter = get_adapter("duckdb")

    assert adapter is duckdb_udf
    assert "duckdb" in available_adapters()
    assert adapter_spec("duckdb").extra == "duckdb"
    assert hasattr(adapter, "register_openmed_udfs")


def test_register_openmed_udfs_redacts_fixture_column_in_query():
    calls: list[tuple[str, str]] = []

    def deidentifier(text: str, **kwargs):
        calls.append((text, kwargs["policy"]))
        return fake_deidentifier(text, **kwargs)

    con = duckdb.connect(":memory:")
    register_openmed_udfs(con, deidentifier=deidentifier)
    con.execute(
        """
        CREATE TABLE notes AS
        SELECT *
        FROM (
            VALUES
                (1, 'Patient Jane Roe emailed jane.roe@example.com.'),
                (2, 'No direct identifier here.')
        ) AS fixture(id, note)
        """
    )

    rows = con.execute(
        """
        SELECT id, openmed_deidentify(note, 'safe_harbor') AS redacted
        FROM notes
        ORDER BY id
        """
    ).fetchall()

    assert rows == [
        (1, "Patient [PERSON] emailed [EMAIL]."),
        (2, "No direct identifier here."),
    ]
    assert calls == [
        ("Patient Jane Roe emailed jane.roe@example.com.", "hipaa_safe_harbor"),
        ("No direct identifier here.", "hipaa_safe_harbor"),
    ]


def test_register_openmed_udfs_adds_helper_pii_count():
    con = duckdb.connect(":memory:")
    register_openmed_udfs(con, deidentifier=fake_deidentifier)

    count = con.execute(
        "SELECT openmed_pii_count('Call Jane Roe at 555-0100.')"
    ).fetchone()[0]

    assert count == 2


def test_default_deidentifier_reuses_cached_loader(monkeypatch):
    loader = object()
    calls: list[object] = []

    def deidentifier(text: str, **kwargs):
        del text
        calls.append(kwargs["loader"])
        return SimpleNamespace(deidentified_text="[PERSON]", pii_entities=[])

    monkeypatch.setattr(duckdb_udf, "_cached_model_loader", lambda: loader)
    monkeypatch.setattr(duckdb_udf, "_default_deidentifier", lambda: deidentifier)
    con = duckdb.connect(":memory:")
    register_openmed_udfs(
        con,
        config=DuckDBUDFConfig(default_policy="hipaa_safe_harbor"),
    )

    rows = con.execute(
        """
        SELECT openmed_deidentify(note, 'hipaa_safe_harbor')
        FROM (VALUES ('Patient Jane Roe'), ('Patient John Doe')) AS notes(note)
        """
    ).fetchall()

    assert rows == [("[PERSON]",), ("[PERSON]",)]
    assert calls == [loader, loader]


def test_register_openmed_udfs_raises_clear_error_without_duckdb(monkeypatch):
    def missing_dependency(name: str):
        raise ImportError(name)

    monkeypatch.setattr(duckdb_udf, "_import_module", missing_dependency)

    with pytest.raises(ImportError, match=r"openmed\[duckdb\]"):
        register_openmed_udfs(object(), deidentifier=fake_deidentifier)
