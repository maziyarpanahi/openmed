from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from openmed.integrations import postgres_plpython

_DDL_PATH = (
    Path(__file__).resolve().parents[3]
    / "openmed"
    / "integrations"
    / "sql"
    / "postgres_deidentify.sql"
)
_CREATE_FUNCTION = re.compile(
    r"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+"
    r"(?P<name>[a-z_][a-z0-9_]*)\s*\("
    r"(?P<argument_name>[a-z_][a-z0-9_]*)\s+"
    r"(?P<argument_type>text(?:\[\])?)\s*\)\s*"
    r"RETURNS\s+(?P<return_type>SETOF\s+text|text)\b"
    r"(?P<attributes>.*?)"
    r"AS\s+(?P<tag>\$[a-z_][a-z0-9_]*\$)"
    r"(?P<body>.*?)"
    r"(?P=tag)\s*;",
    flags=re.IGNORECASE | re.DOTALL,
)


def _fake_batch_result(texts: list[str]) -> SimpleNamespace:
    replacements = {
        "Jane Roe": "[PERSON]",
        "jane.roe@example.com": "[EMAIL]",
        "555-0100": "[PHONE]",
        "John Doe": "[PERSON]",
        "555-0199": "[PHONE]",
    }
    items = []
    for text in texts:
        redacted = text
        for source, replacement in replacements.items():
            redacted = redacted.replace(source, replacement)
        items.append(
            SimpleNamespace(
                success=True,
                result=SimpleNamespace(deidentified_text=redacted),
            )
        )
    return SimpleNamespace(items=items)


def test_function_bodies_redact_synthetic_rows_and_reuse_session_pipeline():
    session_globals: dict[str, Any] = {}
    loader_creations = 0
    calls: list[tuple[list[str], dict[str, Any]]] = []
    pipelines: list[object] = []

    class FakeLoader:
        def __init__(self) -> None:
            self.pipeline: object | None = None

        def create_pipeline(self) -> object:
            if self.pipeline is None:
                self.pipeline = object()
                pipelines.append(self.pipeline)
            return self.pipeline

    loader = FakeLoader()

    def loader_factory() -> Any:
        nonlocal loader_creations
        loader_creations += 1
        return loader

    def process_batch(texts: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append((list(texts), kwargs))
        kwargs["loader"].create_pipeline()
        return _fake_batch_result(texts)

    scalar = postgres_plpython.deidentify_text(
        "Patient Jane Roe emailed jane.roe@example.com.",
        session_globals,
        process_batch_fn=process_batch,
        loader_factory=loader_factory,
    )
    rows = postgres_plpython.deidentify_batch(
        [
            "John Doe called 555-0199.",
            None,
            "",
            "Call Jane Roe at 555-0100.",
        ],
        session_globals,
        process_batch_fn=process_batch,
        loader_factory=loader_factory,
    )

    assert scalar == "Patient [PERSON] emailed [EMAIL]."
    assert rows == [
        "[PERSON] called [PHONE].",
        None,
        "",
        "Call [PERSON] at [PHONE].",
    ]
    assert loader_creations == 1
    assert len(pipelines) == 1
    assert len(calls) == 2
    assert calls[0][1]["loader"] is loader
    assert calls[1][1]["loader"] is loader
    assert calls[1][0] == [
        "John Doe called 555-0199.",
        "Call Jane Roe at 555-0100.",
    ]
    for _, kwargs in calls:
        assert kwargs["operation"] == "deidentify"
        assert kwargs["method"] == "mask"
        assert kwargs["policy"] == "hipaa_safe_harbor"
        assert kwargs["model_name"] == postgres_plpython.DEFAULT_MODEL_NAME


def test_install_migration_parses_and_matches_documented_signatures():
    ddl = _DDL_PATH.read_text(encoding="utf-8")
    functions = {
        match.group("name"): match.groupdict()
        for match in _CREATE_FUNCTION.finditer(ddl)
    }

    assert re.search(r"\bBEGIN\s*;", ddl, flags=re.IGNORECASE)
    assert re.search(
        r"CREATE\s+EXTENSION\s+IF\s+NOT\s+EXISTS\s+plpython3u\s*;",
        ddl,
        flags=re.IGNORECASE,
    )
    assert re.search(r"\bCOMMIT\s*;", ddl, flags=re.IGNORECASE)
    assert set(functions) == {"openmed_deidentify", "openmed_deidentify_batch"}
    assert functions["openmed_deidentify"]["argument_type"].lower() == "text"
    assert functions["openmed_deidentify"]["return_type"].lower() == "text"
    assert functions["openmed_deidentify_batch"]["argument_type"].lower() == ("text[]")
    assert functions["openmed_deidentify_batch"]["return_type"].lower() == (
        "setof text"
    )

    argument_names = {
        "openmed_deidentify": "input_text",
        "openmed_deidentify_batch": "input_texts",
    }
    for name, function in functions.items():
        attributes = function["attributes"].upper()
        assert "LANGUAGE PLPYTHON3U" in attributes
        assert "STRICT" in attributes
        assert "VOLATILE" in attributes
        assert "PARALLEL UNSAFE" in attributes
        wrapped_body = "def _plpython_body(" + argument_names[name] + ", GD):\n"
        wrapped_body += textwrap.indent(function["body"].strip(), "    ")
        ast.parse(wrapped_body)


def test_function_failures_and_sql_bodies_never_log_raw_text(caplog):
    raw_text = "Patient Jane Roe has MRN 12345678"

    def failing_process_batch(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise ValueError(raw_text)

    with pytest.raises(RuntimeError, match="OpenMed de-identification failed") as exc:
        postgres_plpython.deidentify_text(
            raw_text,
            {},
            process_batch_fn=failing_process_batch,
            loader_factory=object,
        )

    assert raw_text not in str(exc.value)
    assert raw_text not in caplog.text

    ddl = _DDL_PATH.read_text(encoding="utf-8")
    bodies = [match.group("body") for match in _CREATE_FUNCTION.finditer(ddl)]
    assert len(bodies) == 2
    for body in bodies:
        assert not re.search(
            r"\bplpy\.(?:debug|log|info|notice|warning|error|fatal)\s*\(",
            body,
            flags=re.IGNORECASE,
        )
