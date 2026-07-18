from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from openmed.core.pii import DeidentificationResult, PIIEntity
from openmed.interop import adapter_spec, available_adapters


class _StubPrefectTask:
    """Minimal stand-in for ``prefect.Task`` exposing ``.fn``."""

    def __init__(self, fn, **options):
        self.fn = fn
        self.options = options

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class _StubPrefectFlow(_StubPrefectTask):
    """Minimal stand-in for ``prefect.Flow`` exposing ``.fn``."""


def _stub_prefect_decorator(wrapper_cls):
    def decorator(fn=None, **options):
        if fn is None:
            return lambda inner: wrapper_cls(inner, **options)
        return wrapper_cls(fn, **options)

    return decorator


def _install_prefect_stub_if_missing() -> None:
    if "prefect" in sys.modules:
        return
    if importlib.util.find_spec("prefect") is not None:
        return
    module = ModuleType("prefect")
    module.task = _stub_prefect_decorator(_StubPrefectTask)
    module.flow = _stub_prefect_decorator(_StubPrefectFlow)
    sys.modules["prefect"] = module


@pytest.fixture(scope="module")
def prefect_tasks():
    _install_prefect_stub_if_missing()
    from openmed.interop import prefect_tasks as module

    return module


def test_registry_lists_prefect_adapter_without_importing_prefect():
    assert "prefect" in available_adapters()
    assert adapter_spec("prefect").extra == "prefect"
    assert adapter_spec("prefect").module == "openmed.interop.prefect_tasks"

    code = """
import builtins

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "prefect" or name.startswith("prefect."):
        raise AssertionError(f"unexpected Prefect import: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import

import openmed
from openmed.interop import available_adapters

assert openmed is not None
assert "prefect" in available_adapters()
"""

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=False,
        cwd=".",
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_deidentify_file_task_fn_redacts_temp_file_and_summarizes(
    tmp_path: Path,
    monkeypatch,
    prefect_tasks,
) -> None:
    input_path = tmp_path / "notes.csv"
    input_path.write_text(
        "id,note,age\n"
        "1,Patient John Doe called 555-0101,42\n"
        "2,Patient Jane Roe emailed jane@example.test,37\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("openmed.core.pii.deidentify", _fake_deidentify)

    summary = prefect_tasks.deidentify_file_task.fn(
        input_path,
        text_columns=["note"],
        policy="strict_no_leak",
    )

    output_path = tmp_path / "notes.redacted.csv"
    rows = list(csv.DictReader(output_path.open(encoding="utf-8", newline="")))
    assert rows == [
        {"id": "1", "note": "Patient [PERSON] called [PHONE]", "age": "42"},
        {"id": "2", "note": "Patient [PERSON] emailed [EMAIL]", "age": "37"},
    ]
    assert summary == {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "files_processed": 1,
        "rows_processed": 2,
        "cells_redacted": 2,
        "spans_redacted": 4,
    }
    _assert_summary_has_no_fixture_phi(summary)


def test_deidentify_dataset_flow_fn_fans_task_and_aggregates(
    tmp_path: Path,
    monkeypatch,
    prefect_tasks,
) -> None:
    input_paths = [tmp_path / "batch-a.csv", tmp_path / "batch-b.csv"]
    for input_path in input_paths:
        input_path.write_text("id,note\n1,synthetic\n", encoding="utf-8")
    output_dir = tmp_path / "redacted"
    output_dir.mkdir()
    calls: list[dict] = []
    monkeypatch.setattr(
        prefect_tasks,
        "_load_redact_dataset",
        lambda: _recording_redactor(calls),
    )

    result = prefect_tasks.deidentify_dataset_flow.fn(
        input_paths,
        text_columns=["note"],
        output_dir=output_dir,
        policy="hipaa_safe_harbor",
        method="replace",
        confidence_threshold=0.9,
    )

    assert [call["path"] for call in calls] == [str(path) for path in input_paths]
    for call in calls:
        assert call["text_columns"] == ["note"]
        assert call["policy"] == "hipaa_safe_harbor"
        assert call["method"] == "replace"
        assert call["confidence_threshold"] == 0.9
    assert result["files_processed"] == 2
    assert result["rows_processed"] == 4
    assert result["cells_redacted"] == 2
    assert result["spans_redacted"] == 6
    assert [item["output_path"] for item in result["files"]] == [
        str(output_dir / "batch-a.redacted.csv"),
        str(output_dir / "batch-b.redacted.csv"),
    ]
    for item in result["files"]:
        assert item["files_processed"] == 1
        assert Path(item["output_path"]).exists()


def _recording_redactor(calls: list[dict]):
    def fake_redact_dataset(
        path,
        text_columns,
        *,
        output_path=None,
        policy=None,
        method="mask",
        confidence_threshold=0.7,
        **kwargs,
    ):
        calls.append(
            {
                "path": str(path),
                "text_columns": list(text_columns),
                "policy": policy,
                "method": method,
                "confidence_threshold": confidence_threshold,
            }
        )
        destination = Path(output_path)
        destination.write_text("id,note\n1,[REDACTED]\n", encoding="utf-8")
        return SimpleNamespace(
            output_path=destination,
            summary=SimpleNamespace(total_rows=2, redacted_cells=1, total_spans=3),
        )

    return fake_redact_dataset


def _fake_deidentify(text: str, **kwargs) -> DeidentificationResult:
    replacements = {
        "John Doe": ("[PERSON]", "PERSON"),
        "Jane Roe": ("[PERSON]", "PERSON"),
        "555-0101": ("[PHONE]", "PHONE"),
        "jane@example.test": ("[EMAIL]", "EMAIL"),
    }
    redacted = text
    entities: list[PIIEntity] = []
    for surface, (replacement, label) in replacements.items():
        start = text.find(surface)
        if start == -1:
            continue
        entities.append(
            PIIEntity(
                text=surface,
                label=label,
                confidence=0.99,
                start=start,
                end=start + len(surface),
                entity_type=label,
                original_text=surface,
                redacted_text=replacement,
            )
        )
        redacted = redacted.replace(surface, replacement)

    return DeidentificationResult(
        original_text=text,
        deidentified_text=redacted,
        pii_entities=entities,
        method=kwargs.get("method", "mask"),
        timestamp=datetime.now(),
    )


def _assert_summary_has_no_fixture_phi(summary: dict) -> None:
    payload = json.dumps(summary, sort_keys=True)
    for token in ("John Doe", "Jane Roe", "555-0101", "jane@example.test"):
        assert token not in payload
