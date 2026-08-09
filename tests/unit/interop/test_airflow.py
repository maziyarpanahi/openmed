from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from openmed.interop import adapter_spec, available_adapters, get_adapter
from openmed.interop import airflow as airflow_adapter
from openmed.interop.airflow import OpenMedRedactionOperator, RedactionOperatorError

_SYNTHETIC_VALUE = "synthetic-person-001@example.test"


def _fake_deidentifier(text: str, **kwargs):
    del kwargs
    redacted = text.replace(_SYNTHETIC_VALUE, "[EMAIL]")
    return SimpleNamespace(
        deidentified_text=redacted,
        pii_entities=[object()] if redacted != text else [],
    )


def test_registry_lists_airflow_without_importing_optional_dependency():
    assert "airflow" in available_adapters()
    assert adapter_spec("airflow").extra == "airflow"
    assert adapter_spec("airflow").module == "openmed.interop.airflow"

    code = """
import builtins

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "airflow" or name.startswith("airflow."):
        raise AssertionError(f"unexpected Airflow import: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import

import openmed
from openmed.interop import available_adapters

assert openmed is not None
assert "airflow" in available_adapters()
"""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=False,
        cwd=".",
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_airflow_extra_declares_optional_dependency():
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10
        import tomli as tomllib  # type: ignore[no-redef]

    root = Path(__file__).resolve().parents[3]
    with (root / "pyproject.toml").open("rb") as handle:
        dependencies = tomllib.load(handle)["project"]["optional-dependencies"][
            "airflow"
        ]

    assert any(requirement.startswith("apache-airflow") for requirement in dependencies)


def test_file_operator_redacts_atomically_and_emits_phi_free_fingerprint(
    tmp_path: Path,
    caplog,
):
    source = tmp_path / "notes.txt"
    output = tmp_path / "notes.redacted.txt"
    source.write_text(f"contact {_SYNTHETIC_VALUE}\n", encoding="utf-8")
    operator = OpenMedRedactionOperator(
        input_path=source,
        output_path=output,
        deidentifier=_fake_deidentifier,
    )

    with caplog.at_level("INFO"):
        result = operator.execute({})

    assert output.read_text(encoding="utf-8") == "contact [EMAIL]\n"
    assert result["status"] == "success"
    assert result["records_processed"] == 1
    assert result["records_redacted"] == 1
    assert result["spans_redacted"] == 1
    assert result["output_fingerprint"].startswith("sha256:")

    manifest_path = Path(f"{output}.openmed-fingerprint.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["output_fingerprint"] == result["output_fingerprint"]
    assert _SYNTHETIC_VALUE not in json.dumps(manifest)
    assert _SYNTHETIC_VALUE not in caplog.text


def test_matching_output_fingerprint_makes_retries_idempotent(tmp_path: Path):
    source = tmp_path / "notes.txt"
    output = tmp_path / "notes.redacted.txt"
    source.write_text(_SYNTHETIC_VALUE, encoding="utf-8")
    calls: list[str] = []

    def recording_deidentifier(text: str, **kwargs):
        calls.append(text)
        return _fake_deidentifier(text, **kwargs)

    operator = OpenMedRedactionOperator(
        input_path=source,
        output_path=output,
        deidentifier=recording_deidentifier,
    )

    first = operator.execute({})
    second = operator.execute({})

    assert calls == [_SYNTHETIC_VALUE]
    assert first["output_fingerprint"] == second["output_fingerprint"]
    assert second["status"] == "skipped"


def test_record_batch_returns_redacted_records_without_writing_raw_values():
    operator = OpenMedRedactionOperator(
        records=[
            {"text": _SYNTHETIC_VALUE, "kind": "synthetic"},
            {"text": None, "kind": "empty"},
        ],
        deidentifier=_fake_deidentifier,
    )

    result = operator.execute({})

    assert result["mode"] == "records"
    assert result["records_processed"] == 2
    assert result["records_redacted"] == 1
    assert result["redacted_records"] == [
        {"text": "[EMAIL]", "kind": "synthetic"},
        {"text": None, "kind": "empty"},
    ]


def test_jsonl_file_input_writes_a_bounded_record_output(tmp_path: Path):
    source = tmp_path / "notes.jsonl"
    output = tmp_path / "notes.redacted.jsonl"
    source.write_text(
        json.dumps({"record_id": "stable-1", "note": _SYNTHETIC_VALUE}) + "\n",
        encoding="utf-8",
    )

    result = OpenMedRedactionOperator(
        input_path=source,
        output_path=output,
        text_field="note",
        deidentifier=_fake_deidentifier,
    ).execute({})

    assert result["mode"] == "file-records"
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "record_id": "stable-1",
        "note": "[EMAIL]",
    }


def test_record_batch_bound_and_deidentifier_failures_are_value_free():
    with pytest.raises(RedactionOperatorError, match="configured limit") as limit_error:
        OpenMedRedactionOperator(
            records=["one", "two"],
            max_records=1,
            deidentifier=_fake_deidentifier,
        ).execute({})
    assert _SYNTHETIC_VALUE not in str(limit_error.value)

    def broken_deidentifier(text: str, **kwargs):
        del kwargs
        raise RuntimeError(f"leaked value: {text}")

    with pytest.raises(RedactionOperatorError) as redaction_error:
        OpenMedRedactionOperator(
            records=[_SYNTHETIC_VALUE],
            deidentifier=broken_deidentifier,
        ).execute({})
    assert _SYNTHETIC_VALUE not in str(redaction_error.value)


def test_default_deidentifier_is_configured_cache_only(monkeypatch):
    captured: dict[str, object] = {}

    def fake_default(text: str, **kwargs):
        captured.update(kwargs)
        return text

    monkeypatch.setattr(airflow_adapter, "_default_deidentifier", fake_default)
    operator = OpenMedRedactionOperator(records=["synthetic note"])

    operator.execute({})

    config = captured["config"]
    assert getattr(config, "local_only") is True
    assert captured["method"] == "mask"
    assert captured["policy"] == "hipaa_safe_harbor"


def test_get_adapter_loads_airflow_module_without_requiring_airflow_extra():
    adapter = get_adapter("airflow")

    assert adapter is airflow_adapter
    assert adapter.OpenMedRedactionOperator is OpenMedRedactionOperator
