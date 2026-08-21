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


def test_retry_fingerprint_distinguishes_callback_closure_state(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    output = tmp_path / "notes.redacted.txt"
    source.write_text("synthetic note", encoding="utf-8")

    def make_deidentifier(replacement: str):
        def deidentifier(text: str, **kwargs):
            del text, kwargs
            return replacement

        return deidentifier

    OpenMedRedactionOperator(
        input_path=source,
        output_path=output,
        deidentifier=make_deidentifier("[FIRST]"),
    ).execute({})

    with pytest.raises(RedactionOperatorError, match="does not match this run"):
        OpenMedRedactionOperator(
            input_path=source,
            output_path=output,
            deidentifier=make_deidentifier("[SECOND]"),
        ).execute({})


def test_retry_fingerprint_distinguishes_callable_object_state(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    output = tmp_path / "notes.redacted.txt"
    source.write_text("synthetic note", encoding="utf-8")

    class Replacer:
        def __init__(self, replacement: str) -> None:
            self.replacement = replacement

        def __call__(self, text: str, **kwargs):
            del text, kwargs
            return self.replacement

    OpenMedRedactionOperator(
        input_path=source,
        output_path=output,
        deidentifier=Replacer("[FIRST]"),
    ).execute({})

    with pytest.raises(RedactionOperatorError, match="does not match this run"):
        OpenMedRedactionOperator(
            input_path=source,
            output_path=output,
            deidentifier=Replacer("[SECOND]"),
        ).execute({})


def test_deidentifier_options_are_snapshotted_and_fresh_per_record(
    tmp_path: Path,
) -> None:
    supplied_tags = ["en"]
    observed_tags: list[list[str]] = []

    def mutating_deidentifier(text: str, **kwargs):
        tags = kwargs["token_language_tags"]
        observed_tags.append(list(tags))
        tags.append("mutated")
        return text

    operator = OpenMedRedactionOperator(
        records=["first", "second"],
        output_path=tmp_path / "redacted.jsonl",
        deidentifier=mutating_deidentifier,
        deidentify_kwargs={"token_language_tags": supplied_tags},
    )
    supplied_tags.append("fr")

    operator.execute({})

    assert observed_tags == [["en"], ["en"]]


def test_record_batch_writes_output_and_returns_only_phi_free_counts(
    tmp_path: Path,
):
    output = tmp_path / f"{_SYNTHETIC_VALUE}.redacted.jsonl"
    operator = OpenMedRedactionOperator(
        records=[
            {"text": _SYNTHETIC_VALUE, "kind": "synthetic"},
            {"text": None, "kind": "empty"},
        ],
        output_path=output,
        deidentifier=_fake_deidentifier,
    )

    result = operator.execute({})

    assert result["mode"] == "records"
    assert result["records_processed"] == 2
    assert result["records_redacted"] == 1
    assert "redacted_records" not in result
    assert _SYNTHETIC_VALUE not in json.dumps(result)
    assert [json.loads(line) for line in output.read_text().splitlines()] == [
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


def test_record_batch_bound_and_deidentifier_failures_are_value_free(
    tmp_path: Path,
):
    with pytest.raises(RedactionOperatorError, match="configured limit") as limit_error:
        OpenMedRedactionOperator(
            records=["one", "two"],
            output_path=tmp_path / "bounded.jsonl",
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
            output_path=tmp_path / "failed.jsonl",
            deidentifier=broken_deidentifier,
        ).execute({})
    assert _SYNTHETIC_VALUE not in str(redaction_error.value)


def test_deidentifier_cannot_inject_a_public_contract_error(tmp_path: Path) -> None:
    def broken_deidentifier(text: str, **kwargs):
        del kwargs
        raise RedactionOperatorError(f"leaked value: {text}")

    with pytest.raises(RedactionOperatorError) as redaction_error:
        OpenMedRedactionOperator(
            records=[_SYNTHETIC_VALUE],
            output_path=tmp_path / "failed-public-error.jsonl",
            deidentifier=broken_deidentifier,
        ).execute({})

    assert _SYNTHETIC_VALUE not in str(redaction_error.value)
    assert str(redaction_error.value).startswith("redaction failed; input_fingerprint=")


def test_result_metadata_failure_is_value_free(tmp_path: Path) -> None:
    class ExplodingResult:
        deidentified_text = "[EMAIL]"

        @property
        def pii_entities(self):
            raise RuntimeError(f"leaked value: {_SYNTHETIC_VALUE}")

    def deidentifier(text: str, **kwargs):
        del text, kwargs
        return ExplodingResult()

    with pytest.raises(RedactionOperatorError) as redaction_error:
        OpenMedRedactionOperator(
            records=[_SYNTHETIC_VALUE],
            output_path=tmp_path / "failed-metadata.jsonl",
            deidentifier=deidentifier,
        ).execute({})

    assert _SYNTHETIC_VALUE not in str(redaction_error.value)


def test_record_iterator_failure_is_value_free(tmp_path: Path) -> None:
    class ExplodingRecords:
        def __iter__(self):
            yield "synthetic note"
            raise RuntimeError(f"leaked value: {_SYNTHETIC_VALUE}")

    with pytest.raises(RedactionOperatorError) as record_error:
        OpenMedRedactionOperator(
            records=ExplodingRecords(),  # type: ignore[arg-type]
            output_path=tmp_path / "failed-records.jsonl",
            deidentifier=_fake_deidentifier,
        ).execute({})

    assert _SYNTHETIC_VALUE not in str(record_error.value)


def test_record_metadata_failure_is_value_free(tmp_path: Path) -> None:
    class ExplodingRecord(dict[str, str]):
        reads = 0

        def __getitem__(self, key: str) -> str:
            self.reads += 1
            if self.reads > 1:
                raise RuntimeError(f"leaked value: {_SYNTHETIC_VALUE}")
            return super().__getitem__(key)

    with pytest.raises(RedactionOperatorError) as record_error:
        OpenMedRedactionOperator(
            records=[ExplodingRecord(text=_SYNTHETIC_VALUE)],
            output_path=tmp_path / "failed-record-metadata.jsonl",
            deidentifier=_fake_deidentifier,
        ).execute({})

    assert _SYNTHETIC_VALUE not in str(record_error.value)


def test_default_deidentifier_is_configured_cache_only(
    monkeypatch,
    tmp_path: Path,
):
    captured: dict[str, object] = {}

    def fake_default(text: str, **kwargs):
        captured.update(kwargs)
        return text

    monkeypatch.setattr(airflow_adapter, "_default_deidentifier", fake_default)
    operator = OpenMedRedactionOperator(
        records=["synthetic note"],
        output_path=tmp_path / "redacted.jsonl",
    )

    operator.execute({})

    config = captured["config"]
    assert getattr(config, "local_only") is True
    assert captured["method"] == "mask"
    assert captured["policy"] == "hipaa_safe_harbor"


def test_file_read_is_bounded_before_loading_the_complete_input(
    tmp_path: Path,
) -> None:
    source = tmp_path / "oversized.txt"
    source.write_bytes(b"x" * 33)

    with pytest.raises(RedactionOperatorError, match="byte bound"):
        OpenMedRedactionOperator(
            input_path=source,
            output_path=tmp_path / "redacted.txt",
            max_input_bytes=32,
            deidentifier=_fake_deidentifier,
        ).execute({})


def test_file_input_rejects_final_symlinks(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    link = tmp_path / "notes-link.txt"
    source.write_text(_SYNTHETIC_VALUE, encoding="utf-8")
    try:
        link.symlink_to(source)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are unavailable")

    with pytest.raises(RedactionOperatorError, match="unable to read input file"):
        OpenMedRedactionOperator(
            input_path=link,
            output_path=tmp_path / "redacted.txt",
            deidentifier=_fake_deidentifier,
        ).execute({})


def test_output_expansion_is_bounded(tmp_path: Path) -> None:
    source = tmp_path / "one-byte.txt"
    source.write_bytes(b"x")

    with pytest.raises(RedactionOperatorError, match="expansion bound"):
        OpenMedRedactionOperator(
            input_path=source,
            output_path=tmp_path / "expanded.txt",
            max_input_bytes=1,
            deidentifier=lambda text, **kwargs: "x" * 9,
        ).execute({})


def test_retry_rejects_an_unbounded_manifest_output_size(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    output = tmp_path / "notes.redacted.txt"
    source.write_text(_SYNTHETIC_VALUE, encoding="utf-8")
    operator = OpenMedRedactionOperator(
        input_path=source,
        output_path=output,
        deidentifier=_fake_deidentifier,
    )
    operator.execute({})
    manifest_path = Path(f"{output}.openmed-fingerprint.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_size"] = operator.max_input_bytes * 9
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RedactionOperatorError, match="invalid counts"):
        operator.execute({})


def test_fingerprint_path_cannot_overwrite_the_input(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text(_SYNTHETIC_VALUE, encoding="utf-8")

    with pytest.raises(ValueError, match="input and fingerprint paths"):
        OpenMedRedactionOperator(
            input_path=source,
            output_path=tmp_path / "redacted.txt",
            fingerprint_path=source,
            deidentifier=_fake_deidentifier,
        )

    assert source.read_text(encoding="utf-8") == _SYNTHETIC_VALUE


def test_get_adapter_loads_airflow_module_without_requiring_airflow_extra():
    adapter = get_adapter("airflow")

    assert adapter is airflow_adapter
    assert adapter.OpenMedRedactionOperator is OpenMedRedactionOperator
