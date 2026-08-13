from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as _toml  # type: ignore[no-redef]

from openmed.interop import (
    adapter_spec,
    available_adapters,
    beam_transform,
    get_adapter,
)
from openmed.interop.beam_transform import DeidentifyText, _DeidentifyTextDoFn

ROOT = Path(__file__).resolve().parents[3]


def fake_deidentifier(text: str, **kwargs):
    assert kwargs["policy"] == "hipaa_safe_harbor"
    assert kwargs["loader"] is not None
    redacted = text.replace("Jane Roe", "[PERSON]").replace("555-0100", "[PHONE]")
    return SimpleNamespace(deidentified_text=redacted)


def test_process_redacts_strings_and_dictionary_text_fields():
    dofn = _DeidentifyTextDoFn(
        text_field="note",
        deidentifier=fake_deidentifier,
        loader_factory=object,
    )
    dofn.setup()
    record = {"id": 7, "note": "Jane Roe called 555-0100"}

    redacted_string = list(dofn.process("Jane Roe called 555-0100"))
    redacted_record = list(dofn.process(record))

    assert redacted_string == ["[PERSON] called [PHONE]"]
    assert redacted_record == [{"id": 7, "note": "[PERSON] called [PHONE]"}]
    assert record == {"id": 7, "note": "Jane Roe called 555-0100"}


def test_setup_loads_model_once_across_multiple_process_calls():
    load_count = 0
    loader = object()
    captured_loaders: list[object] = []

    def loader_factory():
        nonlocal load_count
        load_count += 1
        return loader

    def deidentifier(text: str, **kwargs):
        del text
        captured_loaders.append(kwargs["loader"])
        return "[PERSON]"

    dofn = _DeidentifyTextDoFn(
        deidentifier=deidentifier,
        loader_factory=loader_factory,
    )

    dofn.setup()
    list(dofn.process("Jane Roe"))
    list(dofn.process("John Doe"))
    dofn.setup()

    assert load_count == 1
    assert captured_loaders == [loader, loader]


def test_registry_exposes_beam_without_requiring_dependency():
    adapter = get_adapter("beam")

    assert adapter is beam_transform
    assert "beam" in available_adapters()
    assert adapter_spec("beam").extra == "beam"
    assert adapter.DeidentifyText is DeidentifyText


def test_beam_extra_declares_apache_beam_dependency():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        dependencies = _toml.load(handle)["project"]["optional-dependencies"]["beam"]

    assert any(requirement.startswith("apache-beam") for requirement in dependencies)


def test_beam_resolution_uses_patched_security_dependencies():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        overrides = _toml.load(handle)["tool"]["uv"]["override-dependencies"]

    assert any(requirement.startswith("cryptography>=50") for requirement in overrides)
    assert any(requirement.startswith("httplib2>=0.32") for requirement in overrides)


def test_expand_raises_clear_error_without_apache_beam(monkeypatch):
    monkeypatch.setattr(beam_transform, "_beam", None)

    with pytest.raises(ImportError, match=r"openmed\[beam\]"):
        DeidentifyText().expand(object())


def test_direct_runner_redacts_string_elements_when_beam_is_available(monkeypatch):
    beam = pytest.importorskip("apache_beam")
    from apache_beam.testing.test_pipeline import TestPipeline
    from apache_beam.testing.util import assert_that, equal_to

    monkeypatch.setattr(beam_transform, "_new_model_loader", object)
    monkeypatch.setattr(
        beam_transform,
        "_default_deidentifier",
        lambda: fake_deidentifier,
    )

    with TestPipeline() as pipeline:
        output = pipeline | beam.Create(["Jane Roe called 555-0100"]) | DeidentifyText()
        assert_that(output, equal_to(["[PERSON] called [PHONE]"]))
