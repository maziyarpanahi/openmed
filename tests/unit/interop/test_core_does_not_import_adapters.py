from __future__ import annotations

import sys

import pytest

OPTIONAL_ADAPTER_MODULE_PREFIXES = (
    "presidio",
    "philter_ucsf",
    "pyDeid",
    "pydeid",
    "gliner",
)


def _clear_optional_adapter_modules() -> None:
    for name in list(sys.modules):
        if _is_optional_adapter_module(name):
            sys.modules.pop(name, None)


def _is_optional_adapter_module(name: str) -> bool:
    return any(
        name == prefix or name.startswith(f"{prefix}.")
        for prefix in OPTIONAL_ADAPTER_MODULE_PREFIXES
    )


def test_import_openmed_does_not_import_optional_adapter_dependencies():
    _clear_optional_adapter_modules()

    import openmed  # noqa: F401

    assert not any(_is_optional_adapter_module(name) for name in sys.modules)


def test_import_interop_registry_does_not_import_optional_adapter_dependencies():
    _clear_optional_adapter_modules()

    from openmed.interop import adapter_spec, available_adapters

    assert available_adapters() == (
        "gliner_biomed",
        "philter",
        "presidio",
        "pydeid",
    )
    assert adapter_spec("presidio").extra == "presidio"
    assert adapter_spec("philter").extra == "philter"
    assert adapter_spec("pydeid").extra == "pydeid"
    assert adapter_spec("gliner_biomed").extra == "gliner"
    assert not any(_is_optional_adapter_module(name) for name in sys.modules)


def test_presidio_adapter_missing_extra_raises_clear_import_error(monkeypatch):
    from openmed.interop import presidio

    def missing_dependency(name: str):
        raise ImportError(name)

    monkeypatch.setattr(presidio, "_import_module", missing_dependency)

    with pytest.raises(ImportError, match=r"openmed\[presidio\]"):
        presidio.from_canonical([])
