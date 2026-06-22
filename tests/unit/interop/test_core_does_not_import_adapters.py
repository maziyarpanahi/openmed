from __future__ import annotations

import sys

import pytest


def _clear_presidio_modules() -> None:
    for name in list(sys.modules):
        if name.startswith("presidio"):
            sys.modules.pop(name, None)


def test_import_openmed_does_not_import_presidio():
    _clear_presidio_modules()

    import openmed  # noqa: F401

    assert not any(name.startswith("presidio") for name in sys.modules)


def test_import_interop_registry_does_not_import_presidio():
    _clear_presidio_modules()

    from openmed.interop import adapter_spec, available_adapters

    assert "presidio" in available_adapters()
    assert adapter_spec("presidio").extra == "presidio"
    assert not any(name.startswith("presidio") for name in sys.modules)


def test_presidio_adapter_missing_extra_raises_clear_import_error(monkeypatch):
    from openmed.interop import presidio

    def missing_dependency(name: str):
        raise ImportError(name)

    monkeypatch.setattr(presidio, "_import_module", missing_dependency)

    with pytest.raises(ImportError, match=r"openmed\[presidio\]"):
        presidio.from_canonical([])
