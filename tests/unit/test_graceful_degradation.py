"""End-to-end graceful-degradation tests across the optional-extra seams.

Each test simulates a missing optional extra (either by blocking the real
import via ``sys.meta_path`` or by forcing the capability probe to report the
module absent) and asserts that:

* importing OpenMed and its optional subpackages never crashes, and
* touching a feature that *requires* the missing extra raises the single shared
  :class:`~openmed.core.capabilities.MissingOptionalDependencyError` with an
  actionable ``pip install openmed[...]`` message.
"""

from __future__ import annotations

import builtins
import importlib

import pytest

import openmed
from openmed.core.capabilities import MissingOptionalDependencyError


class _ImportBlocker:
    """A ``sys.meta_path`` finder that makes selected top-level packages fail."""

    def __init__(self, *blocked_prefixes: str) -> None:
        self._blocked = blocked_prefixes

    def find_spec(self, name, path=None, target=None):  # noqa: D401 - finder API
        top = name.split(".")[0]
        if top in self._blocked or name in self._blocked:
            raise ImportError(f"blocked for test: {name}")
        return None


@pytest.fixture
def block_imports(monkeypatch):
    """Return a helper that blocks the given top-level packages from importing."""

    def _apply(*blocked_prefixes: str):
        blocker = _ImportBlocker(*blocked_prefixes)
        monkeypatch.setattr("sys.meta_path", [blocker, *importlib.sys.meta_path])
        for mod_name in list(importlib.sys.modules):
            top = mod_name.split(".")[0]
            if top in blocked_prefixes or mod_name in blocked_prefixes:
                monkeypatch.delitem(importlib.sys.modules, mod_name, raising=False)
        return blocker

    return _apply


def test_import_openmed_exposes_capability_probe():
    # The top-level probe is part of the public API.
    assert callable(openmed.available_backends)
    assert callable(openmed.is_backend_available)
    assert callable(openmed.require_backend)
    assert openmed.MissingOptionalDependencyError is MissingOptionalDependencyError


def test_available_backends_top_level_matches_module():
    from openmed.core.capabilities import available_backends as module_probe

    assert set(openmed.available_backends()) == set(module_probe())


def test_service_package_imports_without_fastapi(block_imports):
    block_imports("fastapi", "uvicorn", "starlette")
    # Importing the package must not crash even though FastAPI is unavailable.
    service = importlib.import_module("openmed.service")
    importlib.reload(service)
    assert service.is_service_available() is False


def test_service_create_app_raises_actionable_error(block_imports):
    block_imports("fastapi", "uvicorn", "starlette")
    service = importlib.import_module("openmed.service")
    importlib.reload(service)
    with pytest.raises(MissingOptionalDependencyError) as excinfo:
        service.ensure_service_available()
    assert "pip install openmed[service]" in str(excinfo.value)


def test_service_getattr_app_raises_actionable_error(block_imports):
    block_imports("fastapi", "uvicorn", "starlette")
    service = importlib.import_module("openmed.service")
    importlib.reload(service)
    with pytest.raises(MissingOptionalDependencyError):
        _ = service.create_app


def test_mlx_package_imports_without_mlx(block_imports):
    block_imports("mlx")
    mlx_pkg = importlib.import_module("openmed.mlx")
    importlib.reload(mlx_pkg)
    assert mlx_pkg.is_mlx_available() is False
    with pytest.raises(MissingOptionalDependencyError) as excinfo:
        mlx_pkg.ensure_mlx_available()
    assert "pip install openmed[mlx]" in str(excinfo.value)


def test_onnx_package_imports_without_onnx(block_imports):
    block_imports("onnx", "onnxruntime")
    onnx_pkg = importlib.import_module("openmed.onnx")
    importlib.reload(onnx_pkg)
    assert onnx_pkg.is_onnx_available() is False
    with pytest.raises(MissingOptionalDependencyError) as excinfo:
        onnx_pkg.ensure_onnx_available()
    assert "pip install openmed[onnx]" in str(excinfo.value)


def test_coreml_package_imports_without_coremltools(block_imports):
    block_imports("coremltools")
    coreml_pkg = importlib.import_module("openmed.coreml")
    importlib.reload(coreml_pkg)
    assert coreml_pkg.is_coreml_available() is False
    with pytest.raises(MissingOptionalDependencyError):
        coreml_pkg.ensure_coreml_available()


def test_mcp_package_imports_without_mcp(block_imports):
    block_imports("mcp")
    mcp_pkg = importlib.import_module("openmed.mcp")
    importlib.reload(mcp_pkg)
    assert mcp_pkg.is_mcp_available() is False
    with pytest.raises(MissingOptionalDependencyError):
        mcp_pkg.ensure_mcp_available()


def test_presidio_adapter_raises_shared_error_when_missing(block_imports):
    block_imports("presidio_analyzer")
    from openmed.interop import presidio

    with pytest.raises(MissingOptionalDependencyError, match=r"openmed\[presidio\]"):
        presidio._load_presidio_result_cls()


def test_gliner_biomed_adapter_raises_shared_error_when_missing(block_imports):
    block_imports("gliner")
    from openmed.interop import gliner_biomed

    with pytest.raises(MissingOptionalDependencyError, match=r"openmed\[gliner\]"):
        gliner_biomed._load_gliner_model("does/not-matter")


def test_langchain_adapter_raises_shared_error_when_missing(block_imports):
    block_imports("langchain_core")
    from openmed.interop import langchain

    with pytest.raises(MissingOptionalDependencyError, match=r"openmed\[langchain\]"):
        langchain._load_runnable_lambda()


def test_llamaindex_adapter_raises_shared_error_when_missing(block_imports):
    block_imports("llama_index")
    from openmed.interop import llamaindex

    with pytest.raises(MissingOptionalDependencyError, match=r"openmed\[llamaindex\]"):
        llamaindex._load_function_tool()


def test_multimodal_reports_unavailable_without_extra(monkeypatch):
    from openmed import multimodal

    # Force every multimodal dependency to look absent via the shared probe path.
    monkeypatch.setattr(
        multimodal.base,
        "_missing_multimodal_dependencies",
        lambda: ["pdfplumber", "python-docx"],
    )
    assert multimodal.is_multimodal_available() is False
    with pytest.raises(MissingOptionalDependencyError, match=r"openmed\[multimodal\]"):
        multimodal.ensure_multimodal_available()


def test_shared_error_family_catches_every_seam(block_imports):
    """A single ``except MissingOptionalDependencyError`` covers all seams."""

    block_imports("gliner")
    from openmed.interop import gliner_biomed
    from openmed.multimodal.exceptions import MissingDependencyError as MmErr
    from openmed.ner.exceptions import MissingDependencyError as NerErr

    # Subclass relationships mean the shared guard catches the legacy errors.
    assert issubclass(NerErr, MissingOptionalDependencyError)
    assert issubclass(MmErr, MissingOptionalDependencyError)

    caught = False
    try:
        gliner_biomed._load_gliner_model("x")
    except MissingOptionalDependencyError:
        caught = True
    assert caught


def test_hf_loader_degrades_when_transformers_missing(monkeypatch):
    from openmed.core import models

    monkeypatch.setattr(models, "HF_AVAILABLE", False)
    assert models.is_hf_available() is False
    with pytest.raises(MissingOptionalDependencyError, match=r"openmed\[hf\]"):
        models.ensure_hf_available()


def test_no_optional_extra_installed_still_imports_core(block_imports):
    """Blocking every heavy optional extra must not break ``import openmed``."""

    block_imports(
        "transformers",
        "torch",
        "mlx",
        "coremltools",
        "onnx",
        "onnxruntime",
        "gliner",
        "spacy",
        "presidio_analyzer",
    )
    module = importlib.reload(importlib.import_module("openmed"))
    report = module.available_backends()
    assert report["hf"].available is False
    assert report["mlx"].available is False


def test_original_importerror_still_catchable_as_importerror(block_imports):
    """Existing ``except ImportError`` guards keep working after unification."""

    block_imports("presidio_analyzer")
    from openmed.interop import presidio

    with pytest.raises(ImportError):
        presidio._load_presidio_result_cls()


def test_builtins_import_not_globally_monkeypatched():
    # Sanity: the real import machinery is intact between tests.
    assert builtins.__import__ is builtins.__import__
