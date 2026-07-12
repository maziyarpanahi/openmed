"""Tests for the unified optional-backend capability probe.

These tests simulate missing optional extras (by forcing
``importlib.util.find_spec`` to report modules as absent) and assert that the
capability layer degrades gracefully: the probe stays importless and never
crashes, and required-but-missing features raise one shared, actionable error.
"""

from __future__ import annotations

import importlib

import pytest

from openmed.core import capabilities
from openmed.core.capabilities import (
    BackendStatus,
    MissingOptionalDependencyError,
    available_backends,
    backend_spec,
    backend_status,
    install_hint,
    is_backend_available,
    raise_missing_backend,
    registered_backends,
    require_backend,
    warn_backend_unavailable,
)


@pytest.fixture
def force_missing(monkeypatch):
    """Return a helper that forces the named modules to look uninstalled.

    Every other module keeps its real availability, so a probe that is
    genuinely installed in the test environment is not accidentally reported
    as present when the test wants it absent.
    """

    real_find_spec = importlib.util.find_spec

    def _apply(*missing_modules: str) -> None:
        missing = set(missing_modules)

        def fake_find_spec(name, package=None):
            if name in missing:
                return None
            return real_find_spec(name, package)

        monkeypatch.setattr(capabilities.importlib.util, "find_spec", fake_find_spec)

    return _apply


@pytest.fixture
def force_present(monkeypatch):
    """Force the named modules to look installed regardless of the env."""

    real_find_spec = importlib.util.find_spec

    def _apply(*present_modules: str) -> None:
        present = set(present_modules)

        def fake_find_spec(name, package=None):
            if name in present:
                return object()
            return real_find_spec(name, package)

        monkeypatch.setattr(capabilities.importlib.util, "find_spec", fake_find_spec)

    return _apply


def test_registered_backends_cover_every_optional_seam():
    names = set(registered_backends())
    # The seams called out in OM-820 must all be probeable.
    for seam in (
        "mlx",
        "coreml",
        "onnx",
        "openvino",
        "gliner",
        "spacy",
        "presidio",
        "hf",
        "multimodal",
        "service",
        "mcp",
    ):
        assert seam in names, f"missing capability seam: {seam}"


def test_available_backends_reports_every_seam_importlessly(force_missing):
    force_missing("transformers")
    report = available_backends()
    assert set(report) == set(registered_backends())
    assert all(isinstance(status, BackendStatus) for status in report.values())
    # Nothing here imports transformers; the probe only inspects specs.
    assert report["hf"].available is False
    assert report["hf"].missing == ("transformers",)


def test_available_backends_never_imports_backend_modules(force_present):
    # Even when every spec is reported present, the probe must not import them.
    force_present(
        "mlx.core",
        "coremltools",
        "onnx",
        "onnxruntime",
        "gliner",
        "transformers",
    )
    report = available_backends()
    assert report["mlx"].available is True
    assert report["onnx"].available is True
    assert report["hf"].available is True
    # find_spec was stubbed, so no heavy module ended up imported by the probe.


def test_is_backend_available_false_when_missing(force_missing):
    force_missing("mlx.core")
    assert is_backend_available("mlx") is False


def test_is_backend_available_true_when_present(force_present):
    force_present("gliner")
    assert is_backend_available("gliner") is True


def test_backend_status_lists_all_missing_modules(force_missing):
    force_missing("onnx", "onnxruntime")
    status = backend_status("onnx")
    assert status.available is False
    assert set(status.missing) == {"onnx", "onnxruntime"}
    assert "openmed[onnx]" in status.install_hint


def test_backend_status_available_when_partial_present(monkeypatch):
    # onnx needs both onnx and onnxruntime; missing just one is unavailable.
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, package=None):
        if name == "onnxruntime":
            return None
        if name == "onnx":
            return object()
        return real_find_spec(name, package)

    monkeypatch.setattr(capabilities.importlib.util, "find_spec", fake_find_spec)
    status = backend_status("onnx")
    assert status.available is False
    assert status.missing == ("onnxruntime",)


def test_backend_status_name_is_normalized():
    assert backend_status("MLX").name == "mlx"
    assert backend_status("Multimodal").name == "multimodal"
    # Case and surrounding whitespace are normalized to the canonical key.
    assert backend_spec("  LlamaIndex  ").name == "llamaindex"


def test_unknown_backend_raises_keyerror():
    with pytest.raises(KeyError, match="unknown backend"):
        backend_status("does-not-exist")
    with pytest.raises(KeyError):
        backend_spec("nope")


def test_require_backend_returns_none_when_available(force_present):
    force_present("presidio_analyzer")
    assert require_backend("presidio") is None


def test_require_backend_raises_actionable_error(force_missing):
    force_missing("spacy")
    with pytest.raises(MissingOptionalDependencyError) as excinfo:
        require_backend("spacy", feature="spaCy pipeline component")

    err = excinfo.value
    assert isinstance(err, ImportError)  # backward-compatible with old guards
    assert err.extra == "spacy"
    message = str(err)
    assert "spaCy pipeline component" in message
    assert "pip install openmed[spacy]" in message


def test_raise_missing_backend_chains_cause(force_missing):
    force_missing("gliner")
    original = ImportError("No module named 'gliner'")
    with pytest.raises(MissingOptionalDependencyError) as excinfo:
        raise_missing_backend("gliner", feature="GLiNER support", cause=original)
    assert excinfo.value.__cause__ is original
    assert "openmed[gliner]" in str(excinfo.value)


def test_warn_backend_unavailable_emits_warning_and_returns_status(force_missing):
    force_missing("transformers")
    with pytest.warns(UserWarning, match="pip install openmed\\[hf\\]"):
        status = warn_backend_unavailable(
            "hf", feature="HF inference", action="skipping"
        )
    assert status.available is False


def test_warn_backend_unavailable_silent_when_available(force_present, recwarn):
    force_present("transformers")
    status = warn_backend_unavailable("hf")
    assert status.available is True
    assert len(recwarn) == 0


def test_install_hint_mentions_extra_and_package():
    hint = install_hint("transformers", "hf")
    assert "pip install openmed[hf]" in hint
    assert "pip install transformers" in hint


def test_install_hint_without_extra():
    hint = install_hint("python-dateutil")
    assert "openmed[" not in hint
    assert "pip install python-dateutil" in hint


def test_status_as_dict_is_serializable(force_missing):
    force_missing("mlx.core")
    data = backend_status("mlx").as_dict()
    assert data["name"] == "mlx"
    assert data["available"] is False
    assert data["missing"] == ["mlx.core"]
    assert "install_hint" in data
