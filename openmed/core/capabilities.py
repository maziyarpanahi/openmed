"""Unified capability probe for OpenMed's optional backends and extras.

OpenMed ships many optional integrations behind ``pip install openmed[...]``
extras (``mlx``, ``coreml``, ``onnx``, ``gliner``, ``spacy``, ``presidio``,
``hf``, ``multimodal``, ``service``, ``mcp``, ...). Historically each seam
guarded its optional imports differently: some raised a bare :class:`ImportError`
with an ad-hoc message, others raised one of several ``MissingDependencyError``
variants, and there was no single, importless way to probe which backends are
available before use.

This module is that single source of truth. It is deliberately dependency-free
(standard library only) so importing it never drags in heavy optional packages,
and every probe uses :func:`importlib.util.find_spec` so checking availability
never imports the backend either.

Two complementary contracts are provided:

* A **capability probe** -- :func:`available_backends`, :func:`backend_status`,
  and :func:`is_backend_available` -- for code that wants to *branch* on whether
  an optional seam is installed and degrade gracefully (skip-with-warning or
  fall back) when it is not.
* An **actionable requirement** -- :func:`require_backend` raising
  :class:`MissingOptionalDependencyError` -- for code paths where a feature
  genuinely *requires* a missing extra and should fail with one consistent
  message naming the exact ``pip install`` command.

Example:
    >>> from openmed import available_backends
    >>> report = available_backends()
    >>> report["hf"].available  # doctest: +SKIP
    False
    >>> report["hf"].install_hint  # doctest: +SKIP
    "Install with `pip install openmed[hf]` or `pip install transformers`."
"""

from __future__ import annotations

import importlib.util
import warnings
from dataclasses import dataclass
from typing import Final, NoReturn


class MissingOptionalDependencyError(ImportError):
    """Raised when a requested optional capability needs an unavailable package.

    This is the canonical, shared "missing extra" error for the whole package.
    It subclasses :class:`ImportError` so existing ``except ImportError`` guards
    keep working, and it carries structured ``feature``/``package``/``extra``
    fields plus an actionable install message.
    """

    def __init__(
        self,
        *,
        package: str,
        feature: str,
        extra: str | None = None,
    ) -> None:
        instruction = install_hint(package, extra)
        super().__init__(
            f"{feature} requires optional dependency '{package}'. {instruction}"
        )
        self.package = package
        self.feature = feature
        self.extra = extra


@dataclass(frozen=True)
class BackendSpec:
    """Registry metadata for one optional backend/extra seam.

    Attributes:
        name: Stable capability key (e.g. ``"mlx"``, ``"hf"``).
        extra: The ``pip install openmed[<extra>]`` extra that provides it.
        modules: Import module names that must all be importable for the
            capability to be considered available.
        description: Human-readable description of what the backend enables.
        install: Optional distribution name to suggest with a bare
            ``pip install`` when the extra alias is not enough (defaults to the
            first entry of ``modules``).
    """

    name: str
    extra: str
    modules: tuple[str, ...]
    description: str
    install: str | None = None

    @property
    def primary_package(self) -> str:
        """Return the package name to name in a bare ``pip install`` hint."""

        return self.install or self.modules[0]


# Canonical extra names mirror ``[project.optional-dependencies]`` in
# ``pyproject.toml``. Keep this registry in sync when adding an optional extra.
_BACKENDS: Final[dict[str, BackendSpec]] = {
    "hf": BackendSpec(
        name="hf",
        extra="hf",
        modules=("transformers",),
        description="HuggingFace Transformers token-classification inference",
        install="transformers",
    ),
    "mlx": BackendSpec(
        name="mlx",
        extra="mlx",
        modules=("mlx.core",),
        description="Apple MLX hardware-accelerated inference (Apple Silicon)",
        install="mlx",
    ),
    "coreml": BackendSpec(
        name="coreml",
        extra="coreml",
        modules=("coremltools",),
        description="CoreML export for iOS/macOS deployment",
        install="coremltools",
    ),
    "onnx": BackendSpec(
        name="onnx",
        extra="onnx",
        modules=("onnx", "onnxruntime"),
        description="ONNX export and ONNX Runtime inference",
        install="onnxruntime",
    ),
    "openvino": BackendSpec(
        name="openvino",
        extra="openvino",
        modules=("openvino",),
        description="OpenVINO IR export and inference",
        install="openvino",
    ),
    "gliner": BackendSpec(
        name="gliner",
        extra="gliner",
        modules=("gliner",),
        description="GLiNER zero-shot entity recognition",
        install="gliner",
    ),
    "spacy": BackendSpec(
        name="spacy",
        extra="spacy",
        modules=("spacy",),
        description="spaCy pipeline component for OpenMed PII spans",
        install="spacy",
    ),
    "presidio": BackendSpec(
        name="presidio",
        extra="presidio",
        modules=("presidio_analyzer",),
        description="Presidio RecognizerResult interoperability",
        install="presidio-analyzer",
    ),
    "philter": BackendSpec(
        name="philter",
        extra="philter",
        modules=("philter",),
        description="Philter PHI span interoperability",
        install="philter-ucsf",
    ),
    "pydeid": BackendSpec(
        name="pydeid",
        extra="pydeid",
        modules=("pyDeid",),
        description="pyDeid PHI span interoperability",
        install="pyDeid",
    ),
    "multimodal": BackendSpec(
        name="multimodal",
        extra="multimodal",
        modules=("pdfplumber", "docx", "PIL"),
        description="Multimodal document/image ingestion and redaction",
        install="pdfplumber",
    ),
    "service": BackendSpec(
        name="service",
        extra="service",
        modules=("fastapi", "uvicorn"),
        description="FastAPI REST service surface",
        install="fastapi",
    ),
    "mcp": BackendSpec(
        name="mcp",
        extra="mcp",
        modules=("mcp",),
        description="Model Context Protocol server integration",
        install="mcp",
    ),
    "langchain": BackendSpec(
        name="langchain",
        extra="langchain",
        modules=("langchain_core",),
        description="LangChain redaction runnable adapter",
        install="langchain-core",
    ),
    "llamaindex": BackendSpec(
        name="llamaindex",
        extra="llamaindex",
        modules=("llama_index.core",),
        description="LlamaIndex FunctionTool adapter",
        install="llama-index-core",
    ),
}


@dataclass(frozen=True)
class BackendStatus:
    """Importless availability report for one optional backend/extra seam.

    Attributes:
        name: The capability key.
        available: True when every required module is importable.
        extra: The ``openmed[<extra>]`` extra that provides the backend.
        description: Human-readable description of the backend.
        missing: Module names that are not importable (empty when available).
        install_hint: Actionable ``pip install`` instruction.
    """

    name: str
    available: bool
    extra: str
    description: str
    missing: tuple[str, ...]
    install_hint: str

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping of this status."""

        return {
            "name": self.name,
            "available": self.available,
            "extra": self.extra,
            "description": self.description,
            "missing": list(self.missing),
            "install_hint": self.install_hint,
        }


def install_hint(package: str, extra: str | None = None) -> str:
    """Return an actionable install instruction for a missing optional package.

    Args:
        package: The distribution name to suggest with a bare ``pip install``.
        extra: The ``openmed[<extra>]`` extra that also provides it, if any.

    Returns:
        A short sentence naming the exact command(s) to install the dependency.
    """

    if extra:
        return (
            f"Install with `pip install openmed[{extra}]` or `pip install {package}`."
        )
    return f"Install with `pip install {package}`."


def registered_backends() -> tuple[str, ...]:
    """Return every registered capability key, sorted, without importing them."""

    return tuple(sorted(_BACKENDS))


def backend_spec(name: str) -> BackendSpec:
    """Return the :class:`BackendSpec` for *name* without importing it.

    Raises:
        KeyError: If *name* is not a registered capability.
    """

    key = _normalize(name)
    try:
        return _BACKENDS[key]
    except KeyError as exc:
        known = ", ".join(registered_backends())
        raise KeyError(f"unknown backend {name!r}; available: {known}") from exc


def _missing_modules(spec: BackendSpec) -> tuple[str, ...]:
    """Return required module names that are not importable, without importing."""

    missing: list[str] = []
    for module in spec.modules:
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            # A parent package that itself fails to import, or a submodule of a
            # namespace that cannot be resolved, is treated as unavailable.
            found = False
        if not found:
            missing.append(module)
    return tuple(missing)


def backend_status(name: str) -> BackendStatus:
    """Return an importless :class:`BackendStatus` for a single backend.

    Args:
        name: A registered capability key (case-insensitive; ``-`` and ``_``
            are treated the same).

    Returns:
        The availability report, computed via :func:`importlib.util.find_spec`
        so the backend itself is never imported.

    Raises:
        KeyError: If *name* is not a registered capability.
    """

    spec = backend_spec(name)
    missing = _missing_modules(spec)
    return BackendStatus(
        name=spec.name,
        available=not missing,
        extra=spec.extra,
        description=spec.description,
        missing=missing,
        install_hint=install_hint(spec.primary_package, spec.extra),
    )


def is_backend_available(name: str) -> bool:
    """Return True when every module for *name* is importable, importlessly.

    Args:
        name: A registered capability key.

    Returns:
        True if the backend can be used, False otherwise. Never raises for an
        unavailable backend; only raises :class:`KeyError` for an unknown key.
    """

    return backend_status(name).available


def available_backends() -> dict[str, BackendStatus]:
    """Return an importless availability report for every optional backend.

    This is the top-level capability probe. It imports none of the optional
    dependencies (each is checked with :func:`importlib.util.find_spec`), so it
    is safe to call on a minimal, local-first install to decide which features
    to enable.

    Returns:
        A mapping of capability key to its :class:`BackendStatus`, ordered by
        capability key.
    """

    return {name: backend_status(name) for name in registered_backends()}


def require_backend(name: str, *, feature: str | None = None) -> None:
    """Ensure a required backend is importable, or raise an actionable error.

    Use this on code paths where the feature genuinely *cannot* proceed without
    the optional extra. For paths that can degrade gracefully, prefer
    :func:`is_backend_available` (or :func:`warn_backend_unavailable`) instead.

    Args:
        name: A registered capability key.
        feature: Human-readable description of the feature being attempted;
            defaults to the backend's registry description.

    Raises:
        MissingOptionalDependencyError: If any required module is missing.
        KeyError: If *name* is not a registered capability.
    """

    status = backend_status(name)
    if status.available:
        return
    spec = backend_spec(name)
    raise MissingOptionalDependencyError(
        package=", ".join(status.missing) or spec.primary_package,
        feature=feature or spec.description,
        extra=spec.extra,
    )


def raise_missing_backend(
    name: str,
    *,
    feature: str | None = None,
    cause: BaseException | None = None,
) -> NoReturn:
    """Unconditionally raise the shared missing-extra error for *name*.

    Helper for ``except ImportError`` blocks that have already discovered the
    backend is unavailable and want to re-raise a consistent, actionable error
    chained from the original import failure.

    Args:
        name: A registered capability key.
        feature: Human-readable description of the feature being attempted.
        cause: The original :class:`ImportError`, chained via ``from``.

    Raises:
        MissingOptionalDependencyError: Always.
    """

    spec = backend_spec(name)
    status = backend_status(name)
    error = MissingOptionalDependencyError(
        package=", ".join(status.missing) or spec.primary_package,
        feature=feature or spec.description,
        extra=spec.extra,
    )
    raise error from cause


def warn_backend_unavailable(
    name: str,
    *,
    feature: str | None = None,
    action: str = "skipping",
) -> BackendStatus:
    """Emit a clear one-line warning that an optional backend is unavailable.

    For graceful-degradation paths: call this to inform the user that a feature
    is being skipped or falling back because the extra is not installed, then
    continue. Returns the status so callers can branch further.

    Args:
        name: A registered capability key.
        feature: Human-readable description of the affected feature; defaults to
            the backend's registry description.
        action: What the caller is doing instead (e.g. ``"skipping"`` or
            ``"falling back"``).

    Returns:
        The :class:`BackendStatus` for *name*.
    """

    status = backend_status(name)
    if not status.available:
        warnings.warn(
            f"OpenMed: {feature or status.description} is unavailable "
            f"({action}) because optional dependency "
            f"{', '.join(status.missing) or status.name} is not installed. "
            f"{status.install_hint}",
            UserWarning,
            stacklevel=2,
        )
    return status


def _normalize(name: str) -> str:
    return str(name or "").strip().lower().replace("-", "_")


__all__ = [
    "BackendSpec",
    "BackendStatus",
    "MissingOptionalDependencyError",
    "available_backends",
    "backend_spec",
    "backend_status",
    "install_hint",
    "is_backend_available",
    "raise_missing_backend",
    "registered_backends",
    "require_backend",
    "warn_backend_unavailable",
]
