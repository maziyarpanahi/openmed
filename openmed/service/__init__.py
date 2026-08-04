"""OpenMed REST service package.

The REST surface depends on the optional ``service`` extra (FastAPI, uvicorn,
...). Importing this package stays lightweight and never crashes when the extra
is absent: ``app`` and ``create_app`` are resolved lazily, and touching them
without the extra raises a single actionable
:class:`~openmed.core.capabilities.MissingOptionalDependencyError`.

Install with: ``pip install openmed[service]``.
"""

from __future__ import annotations

from typing import Any

from openmed.core.capabilities import (
    is_backend_available,
    require_backend,
)

__all__ = ["app", "create_app", "ensure_service_available", "is_service_available"]


def is_service_available() -> bool:
    """Return True when the ``service`` extra (FastAPI/uvicorn) is importable."""

    return is_backend_available("service")


def ensure_service_available() -> None:
    """Raise an actionable error when the ``service`` extra is not installed."""

    require_backend("service", feature="The OpenMed REST service")


def __getattr__(name: str) -> Any:
    if name in {"app", "create_app"}:
        ensure_service_available()
        from .app import app, create_app

        return {"app": app, "create_app": create_app}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
