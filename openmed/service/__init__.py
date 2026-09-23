"""OpenMed REST service package.

The REST surface depends on the optional ``service`` extra (FastAPI, uvicorn,
...). Importing this package stays lightweight and never crashes when the extra
is absent: ``app`` and ``create_app`` are resolved lazily, and touching them
without the extra raises a single actionable
:class:`~openmed.core.capabilities.MissingOptionalDependencyError`.

Install with: ``pip install openmed[service]``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from openmed.core.capabilities import (
    is_backend_available,
    require_backend,
)

__all__ = [
    "JourneyAccessPolicy",
    "JourneyResourceCatalog",
    "JourneyResourceKind",
    "JourneyResourcePage",
    "JourneyResourceQuery",
    "JourneyResourceRecord",
    "JourneyResourceState",
    "OperationalAlert",
    "OperationalCategory",
    "OperationalEvent",
    "OperationalState",
    "app",
    "create_app",
    "ensure_service_available",
    "is_service_available",
]

_LAZY_IMPORTS = {
    "JourneyAccessPolicy": ".journey_resources",
    "JourneyResourceCatalog": ".journey_resources",
    "JourneyResourceKind": ".journey_resources",
    "JourneyResourcePage": ".journey_resources",
    "JourneyResourceQuery": ".journey_resources",
    "JourneyResourceRecord": ".journey_resources",
    "JourneyResourceState": ".journey_resources",
    "OperationalAlert": ".operational",
    "OperationalCategory": ".operational",
    "OperationalEvent": ".operational",
    "OperationalState": ".operational",
    "app": ".app",
    "create_app": ".app",
}


def is_service_available() -> bool:
    """Return True when the ``service`` extra (FastAPI/uvicorn) is importable."""

    return is_backend_available("service")


def ensure_service_available() -> None:
    """Raise an actionable error when the ``service`` extra is not installed."""

    require_backend("service", feature="The OpenMed REST service")


def __getattr__(name: str) -> Any:
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    ensure_service_available()
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
