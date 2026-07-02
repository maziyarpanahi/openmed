"""Linker registry so ``ground(systems=[...])`` (a later task) can dispatch.

Linkers register themselves under a vocabulary system key (e.g. ``"rxnorm"``).
The registry stores a callable ``(vocab) -> linker`` factory so callers can
build a linker with their own vocabulary data.
"""

from __future__ import annotations

from typing import Any, Callable

LinkerFactory = Callable[..., Any]

_LINKERS: dict[str, LinkerFactory] = {}


def register_linker(system: str, factory: LinkerFactory) -> None:
    """Register a linker factory under a vocabulary ``system`` key."""
    _LINKERS[system] = factory


def get_linker(system: str) -> LinkerFactory:
    """Return the linker factory registered for ``system`` (raises KeyError)."""
    return _LINKERS[system]


def available_linkers() -> list[str]:
    """Return the sorted list of registered linker system keys."""
    return sorted(_LINKERS)
