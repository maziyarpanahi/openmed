"""Exceptions shared across OpenMed zero-shot NER modules."""

from __future__ import annotations

from openmed.core.capabilities import MissingOptionalDependencyError


class MissingDependencyError(MissingOptionalDependencyError):
    """Raised when an optional NER dependency is required but unavailable.

    Subclasses the shared :class:`MissingOptionalDependencyError` so a single
    ``except MissingOptionalDependencyError`` guard catches every optional-extra
    failure across OpenMed, while keeping the historical
    ``(dependency, instruction)`` constructor and message for callers that rely
    on it.
    """

    def __init__(self, dependency: str, instruction: str) -> None:
        message = (
            f"Optional dependency '{dependency}' is required for this operation. "
            f"{instruction}"
        )
        # Bypass the parent's keyword-only constructor to preserve the legacy
        # message/attributes while remaining part of the shared error family.
        ImportError.__init__(self, message)
        self.dependency = dependency
        self.instruction = instruction
        self.package = dependency
        self.feature = "This operation"
        self.extra = None
