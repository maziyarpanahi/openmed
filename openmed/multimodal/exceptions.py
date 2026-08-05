"""Exceptions for the multimodal ingest/redact subsystem.

Kept dependency-free so importing it never drags in heavy ingestion packages.
The shared :class:`~openmed.core.capabilities.MissingOptionalDependencyError`
it builds on is standard-library only, so this stays lightweight.
"""

from __future__ import annotations

from openmed.core.capabilities import MissingOptionalDependencyError


class MissingDependencyError(MissingOptionalDependencyError):
    """Raised when the multimodal extra is required but not installed.

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


class UnsupportedDocumentError(ValueError):
    """Raised when no handler is registered for a document's file type."""
