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
        super().__init__(
            package=dependency,
            feature="This operation",
        )
        # Preserve the historical sentence exactly while retaining the shared
        # taxonomy fields initialized by the parent.
        self.args = (message,)
        self.message = message
        self.dependency = dependency
        self.instruction = instruction
        # Keep the v2.1 constructor-owned attributes visible on this concrete
        # compatibility class as well as initialized by the shared parent.
        self.package = dependency
        self.feature = "This operation"
        self.extra = None


class UnsupportedDocumentError(ValueError):
    """Raised when no handler is registered for a document's file type."""


class DocumentGraphError(ValueError):
    """Raised when a document cannot be converted into a safe graph."""


class MalformedDocumentError(DocumentGraphError):
    """Raised when a document is truncated, invalid, or structurally unsafe."""


class EncryptedDocumentError(DocumentGraphError):
    """Raised when a document requires a password or encrypted content access."""
