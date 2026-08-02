"""Exceptions for the multimodal ingest/redact subsystem.

Kept dependency-free so importing it never drags in heavy ingestion packages.
"""

from __future__ import annotations


class MissingDependencyError(ImportError):
    """Raised when the multimodal extra is required but not installed."""

    def __init__(self, dependency: str, instruction: str) -> None:
        message = (
            f"Optional dependency '{dependency}' is required for this operation. "
            f"{instruction}"
        )
        super().__init__(message)
        self.dependency = dependency
        self.instruction = instruction


class UnsupportedDocumentError(ValueError):
    """Raised when no handler is registered for a document's file type."""


class DocumentGraphError(ValueError):
    """Raised when a document cannot be converted into a safe graph."""


class MalformedDocumentError(DocumentGraphError):
    """Raised when a document is truncated, invalid, or structurally unsafe."""


class EncryptedDocumentError(DocumentGraphError):
    """Raised when a document requires a password or encrypted content access."""
