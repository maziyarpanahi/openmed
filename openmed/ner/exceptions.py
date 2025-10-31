"""Exceptions shared across OpenMed zero-shot NER modules."""

from __future__ import annotations


class MissingDependencyError(ImportError):
    """Raised when an optional dependency is required but unavailable."""

    def __init__(self, dependency: str, instruction: str) -> None:
        message = (
            f"Optional dependency '{dependency}' is required for this operation. "
            f"{instruction}"
        )
        super().__init__(message)
        self.dependency = dependency
        self.instruction = instruction
