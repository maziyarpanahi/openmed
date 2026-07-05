"""Exceptions shared across OpenMed zero-shot NER modules.

This module re-exports the structured error taxonomy defined in
:mod:`openmed.core.errors` so the whole hierarchy is reachable from the NER
surface, and defines :class:`MissingDependencyError` as a taxonomy member.
"""

from __future__ import annotations

from ..core.errors import (
    BudgetExceededError,
    CapabilityError,
    ConfigurationError,
    InferenceError,
    InputError,
    InternalError,
    MissingExtraError,
    ModelLoadError,
    OpenMedError,
    PolicyError,
    redact_detail,
)

__all__ = [
    "OpenMedError",
    "InputError",
    "ConfigurationError",
    "CapabilityError",
    "MissingExtraError",
    "ModelLoadError",
    "PolicyError",
    "BudgetExceededError",
    "InternalError",
    "InferenceError",
    "MissingDependencyError",
    "redact_detail",
]


class MissingDependencyError(MissingExtraError):
    """Raised when an optional dependency is required but unavailable.

    Part of the structured error taxonomy: subclasses
    :class:`openmed.core.errors.MissingExtraError` (hence ``OpenMedError`` and
    ``ImportError``) so both ``except OpenMedError`` and legacy ``except
    ImportError`` handlers catch it. The two-positional-argument signature is
    preserved for backwards compatibility.

    Args:
        dependency: The missing distribution name.
        instruction: Actionable install remediation to append to the message.
    """

    def __init__(self, dependency: str, instruction: str) -> None:
        message = (
            f"Optional dependency '{dependency}' is required for this operation. "
            f"{instruction}"
        )
        super().__init__(message, package=dependency)
        self.dependency = dependency
        self.instruction = instruction
