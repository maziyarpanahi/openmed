"""Exceptions shared across OpenMed zero-shot NER modules.

The public taxonomy is re-exported here for callers that historically imported
NER exceptions from this module.
"""

from __future__ import annotations

from openmed.core.capabilities import MissingOptionalDependencyError
from openmed.core.errors import (
    ERROR_CODES,
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
    "ERROR_CODES",
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
        super().__init__(
            package=dependency,
            feature="This operation",
        )
        # Preserve the historical sentence exactly while retaining the shared
        # structured fields initialized by the taxonomy base.
        self.args = (message,)
        self.message = message
        self.dependency = dependency
        self.instruction = instruction
