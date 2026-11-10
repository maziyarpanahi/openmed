"""Canonical developer-authored identifiers for agent governance metadata.

Governance identifiers are syntax-checked names. They are not derived from
patient, clinician, tenant, device, or clinical content, and successful
parsing cannot establish provenance, ownership, or authorization.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar, Final, TypeVar, cast

_MAX_IDENTIFIER_LENGTH: Final = 512
_MAX_NAMESPACE_LENGTH: Final = 253

_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_NAMESPACE = rf"{_LABEL}(?:\.{_LABEL})+"
_LOCAL_NAME = r"[a-z][a-z0-9-]{0,63}"
_NUMBER = r"(?:0|[1-9][0-9]*)"
_VERSION = rf"{_NUMBER}\.{_NUMBER}\.{_NUMBER}"

# The kind is intentionally parsed as lowercase ASCII before the closed
# vocabulary check so that unknown kinds and wrong typed kinds remain distinct
# diagnostics without weakening the accepted identifier grammar.
_IDENTIFIER_RE = re.compile(
    rf"(?P<kind>[a-z]+):"
    rf"(?P<namespace>{_NAMESPACE})/"
    rf"(?P<local_name>{_LOCAL_NAME})"
    rf"(?:@(?P<version>{_VERSION}))?"
)
_ALLOWED_KINDS: Final = frozenset(
    {"capability", "purpose", "policy", "workflow", "tool"}
)

_GovernanceIdT = TypeVar("_GovernanceIdT", bound="_GovernanceId")


class GovernanceIdError(ValueError):
    """Raised when a governance identifier fails closed validation.

    Args:
        code: Stable machine-readable validation code.
        field_name: Optional fixed public field associated with the failure.

    Error messages and attributes contain only controlled diagnostic metadata;
    they never retain or echo the rejected identifier.
    """

    def __init__(self, code: str, field_name: str | None = None) -> None:
        self.code = code
        self.field_name = field_name
        message = code if field_name is None else f"{field_name}: {code}"
        super().__init__(message)


@dataclass(frozen=True, slots=True, repr=False)
class _GovernanceId:
    """Shared immutable implementation for typed governance identifiers."""

    value: str
    _expected_kind: ClassVar[str] = ""

    def __post_init__(self) -> None:
        _validate_identifier(self.value, expected_kind=self._expected_kind)

    @classmethod
    def parse(cls: type[_GovernanceIdT], value: object) -> _GovernanceIdT:
        """Parse a canonical identifier of this typed kind."""

        return cls(cast(str, value))

    @property
    def kind(self) -> str:
        """Return the canonical kind component."""

        return _components(self.value)["kind"]

    @property
    def namespace(self) -> str:
        """Return the canonical reverse-domain namespace component."""

        return _components(self.value)["namespace"]

    @property
    def local_name(self) -> str:
        """Return the canonical local-name component."""

        return _components(self.value)["local_name"]

    @property
    def version(self) -> str | None:
        """Return the optional canonical version, or ``None`` when absent."""

        return _components(self.value)["version"]

    def serialize(self) -> str:
        """Return the original canonical identifier unchanged."""

        return self.value

    def __str__(self) -> str:
        """Return the canonical identifier for explicit string conversion."""

        return self.value

    def __repr__(self) -> str:
        """Return a value-free representation for diagnostic output."""

        return f"{type(self).__name__}(<redacted>)"


class CapabilityId(_GovernanceId):
    """Typed governance identifier for an agent capability."""

    __slots__ = ()
    _expected_kind = "capability"


class PurposeId(_GovernanceId):
    """Typed governance identifier for an agent purpose."""

    __slots__ = ()
    _expected_kind = "purpose"


class PolicyId(_GovernanceId):
    """Typed governance identifier for an agent policy."""

    __slots__ = ()
    _expected_kind = "policy"


class WorkflowId(_GovernanceId):
    """Typed governance identifier for an agent workflow."""

    __slots__ = ()
    _expected_kind = "workflow"


class ToolId(_GovernanceId):
    """Typed governance identifier for an agent tool."""

    __slots__ = ()
    _expected_kind = "tool"


def _components(value: str) -> re.Match[str]:
    match = _IDENTIFIER_RE.fullmatch(value)
    if match is None:
        # Every public instance has already passed this validation. Keep this
        # guard value-free in case a caller deliberately bypasses invariants.
        raise GovernanceIdError("invalid_identifier", "identifier")
    return match


def _validate_identifier(value: object, *, expected_kind: str) -> None:
    if type(value) is not str:
        raise GovernanceIdError("invalid_identifier_type", "identifier")
    if len(value) > _MAX_IDENTIFIER_LENGTH:
        raise GovernanceIdError("identifier_too_long", "identifier")

    match = _IDENTIFIER_RE.fullmatch(value)
    if match is None:
        raise GovernanceIdError("invalid_identifier", "identifier")

    kind = match.group("kind")
    if kind not in _ALLOWED_KINDS:
        raise GovernanceIdError("unknown_kind", "kind")
    if kind != expected_kind:
        raise GovernanceIdError("wrong_kind", "kind")
    if len(match.group("namespace")) > _MAX_NAMESPACE_LENGTH:
        raise GovernanceIdError("namespace_too_long", "namespace")


__all__ = [
    "CapabilityId",
    "GovernanceIdError",
    "PolicyId",
    "PurposeId",
    "ToolId",
    "WorkflowId",
]
