"""Offline plans for FHIR R4 conditional creates and updates.

Predicates may contain clinical identifiers. They stay in memory for the
caller's explicit FHIR request; representations and decisions omit their values.
This module never searches or writes to a server.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import quote, urlencode
from uuid import UUID

__all__ = [
    "ConditionalWriteKind",
    "ConditionalWritePlan",
    "MatchDecision",
    "MatchDisposition",
    "SearchPredicate",
    "build_conditional_write_plan",
    "assess_matches",
]

_RESOURCE_TYPE = re.compile(r"[A-Z][A-Za-z0-9]{0,63}\Z")
_SEARCH_NAME = re.compile(r"_?[A-Za-z][A-Za-z0-9-]*(?::[A-Za-z][A-Za-z0-9-]*)?\Z")
_CONTROL_PARAMETERS = frozenset(
    {
        "_count",
        "_sort",
        "_include",
        "_revinclude",
        "_summary",
        "_elements",
        "_total",
        "_format",
        "_pretty",
        "_contained",
        "_containedType",
        "_since",
    }
)
_KEY_DOMAIN = b"openmed.fhir.conditional-write.v1\x00"


class ConditionalWriteKind(str, Enum):
    """FHIR conditional write interaction."""

    CREATE = "create"
    UPDATE = "update"


@dataclass(frozen=True, slots=True, repr=False)
class SearchPredicate:
    """Canonical, single-valued FHIR search predicate kept in memory.

    Distinct parameter names are joined with AND. Repeated parameters and
    comma-separated OR values are rejected because their meaning can vary by
    server. Values are never normalized, logged, or included in an exception.
    """

    parameters: tuple[tuple[str, str], ...] = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.parameters) is not tuple or not self.parameters:
            raise ValueError("predicate requires search parameters")
        names: set[str] = set()
        ordered: list[tuple[str, str]] = []
        for parameter in self.parameters:
            if type(parameter) is not tuple or len(parameter) != 2:
                raise ValueError("search parameter is invalid")
            name, value = parameter
            if (
                type(name) is not str
                or _SEARCH_NAME.fullmatch(name) is None
                or name in _CONTROL_PARAMETERS
            ):
                raise ValueError("search parameter name is invalid")
            if name in names:
                raise ValueError("duplicate search parameter")
            if (
                type(value) is not str
                or not value
                or len(value) > 2048
                or any(character in value for character in ",\r\n\x00")
            ):
                raise ValueError("search parameter value is invalid")
            names.add(name)
            ordered.append((name, value))
        object.__setattr__(self, "parameters", tuple(sorted(ordered)))

    @classmethod
    def from_mapping(cls, parameters: Mapping[str, str]) -> SearchPredicate:
        """Build a predicate from one value per search parameter."""

        if not isinstance(parameters, Mapping):
            raise TypeError("search parameters must be a mapping")
        return cls(tuple(parameters.items()))

    @property
    def canonical_query(self) -> str:
        """Return the FHIR query; treat it as sensitive request content."""

        return urlencode(self.parameters, quote_via=quote, safe="")

    def __repr__(self) -> str:
        return "SearchPredicate(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class ConditionalWritePlan:
    """Typed request metadata without resource payload or credentials.

    The key is a keyed commitment to the operation identity, type, and predicate. A
    caller must separately check server capability, search matches, and obtain
    authorization before constructing a request or performing a write.
    """

    kind: ConditionalWriteKind
    resource_type: str
    predicate: SearchPredicate = field(repr=False)
    idempotency_key: str = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.kind) is not ConditionalWriteKind:
            raise TypeError("kind must be ConditionalWriteKind")
        if type(self.resource_type) is not str or not _RESOURCE_TYPE.fullmatch(
            self.resource_type
        ):
            raise ValueError("resource type is invalid")
        if not isinstance(self.predicate, SearchPredicate):
            raise TypeError("predicate must be SearchPredicate")
        if not re.fullmatch(r"fhir-cw-v1-[0-9a-f]{64}", self.idempotency_key):
            raise ValueError("idempotency key is invalid")

    def __repr__(self) -> str:
        return (
            f"ConditionalWritePlan(kind={self.kind.value!r}, "
            f"resource_type={self.resource_type!r}, predicate=<redacted>, "
            "idempotency_key=<redacted>)"
        )


def build_conditional_write_plan(
    kind: ConditionalWriteKind,
    resource_type: str,
    search_parameters: Mapping[str, str],
    *,
    operation_id: UUID,
    secret: bytes,
) -> ConditionalWritePlan:
    """Build a deterministic conditional write plan without I/O.

    Args:
        kind: Conditional create or update interaction.
        resource_type: FHIR resource type.
        search_parameters: One value per search parameter. Values may be PHI.
        operation_id: Stable opaque UUID for retries of this one planned write.
        secret: Stable, private 32-byte-or-longer key for this deployment.

    Returns:
        A typed plan with a domain-separated HMAC-SHA256 idempotency key.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If the plan cannot be safely formed.
    """

    if type(kind) is not ConditionalWriteKind:
        raise TypeError("kind must be ConditionalWriteKind")
    if type(resource_type) is not str or not _RESOURCE_TYPE.fullmatch(resource_type):
        raise ValueError("resource type is invalid")
    if type(operation_id) is not UUID:
        raise TypeError("operation_id must be UUID")
    if type(secret) is not bytes or len(secret) < 32:
        raise ValueError("secret must contain at least 32 bytes")
    predicate = SearchPredicate.from_mapping(search_parameters)
    message = (
        _KEY_DOMAIN
        + kind.value.encode("ascii")
        + b"\x00"
        + operation_id.bytes
        + b"\x00"
        + resource_type.encode("ascii")
        + b"\x00"
        + predicate.canonical_query.encode("utf-8")
    )
    digest = hmac.new(secret, message, hashlib.sha256).hexdigest()
    return ConditionalWritePlan(
        kind=kind,
        resource_type=resource_type,
        predicate=predicate,
        idempotency_key=f"fhir-cw-v1-{digest}",
    )


class MatchDisposition(str, Enum):
    """Count-only disposition before an explicit conditional request."""

    READY = "ready"
    NO_OP = "no_op"
    REVIEW = "review"


@dataclass(frozen=True, slots=True)
class MatchDecision:
    """Privacy-safe pre-write decision with a fixed reason code."""

    disposition: MatchDisposition
    reason_code: str

    def __post_init__(self) -> None:
        allowed = {
            MatchDisposition.READY: {"no_existing_match", "unique_match"},
            MatchDisposition.NO_OP: {"existing_match"},
            MatchDisposition.REVIEW: {
                "ambiguous_matches",
                "search_incomplete",
                "update_match_missing",
            },
        }
        if (
            type(self.disposition) is not MatchDisposition
            or type(self.reason_code) is not str
            or self.reason_code not in allowed[self.disposition]
        ):
            raise ValueError("match decision is invalid")

    @property
    def is_ready(self) -> bool:
        """Return whether the complete match result permits this plan."""

        return self.disposition is MatchDisposition.READY

    def require_ready(self) -> None:
        """Reject review and no-op decisions with a value-free exception."""

        if not self.is_ready:
            raise ValueError(f"conditional write is not ready: {self.reason_code}")


def assess_matches(
    plan: ConditionalWritePlan,
    match_count: int,
    *,
    search_complete: bool,
) -> MatchDecision:
    """Classify a complete caller-supplied search result without identifiers.

    A conditional update with no match needs review because some servers
    create a resource in that case. Multiple matches always need reviewer
    resolution. The caller must still handle races and server-side failures.
    """

    if not isinstance(plan, ConditionalWritePlan):
        raise TypeError("plan must be ConditionalWritePlan")
    if type(match_count) is not int or match_count < 0:
        raise ValueError("match count is invalid")
    if type(search_complete) is not bool:
        raise TypeError("search_complete must be a boolean")
    if not search_complete:
        return MatchDecision(MatchDisposition.REVIEW, "search_incomplete")
    if match_count > 1:
        return MatchDecision(MatchDisposition.REVIEW, "ambiguous_matches")
    if plan.kind is ConditionalWriteKind.CREATE:
        if match_count == 1:
            return MatchDecision(MatchDisposition.NO_OP, "existing_match")
        return MatchDecision(MatchDisposition.READY, "no_existing_match")
    if match_count == 0:
        return MatchDecision(MatchDisposition.REVIEW, "update_match_missing")
    return MatchDecision(MatchDisposition.READY, "unique_match")
