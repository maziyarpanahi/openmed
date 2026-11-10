"""Fail-closed ECL validation against a caller-supplied SNOMED CT edition.

This module deliberately contains no terminology content and no network client.
Deployments provide an :class:`ECLResolver` backed by the SNOMED CT edition they
are licensed to use.  OpenMed only validates the constraint envelope and asks
that resolver whether concept identifiers belong to the configured domains.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .postcoordination import SnomedExpression

__all__ = [
    "ECLConstraint",
    "ECLResolver",
    "ECLValidationError",
    "ECLValidationIssue",
    "ECLValidationResult",
    "ECLValidator",
    "validate_ecl_syntax",
]

_CONCEPT_ID_RE = re.compile(r"^[1-9][0-9]{5,17}$")
_OPENING = {"(": ")", "[": "]", "{": "}"}
_CLOSING = {value: key for key, value in _OPENING.items()}


class ECLResolver(Protocol):
    """Resolve concept membership in ECL using a user-supplied edition.

    Implementations may call a local terminology server or query an in-process
    index, but OpenMed neither supplies nor downloads that index.
    """

    def matches(self, concept_id: str, constraint: str, edition_uri: str) -> bool:
        """Return whether ``concept_id`` satisfies ``constraint`` in an edition."""


@dataclass(frozen=True)
class ECLConstraint:
    """Allowed focus and value domains for one post-coordination slot.

    Args:
        slot: Closed caller-facing attribute type, such as ``"laterality"``.
        attribute_id: Attribute concept identifier required for this slot.
        value_domain: ECL selecting allowed attribute values.
        focus_domain: ECL selecting focus concepts on which the attribute may
            be used. ``"*"`` delegates the unrestricted domain to the resolver.
    """

    slot: str
    attribute_id: str
    value_domain: str
    focus_domain: str = "*"

    def __post_init__(self) -> None:
        slot = self.slot.strip().casefold().replace("-", "_").replace(" ", "_")
        if not slot:
            raise ValueError("ECL constraint slot must not be blank")
        if not _CONCEPT_ID_RE.fullmatch(self.attribute_id):
            raise ValueError(
                "ECL constraint attribute_id must be a 6-18 digit concept identifier"
            )
        validate_ecl_syntax(self.value_domain)
        validate_ecl_syntax(self.focus_domain)
        object.__setattr__(self, "slot", slot)


@dataclass(frozen=True)
class ECLValidationIssue:
    """One safe, structured reason an expression was rejected."""

    code: str
    message: str
    slot: str | None = None

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-ready issue without resolver internals or raw text."""

        result = {"code": self.code, "message": self.message}
        if self.slot is not None:
            result["slot"] = self.slot
        return result


@dataclass(frozen=True)
class ECLValidationResult:
    """Result of validating one expression against a pinned edition."""

    edition_uri: str
    issues: tuple[ECLValidationIssue, ...] = ()

    @property
    def valid(self) -> bool:
        """Return whether no syntactic or semantic issue was found."""

        return not self.issues

    def require_valid(self) -> None:
        """Raise :class:`ECLValidationError` unless the expression is valid."""

        if self.issues:
            raise ECLValidationError(self)


class ECLValidationError(ValueError):
    """Raised when an expression does not satisfy its ECL constraints."""

    def __init__(self, result: ECLValidationResult) -> None:
        self.result = result
        reasons = "; ".join(issue.message for issue in result.issues)
        super().__init__(f"post-coordinated expression rejected: {reasons}")


class ECLValidator:
    """Validate expression refinements with a caller-owned ECL resolver.

    The validator is intentionally edition-specific. It fails closed when the
    resolver raises, a slot lacks a constraint, an attribute does not match its
    configured slot, or either the focus or value falls outside its ECL domain.
    """

    def __init__(
        self,
        *,
        edition_uri: str,
        constraints: Mapping[str, ECLConstraint],
        resolver: ECLResolver,
    ) -> None:
        if not isinstance(edition_uri, str) or not edition_uri.strip():
            raise ValueError("SNOMED edition_uri must be caller-supplied")
        if resolver is None or not callable(getattr(resolver, "matches", None)):
            raise TypeError("resolver must implement matches(concept_id, ECL, edition)")
        normalized: dict[str, ECLConstraint] = {}
        for raw_slot, constraint in constraints.items():
            if not isinstance(constraint, ECLConstraint):
                raise TypeError("constraints must contain ECLConstraint values")
            slot = str(raw_slot).strip().casefold().replace("-", "_").replace(" ", "_")
            if slot != constraint.slot:
                raise ValueError(
                    f"constraint key {raw_slot!r} does not match slot "
                    f"{constraint.slot!r}"
                )
            normalized[slot] = constraint
        if not normalized:
            raise ValueError("at least one ECL attribute constraint is required")
        self.edition_uri = edition_uri.strip()
        self.constraints = normalized
        self._resolver = resolver

    def validate(self, expression: SnomedExpression) -> ECLValidationResult:
        """Validate every refinement and return all safe rejection reasons."""

        issues: list[ECLValidationIssue] = []
        for refinement in expression.refinements:
            constraint = self.constraints.get(refinement.slot)
            if constraint is None:
                issues.append(
                    ECLValidationIssue(
                        code="missing_constraint",
                        message="attribute slot has no caller-supplied ECL constraint",
                        slot=refinement.slot,
                    )
                )
                continue
            if refinement.attribute.concept_id != constraint.attribute_id:
                issues.append(
                    ECLValidationIssue(
                        code="attribute_mismatch",
                        message="attribute concept does not match the configured slot",
                        slot=refinement.slot,
                    )
                )
                continue
            if not self._matches(
                expression.focus.concept_id,
                constraint.focus_domain,
                slot=refinement.slot,
                domain="focus",
                issues=issues,
            ):
                continue
            self._matches(
                refinement.value.concept_id,
                constraint.value_domain,
                slot=refinement.slot,
                domain="value",
                issues=issues,
            )
        return ECLValidationResult(self.edition_uri, tuple(issues))

    def require_valid(self, expression: SnomedExpression) -> ECLValidationResult:
        """Validate ``expression`` and raise with reasons on any failure."""

        result = self.validate(expression)
        result.require_valid()
        return result

    def _matches(
        self,
        concept_id: str,
        ecl: str,
        *,
        slot: str,
        domain: str,
        issues: list[ECLValidationIssue],
    ) -> bool:
        try:
            matches = bool(self._resolver.matches(concept_id, ecl, self.edition_uri))
        except Exception:
            issues.append(
                ECLValidationIssue(
                    code="resolver_error",
                    message="caller-supplied ECL resolver failed closed",
                    slot=slot,
                )
            )
            return False
        if not matches:
            issues.append(
                ECLValidationIssue(
                    code=f"{domain}_outside_domain",
                    message=f"{domain} concept is outside the allowed ECL domain",
                    slot=slot,
                )
            )
            return False
        return True


def validate_ecl_syntax(constraint: str) -> None:
    """Reject malformed ECL envelopes before calling a user resolver.

    This is a deliberately small safety check, not an ECL evaluator. The
    caller-supplied resolver remains authoritative for the full ECL grammar and
    semantics of its edition.
    """

    if not isinstance(constraint, str) or not constraint.strip():
        raise ValueError("ECL constraint must be non-empty text")
    if any(
        ord(character) < 32 and character not in "\t\r\n" for character in constraint
    ):
        raise ValueError("ECL constraint contains a forbidden control character")
    stack: list[str] = []
    in_term = False
    escaped = False
    for character in constraint:
        if character == "\\" and in_term:
            escaped = not escaped
            continue
        if character == "|" and not escaped:
            in_term = not in_term
            continue
        escaped = False
        if in_term:
            continue
        if character in _OPENING:
            stack.append(character)
        elif character in _CLOSING:
            if not stack or stack.pop() != _CLOSING[character]:
                raise ValueError("ECL constraint has unbalanced delimiters")
    if in_term:
        raise ValueError("ECL constraint has an unterminated description term")
    if stack:
        raise ValueError("ECL constraint has unbalanced delimiters")
