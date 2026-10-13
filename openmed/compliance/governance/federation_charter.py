"""Offline, version-bound federation rules over anonymous round declarations."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal

_CODE = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_ROLES = frozenset({"participant", "coordinator", "reviewer", "incident_responder"})
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _invalid() -> None:
    raise ValueError("federation_charter_invalid_contract")


def _count(value: object, minimum: int = 0) -> None:
    if type(value) is not int or not minimum <= value <= 1_000_000:
        _invalid()


def _codes(values: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)) or not 1 <= len(values) <= 128:
        _invalid()
    if any(type(v) is not str or not _CODE.fullmatch(v) for v in values):
        _invalid()
    if len(set(values)) != len(values):
        _invalid()
    return tuple(sorted(values))


def _privacy(epsilon: Decimal, delta: Decimal) -> None:
    if not all(type(v) is Decimal and v.is_finite() for v in (epsilon, delta)):
        _invalid()
    if not Decimal(0) <= epsilon <= Decimal(1000) or not Decimal(0) <= delta < 1:
        _invalid()


def _decimal(value: Decimal) -> str:
    # Fixed precision cap avoids reports expanding arbitrary Decimal exponents.
    if value.as_tuple().exponent < -18:
        _invalid()
    rendered = format(value, "f") if value else "0"
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


@dataclass(frozen=True, slots=True)
class FederationCharter:
    """Explicit task, membership, decision, privacy and shutdown constraints.

    Roles and task codes are local metadata vocabulary. Privacy ceilings are
    declared budgets, not evidence of a privacy mechanism or certification.
    Amendments take effect at an integer round sequence, without wall clocks.
    """

    version: int
    effective_round: int
    permitted_tasks: tuple[str, ...]
    permitted_roles: tuple[str, ...]
    decision_quorum: int
    minimum_reviewers: int
    epsilon_ceiling: Decimal
    delta_ceiling: Decimal
    incident_response: str
    appeals: str
    termination: str
    maximum_round: int

    def __post_init__(self) -> None:
        _count(self.version, 1)
        _count(self.effective_round)
        _count(self.decision_quorum, 1)
        _count(self.minimum_reviewers, 1)
        _count(self.maximum_round)
        tasks = _codes(self.permitted_tasks)
        roles = _codes(self.permitted_roles)
        if not set(roles) <= _ROLES or not {"participant", "reviewer"} <= set(roles):
            _invalid()
        if self.minimum_reviewers > self.decision_quorum:
            _invalid()
        if self.maximum_round < self.effective_round:
            _invalid()
        _privacy(self.epsilon_ceiling, self.delta_ceiling)
        _decimal(self.epsilon_ceiling)
        _decimal(self.delta_ceiling)
        if self.incident_response != "hold_pending_review":
            _invalid()
        if type(self.appeals) is not str or self.appeals not in {
            "independent_review",
            "joint_review",
        }:
            _invalid()
        if type(self.termination) is not str or self.termination not in {
            "archive_metadata",
            "erase_round_artifacts",
        }:
            _invalid()
        object.__setattr__(self, "permitted_tasks", tasks)
        object.__setattr__(self, "permitted_roles", roles)

    def to_dict(self) -> dict[str, object]:
        """Return canonical declared policy, with no membership identities."""
        return {
            "schema_version": 1,
            "version": self.version,
            "effective_round": self.effective_round,
            "permitted_tasks": list(self.permitted_tasks),
            "permitted_roles": list(self.permitted_roles),
            "decision_quorum": self.decision_quorum,
            "minimum_reviewers": self.minimum_reviewers,
            "epsilon_ceiling": _decimal(self.epsilon_ceiling),
            "delta_ceiling": _decimal(self.delta_ceiling),
            "incident_response": self.incident_response,
            "appeals": self.appeals,
            "termination": self.termination,
            "maximum_round": self.maximum_round,
        }

    @property
    def digest(self) -> str:
        """Bind all policy fields to a domain-separated canonical digest."""
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return (
            "sha256:"
            + hashlib.sha256(b"openmed.federation.charter.v1\0" + encoded).hexdigest()
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> FederationCharter:
        """Read a closed versioned JSON object, with bounded decimal strings."""
        fields = set(cls.__dataclass_fields__)
        if not isinstance(payload, Mapping) or set(payload) != fields | {
            "schema_version"
        }:
            _invalid()
        if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
            _invalid()
        values = {name: payload[name] for name in fields}
        for name in ("epsilon_ceiling", "delta_ceiling"):
            raw = values[name]
            if type(raw) is not str or not re.fullmatch(
                r"[0-9]{1,4}(?:\.[0-9]{1,18})?", raw
            ):
                _invalid()
            values[name] = Decimal(raw)
        return cls(**values)


@dataclass(frozen=True, slots=True)
class CharterHistory:
    """Append-only sequential amendments with deterministic activation."""

    amendments: tuple[FederationCharter, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.amendments, (tuple, list))
            or not 1 <= len(self.amendments) <= 128
        ):
            _invalid()
        items = tuple(self.amendments)
        if any(not isinstance(item, FederationCharter) for item in items):
            _invalid()
        if items[0].version != 1 or items[0].effective_round != 0:
            _invalid()
        if any(
            b.version != a.version + 1 or b.effective_round <= a.effective_round
            for a, b in zip(items, items[1:])
        ):
            _invalid()
        object.__setattr__(self, "amendments", items)

    def append(self, charter: FederationCharter) -> CharterHistory:
        """Return a new validated history, retaining every previous policy."""
        return CharterHistory((*self.amendments, charter))

    def active(self, round_sequence: int) -> FederationCharter:
        """Resolve exactly the latest amendment effective for this round."""
        _count(round_sequence)
        return next(
            c for c in reversed(self.amendments) if c.effective_round <= round_sequence
        )

    def to_dict(self) -> dict[str, object]:
        """Export linked amendment commitments and their declared policies."""
        previous = None
        records = []
        for charter in self.amendments:
            records.append(
                {
                    "charter": charter.to_dict(),
                    "digest": charter.digest,
                    "previous_digest": previous,
                }
            )
            previous = charter.digest
        return {"schema_version": 1, "amendments": records}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> CharterHistory:
        """Verify exported commitments and previous-policy links, not origin trust."""
        if not isinstance(payload, Mapping) or set(payload) != {
            "schema_version",
            "amendments",
        }:
            _invalid()
        if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
            _invalid()
        records = payload["amendments"]
        if not isinstance(records, (list, tuple)) or not 1 <= len(records) <= 128:
            _invalid()
        previous = None
        amendments = []
        for record in records:
            if not isinstance(record, Mapping) or set(record) != {
                "charter",
                "digest",
                "previous_digest",
            }:
                _invalid()
            charter = FederationCharter.from_dict(record["charter"])
            if (
                record["digest"] != charter.digest
                or record["previous_digest"] != previous
            ):
                _invalid()
            amendments.append(charter)
            previous = charter.digest
        return cls(tuple(amendments))


@dataclass(frozen=True, slots=True)
class FederationRoundDeclaration:
    """Anonymous proposed round metadata supplied by a trusted orchestrator."""

    sequence: int
    charter_version: int
    charter_digest: str
    task: str
    roles: tuple[str, ...]
    decision_votes: int
    reviewer_votes: int
    epsilon: Decimal
    delta: Decimal
    incident_open: bool = False
    appeal_open: bool = False
    terminated: bool = False

    def __post_init__(self) -> None:
        for value in (self.sequence, self.decision_votes, self.reviewer_votes):
            _count(value)
        _count(self.charter_version, 1)
        if type(self.charter_digest) is not str or not _DIGEST.fullmatch(
            self.charter_digest
        ):
            _invalid()
        _codes((self.task,))
        roles = _codes(self.roles)
        if not set(roles) <= _ROLES or self.reviewer_votes > self.decision_votes:
            _invalid()
        _privacy(self.epsilon, self.delta)
        _decimal(self.epsilon)
        _decimal(self.delta)
        if any(
            type(v) is not bool
            for v in (self.incident_open, self.appeal_open, self.terminated)
        ):
            _invalid()
        object.__setattr__(self, "roles", roles)


def validate_federation_round(
    history: CharterHistory, declaration: FederationRoundDeclaration
) -> dict[str, object]:
    """Gate a round against its active exact policy; return fixed reason codes.

    This validates declarations only. It does not authenticate voters, enroll
    people, compose privacy accountants, resolve appeals or execute termination.
    """
    if not isinstance(history, CharterHistory) or not isinstance(
        declaration, FederationRoundDeclaration
    ):
        _invalid()
    charter = history.active(declaration.sequence)
    reasons = []
    if (
        declaration.charter_version != charter.version
        or declaration.charter_digest != charter.digest
    ):
        reasons.append("inactive_charter")
    if declaration.task not in charter.permitted_tasks:
        reasons.append("task_not_permitted")
    if not set(declaration.roles) <= set(charter.permitted_roles):
        reasons.append("role_not_permitted")
    if not {"participant", "reviewer"} <= set(declaration.roles):
        reasons.append("required_role_missing")
    if (
        declaration.decision_votes < charter.decision_quorum
        or declaration.reviewer_votes < charter.minimum_reviewers
    ):
        reasons.append("decision_quorum_missing")
    if (
        declaration.epsilon > charter.epsilon_ceiling
        or declaration.delta > charter.delta_ceiling
    ):
        reasons.append("privacy_ceiling_exceeded")
    if declaration.incident_open:
        reasons.append("incident_review_required")
    if declaration.appeal_open:
        reasons.append("appeal_review_required")
    if declaration.terminated or declaration.sequence > charter.maximum_round:
        reasons.append("federation_terminated")
    return {
        "schema_version": 1,
        "outcome": "hold" if reasons else "accept",
        "reasons": sorted(reasons),
        "active_charter_version": charter.version,
        "active_charter_digest": charter.digest,
    }
