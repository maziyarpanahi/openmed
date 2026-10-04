"""Immutable, versioned privacy-budget policy contracts for federated rounds.

The contract in this module is deliberately narrow. It describes the policy a
coordinator commits to before a federated training round starts: the accountant
family, the composition rule, the numeric budget bounds, the unit scope of each
bound, and what happens when the budget is exhausted. It never computes
privacy loss, selects a budget, or claims a formal guarantee -- those remain
explicitly out of scope.

Every value is validated on construction and reported, never assumed safe. A
policy that cannot be evaluated is rejected with a value-free report: reason
codes and field names only, so no site, cohort, or participant identifier can
reach a log or an artifact through a rejected policy.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

DP_BUDGET_POLICY_SCHEMA_VERSION = "openmed.training.dp_budget_policy.v1"

__all__ = [
    "DP_BUDGET_MAX_EPSILON",
    "DP_BUDGET_POLICY_FIELDS",
    "DP_BUDGET_POLICY_REASON_CODES",
    "DP_BUDGET_POLICY_SCHEMA_VERSION",
    "DPAccountant",
    "DPBudgetPolicy",
    "DPBudgetPolicyError",
    "DPBudgetPolicyFinding",
    "DPBudgetPolicyRejected",
    "DPBudgetPolicyReport",
    "DPBudgetScope",
    "DPComposition",
    "DPExhaustion",
    "build_dp_budget_policy",
    "fingerprint_dp_budget_policy",
    "validate_dp_budget_policy",
]

#: Largest epsilon any single policy may declare. Bounded so that a typo or a
#: unit mix-up cannot silently unlock an effectively unbounded budget.
DP_BUDGET_MAX_EPSILON: Final = 1e3

#: Fixed serialization order of :class:`DPBudgetPolicy`.
DP_BUDGET_POLICY_FIELDS: Final = (
    "policy_id",
    "max_epsilon",
    "max_delta",
    "epsilon_scope",
    "delta_scope",
    "accountant",
    "composition",
    "exhaustion",
    "delta_prime",
    "min_rounds",
    "max_rounds",
    "schema_version",
)

#: Closed finding vocabulary. Every member fails the policy closed.
DP_BUDGET_POLICY_REASON_CODES: Final = frozenset(
    {
        "invalid_field_type",
        "unsupported_field",
        "missing_field",
        "unsupported_schema_version",
        "non_finite_value",
        "epsilon_out_of_range",
        "delta_out_of_range",
        "delta_prime_out_of_range",
        "ambiguous_scope",
        "unsupported_accountant",
        "unsupported_composition",
        "unsafe_exhaustion",
        "invalid_policy_id",
        "invalid_round_bounds",
    }
)

_REASON_ORDER: Final = {
    "missing_field": 0,
    "unsupported_field": 1,
    "invalid_field_type": 2,
    "unsupported_schema_version": 3,
    "non_finite_value": 4,
    "epsilon_out_of_range": 5,
    "delta_out_of_range": 6,
    "delta_prime_out_of_range": 7,
    "ambiguous_scope": 8,
    "unsupported_accountant": 9,
    "unsupported_composition": 10,
    "unsafe_exhaustion": 11,
    "invalid_policy_id": 12,
    "invalid_round_bounds": 13,
}

_REQUIRED_FIELDS: Final = (
    "policy_id",
    "max_epsilon",
    "max_delta",
    "epsilon_scope",
    "delta_scope",
    "accountant",
    "composition",
    "exhaustion",
)

_OPTIONAL_DEFAULTS: Final = {
    "delta_prime": 0.0,
    "min_rounds": 1,
    "max_rounds": None,
    "schema_version": DP_BUDGET_POLICY_SCHEMA_VERSION,
}

_REPORT_FIELDS: Final = frozenset(
    {"schema_version", "policy_id", "valid", "findings", "policy_digest"}
)
_FINDING_FIELDS: Final = frozenset({"reason_code", "field_name", "observed"})

_POLICY_ID = re.compile(r"[a-z][a-z0-9_.-]{0,63}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_FINGERPRINT_DOMAIN: Final = b"openmed.training.dp_budget_policy.v1\0"


class DPBudgetPolicyError(ValueError):
    """Raised when a policy payload or report violates the contract."""


class DPBudgetPolicyRejected(DPBudgetPolicyError):
    """Raised when a policy is rejected; carries the value-free report."""

    def __init__(self, report: DPBudgetPolicyReport) -> None:
        """Store the rejection report and keep the message value free.

        Args:
            report: The rejected policy report describing every finding.
        """

        self.report = report
        super().__init__(
            f"privacy budget policy rejected with {len(report.findings)} finding(s)"
        )


class DPAccountant(str, Enum):
    """Closed accountant families a policy may name."""

    BASIC = "basic"
    RENYI = "renyi"
    ZCDP = "zcdp"
    GAUSSIAN = "gaussian"


class DPComposition(str, Enum):
    """Closed composition rules a policy may name."""

    BASIC = "basic"
    ADVANCED = "advanced"


class DPBudgetScope(str, Enum):
    """Closed unit scopes for a numeric budget bound."""

    TOTAL = "total"
    PER_ROUND = "per_round"


class DPExhaustion(str, Enum):
    """Closed exhaustion behaviors. No permissive member exists."""

    FAIL_CLOSED = "fail_closed"
    REFUSE_ROUND = "refuse_round"
    REQUIRE_REVIEW = "require_review"


@dataclass(frozen=True, slots=True)
class DPBudgetPolicyFinding:
    """A single value-free reason a policy was rejected.

    Attributes:
        reason_code: Member of :data:`DP_BUDGET_POLICY_REASON_CODES`.
        field_name: Contract field name, or an empty string when the rejected
            input was an unsupported key that must not be echoed.
        observed: Optional bounded integer observation, such as a round bound.
            Free-form and floating-point values are never echoed.
    """

    reason_code: str
    field_name: str
    observed: int | None = None

    def __post_init__(self) -> None:
        if self.reason_code not in DP_BUDGET_POLICY_REASON_CODES:
            raise DPBudgetPolicyError("reason_code is not a supported policy reason")
        if self.field_name != "" and self.field_name not in DP_BUDGET_POLICY_FIELDS:
            raise DPBudgetPolicyError("field_name is not a policy field")
        if self.observed is not None and type(self.observed) is not int:
            raise DPBudgetPolicyError("observed must be an integer or None")

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic mapping for this finding."""

        return {
            "reason_code": self.reason_code,
            "field_name": self.field_name,
            "observed": self.observed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DPBudgetPolicyFinding:
        """Rebuild a finding from :meth:`to_dict` output.

        Args:
            payload: Mapping with exactly the documented finding keys.

        Returns:
            The rebuilt finding.

        Raises:
            DPBudgetPolicyError: If the payload shape or values are invalid.
        """

        if not isinstance(payload, Mapping) or set(payload) != _FINDING_FIELDS:
            raise DPBudgetPolicyError("invalid policy finding fields")
        reason_code = payload["reason_code"]
        field_name = payload["field_name"]
        observed = payload["observed"]
        if type(reason_code) is not str or type(field_name) is not str:
            raise DPBudgetPolicyError("invalid policy finding values")
        if observed is not None and type(observed) is not int:
            raise DPBudgetPolicyError("invalid policy finding observation")
        return cls(reason_code=reason_code, field_name=field_name, observed=observed)


@dataclass(frozen=True, slots=True)
class DPBudgetPolicy:
    """Immutable, versioned privacy-budget policy for federated rounds.

    Attributes:
        policy_id: Lowercase identifier for the committed policy.
        max_epsilon: Bounded epsilon the policy grants.
        max_delta: Bounded delta the policy grants, strictly below one.
        epsilon_scope: Unit scope of ``max_epsilon``.
        delta_scope: Unit scope of ``max_delta``.
        accountant: Closed accountant family the policy names.
        composition: Closed composition rule the policy names.
        exhaustion: Behavior once the budget is exhausted.
        delta_prime: Advanced-composition slack, zero for basic composition.
        min_rounds: Inclusive lower bound on the rounds the policy covers.
        max_rounds: Optional inclusive upper bound; ``None`` means unbounded.
        schema_version: Schema version of this contract.

    Raises:
        DPBudgetPolicyRejected: If any field fails the closed contract. The
            rejection carries a value-free :class:`DPBudgetPolicyReport`.
    """

    policy_id: str
    max_epsilon: float
    max_delta: float
    epsilon_scope: DPBudgetScope
    delta_scope: DPBudgetScope
    accountant: DPAccountant
    composition: DPComposition
    exhaustion: DPExhaustion
    delta_prime: float = 0.0
    min_rounds: int = 1
    max_rounds: int | None = None
    schema_version: str = DP_BUDGET_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field, enum_type in (
            ("epsilon_scope", DPBudgetScope),
            ("delta_scope", DPBudgetScope),
            ("accountant", DPAccountant),
            ("composition", DPComposition),
            ("exhaustion", DPExhaustion),
        ):
            member = _enum_member(enum_type, getattr(self, field))
            if member is not None:
                object.__setattr__(self, field, member)
        findings = _collect_findings(self._payload())
        if findings:
            raise DPBudgetPolicyRejected(_rejected_report(self._payload(), findings))

    def _payload(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in DP_BUDGET_POLICY_FIELDS}

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic mapping in the documented field order."""

        return {
            "policy_id": self.policy_id,
            "max_epsilon": self.max_epsilon,
            "max_delta": self.max_delta,
            "epsilon_scope": self.epsilon_scope.value,
            "delta_scope": self.delta_scope.value,
            "accountant": self.accountant.value,
            "composition": self.composition.value,
            "exhaustion": self.exhaustion.value,
            "delta_prime": self.delta_prime,
            "min_rounds": self.min_rounds,
            "max_rounds": self.max_rounds,
            "schema_version": self.schema_version,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize the policy deterministically.

        Args:
            indent: Optional indentation. ``None`` emits compact bytes.

        Returns:
            JSON text ending with a newline.
        """

        separators = None if indent else (",", ":")
        return (
            json.dumps(
                self.to_dict(),
                allow_nan=False,
                ensure_ascii=True,
                indent=indent,
                separators=separators,
                sort_keys=True,
            )
            + "\n"
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DPBudgetPolicy:
        """Rebuild a policy from :meth:`to_dict` output.

        Args:
            payload: Mapping with exactly the documented policy keys.

        Returns:
            The validated policy.

        Raises:
            DPBudgetPolicyError: If the payload shape is invalid.
            DPBudgetPolicyRejected: If any field fails the closed contract.
        """

        if not isinstance(payload, Mapping) or set(payload) != set(
            DP_BUDGET_POLICY_FIELDS
        ):
            raise DPBudgetPolicyError("invalid privacy budget policy fields")
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class DPBudgetPolicyReport:
    """Value-free outcome of validating a privacy-budget policy.

    Attributes:
        schema_version: Schema version of the report contract.
        policy_id: Committed policy identifier, empty for a rejected input.
        valid: ``True`` only when no finding was raised.
        findings: Sorted, duplicate-free findings.
        policy_digest: Domain-separated digest of a valid policy, else empty.

    Raises:
        DPBudgetPolicyError: If the report violates its own invariants.
    """

    schema_version: str
    policy_id: str
    valid: bool
    findings: tuple[DPBudgetPolicyFinding, ...]
    policy_digest: str

    def __post_init__(self) -> None:
        if self.schema_version != DP_BUDGET_POLICY_SCHEMA_VERSION:
            raise DPBudgetPolicyError("unsupported report schema version")
        if type(self.policy_id) is not str or type(self.valid) is not bool:
            raise DPBudgetPolicyError("invalid report scalar fields")
        if type(self.findings) is not tuple or any(
            type(finding) is not DPBudgetPolicyFinding for finding in self.findings
        ):
            raise DPBudgetPolicyError("report findings must be a tuple of findings")
        if self.findings != _sorted_findings(self.findings):
            raise DPBudgetPolicyError(
                "report findings must be sorted and free of duplicates"
            )
        if self.valid:
            if self.findings:
                raise DPBudgetPolicyError("a valid report cannot carry findings")
            if not _POLICY_ID.fullmatch(self.policy_id):
                raise DPBudgetPolicyError("a valid report needs a policy identifier")
            if type(self.policy_digest) is not str or not _DIGEST.fullmatch(
                self.policy_digest
            ):
                raise DPBudgetPolicyError("a valid report needs a policy digest")
        else:
            if not self.findings:
                raise DPBudgetPolicyError(
                    "a rejected report needs at least one finding"
                )
            if self.policy_digest != "":
                raise DPBudgetPolicyError("a rejected report cannot carry a digest")

    @property
    def ok(self) -> bool:
        """Return ``True`` when the policy passed validation."""

        return self.valid

    @property
    def reason_codes(self) -> tuple[str, ...]:
        """Return the sorted, duplicate-free reason codes of the findings."""

        return tuple(sorted({finding.reason_code for finding in self.findings}))

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic mapping for the report."""

        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "valid": self.valid,
            "findings": [finding.to_dict() for finding in self.findings],
            "policy_digest": self.policy_digest,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize the report deterministically.

        Args:
            indent: Optional indentation. ``None`` emits compact bytes.

        Returns:
            JSON text ending with a newline.
        """

        separators = None if indent else (",", ":")
        return (
            json.dumps(
                self.to_dict(),
                allow_nan=False,
                ensure_ascii=True,
                indent=indent,
                separators=separators,
                sort_keys=True,
            )
            + "\n"
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DPBudgetPolicyReport:
        """Rebuild a report from :meth:`to_dict` output.

        Args:
            payload: Mapping with exactly the documented report keys.

        Returns:
            The rebuilt report.

        Raises:
            DPBudgetPolicyError: If the payload shape or values are invalid.
        """

        if not isinstance(payload, Mapping) or set(payload) != _REPORT_FIELDS:
            raise DPBudgetPolicyError("invalid policy report fields")
        if type(payload["findings"]) is not list:
            raise DPBudgetPolicyError("invalid policy report findings")
        findings = tuple(
            DPBudgetPolicyFinding.from_dict(row) for row in payload["findings"]
        )
        return cls(
            schema_version=payload["schema_version"],
            policy_id=payload["policy_id"],
            valid=payload["valid"],
            findings=findings,
            policy_digest=payload["policy_digest"],
        )


def _enum_member(enum_type: type[Enum], value: object) -> Enum | None:
    """Return the member of ``enum_type`` matching ``value``, else ``None``."""

    for member in enum_type:
        if value is member or value == member.value:
            return member
    return None


def _is_number(value: object) -> bool:
    if type(value) is int or type(value) is float:
        return True
    return False


def _finite(value: object) -> bool:
    if type(value) is int:
        return True
    if type(value) is float:
        return math.isfinite(value)
    return False


def _enum_value(enum_type: type[Enum], value: object) -> bool:
    return _enum_member(enum_type, value) is not None


def _collect_findings(
    payload: Mapping[str, Any],
) -> tuple[DPBudgetPolicyFinding, ...]:
    """Collect every finding for a candidate policy payload.

    Args:
        payload: Mapping that may mix supported, unsupported, and missing keys.

    Returns:
        Sorted, duplicate-free findings. An empty tuple means the payload
        satisfies the closed contract.
    """

    findings: list[DPBudgetPolicyFinding] = []
    supported = set(DP_BUDGET_POLICY_FIELDS)

    for key in payload:
        if type(key) is not str or key not in supported:
            findings.append(DPBudgetPolicyFinding("unsupported_field", ""))
    for field in _REQUIRED_FIELDS:
        if field not in payload:
            findings.append(DPBudgetPolicyFinding("missing_field", field))

    if "schema_version" in payload:
        schema_version = payload["schema_version"]
        if type(schema_version) is not str:
            findings.append(
                DPBudgetPolicyFinding("invalid_field_type", "schema_version")
            )
        elif schema_version != DP_BUDGET_POLICY_SCHEMA_VERSION:
            findings.append(
                DPBudgetPolicyFinding("unsupported_schema_version", "schema_version")
            )

    policy_id = payload.get("policy_id", None)
    if "policy_id" in payload:
        if type(policy_id) is not str:
            findings.append(DPBudgetPolicyFinding("invalid_field_type", "policy_id"))
        elif not _POLICY_ID.fullmatch(policy_id):
            findings.append(DPBudgetPolicyFinding("invalid_policy_id", "policy_id"))

    for field in ("max_epsilon", "max_delta", "delta_prime"):
        if field not in payload:
            continue
        value = payload[field]
        if not _is_number(value) or type(value) is bool:
            findings.append(DPBudgetPolicyFinding("invalid_field_type", field))
        elif not _finite(value):
            findings.append(DPBudgetPolicyFinding("non_finite_value", field))

    if "epsilon_scope" in payload and not _enum_value(
        DPBudgetScope, payload["epsilon_scope"]
    ):
        findings.append(DPBudgetPolicyFinding("ambiguous_scope", "epsilon_scope"))
    if "delta_scope" in payload and not _enum_value(
        DPBudgetScope, payload["delta_scope"]
    ):
        findings.append(DPBudgetPolicyFinding("ambiguous_scope", "delta_scope"))
    if "accountant" in payload and not _enum_value(DPAccountant, payload["accountant"]):
        findings.append(DPBudgetPolicyFinding("unsupported_accountant", "accountant"))
    if "composition" in payload and not _enum_value(
        DPComposition, payload["composition"]
    ):
        findings.append(DPBudgetPolicyFinding("unsupported_composition", "composition"))
    if "exhaustion" in payload and not _enum_value(DPExhaustion, payload["exhaustion"]):
        findings.append(DPBudgetPolicyFinding("unsafe_exhaustion", "exhaustion"))

    max_epsilon = payload.get("max_epsilon", None)
    if _is_number(max_epsilon) and _finite(max_epsilon):
        if not 0 < max_epsilon <= DP_BUDGET_MAX_EPSILON:
            findings.append(
                DPBudgetPolicyFinding("epsilon_out_of_range", "max_epsilon")
            )

    max_delta = payload.get("max_delta", None)
    if _is_number(max_delta) and _finite(max_delta):
        if not 0 < max_delta < 1:
            findings.append(DPBudgetPolicyFinding("delta_out_of_range", "max_delta"))

    delta_prime = payload.get("delta_prime", None)
    if _is_number(delta_prime) and _finite(delta_prime):
        if delta_prime < 0:
            findings.append(
                DPBudgetPolicyFinding("delta_prime_out_of_range", "delta_prime")
            )
        elif _is_number(max_delta) and _finite(max_delta) and delta_prime >= max_delta:
            findings.append(
                DPBudgetPolicyFinding("delta_prime_out_of_range", "delta_prime")
            )

    composition = payload.get("composition", None)
    if _enum_value(DPComposition, composition) and _is_number(delta_prime):
        advanced = composition is DPComposition.ADVANCED or (
            composition == DPComposition.ADVANCED.value
        )
        if advanced and delta_prime <= 0:
            findings.append(
                DPBudgetPolicyFinding("unsupported_composition", "composition")
            )
        elif not advanced and delta_prime != 0:
            findings.append(
                DPBudgetPolicyFinding("unsupported_composition", "composition")
            )

    for field in ("min_rounds", "max_rounds"):
        if field not in payload:
            continue
        value = payload[field]
        if field == "min_rounds":
            if type(value) is not int or type(value) is bool:
                findings.append(DPBudgetPolicyFinding("invalid_round_bounds", field))
            elif value < 1:
                findings.append(
                    DPBudgetPolicyFinding("invalid_round_bounds", field, observed=value)
                )
        elif value is not None:
            if type(value) is not int or type(value) is bool:
                findings.append(DPBudgetPolicyFinding("invalid_round_bounds", field))
            elif value < 1:
                findings.append(
                    DPBudgetPolicyFinding("invalid_round_bounds", field, observed=value)
                )
            else:
                min_rounds = payload.get("min_rounds", None)
                if type(min_rounds) is int and value < min_rounds:
                    findings.append(
                        DPBudgetPolicyFinding(
                            "invalid_round_bounds", field, observed=value
                        )
                    )

    return _sorted_findings(tuple(findings))


def _sorted_findings(
    findings: tuple[DPBudgetPolicyFinding, ...],
) -> tuple[DPBudgetPolicyFinding, ...]:
    unique = {
        (finding.reason_code, finding.field_name, finding.observed)
        for finding in findings
    }
    ordered = sorted(
        (
            DPBudgetPolicyFinding(reason_code, field_name, observed)
            for reason_code, field_name, observed in unique
        ),
        key=lambda finding: (
            DP_BUDGET_POLICY_FIELDS.index(finding.field_name)
            if finding.field_name in DP_BUDGET_POLICY_FIELDS
            else len(DP_BUDGET_POLICY_FIELDS),
            _REASON_ORDER[finding.reason_code],
            -1 if finding.observed is None else finding.observed,
        ),
    )
    return tuple(ordered)


def _rejected_report(
    payload: Mapping[str, Any],
    findings: tuple[DPBudgetPolicyFinding, ...],
) -> DPBudgetPolicyReport:
    return DPBudgetPolicyReport(
        schema_version=DP_BUDGET_POLICY_SCHEMA_VERSION,
        policy_id="",
        valid=False,
        findings=findings,
        policy_digest="",
    )


def build_dp_budget_policy(**fields: Any) -> DPBudgetPolicy:
    """Build a validated privacy-budget policy from keyword fields.

    Args:
        **fields: The documented policy fields. Optional fields default to a
            bounded, non-permissive value; the safety-relevant fields
            ``epsilon_scope``, ``delta_scope``, ``accountant``, ``composition``
            and ``exhaustion`` have no default and must be supplied.

    Returns:
        The validated policy.

    Raises:
        DPBudgetPolicyRejected: If any field is unsupported, missing, or fails
            the closed contract.
    """

    unknown = sorted(
        key
        for key in fields
        if type(key) is not str or key not in DP_BUDGET_POLICY_FIELDS
    )
    if unknown:
        findings = tuple(
            DPBudgetPolicyFinding("unsupported_field", "") for _key in unknown
        )
        raise DPBudgetPolicyRejected(_rejected_report({}, _sorted_findings(findings)))
    missing = tuple(field for field in _REQUIRED_FIELDS if field not in fields)
    if missing:
        findings = tuple(
            DPBudgetPolicyFinding("missing_field", field) for field in missing
        )
        raise DPBudgetPolicyRejected(
            _rejected_report(dict(fields), _sorted_findings(findings))
        )
    merged: dict[str, Any] = dict(_OPTIONAL_DEFAULTS)
    merged.update(fields)
    return DPBudgetPolicy(**merged)


def validate_dp_budget_policy(payload: Mapping[str, Any]) -> DPBudgetPolicyReport:
    """Validate a mapping as a privacy-budget policy without raising on content.

    Args:
        payload: Candidate policy mapping, for example ``json.loads`` output.

    Returns:
        A report carrying the valid policy digest, or sorted findings.

    Raises:
        DPBudgetPolicyError: If ``payload`` is not a mapping.
    """

    if not isinstance(payload, Mapping):
        raise DPBudgetPolicyError("policy payload must be a mapping")
    candidate = {key: payload[key] for key in payload}
    findings = _collect_findings(candidate)
    if findings:
        return _rejected_report(candidate, findings)
    merged: dict[str, Any] = dict(_OPTIONAL_DEFAULTS)
    merged.update(candidate)
    policy = DPBudgetPolicy(
        **{field: merged[field] for field in DP_BUDGET_POLICY_FIELDS}
    )
    return DPBudgetPolicyReport(
        schema_version=DP_BUDGET_POLICY_SCHEMA_VERSION,
        policy_id=policy.policy_id,
        valid=True,
        findings=(),
        policy_digest=fingerprint_dp_budget_policy(policy),
    )


def fingerprint_dp_budget_policy(policy: DPBudgetPolicy) -> str:
    """Return the domain-separated digest of a validated policy.

    Args:
        policy: A validated :class:`DPBudgetPolicy`.

    Returns:
        ``"sha256:"`` prefixed digest over the compact, sorted policy mapping.

    Raises:
        DPBudgetPolicyError: If ``policy`` is not a :class:`DPBudgetPolicy`.
    """

    if type(policy) is not DPBudgetPolicy:
        raise DPBudgetPolicyError("fingerprinting requires a privacy budget policy")
    canonical = json.dumps(
        policy.to_dict(),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(_FINGERPRINT_DOMAIN + canonical).hexdigest()
