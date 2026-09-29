"""Content-free batch preflight for anonymous federated update metadata.

Each submitted envelope is validated independently against one trusted
``FederatedUpdatePolicy``. Accepted envelopes are grouped by schema fingerprint,
groups below the disclosure floor are suppressed, and declared update digests
stay hidden unless the caller explicitly permits disclosure. The checker never
reads tensors, identifiers, endpoints, or local values.
"""

from __future__ import annotations

import hmac
import json
import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Final

from .federated_schema_fingerprint import fingerprint_update_schema
from .federated_status import DEFAULT_FEDERATED_MINIMUM_GROUP_SIZE
from .federated_update_metadata import (
    FederatedUpdateMetadata,
    FederatedUpdateMetadataError,
    FederatedUpdatePolicy,
)

FEDERATED_UPDATE_BATCH_SCHEMA_VERSION = "openmed.training.federated_update_batch.v1"
DEFAULT_MAX_BATCH_UPDATES: Final = 64
DEFAULT_FEDERATED_UPDATE_MINIMUM_GROUP_SIZE: Final = (
    DEFAULT_FEDERATED_MINIMUM_GROUP_SIZE
)
MAX_BATCH_UPDATES: Final = 512

_DIGEST: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")


class FederatedUpdateBatchError(ValueError):
    """Raised for invalid batch inputs without including submitted values."""


class FederatedUpdateBatchStatus(str, Enum):
    """Verdict applied to one submitted envelope or grouped finding."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SKIPPED = "skipped"

    def __str__(self) -> str:
        return self.value


class FederatedUpdateBatchReasonCode(str, Enum):
    """Stable, closed reasons emitted by the batch preflight."""

    BATCH_DIGEST_WITHHELD = "batch_digest_withheld"
    BATCH_EMPTY = "batch_empty"
    BATCH_GROUP_SUPPRESSED = "batch_group_suppressed"
    BATCH_LIMIT_EXCEEDED = "batch_limit_exceeded"
    UPDATE_ACCEPTED = "update_accepted"
    UPDATE_DUPLICATE = "update_duplicate"
    UPDATE_FINGERPRINT_MISMATCH = "update_fingerprint_mismatch"
    UPDATE_SCHEMA_INVALID = "update_schema_invalid"

    def __str__(self) -> str:
        return self.value


FEDERATED_UPDATE_BATCH_REASON_CODES: Final[
    tuple[FederatedUpdateBatchReasonCode, ...]
] = tuple(sorted(FederatedUpdateBatchReasonCode, key=lambda reason: reason.value))

_STATUS_BY_REASON: Final[
    dict[FederatedUpdateBatchReasonCode, FederatedUpdateBatchStatus]
] = {
    FederatedUpdateBatchReasonCode.BATCH_DIGEST_WITHHELD: (
        FederatedUpdateBatchStatus.SKIPPED
    ),
    FederatedUpdateBatchReasonCode.BATCH_EMPTY: FederatedUpdateBatchStatus.SKIPPED,
    FederatedUpdateBatchReasonCode.BATCH_GROUP_SUPPRESSED: (
        FederatedUpdateBatchStatus.SKIPPED
    ),
    FederatedUpdateBatchReasonCode.BATCH_LIMIT_EXCEEDED: (
        FederatedUpdateBatchStatus.SKIPPED
    ),
    FederatedUpdateBatchReasonCode.UPDATE_ACCEPTED: (
        FederatedUpdateBatchStatus.ACCEPTED
    ),
    FederatedUpdateBatchReasonCode.UPDATE_DUPLICATE: (
        FederatedUpdateBatchStatus.REJECTED
    ),
    FederatedUpdateBatchReasonCode.UPDATE_FINGERPRINT_MISMATCH: (
        FederatedUpdateBatchStatus.REJECTED
    ),
    FederatedUpdateBatchReasonCode.UPDATE_SCHEMA_INVALID: (
        FederatedUpdateBatchStatus.REJECTED
    ),
}


@dataclass(frozen=True, slots=True)
class FederatedUpdateBatchPolicy:
    """Trusted batch expectations supplied independently of the envelopes.

    Args:
        update_policy: Per-envelope policy reused for every submission.
        expected_fingerprint: Optional ``sha256:`` schema fingerprint that every
            accepted envelope must match; ``None`` accepts any valid schema.
        max_updates: Bounded number of envelopes accepted in one batch, at most
            ``MAX_BATCH_UPDATES``.
        minimum_group_size: Smallest acceptable group per schema fingerprint;
            smaller groups are suppressed and their digests withheld. Defaults
            to the shared federated floor.
        disclose_digests: Whether declared per-update digests may appear in the
            report. Disabled by default because digests are disclosure-sensitive.
    """

    update_policy: FederatedUpdatePolicy
    expected_fingerprint: str | None = None
    max_updates: int = DEFAULT_MAX_BATCH_UPDATES
    minimum_group_size: int = DEFAULT_FEDERATED_UPDATE_MINIMUM_GROUP_SIZE
    disclose_digests: bool = False

    def __post_init__(self) -> None:
        if type(self.update_policy) is not FederatedUpdatePolicy:
            raise FederatedUpdateBatchError(
                "batch policy must be a FederatedUpdateBatchPolicy"
            )
        if self.expected_fingerprint is not None and (
            type(self.expected_fingerprint) is not str
            or _DIGEST.fullmatch(self.expected_fingerprint) is None
        ):
            raise FederatedUpdateBatchError(
                "expected_fingerprint must be a sha256 digest reference"
            )
        _require_bounded_int(self.max_updates, 1, MAX_BATCH_UPDATES, "max_updates")
        _require_bounded_int(
            self.minimum_group_size,
            2,
            MAX_BATCH_UPDATES,
            "minimum_group_size",
        )
        if type(self.disclose_digests) is not bool:
            raise FederatedUpdateBatchError("disclose_digests must be a boolean")


@dataclass(frozen=True, slots=True)
class FederatedUpdateOutcome:
    """Per-envelope verdict in submitted order.

    Args:
        index: Zero-based submitted position of the envelope.
        status: Accepted, rejected, or skipped verdict.
        reason: Stable reason code explaining the verdict.
        fingerprint: Schema fingerprint of an accepted envelope, otherwise None.
        update_digest: Declared update digest, present only for an accepted
            envelope when the policy permits disclosure.
    """

    index: int
    status: FederatedUpdateBatchStatus
    reason: FederatedUpdateBatchReasonCode
    fingerprint: str | None = None
    update_digest: str | None = None

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise FederatedUpdateBatchError(
                "outcome index must be a non-negative integer"
            )
        if type(self.status) is not FederatedUpdateBatchStatus:
            raise FederatedUpdateBatchError(
                "outcome status must be a FederatedUpdateBatchStatus"
            )
        if type(self.reason) is not FederatedUpdateBatchReasonCode:
            raise FederatedUpdateBatchError(
                "outcome reason must be a FederatedUpdateBatchReasonCode"
            )
        if _STATUS_BY_REASON[self.reason] is not self.status:
            raise FederatedUpdateBatchError(
                "outcome reason does not match outcome status"
            )
        for value in (self.fingerprint, self.update_digest):
            if value is not None and (
                type(value) is not str or _DIGEST.fullmatch(value) is None
            ):
                raise FederatedUpdateBatchError(
                    "outcome digests must be sha256 digest references"
                )
        if self.status is not FederatedUpdateBatchStatus.ACCEPTED and (
            self.fingerprint is not None or self.update_digest is not None
        ):
            raise FederatedUpdateBatchError("rejected outcomes must not carry digests")

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh mapping containing no submitted values."""

        return {
            "fingerprint": self.fingerprint,
            "index": self.index,
            "reason": str(self.reason),
            "status": str(self.status),
            "update_digest": self.update_digest,
        }


@dataclass(frozen=True, slots=True)
class FederatedUpdateGroup:
    """Accepted envelopes sharing one schema fingerprint.

    Args:
        fingerprint: Shared content-free schema fingerprint.
        accepted: Number of accepted envelopes in the group.
        suppressed: Whether the group fell below the disclosure floor.
        update_digests: Sorted declared digests, empty when the group is
            suppressed or when disclosure is not permitted.
    """

    fingerprint: str
    accepted: int
    suppressed: bool
    update_digests: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.fingerprint) is not str
            or _DIGEST.fullmatch(self.fingerprint) is None
        ):
            raise FederatedUpdateBatchError(
                "group fingerprint must be a sha256 digest reference"
            )
        if type(self.accepted) is not int or self.accepted < 1:
            raise FederatedUpdateBatchError(
                "group accepted count must be a positive integer"
            )
        if type(self.suppressed) is not bool:
            raise FederatedUpdateBatchError("group suppressed flag must be a boolean")
        if type(self.update_digests) is not tuple or any(
            type(digest) is not str or _DIGEST.fullmatch(digest) is None
            for digest in self.update_digests
        ):
            raise FederatedUpdateBatchError(
                "group digests must be sha256 digest references"
            )
        if self.suppressed and self.update_digests:
            raise FederatedUpdateBatchError("suppressed groups must not carry digests")
        if self.update_digests and len(self.update_digests) != self.accepted:
            raise FederatedUpdateBatchError(
                "group digest count does not match the accepted count"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh mapping with disclosed digests in sorted order."""

        return {
            "accepted": self.accepted,
            "fingerprint": self.fingerprint,
            "suppressed": self.suppressed,
            "update_digests": list(self.update_digests),
        }


@dataclass(frozen=True, slots=True)
class FederatedUpdateBatchFinding:
    """Counts for one reason code, grouped across the whole batch.

    Args:
        reason: Stable reason code shared by the counted envelopes.
        status: Verdict implied by the reason code.
        count: Number of envelopes or groups counted under the reason.
    """

    reason: FederatedUpdateBatchReasonCode
    status: FederatedUpdateBatchStatus
    count: int

    def __post_init__(self) -> None:
        if type(self.reason) is not FederatedUpdateBatchReasonCode:
            raise FederatedUpdateBatchError(
                "finding reason must be a FederatedUpdateBatchReasonCode"
            )
        if type(self.status) is not FederatedUpdateBatchStatus:
            raise FederatedUpdateBatchError(
                "finding status must be a FederatedUpdateBatchStatus"
            )
        if _STATUS_BY_REASON[self.reason] is not self.status:
            raise FederatedUpdateBatchError(
                "finding reason does not match finding status"
            )
        if type(self.count) is not int or self.count < 1:
            raise FederatedUpdateBatchError("finding count must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh mapping for the grouped finding."""

        return {
            "count": self.count,
            "reason": str(self.reason),
            "status": str(self.status),
        }


@dataclass(frozen=True, slots=True)
class FederatedUpdateBatchReport:
    """Deterministic, content-free verdict for a submitted batch.

    Args:
        received: Number of submitted envelopes.
        accepted: Number accepted after independent validation.
        rejected: Number rejected with a per-envelope reason code.
        suppressed_groups: Number of schema groups below the disclosure floor.
        limit_exceeded: Whether the submission exceeded ``max_updates``.
        findings: Counts per reason code in ascending reason-code order.
        groups: Accepted groups in ascending fingerprint order.
        outcomes: Per-envelope verdicts in submitted order.
        schema_version: Versioned report schema identifier.
    """

    received: int
    accepted: int
    rejected: int
    suppressed_groups: int
    limit_exceeded: bool
    findings: tuple[FederatedUpdateBatchFinding, ...]
    groups: tuple[FederatedUpdateGroup, ...]
    outcomes: tuple[FederatedUpdateOutcome, ...]
    schema_version: str = FEDERATED_UPDATE_BATCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != FEDERATED_UPDATE_BATCH_SCHEMA_VERSION:
            raise FederatedUpdateBatchError("unsupported batch report schema")
        for count in (
            self.received,
            self.accepted,
            self.rejected,
            self.suppressed_groups,
        ):
            if type(count) is not int or count < 0:
                raise FederatedUpdateBatchError(
                    "batch report counts must be non-negative integers"
                )
        if type(self.limit_exceeded) is not bool:
            raise FederatedUpdateBatchError("batch report limit flag must be a boolean")
        expected = 0 if self.limit_exceeded else self.received
        if self.accepted + self.rejected != expected:
            raise FederatedUpdateBatchError("batch report counts do not reconcile")
        if type(self.findings) is not tuple or any(
            type(finding) is not FederatedUpdateBatchFinding
            for finding in self.findings
        ):
            raise FederatedUpdateBatchError(
                "batch report findings must be FederatedUpdateBatchFinding records"
            )
        reasons = [finding.reason for finding in self.findings]
        if reasons != sorted(reasons, key=lambda reason: reason.value) or len(
            set(reasons)
        ) != len(reasons):
            raise FederatedUpdateBatchError(
                "batch report findings must be ordered reason codes"
            )
        if type(self.groups) is not tuple or any(
            type(group) is not FederatedUpdateGroup for group in self.groups
        ):
            raise FederatedUpdateBatchError(
                "batch report groups must be FederatedUpdateGroup records"
            )
        fingerprints = [group.fingerprint for group in self.groups]
        if fingerprints != sorted(fingerprints):
            raise FederatedUpdateBatchError(
                "batch report groups must be ordered fingerprints"
            )
        if sum(group.accepted for group in self.groups) != self.accepted:
            raise FederatedUpdateBatchError(
                "batch report groups do not reconcile with the accepted count"
            )
        if sum(1 for group in self.groups if group.suppressed) != (
            self.suppressed_groups
        ):
            raise FederatedUpdateBatchError(
                "batch report suppression count does not reconcile"
            )
        if type(self.outcomes) is not tuple or any(
            type(outcome) is not FederatedUpdateOutcome for outcome in self.outcomes
        ):
            raise FederatedUpdateBatchError(
                "batch report outcomes must be FederatedUpdateOutcome records"
            )
        if len(self.outcomes) != expected:
            raise FederatedUpdateBatchError(
                "batch report outcomes do not reconcile with the received count"
            )

    @property
    def ok(self) -> bool:
        """Return whether the batch was bounded and free of rejected envelopes."""

        return not self.limit_exceeded and self.rejected == 0

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh versioned mapping without submitted values."""

        return {
            "schema_version": self.schema_version,
            "received": self.received,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "suppressed_groups": self.suppressed_groups,
            "limit_exceeded": self.limit_exceeded,
            "findings": [finding.to_dict() for finding in self.findings],
            "groups": [group.to_dict() for group in self.groups],
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
        }

    def to_json(self) -> str:
        """Return byte-stable JSON with a trailing newline."""

        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def check_federated_update_batch(
    payloads: object, *, policy: FederatedUpdateBatchPolicy
) -> FederatedUpdateBatchReport:
    """Validate a bounded batch of anonymous envelopes independently.

    Args:
        payloads: Submitted envelopes as JSON text or JSON-style dictionaries
            with exactly the update metadata fields. One invalid envelope never
            stops the remaining submissions.
        policy: Trusted batch and per-envelope expectations.

    Returns:
        A deterministic report containing counts, grouped reason codes, schema
        groups, and per-envelope verdicts. No submitted value, identifier,
        parameter name, path, endpoint, or message is retained.

    Raises:
        FederatedUpdateBatchError: If ``policy`` is not a batch policy or
            ``payloads`` is not a list or tuple.
    """

    if type(policy) is not FederatedUpdateBatchPolicy:
        raise FederatedUpdateBatchError(
            "batch policy must be a FederatedUpdateBatchPolicy"
        )
    if type(payloads) is not list and type(payloads) is not tuple:
        raise FederatedUpdateBatchError(
            "payloads must be a list or tuple of update envelopes"
        )
    received = len(payloads)
    if received > policy.max_updates:
        findings: tuple[FederatedUpdateBatchFinding, ...] = (
            FederatedUpdateBatchFinding(
                FederatedUpdateBatchReasonCode.BATCH_LIMIT_EXCEEDED,
                FederatedUpdateBatchStatus.SKIPPED,
                1,
            ),
        )
        return FederatedUpdateBatchReport(
            received=received,
            accepted=0,
            rejected=0,
            suppressed_groups=0,
            limit_exceeded=True,
            findings=findings,
            groups=(),
            outcomes=(),
        )

    outcomes: list[FederatedUpdateOutcome] = []
    accepted_digests: list[str] = []
    accepted_by_index: dict[int, tuple[str, str]] = {}
    by_fingerprint: dict[str, list[str]] = {}
    seen_digests: set[str] = set()
    for index, payload in enumerate(payloads):
        try:
            metadata = _parse_envelope(payload, policy.update_policy)
        except FederatedUpdateMetadataError:
            outcomes.append(
                FederatedUpdateOutcome(
                    index,
                    FederatedUpdateBatchStatus.REJECTED,
                    FederatedUpdateBatchReasonCode.UPDATE_SCHEMA_INVALID,
                )
            )
            continue
        fingerprint = fingerprint_update_schema(metadata)
        if policy.expected_fingerprint is not None and not hmac.compare_digest(
            fingerprint, policy.expected_fingerprint
        ):
            outcomes.append(
                FederatedUpdateOutcome(
                    index,
                    FederatedUpdateBatchStatus.REJECTED,
                    FederatedUpdateBatchReasonCode.UPDATE_FINGERPRINT_MISMATCH,
                )
            )
            continue
        if metadata.update_digest in seen_digests:
            outcomes.append(
                FederatedUpdateOutcome(
                    index,
                    FederatedUpdateBatchStatus.REJECTED,
                    FederatedUpdateBatchReasonCode.UPDATE_DUPLICATE,
                )
            )
            continue
        seen_digests.add(metadata.update_digest)
        accepted_digests.append(metadata.update_digest)
        accepted_by_index[index] = (fingerprint, metadata.update_digest)
        by_fingerprint.setdefault(fingerprint, []).append(metadata.update_digest)
        outcomes.append(
            FederatedUpdateOutcome(
                index,
                FederatedUpdateBatchStatus.ACCEPTED,
                FederatedUpdateBatchReasonCode.UPDATE_ACCEPTED,
                fingerprint=fingerprint,
            )
        )

    groups: list[FederatedUpdateGroup] = []
    suppressed_groups = 0
    for fingerprint in sorted(by_fingerprint):
        digests = by_fingerprint[fingerprint]
        suppressed = len(digests) < policy.minimum_group_size
        if suppressed:
            suppressed_groups += 1
        groups.append(
            FederatedUpdateGroup(
                fingerprint=fingerprint,
                accepted=len(digests),
                suppressed=suppressed,
                update_digests=(
                    ()
                    if suppressed or not policy.disclose_digests
                    else tuple(sorted(digests))
                ),
            )
        )

    if policy.disclose_digests:
        disclosed_fingerprints = {
            group.fingerprint for group in groups if not group.suppressed
        }
        outcomes = [
            replace(outcome, update_digest=accepted_by_index[outcome.index][1])
            if outcome.index in accepted_by_index
            and accepted_by_index[outcome.index][0] in disclosed_fingerprints
            else outcome
            for outcome in outcomes
        ]

    counts: dict[FederatedUpdateBatchReasonCode, int] = {}
    for outcome in outcomes:
        counts[outcome.reason] = counts.get(outcome.reason, 0) + 1
    if suppressed_groups:
        counts[FederatedUpdateBatchReasonCode.BATCH_GROUP_SUPPRESSED] = (
            suppressed_groups
        )
    disclosed = sum(len(group.update_digests) for group in groups)
    withheld = len(accepted_digests) - disclosed
    if withheld:
        counts[FederatedUpdateBatchReasonCode.BATCH_DIGEST_WITHHELD] = withheld
    if received == 0:
        counts[FederatedUpdateBatchReasonCode.BATCH_EMPTY] = 1
    findings = tuple(
        FederatedUpdateBatchFinding(reason, _STATUS_BY_REASON[reason], counts[reason])
        for reason in FEDERATED_UPDATE_BATCH_REASON_CODES
        if reason in counts
    )

    accepted = counts.get(FederatedUpdateBatchReasonCode.UPDATE_ACCEPTED, 0)
    return FederatedUpdateBatchReport(
        received=received,
        accepted=accepted,
        rejected=len(outcomes) - accepted,
        suppressed_groups=suppressed_groups,
        limit_exceeded=False,
        findings=findings,
        groups=tuple(groups),
        outcomes=tuple(outcomes),
    )


def _parse_envelope(
    payload: object, policy: FederatedUpdatePolicy
) -> FederatedUpdateMetadata:
    if type(payload) is str:
        return FederatedUpdateMetadata.from_json(payload, policy=policy)
    return FederatedUpdateMetadata.from_dict(payload, policy=policy)


def _require_bounded_int(value: object, minimum: int, maximum: int, name: str) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise FederatedUpdateBatchError(
            f"{name} must be an integer between {minimum} and {maximum}"
        )
    return value


__all__ = [
    "DEFAULT_FEDERATED_UPDATE_MINIMUM_GROUP_SIZE",
    "DEFAULT_MAX_BATCH_UPDATES",
    "FEDERATED_UPDATE_BATCH_REASON_CODES",
    "FEDERATED_UPDATE_BATCH_SCHEMA_VERSION",
    "MAX_BATCH_UPDATES",
    "FederatedUpdateBatchError",
    "FederatedUpdateBatchFinding",
    "FederatedUpdateBatchPolicy",
    "FederatedUpdateBatchReasonCode",
    "FederatedUpdateBatchReport",
    "FederatedUpdateBatchStatus",
    "FederatedUpdateGroup",
    "FederatedUpdateOutcome",
    "check_federated_update_batch",
]
