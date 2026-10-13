"""Count-only consistency checks before a synthetic federated round is enrolled."""

from __future__ import annotations

import json
from dataclasses import dataclass

MAX_COUNT = 2**31 - 1
_REJECT_REASONS = frozenset(
    {
        "malformed_count",
        "count_overflow",
        "arithmetic_overflow",
        "dropout_exceeds_participants",
        "under_quorum",
        "unrecoverable_dropout",
        "unsafe_small_group",
        "threshold_exceeds_quorum",
        "minimum_group_exceeds_quorum",
    }
)


@dataclass(frozen=True, slots=True)
class SecureAggregationPreflightReport:
    """Immutable controlled outcome without participants, shares, keys or payloads.

    Acceptance means only that the caller's declared integer relationships have
    positive headroom. It is not cryptographic, privacy or production approval.
    """

    outcome: str
    reasons: tuple[str, ...]
    worst_case_survivors: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, str) or self.outcome not in {
            "accept",
            "review",
            "reject",
        }:
            raise ValueError("invalid secure aggregation preflight outcome")
        if not isinstance(self.reasons, (tuple, list)) or len(self.reasons) > len(
            _REJECT_REASONS
        ):
            raise ValueError("invalid secure aggregation preflight reasons")
        reasons = tuple(self.reasons)
        if any(not isinstance(reason, str) for reason in reasons):
            raise ValueError("invalid secure aggregation preflight reasons")
        if len(set(reasons)) != len(reasons):
            raise ValueError("duplicate secure aggregation preflight reason")
        allowed = (
            ()
            if self.outcome == "accept"
            else ("zero_headroom",)
            if self.outcome == "review"
            else _REJECT_REASONS
        )
        if any(reason not in allowed for reason in reasons) or (
            self.outcome != "accept" and not reasons
        ):
            raise ValueError("inconsistent secure aggregation preflight reasons")
        if self.worst_case_survivors is not None and (
            type(self.worst_case_survivors) is not int
            or not 0 <= self.worst_case_survivors <= MAX_COUNT
        ):
            raise ValueError("invalid secure aggregation survivor count")
        if self.outcome in {"accept", "review"} and (
            self.worst_case_survivors is None or self.worst_case_survivors < 2
        ):
            raise ValueError("inconsistent secure aggregation survivor count")
        object.__setattr__(self, "reasons", tuple(sorted(reasons)))

    def to_dict(self) -> dict[str, object]:
        """Return a versioned, content-free report with controlled reason codes."""
        return {
            "schema_version": 1,
            "outcome": self.outcome,
            "reasons": list(self.reasons),
            "worst_case_survivors": self.worst_case_survivors,
        }

    def to_json(self) -> str:
        """Serialize byte-stable JSON without any caller-supplied free text."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"


def secure_aggregation_preflight(
    *,
    participant_count: int,
    quorum: int,
    share_threshold: int,
    dropout_bound: int,
    minimum_aggregation_group: int,
) -> SecureAggregationPreflightReport:
    """Check declared anonymous counts without enrollment or cryptography.

    Args:
        participant_count: Anonymous maximum round participation, positive.
        quorum: Required surviving participant count, positive.
        share_threshold: Required surviving recovery-share count, positive.
        dropout_bound: Maximum expected dropouts, nonnegative.
        minimum_aggregation_group: Declared minimum release group, at least two,
            consistent with OpenMed's existing aggregate count-release contract.

    Returns:
        Reject for malformed or inconsistent relationships; review at exact
        zero-headroom boundaries; accept for consistent positive headroom.
        Counts use a signed 32-bit maximum and checked requirement-plus-dropout
        arithmetic. This does not choose production thresholds, establish
        protocol security or certify the declared dropout assumption.
    """
    counts = (
        participant_count,
        quorum,
        share_threshold,
        dropout_bound,
        minimum_aggregation_group,
    )
    if (
        any(type(value) is not int for value in counts)
        or any(value < 0 for value in counts)
        or any(value == 0 for value in (participant_count, quorum, share_threshold))
    ):
        return SecureAggregationPreflightReport("reject", ("malformed_count",))
    if any(value > MAX_COUNT for value in counts):
        return SecureAggregationPreflightReport("reject", ("count_overflow",))
    if minimum_aggregation_group < 2:
        return SecureAggregationPreflightReport("reject", ("unsafe_small_group",))
    if dropout_bound > participant_count:
        return SecureAggregationPreflightReport(
            "reject", ("dropout_exceeds_participants",)
        )
    requirements = (quorum, share_threshold, minimum_aggregation_group)
    if any(value > MAX_COUNT - dropout_bound for value in requirements):
        return SecureAggregationPreflightReport("reject", ("arithmetic_overflow",))
    # Addition is performed only after proving every sum fits the declared width.
    required_participation = tuple(value + dropout_bound for value in requirements)
    survivors = participant_count - dropout_bound
    reasons = []
    for required, reason in zip(
        required_participation,
        ("under_quorum", "unrecoverable_dropout", "unsafe_small_group"),
    ):
        if participant_count < required:
            reasons.append(reason)
    if share_threshold > quorum:
        reasons.append("threshold_exceeds_quorum")
    if minimum_aggregation_group > quorum:
        reasons.append("minimum_group_exceeds_quorum")
    if reasons:
        return SecureAggregationPreflightReport("reject", tuple(reasons), survivors)
    if any(participant_count == required for required in required_participation):
        return SecureAggregationPreflightReport("review", ("zero_headroom",), survivors)
    return SecureAggregationPreflightReport("accept", (), survivors)
