"""Privacy-safe operator summaries for federated training rounds."""

from __future__ import annotations

import json
import re
from dataclasses import InitVar, dataclass
from enum import Enum
from typing import Any, Final, Sequence

from .federated_round import FederatedRoundState

FEDERATED_ROUND_STATUS_SCHEMA_VERSION = "openmed.training.federated_status.v1"
DEFAULT_FEDERATED_MINIMUM_GROUP_SIZE = 5

_SHA256_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_STATUS_BUILDER_TOKEN: Final = object()


class FederatedQuorumStatus(str, Enum):
    """Whether the aggregate participant count satisfies quorum."""

    MET = "met"
    NOT_MET = "not_met"

    def __str__(self) -> str:
        return self.value


class FederatedCompletionBand(str, Enum):
    """A coarse, non-percentage completion category."""

    NOT_STARTED = "not_started"
    UNDER_HALF = "under_half"
    HALF_OR_MORE = "half_or_more"
    COMPLETE = "complete"

    def __str__(self) -> str:
        return self.value


class FederatedRoundReasonCode(str, Enum):
    """Stable, categorical reasons for held and aborted rounds."""

    PRIVACY_REVIEW_REQUIRED = "privacy_review_required"
    QUALITY_REVIEW_REQUIRED = "quality_review_required"
    POLICY_REVIEW_REQUIRED = "policy_review_required"
    QUORUM_NOT_MET = "quorum_not_met"
    PRIVACY_GATE_FAILED = "privacy_gate_failed"
    QUALITY_GATE_FAILED = "quality_gate_failed"
    POLICY_GATE_FAILED = "policy_gate_failed"
    OPERATOR_CANCELLED = "operator_cancelled"
    ROUND_ERROR = "round_error"

    def __str__(self) -> str:
        return self.value


_HOLD_REASON_CODES: Final[frozenset[FederatedRoundReasonCode]] = frozenset(
    {
        FederatedRoundReasonCode.PRIVACY_REVIEW_REQUIRED,
        FederatedRoundReasonCode.QUALITY_REVIEW_REQUIRED,
        FederatedRoundReasonCode.POLICY_REVIEW_REQUIRED,
    }
)
_ABORT_REASON_CODES: Final[frozenset[FederatedRoundReasonCode]] = frozenset(
    {
        FederatedRoundReasonCode.QUORUM_NOT_MET,
        FederatedRoundReasonCode.PRIVACY_GATE_FAILED,
        FederatedRoundReasonCode.QUALITY_GATE_FAILED,
        FederatedRoundReasonCode.POLICY_GATE_FAILED,
        FederatedRoundReasonCode.OPERATOR_CANCELLED,
        FederatedRoundReasonCode.ROUND_ERROR,
    }
)


class FederatedRoundStatusError(ValueError):
    """Raised when a round summary cannot be produced safely."""


@dataclass(frozen=True)
class FederatedRoundStatus:
    """An immutable summary returned by :func:`build_federated_round_status`."""

    state: FederatedRoundState
    quorum_status: FederatedQuorumStatus
    participant_count: int | None
    completed_participant_count: int | None
    minimum_group_size: int
    completion_band: FederatedCompletionBand
    aggregate_digest_refs: tuple[str, ...] = ()
    reason_code: FederatedRoundReasonCode | None = None
    schema_version: str = FEDERATED_ROUND_STATUS_SCHEMA_VERSION
    _builder_token: InitVar[object | None] = None

    def __post_init__(self, _builder_token: object | None) -> None:
        if _builder_token is not _STATUS_BUILDER_TOKEN:
            raise FederatedRoundStatusError(
                "federated round status must be built from aggregate inputs"
            )
        if not isinstance(self.state, FederatedRoundState):
            raise FederatedRoundStatusError("invalid federated round state")
        if not isinstance(self.quorum_status, FederatedQuorumStatus):
            raise FederatedRoundStatusError("invalid quorum status")
        _require_minimum_group_size(self.minimum_group_size)
        _require_released_count(self.participant_count, self.minimum_group_size)
        _require_released_count(
            self.completed_participant_count, self.minimum_group_size
        )
        if (
            self.participant_count is None
            and self.completed_participant_count is not None
        ):
            raise FederatedRoundStatusError(
                "completed participant count requires a released participant count"
            )
        if (
            self.participant_count is not None
            and self.completed_participant_count is not None
            and self.completed_participant_count > self.participant_count
        ):
            raise FederatedRoundStatusError(
                "completed participant count exceeds participant count"
            )
        if not isinstance(self.completion_band, FederatedCompletionBand):
            raise FederatedRoundStatusError("invalid completion band")
        if (
            type(self.schema_version) is not str
            or self.schema_version != FEDERATED_ROUND_STATUS_SCHEMA_VERSION
        ):
            raise FederatedRoundStatusError("unsupported federated round status schema")
        _require_digest_refs(self.aggregate_digest_refs)
        _require_reason_code(self.state, self.reason_code)

    def to_dict(self) -> dict[str, Any]:
        """Return a privacy-safe deterministic mapping."""

        return {
            "aggregate_digest_refs": list(self.aggregate_digest_refs),
            "completed_participant_count": self.completed_participant_count,
            "completion_band": self.completion_band.value,
            "minimum_group_size": self.minimum_group_size,
            "participant_count": self.participant_count,
            "quorum_status": self.quorum_status.value,
            "reason_code": (
                self.reason_code.value if self.reason_code is not None else None
            ),
            "schema_version": self.schema_version,
            "state": self.state.value,
        }

    def to_json(self) -> str:
        """Return byte-stable JSON with a trailing newline."""

        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_markdown(self) -> str:
        """Return a byte-stable Markdown operator summary."""

        reason = self.reason_code.value if self.reason_code is not None else "none"
        lines = [
            "# Federated round status",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| Schema | `{self.schema_version}` |",
            f"| State | `{self.state.value}` |",
            f"| Quorum | `{self.quorum_status.value}` |",
            (
                "| Participants | "
                f"`{_format_count(self.participant_count, self.minimum_group_size)}` |"
            ),
            (
                "| Completed participants | "
                f"`{_format_count(self.completed_participant_count, self.minimum_group_size)}` |"
            ),
            f"| Completion | `{self.completion_band.value}` |",
            f"| Reason | `{reason}` |",
            "",
            "## Aggregate digests",
            "",
        ]
        if self.aggregate_digest_refs:
            lines.extend(f"- `{digest}`" for digest in self.aggregate_digest_refs)
        else:
            lines.append("None.")
        return "\n".join(lines) + "\n"


def build_federated_round_status(
    *,
    state: FederatedRoundState,
    participant_count: int,
    completed_participant_count: int,
    required_quorum: int,
    minimum_group_size: int = DEFAULT_FEDERATED_MINIMUM_GROUP_SIZE,
    aggregate_digest_refs: Sequence[str] = (),
    reason_code: FederatedRoundReasonCode | None = None,
) -> FederatedRoundStatus:
    """Build a summary while discarding exact sub-threshold counts."""

    if not isinstance(state, FederatedRoundState):
        raise FederatedRoundStatusError("invalid federated round state")
    _require_non_negative_int(participant_count, "participant count")
    _require_non_negative_int(
        completed_participant_count, "completed participant count"
    )
    _require_positive_int(required_quorum, "required quorum")
    _require_minimum_group_size(minimum_group_size)
    if completed_participant_count > participant_count:
        raise FederatedRoundStatusError(
            "completed participant count exceeds participant count"
        )
    digest_refs = _normalize_digest_refs(aggregate_digest_refs)
    _require_reason_code(state, reason_code)

    return FederatedRoundStatus(
        state=state,
        quorum_status=(
            FederatedQuorumStatus.MET
            if participant_count >= required_quorum
            else FederatedQuorumStatus.NOT_MET
        ),
        participant_count=_threshold_count(participant_count, minimum_group_size),
        completed_participant_count=_threshold_count(
            completed_participant_count, minimum_group_size
        ),
        minimum_group_size=minimum_group_size,
        completion_band=_completion_band(
            participant_count, completed_participant_count
        ),
        aggregate_digest_refs=digest_refs,
        reason_code=reason_code,
        _builder_token=_STATUS_BUILDER_TOKEN,
    )


def _threshold_count(count: int, minimum_group_size: int) -> int | None:
    return count if count >= minimum_group_size else None


def _completion_band(
    participant_count: int, completed_participant_count: int
) -> FederatedCompletionBand:
    if completed_participant_count == 0:
        return FederatedCompletionBand.NOT_STARTED
    if completed_participant_count == participant_count:
        return FederatedCompletionBand.COMPLETE
    if completed_participant_count * 2 < participant_count:
        return FederatedCompletionBand.UNDER_HALF
    return FederatedCompletionBand.HALF_OR_MORE


def _normalize_digest_refs(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FederatedRoundStatusError("invalid aggregate digest references")
    normalized = tuple(value)
    if any(
        type(digest) is not str or _SHA256_DIGEST.fullmatch(digest) is None
        for digest in normalized
    ):
        raise FederatedRoundStatusError("invalid aggregate digest references")
    if len(set(normalized)) != len(normalized):
        raise FederatedRoundStatusError("aggregate digest references are not canonical")
    normalized = tuple(sorted(normalized))
    _require_digest_refs(normalized)
    return normalized


def _require_digest_refs(value: object) -> None:
    if type(value) is not tuple:
        raise FederatedRoundStatusError("invalid aggregate digest references")
    if any(
        type(digest) is not str or _SHA256_DIGEST.fullmatch(digest) is None
        for digest in value
    ):
        raise FederatedRoundStatusError("invalid aggregate digest references")
    if tuple(sorted(value)) != value or len(set(value)) != len(value):
        raise FederatedRoundStatusError("aggregate digest references are not canonical")


def _require_reason_code(
    state: FederatedRoundState, reason_code: FederatedRoundReasonCode | None
) -> None:
    if reason_code is not None and not isinstance(
        reason_code, FederatedRoundReasonCode
    ):
        raise FederatedRoundStatusError("invalid federated round reason code")
    if state is FederatedRoundState.HELD:
        if reason_code not in _HOLD_REASON_CODES:
            raise FederatedRoundStatusError("held round requires a hold reason code")
        return
    if state is FederatedRoundState.ABORTED:
        if reason_code not in _ABORT_REASON_CODES:
            raise FederatedRoundStatusError(
                "aborted round requires an abort reason code"
            )
        return
    if reason_code is not None:
        raise FederatedRoundStatusError(
            "reason code is only valid for held or aborted rounds"
        )


def _require_non_negative_int(value: object, field: str) -> None:
    if type(value) is not int or value < 0:
        raise FederatedRoundStatusError(f"{field} must be a non-negative integer")


def _require_positive_int(value: object, field: str) -> None:
    if type(value) is not int or value < 1:
        raise FederatedRoundStatusError(f"{field} must be a positive integer")


def _require_minimum_group_size(value: object) -> None:
    if type(value) is not int or value < 2:
        raise FederatedRoundStatusError(
            "minimum group size must be an integer greater than one"
        )


def _require_released_count(value: object, minimum_group_size: int) -> None:
    if value is None:
        return
    if type(value) is not int or value < minimum_group_size:
        raise FederatedRoundStatusError("invalid released participant count")


def _format_count(count: int | None, minimum_group_size: int) -> str:
    if count is None:
        return f"suppressed (<{minimum_group_size})"
    return str(count)


__all__ = [
    "DEFAULT_FEDERATED_MINIMUM_GROUP_SIZE",
    "FEDERATED_ROUND_STATUS_SCHEMA_VERSION",
    "FederatedCompletionBand",
    "FederatedQuorumStatus",
    "FederatedRoundReasonCode",
    "FederatedRoundStatus",
    "FederatedRoundStatusError",
    "build_federated_round_status",
]
