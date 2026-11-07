"""Deterministic, resource-bounded batch planning for local clinical NLI."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final

NLI_BATCH_PLAN_SCHEMA_VERSION: Final[int] = 1


class NliBatchPlanningError(ValueError):
    """Raised when a batch-planning input violates the value-free contract."""


@dataclass(frozen=True)
class NliRuntimeProfile:
    """Resource ceilings supplied by a local NLI runtime.

    Per-batch ceilings are required. Optional total ceilings represent a hard
    invocation budget; pairs that cannot fit are returned as deferred.
    """

    max_tokens_per_batch: int
    max_pairs_per_batch: int
    max_total_tokens: int | None = None
    max_total_pairs: int | None = None

    def __post_init__(self) -> None:
        _require_positive_int(self.max_tokens_per_batch, "token ceiling")
        _require_positive_int(self.max_pairs_per_batch, "pair ceiling")
        _require_optional_nonnegative_int(self.max_total_tokens, "total token budget")
        _require_optional_nonnegative_int(self.max_total_pairs, "total pair budget")

    @classmethod
    def from_obj(
        cls, value: NliRuntimeProfile | Mapping[str, object]
    ) -> NliRuntimeProfile:
        """Coerce a runtime profile from an instance or mapping."""

        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise NliBatchPlanningError("runtime profile must be a mapping")
        try:
            return cls(
                max_tokens_per_batch=value["max_tokens_per_batch"],  # type: ignore[arg-type]
                max_pairs_per_batch=value["max_pairs_per_batch"],  # type: ignore[arg-type]
                max_total_tokens=value.get("max_total_tokens"),  # type: ignore[arg-type]
                max_total_pairs=value.get("max_total_pairs"),  # type: ignore[arg-type]
            )
        except KeyError as exc:
            raise NliBatchPlanningError(
                "runtime profile is missing a required ceiling"
            ) from exc


@dataclass(frozen=True, repr=False)
class NliPairCost:
    """Opaque pair identifier and caller-computed local token estimate."""

    pair_id: str = field(repr=False)
    token_count: int

    def __post_init__(self) -> None:
        if type(self.pair_id) is not str or not self.pair_id.strip():
            raise NliBatchPlanningError("pair identifier must be a non-empty string")
        try:
            self.pair_id.encode("utf-8")
        except UnicodeEncodeError:
            raise NliBatchPlanningError(
                "pair identifier must be valid Unicode"
            ) from None
        _require_positive_int(self.token_count, "pair token count")

    def __repr__(self) -> str:
        """Return a representation that does not expose the pair identifier."""

        return f"NliPairCost(token_count={self.token_count})"

    @classmethod
    def from_obj(cls, value: NliPairCost | Mapping[str, object]) -> NliPairCost:
        """Coerce a pair cost without accepting premise or hypothesis text."""

        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise NliBatchPlanningError("pair cost must be a mapping")
        try:
            return cls(
                pair_id=value["pair_id"],  # type: ignore[arg-type]
                token_count=value["token_count"],  # type: ignore[arg-type]
            )
        except KeyError as exc:
            raise NliBatchPlanningError(
                "pair cost is missing a required field"
            ) from exc


@dataclass(frozen=True, repr=False)
class PlannedNliBatch:
    """One ordered batch within a resource-bounded plan."""

    index: int
    pair_ids: tuple[str, ...] = field(repr=False)
    token_count: int

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise NliBatchPlanningError("batch index must be a non-negative integer")
        if not self.pair_ids:
            raise NliBatchPlanningError("planned batch cannot be empty")
        if type(self.token_count) is not int or self.token_count < 1:
            raise NliBatchPlanningError("batch token count must be positive")

    @property
    def pair_count(self) -> int:
        """Return the number of pairs in this batch."""

        return len(self.pair_ids)

    def __repr__(self) -> str:
        """Return a representation containing resource counts only."""

        return (
            "PlannedNliBatch("
            f"index={self.index}, pair_count={self.pair_count}, "
            f"token_count={self.token_count})"
        )


@dataclass(frozen=True, repr=False)
class NliBatchPlan:
    """Deterministic batches plus identifiers deferred by a hard budget."""

    batches: tuple[PlannedNliBatch, ...]
    deferred_pair_ids: tuple[str, ...] = field(repr=False)
    schema_version: int = NLI_BATCH_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != NLI_BATCH_PLAN_SCHEMA_VERSION:
            raise NliBatchPlanningError("unsupported batch-plan schema")

    @property
    def planned_pair_ids(self) -> tuple[str, ...]:
        """Return planned identifiers in deterministic execution order."""

        return tuple(pair_id for batch in self.batches for pair_id in batch.pair_ids)

    @property
    def token_count(self) -> int:
        """Return the total token estimate admitted to the plan."""

        return sum(batch.token_count for batch in self.batches)

    def to_audit_dict(self) -> dict[str, object]:
        """Return a JSON-safe report with domain-separated identifier hashes."""

        return {
            "schema_version": self.schema_version,
            "batch_count": len(self.batches),
            "planned_pair_count": len(self.planned_pair_ids),
            "deferred_pair_count": len(self.deferred_pair_ids),
            "token_count": self.token_count,
            "batches": [
                {
                    "index": batch.index,
                    "pair_count": batch.pair_count,
                    "token_count": batch.token_count,
                    "pair_fingerprints": [
                        _fingerprint(pair_id) for pair_id in batch.pair_ids
                    ],
                }
                for batch in self.batches
            ],
            "deferred_pair_fingerprints": [
                _fingerprint(pair_id) for pair_id in self.deferred_pair_ids
            ],
        }

    def __repr__(self) -> str:
        """Return a representation containing aggregate counts only."""

        return (
            "NliBatchPlan("
            f"batch_count={len(self.batches)}, "
            f"planned_pair_count={len(self.planned_pair_ids)}, "
            f"deferred_pair_count={len(self.deferred_pair_ids)}, "
            f"token_count={self.token_count})"
        )


def plan_nli_batches(
    pairs: Iterable[NliPairCost | Mapping[str, Any]],
    profile: NliRuntimeProfile | Mapping[str, object],
) -> NliBatchPlan:
    """Plan ordered local-inference batches under pair and token ceilings.

    The planner is a pure greedy pass over caller order. A pair is deferred if
    it cannot fit in any batch or if admitting it would exceed an optional hard
    total budget. Deferred pairs do not prevent later, smaller pairs from being
    considered.

    Args:
        pairs: Opaque pair identifiers with positive local token estimates.
        profile: Required per-batch ceilings and optional hard total ceilings.

    Returns:
        Immutable ordered batches and deferred pair identifiers.

    Raises:
        NliBatchPlanningError: If inputs are malformed or identifiers repeat.
    """

    runtime = NliRuntimeProfile.from_obj(profile)
    costs = tuple(NliPairCost.from_obj(value) for value in pairs)
    _validate_unique_ids(costs)

    batches: list[PlannedNliBatch] = []
    deferred: list[str] = []
    current_ids: list[str] = []
    current_tokens = 0
    admitted_pairs = 0
    admitted_tokens = 0

    def flush() -> None:
        nonlocal current_ids, current_tokens
        if not current_ids:
            return
        batches.append(
            PlannedNliBatch(
                index=len(batches),
                pair_ids=tuple(current_ids),
                token_count=current_tokens,
            )
        )
        current_ids = []
        current_tokens = 0

    for cost in costs:
        if cost.token_count > runtime.max_tokens_per_batch:
            deferred.append(cost.pair_id)
            continue
        if (
            runtime.max_total_pairs is not None
            and admitted_pairs >= runtime.max_total_pairs
        ):
            deferred.append(cost.pair_id)
            continue
        if (
            runtime.max_total_tokens is not None
            and admitted_tokens + cost.token_count > runtime.max_total_tokens
        ):
            deferred.append(cost.pair_id)
            continue

        if current_ids and (
            len(current_ids) >= runtime.max_pairs_per_batch
            or current_tokens + cost.token_count > runtime.max_tokens_per_batch
        ):
            flush()

        current_ids.append(cost.pair_id)
        current_tokens += cost.token_count
        admitted_pairs += 1
        admitted_tokens += cost.token_count

    flush()
    return NliBatchPlan(
        batches=tuple(batches),
        deferred_pair_ids=tuple(deferred),
    )


def _validate_unique_ids(costs: tuple[NliPairCost, ...]) -> None:
    seen: set[str] = set()
    for cost in costs:
        if cost.pair_id in seen:
            raise NliBatchPlanningError("pair identifiers must be unique")
        seen.add(cost.pair_id)


def _require_positive_int(value: object, field_name: str) -> None:
    if type(value) is not int or value < 1:
        raise NliBatchPlanningError(f"{field_name} must be a positive integer")


def _require_optional_nonnegative_int(value: object, field_name: str) -> None:
    if value is not None and (type(value) is not int or value < 0):
        raise NliBatchPlanningError(
            f"{field_name} must be a non-negative integer or null"
        )


def _fingerprint(pair_id: str) -> str:
    return hashlib.sha256(
        b"openmed:nli-batch-plan:v1\0" + pair_id.encode("utf-8")
    ).hexdigest()


__all__ = [
    "NLI_BATCH_PLAN_SCHEMA_VERSION",
    "NliBatchPlan",
    "NliBatchPlanningError",
    "NliPairCost",
    "NliRuntimeProfile",
    "PlannedNliBatch",
    "plan_nli_batches",
]
