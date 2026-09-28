"""Privacy-safe operational telemetry contracts.

The types in this module deliberately describe work rather than clinical
values.  They are safe to use as metric labels, trace attributes, alert
inputs, and audit metadata because every string is selected from a closed
vocabulary and every numeric field is an aggregate.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Final, Mapping

OPERATIONAL_SCHEMA_VERSION: Final = "1.0.0"
OPERATIONAL_COMPATIBILITY: Final = "same_major"


class OperationalCategory(str, Enum):
    """Bounded operational areas available to telemetry backends."""

    INGESTION = "ingestion"
    MODEL = "model"
    STORE = "store"
    JOB = "job"
    QUEUE = "queue"
    QUERY = "query"
    EXPORT = "export"


class OperationalState(str, Enum):
    """Non-lossy outcomes shared by metrics, traces, and alerts."""

    SUCCESS = "success"
    EMPTY = "empty"
    PARTIAL = "partial"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"
    UNSUPPORTED = "unsupported"
    DENIED = "denied"
    FAILURE = "failure"


OPERATIONS_BY_CATEGORY: Final[Mapping[OperationalCategory, frozenset[str]]] = (
    MappingProxyType(
        {
            OperationalCategory.INGESTION: frozenset(
                {"intake", "parse", "normalize", "commit"}
            ),
            OperationalCategory.MODEL: frozenset(
                {"load", "infer", "unload", "evaluate"}
            ),
            OperationalCategory.STORE: frozenset(
                {"read", "write", "correct", "delete"}
            ),
            OperationalCategory.JOB: frozenset({"create", "run", "cancel", "complete"}),
            OperationalCategory.QUEUE: frozenset(
                {"enqueue", "dispatch", "expire", "shed"}
            ),
            OperationalCategory.QUERY: frozenset(
                {"list", "resolve", "search", "aggregate"}
            ),
            OperationalCategory.EXPORT: frozenset(
                {"authorize", "prepare", "write", "download"}
            ),
        }
    )
)


@dataclass(frozen=True, slots=True)
class OperationalEvent:
    """One value-free, versioned operational observation."""

    category: OperationalCategory
    operation: str
    state: OperationalState
    count: int = 1
    duration_seconds: float = 0.0
    schema_version: str = OPERATIONAL_SCHEMA_VERSION
    compatibility_policy: str = OPERATIONAL_COMPATIBILITY

    def __post_init__(self) -> None:
        try:
            category = OperationalCategory(self.category)
            state = OperationalState(self.state)
        except (TypeError, ValueError):
            raise ValueError("operational category or state is unsupported") from None
        if self.operation not in OPERATIONS_BY_CATEGORY[category]:
            raise ValueError("operation is not valid for the operational category")
        if type(self.count) is not int or not 1 <= self.count <= 1_000_000:
            raise ValueError("operational count must stay within aggregate bounds")
        if (
            type(self.duration_seconds) not in {int, float}
            or not math.isfinite(self.duration_seconds)
            or not 0 <= self.duration_seconds <= 86_400
        ):
            raise ValueError("operational duration must stay within safe bounds")
        if self.schema_version != OPERATIONAL_SCHEMA_VERSION:
            raise ValueError("unsupported operational telemetry schema version")
        if self.compatibility_policy != OPERATIONAL_COMPATIBILITY:
            raise ValueError("unsupported operational compatibility policy")
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "duration_seconds", float(self.duration_seconds))

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic representation containing no work values."""

        return {
            "category": self.category.value,
            "compatibility_policy": self.compatibility_policy,
            "count": self.count,
            "duration_seconds": self.duration_seconds,
            "operation": self.operation,
            "schema_version": self.schema_version,
            "state": self.state.value,
        }


@dataclass(frozen=True, slots=True)
class OperationalAlert:
    """Aggregate alert suitable for logs, paging, or an audit stream."""

    alert_id: str
    category: OperationalCategory
    operation: str
    state: OperationalState
    observed_count: int
    threshold_count: int
    window_seconds: int
    schema_version: str = OPERATIONAL_SCHEMA_VERSION
    compatibility_policy: str = OPERATIONAL_COMPATIBILITY

    def __post_init__(self) -> None:
        if re.fullmatch(r"alert_[0-9a-f]{16,64}", self.alert_id) is None:
            raise ValueError("operational alert ID must be opaque")
        OperationalEvent(
            category=self.category,
            operation=self.operation,
            state=self.state,
        )
        for value in (self.observed_count, self.threshold_count, self.window_seconds):
            if type(value) is not int or value < 1 or value > 86_400_000:
                raise ValueError("operational alert counts must stay within bounds")
        if self.schema_version != OPERATIONAL_SCHEMA_VERSION:
            raise ValueError("unsupported operational alert schema version")
        if self.compatibility_policy != OPERATIONAL_COMPATIBILITY:
            raise ValueError("unsupported operational compatibility policy")
        object.__setattr__(self, "category", OperationalCategory(self.category))
        object.__setattr__(self, "state", OperationalState(self.state))

    def to_dict(self) -> dict[str, object]:
        """Return counts and controlled labels only."""

        return {
            "alert_id": self.alert_id,
            "category": self.category.value,
            "compatibility_policy": self.compatibility_policy,
            "observed_count": self.observed_count,
            "operation": self.operation,
            "schema_version": self.schema_version,
            "state": self.state.value,
            "threshold_count": self.threshold_count,
            "window_seconds": self.window_seconds,
        }


__all__ = [
    "OPERATIONS_BY_CATEGORY",
    "OPERATIONAL_COMPATIBILITY",
    "OPERATIONAL_SCHEMA_VERSION",
    "OperationalAlert",
    "OperationalCategory",
    "OperationalEvent",
    "OperationalState",
]
