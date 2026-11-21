"""Optional, in-memory counts of closed-vocabulary human-factors use errors."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from threading import Lock

from openmed.core.intended_use import RegulatoryTier, load_intended_use_registry
from openmed.core.no_phi_telemetry import (
    MAX_COUNTER_VALUE,
    MAX_SNAPSHOT_SAMPLES,
    _use_error_sink,
)


class UseErrorCode(StrEnum):
    """Stable event codes shared by safety primitives and kit integrations."""

    ACKNOWLEDGEMENT_SKIPPED = "acknowledgement_skipped"
    RESTRICTED_CAPABILITY_REFUSED = "restricted_capability_refused"
    EXPORT_BEFORE_REVIEW = "export_before_review"


class UseErrorRecorder:
    """Aggregate counts in memory with no event log, persistence or transport."""

    def __init__(self) -> None:
        self._counts: Counter[tuple[str, str, str]] = Counter()
        self._lock = Lock()

    def record(
        self, code: UseErrorCode | str, capability: str, tier: RegulatoryTier | str
    ) -> None:
        """Record one event using only registered codes and a known tier."""
        try:
            code = UseErrorCode(code)
            tier = RegulatoryTier(tier)
        except (TypeError, ValueError):
            raise ValueError("unknown_use_error_dimension") from None
        registry = load_intended_use_registry()
        if type(capability) is not str:
            raise ValueError("unknown_use_error_capability")
        capability = registry.entry_points.get(capability, capability)
        if capability not in registry.declarations and capability not in {
            "clinician_mode",
            "notice",
        }:
            raise ValueError("unknown_use_error_capability")
        key = (code.value, capability, tier.value)
        with self._lock:
            if key not in self._counts and len(self._counts) >= MAX_SNAPSHOT_SAMPLES:
                raise ValueError("use_error_dimension_limit")
            if self._counts[key] >= MAX_COUNTER_VALUE:
                raise ValueError("use_error_counter_limit")
            self._counts[key] += 1

    @contextmanager
    def capture(self) -> Iterator[UseErrorRecorder]:
        """Count primitive refusals in this context until the block exits."""
        active = True

        def record(code: str, capability: str, tier: str) -> None:
            if active:
                self.record(code, capability, tier)

        token = _use_error_sink.set(record)
        try:
            yield self
        finally:
            active = False
            _use_error_sink.reset(token)

    def snapshot(self) -> dict:
        """Return a detached, sorted snapshot with counts and bounded dimensions."""
        with self._lock:
            return {
                "schema_version": 1,
                "counts": [
                    {
                        "code": code,
                        "capability": capability,
                        "tier": tier,
                        "count": count,
                    }
                    for (code, capability, tier), count in sorted(self._counts.items())
                ],
            }
