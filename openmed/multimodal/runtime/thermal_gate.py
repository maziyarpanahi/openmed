"""Gate local execution and actual output release on sustained edge budgets.

This cooperative controller never loads a model or samples a platform sensor.
Hosts provide fresh observations in the same monotonic clock domain and already
qualified local modes. A declaration is not a sandbox or a clinical certificate.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Generic, TypeVar, cast

from openmed.multimodal.capability_matrix import (
    ModalityProfile,
    ProviderCapabilityEnvelope,
    capability_matrix,
)
from openmed.multimodal.streaming.privacy.withholding_buffer import (
    BufferState,
    ReleasedSpeechSegment,
    WithholdingBuffer,
)
from openmed.multimodal.vision.providers.base import (
    EdgeVLMProvider,
    LocalProviderMetadata,
    VLMRequest,
    VLMResult,
    run_local_vlm,
)

T = TypeVar("T")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class ThermalGateError(ValueError):
    """Value-free invalid configuration or operation."""


def _require(valid: bool) -> None:
    if not valid:
        raise ThermalGateError("invalid thermal gate contract")


def _number(value: object, maximum: float, *, positive: bool = False) -> None:
    _require(type(value) in (int, float))
    number = cast("int | float", value)
    _require(0 < number <= maximum if positive else 0 <= number <= maximum)
    _require(math.isfinite(number))


def _integer(value: object, minimum: int, maximum: int) -> None:
    _require(type(value) is int and minimum <= value <= maximum)


def _digest(value: object) -> None:
    _require(type(value) is str and _DIGEST.fullmatch(value) is not None)


class ThermalState(IntEnum):
    """Host-normalized platform state; absent observations never mean normal."""

    NOMINAL = 0
    FAIR = 1
    SERIOUS = 2
    CRITICAL = 3


class MemoryPressure(IntEnum):
    """Host-normalized memory pressure, independent of available bytes."""

    NORMAL = 0
    WARNING = 1
    CRITICAL = 2


class GateReason(StrEnum):
    """Closed vocabulary suitable for aggregate diagnostics."""

    READY = "ready"
    QUEUE_EXPIRED = "queue_expired"
    OUTPUT_EXPIRED = "output_expired"
    LATENCY = "latency"
    RESOURCES = "resources"
    OBSERVATION = "observation_unavailable"
    CLOCK = "clock_invalid"
    PROVIDER = "provider_rejected"
    CLOSED = "closed"
    LIMIT = "operation_limit"


@dataclass(frozen=True, slots=True, kw_only=True)
class PlatformObservation:
    """Fresh host measurements without device names or raw workload identifiers.

    Args:
        observed_at: Monotonic seconds in the controller's clock domain.
        thermal: Platform thermal state, or None when unsupported/unavailable.
        memory_pressure: Platform pressure, or None when unavailable.
        available_bytes: Conservative allocation headroom, or None if unknown.
    """

    observed_at: float
    thermal: ThermalState | None
    memory_pressure: MemoryPressure | None
    available_bytes: int | None

    def __post_init__(self) -> None:
        _number(self.observed_at, 1e12)
        _require(self.thermal is None or type(self.thermal) is ThermalState)
        _require(
            self.memory_pressure is None or type(self.memory_pressure) is MemoryPressure
        )
        if self.available_bytes is not None:
            _integer(self.available_bytes, 0, 1 << 60)


@dataclass(frozen=True, slots=True, kw_only=True)
class EdgeBudget:
    """Per-task deadlines and bounded history; all durations use milliseconds.

    Args:
        max_latency_ms: Maximum single execution and rolling p95 latency.
        max_queue_ms: Maximum age before starting work.
        max_output_age_ms: Maximum total age at return, including queue time.
        max_observation_age_ms: Maximum platform measurement age at each check.
        window: Number of measured executions retained per mode.
        max_operations: Lifetime ceiling; create another explicitly qualified session.
    """

    max_latency_ms: float
    max_queue_ms: float
    max_output_age_ms: float
    max_observation_age_ms: float = 500
    window: int = 20
    max_operations: int = 100_000

    def __post_init__(self) -> None:
        for value in (
            self.max_latency_ms,
            self.max_queue_ms,
            self.max_output_age_ms,
            self.max_observation_age_ms,
        ):
            _number(value, 3_600_000, positive=True)
        _require(self.max_output_age_ms >= self.max_latency_ms)
        _integer(self.window, 1, 1024)
        _integer(self.max_operations, 1, 1_000_000)


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalMode:
    """One host-qualified mode in preferred-to-degraded order.

    Args:
        adapter_digest: Exact VLM metadata digest, or speech-screen artifact digest.
        qualification_digest: Host evidence for this task/mode/device; not proof.
        capabilities: Exact supported feature declarations, not transformations.
        required_available_bytes: Conservative extra peak allocation requirement.
        max_thermal: Highest qualified state; CRITICAL is never admitted.
        max_memory_pressure: Highest qualified pressure; CRITICAL never admitted.

    No policy assumes smaller models are cheaper or clinically interchangeable.
    The host must establish those properties before declaring a fallback.
    """

    adapter_digest: str
    qualification_digest: str
    capabilities: ProviderCapabilityEnvelope = field(repr=False)
    required_available_bytes: int
    max_thermal: ThermalState = ThermalState.NOMINAL
    max_memory_pressure: MemoryPressure = MemoryPressure.NORMAL

    def __post_init__(self) -> None:
        _digest(self.adapter_digest)
        _digest(self.qualification_digest)
        _require(type(self.capabilities) is ProviderCapabilityEnvelope)
        _integer(self.required_available_bytes, 1, 1 << 60)
        _require(type(self.max_thermal) is ThermalState)
        _require(self.max_thermal < ThermalState.CRITICAL)
        _require(type(self.max_memory_pressure) is MemoryPressure)
        _require(self.max_memory_pressure < MemoryPressure.CRITICAL)


def vlm_adapter_digest(metadata: LocalProviderMetadata) -> str:
    """Bind a VLM mode to its existing local model and preprocessing declaration."""
    _require(type(metadata) is LocalProviderMetadata)
    metadata.__post_init__()
    return hashlib.sha256(
        json.dumps(
            [
                metadata.backend.value,
                metadata.model_digest,
                metadata.preprocessing_digest,
            ],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ThermalOutcome(Generic[T]):
    """Protected result or abstention; only the aggregate report is exportable.

    Args:
        value: Actual result, absent whenever the execution or release was held.
        reason: Controlled reason, never provider exception text.
        mode_index: Selected declared mode index, if any.
        report_json: Aggregate controller snapshot with no request IDs or values.
    """

    value: T | None = field(repr=False)
    reason: GateReason
    mode_index: int | None
    report_json: str


class ThermalGate:
    """Serialize local calls with preflight and post-execution resource checks.

    Args:
        profile: Validated task requirements shared by every fallback mode.
        modes: Explicit local ladder; every entry must exactly support the task.
        budget: Per-task latency, age and history ceilings.
        observe: Host-owned local sensor callback; no sensors are guessed.
        clock: Monotonic seconds, injectable for deterministic offline replay.

    Modes only degrade during a session. History survives every switch. A failed
    or slow call is never retried with the same protected input automatically.
    There is no background worker, model unload, remote fallback or preemption:
    adapters must independently bound native execution and enforce cancellation.
    """

    def __init__(
        self,
        profile: ModalityProfile,
        modes: tuple[LocalMode, ...],
        budget: EdgeBudget,
        observe: Callable[[], PlatformObservation],
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        _require(type(profile) is ModalityProfile)
        _require(type(modes) is tuple and 0 < len(modes) <= 16)
        _require(type(budget) is EdgeBudget and callable(observe) and callable(clock))
        budget.__post_init__()
        for mode in modes:
            _require(type(mode) is LocalMode)
            mode.__post_init__()
            _require(
                capability_matrix((mode.capabilities,), (profile,))[0].outcome
                == "exact"
            )
        _require(len({mode.adapter_digest for mode in modes}) == len(modes))
        self._profile, self._modes, self._budget = profile, modes, budget
        self._observe, self._clock = observe, clock
        self._last_clock: float | None = None
        self._last_observation: float | None = None
        self._history: list[deque[float]] = [deque(maxlen=budget.window) for _ in modes]
        self._floor = self._operations = self._released = self._held = (
            self._switches
        ) = 0
        self._last_reason = GateReason.READY
        self._closed = self._busy = False

    def _time(self) -> float | None:
        value = None
        try:
            current = self._clock()
            _number(current, 1e12)
            _require(self._last_clock is None or current >= self._last_clock)
            value = float(current)
        except Exception:
            pass
        if value is None:
            self._closed = True
            self._last_reason = GateReason.CLOCK
        else:
            self._last_clock = value
        return value

    def _snapshot(self) -> tuple[float, PlatformObservation] | None:
        observation = None
        try:
            candidate = self._observe()
            _require(type(candidate) is PlatformObservation)
            candidate.__post_init__()
            observation = candidate
        except Exception:
            pass
        now = self._time()
        if now is None:
            return None
        if (
            observation is None
            or observation.thermal is None
            or observation.memory_pressure is None
            or observation.available_bytes is None
            or not 0
            <= (now - observation.observed_at) * 1000
            <= self._budget.max_observation_age_ms
            or (
                self._last_observation is not None
                and observation.observed_at < self._last_observation
            )
        ):
            self._last_reason = GateReason.OBSERVATION
            return None
        self._last_observation = observation.observed_at
        return now, observation

    def _fits(self, index: int, observation: PlatformObservation) -> bool:
        mode = self._modes[index]
        return (
            observation.thermal is not None
            and observation.thermal <= mode.max_thermal
            and observation.memory_pressure is not None
            and observation.memory_pressure <= mode.max_memory_pressure
            and observation.available_bytes is not None
            and observation.available_bytes >= mode.required_available_bytes
        )

    def _p95(self, index: int) -> float:
        history = sorted(self._history[index])
        return history[math.ceil(len(history) * 0.95) - 1] if history else 0.0

    def _advance(self, index: int) -> None:
        if index > self._floor:
            self._switches += 1
            self._floor = index

    def _outcome(
        self, value: T | None, reason: GateReason, index: int | None
    ) -> ThermalOutcome[T]:
        self._last_reason = reason
        if reason is GateReason.READY:
            self._released += 1
        else:
            self._held += 1
        return ThermalOutcome(value, reason, index, self.to_json())

    def _execute(
        self, operation: Callable[[int], T], queued_at: float
    ) -> ThermalOutcome[T]:
        _require(not self._busy)
        _number(queued_at, 1e12)
        if self._closed:
            return self._outcome(None, GateReason.CLOSED, None)
        if self._operations >= self._budget.max_operations:
            self._closed = True
            return self._outcome(None, GateReason.LIMIT, None)
        self._operations += 1
        self._busy = True
        try:
            snapshot = self._snapshot()
            if snapshot is None:
                return self._outcome(None, self._last_reason, None)
            start, observation = snapshot
            age = (start - queued_at) * 1000
            if (
                not 0
                <= age
                <= min(self._budget.max_queue_ms, self._budget.max_output_age_ms)
            ):
                return self._outcome(None, GateReason.QUEUE_EXPIRED, None)
            index = next(
                (
                    i
                    for i in range(self._floor, len(self._modes))
                    if self._fits(i, observation)
                    and self._p95(i) <= self._budget.max_latency_ms
                ),
                None,
            )
            if index is None:
                return self._outcome(None, GateReason.RESOURCES, None)
            self._advance(index)
            result = None
            failed = False
            try:
                result = operation(index)
            except Exception:
                failed = True
            after = self._snapshot()
            # A sensor outage must not erase a measured slow or broken call.
            # Clock failure is terminal; otherwise _snapshot records its time
            # even when the platform observation itself cannot be admitted.
            end = self._last_clock
            duration = 0.0 if end is None else (end - start) * 1000
            if self._last_reason is not GateReason.CLOCK:
                self._history[index].append(duration)
                if failed or duration > self._budget.max_latency_ms:
                    self._advance(index + 1)
            if after is None:
                return self._outcome(None, self._last_reason, index)
            end, observation = after
            if failed:
                # Do not repeatedly call a broken adapter or retry patient input.
                self._advance(index + 1)
                return self._outcome(None, GateReason.PROVIDER, index)
            if (end - queued_at) * 1000 > self._budget.max_output_age_ms:
                if duration > self._budget.max_latency_ms:
                    self._advance(index + 1)
                return self._outcome(None, GateReason.OUTPUT_EXPIRED, index)
            if (
                duration > self._budget.max_latency_ms
                or self._p95(index) > self._budget.max_latency_ms
            ):
                self._advance(index + 1)
                return self._outcome(None, GateReason.LATENCY, index)
            if not self._fits(index, observation):
                return self._outcome(None, GateReason.RESOURCES, index)
            return self._outcome(result, GateReason.READY, index)
        finally:
            self._busy = False

    def run_vlm(
        self,
        providers: tuple[EdgeVLMProvider, ...],
        request: VLMRequest,
        *,
        queued_at: float,
    ) -> ThermalOutcome[VLMResult]:
        """Dispatch the selected pinned VLM and withhold late or unsafe results.

        Args:
            providers: Already-loaded local adapters in exactly the declared order.
            request: Original protected request; no implicit resize or prompt edit.
            queued_at: Host-recorded enqueue time in the gate's monotonic domain.
        """
        _require(self._profile.modality == "image")
        _require(type(providers) is tuple and len(providers) == len(self._modes))
        _require(type(request) is VLMRequest)

        def run(index: int) -> VLMResult:
            provider = providers[index]
            digest = self._modes[index].adapter_digest
            _require(vlm_adapter_digest(provider.metadata) == digest)
            result = run_local_vlm(provider, request)
            _require(vlm_adapter_digest(provider.metadata) == digest)
            return result

        return self._execute(run, queued_at)

    def release_speech(
        self, buffer: WithholdingBuffer, *, queued_at: float
    ) -> ThermalOutcome[tuple[ReleasedSpeechSegment, ...]]:
        """Gate the actual PHI-screened release and discard pending text on hold.

        Args:
            buffer: Existing privacy buffer with its original lineage and deadlines.
            queued_at: Oldest pending input's monotonic enqueue time.

        Use one audio mode bound to the buffer's screen artifact. A live speech
        buffer cannot be switched to another detector or replayed across modes.
        This controls screening/release, not microphone capture or ASR execution.
        """
        _require(self._profile.modality == "audio" and len(self._modes) == 1)
        _require(type(buffer) is WithholdingBuffer)

        def release(index: int) -> tuple[ReleasedSpeechSegment, ...]:
            _require(buffer.screen_artifact_digest == self._modes[index].adapter_digest)
            _require(buffer.tick() in (BufferState.OPEN, BufferState.ENDING))
            result = buffer.release_ready()
            _require(buffer.state is not BufferState.ABSTAINED)
            return result

        outcome = self._execute(release, queued_at)
        if outcome.reason is not GateReason.READY:
            buffer.close()
        return outcome

    def close(self) -> None:
        """Stop admission; caller-owned providers and previously returned values remain owned by the host."""
        _require(not self._busy)
        self._closed = True

    def to_json(self) -> str:
        """Export aggregate performance counters only; never source/model identifiers."""
        return json.dumps(
            {
                "schema_version": "openmed.thermal_gate.v1",
                "closed": self._closed,
                "reason": self._last_reason.value,
                "operations": self._operations,
                "released": self._released,
                "held": self._held,
                "mode_switches": self._switches,
                "mode_floor": self._floor,
                "modes": [
                    {
                        "mode_index": index,
                        "samples": len(history),
                        "p95_ms": self._p95(index),
                    }
                    for index, history in enumerate(self._history)
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
