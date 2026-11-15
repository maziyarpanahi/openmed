"""Bounded ASGI endurance checks with independently measured resource recovery."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from time import monotonic
from typing import Any

from .load_test import DEFAULT_PATH, _post

_RESOURCES = ("permits", "leases", "handles", "queued_jobs", "memory_bytes")
_ACTIONS = ("load", "cancel", "retry", "drain", "model_swap")


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    """Aggregate resource counters, without worker identities or model names."""

    permits: int = 0
    leases: int = 0
    handles: int = 0
    queued_jobs: int = 0
    memory_bytes: int = 0

    def __post_init__(self) -> None:
        if any(
            type(getattr(self, name)) is not int or not 0 <= getattr(self, name) < 2**63
            for name in _RESOURCES
        ):
            raise ValueError("resource_soak_invalid_snapshot")


@dataclass(frozen=True, slots=True)
class SoakProfile:
    """Finite CI defaults and explicit recovery tolerances for a declared runtime."""

    cycles: int = 10
    requests_per_cycle: int = 8
    concurrency: int = 4
    request_deadline_seconds: float = 1.0
    maximum_duration_seconds: float = 60.0
    tolerances: ResourceSnapshot = ResourceSnapshot()
    evidence: str = "runtime_double"

    def __post_init__(self) -> None:
        for value, maximum in (
            (self.cycles, 10_000),
            (self.requests_per_cycle, 10_000),
            (self.concurrency, 64),
        ):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError("resource_soak_invalid_profile")
        if self.cycles * self.requests_per_cycle > 100_000:
            raise ValueError("resource_soak_work_limit")
        for value, maximum in (
            (self.request_deadline_seconds, 60),
            (self.maximum_duration_seconds, 3600),
        ):
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 < value <= maximum
            ):
                raise ValueError("resource_soak_invalid_deadline")
        if (
            not isinstance(self.tolerances, ResourceSnapshot)
            or type(self.evidence) is not str
            or self.evidence not in {"runtime_double", "real_device"}
        ):
            raise ValueError("resource_soak_invalid_evidence")


async def run_resource_soak(
    app: Any,
    *,
    profile: SoakProfile,
    probe: Callable[[], ResourceSnapshot],
    lifecycle: Callable[[str, str], Awaitable[None]],
    payload: dict[str, Any] | None = None,
    path: str = DEFAULT_PATH,
) -> dict[str, object]:
    """Exercise actual ASGI response completion and repeated lifecycle adapters.

    Args:
        app: Synthetic ASGI workload using an injected model runtime.
        profile: Explicit count, duration and recovery limits.
        probe: Trusted aggregate measurement adapter, sampled after each cycle.
        lifecycle: Trusted async adapter invoked with a fixed action and before/
            after phase; it must drive actual runtime retry, drain and swap seams.
        payload: Caller-owned synthetic request; never copied into the report.
        path: Local ASGI route; no network connection is opened.

    Returns:
        Aggregate resource plateaus, recovery failures, throughput separately,
        a pinned profile digest and declared runtime evidence category. Every
        timeout or resource excess fails the profile, regardless of throughput.

    The adapter is responsible for cooperative cancellation. This in-process
    harness is not a sandbox for hostile or cancellation-suppressing runtimes.
    """
    if (
        not isinstance(profile, SoakProfile)
        or not callable(probe)
        or not callable(lifecycle)
    ):
        raise ValueError("resource_soak_invalid_adapter")
    if payload is None:
        payload = {"text": "Synthetic endurance fixture.", "lang": "en"}
    if not isinstance(payload, dict) or type(path) is not str:
        raise ValueError("resource_soak_invalid_request")
    try:
        baseline = probe()
    except Exception:
        failed = True
    else:
        failed = False
    if failed:
        raise ValueError("resource_soak_probe_unavailable")
    if not isinstance(baseline, ResourceSnapshot):
        raise ValueError("resource_soak_invalid_snapshot")
    config = asdict(profile)
    digest = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    started = monotonic()
    completed = errors = cancelled = attempted = timeouts = cycles_finished = 0
    expected_rejections = 0
    peaks = asdict(baseline)
    failures: set[str] = set()
    actions = dict.fromkeys(_ACTIONS, 0)
    cleanup_runs = 0
    for cycle in range(profile.cycles):
        action = _ACTIONS[cycle % len(_ACTIONS)]
        actions[action] += 1
        try:
            await asyncio.wait_for(
                lifecycle(action, "before"), profile.request_deadline_seconds
            )
            for start in range(0, profile.requests_per_cycle, profile.concurrency):
                count = min(profile.concurrency, profile.requests_per_cycle - start)
                tasks = [
                    asyncio.create_task(_post(app, path, payload)) for _ in range(count)
                ]
                attempted += count
                if action == "cancel":
                    await asyncio.sleep(0)
                    tasks[0].cancel()
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        profile.request_deadline_seconds,
                    )
                except TimeoutError:
                    timeouts += 1
                    failures.add("worker_timeout")
                else:
                    for position, result in enumerate(results):
                        if isinstance(result, asyncio.CancelledError):
                            cancelled += 1
                            if action != "cancel" or position != 0:
                                failures.add("unexpected_cancellation")
                        elif (
                            type(result) is int and result == 503 and action == "drain"
                        ):
                            expected_rejections += 1
                        elif isinstance(result, BaseException) or result >= 400:
                            errors += 1
                        else:
                            completed += 1
        except TimeoutError:
            failures.add("lifecycle_timeout")
        except Exception:
            failures.add("lifecycle_error")
        finally:
            try:
                await asyncio.wait_for(
                    lifecycle(action, "after"), profile.request_deadline_seconds
                )
                cleanup_runs += 1
            except TimeoutError:
                failures.add("cleanup_timeout")
            except Exception:
                failures.add("cleanup_error")
        try:
            sample = probe()
            if not isinstance(sample, ResourceSnapshot):
                raise ValueError("invalid")
        except Exception:
            failures.add("probe_error")
        else:
            for name in _RESOURCES:
                value = getattr(sample, name)
                peaks[name] = max(peaks[name], value)
                if value > getattr(baseline, name) + getattr(profile.tolerances, name):
                    failures.add("unrecovered_" + name)
        cycles_finished += 1
        if monotonic() - started > profile.maximum_duration_seconds:
            failures.add("duration_exceeded")
            break
        if failures:
            break
    elapsed = monotonic() - started
    if errors:
        failures.add("request_errors")
    return {
        "schema_version": 1,
        "outcome": "fail" if failures else "pass",
        "reasons": sorted(failures),
        "profile": config,
        "profile_digest": digest,
        "elapsed_seconds": elapsed,
        "cycles_finished": cycles_finished,
        "attempted_requests": attempted,
        "completed_requests": completed,
        "cancelled_requests": cancelled,
        "request_errors": errors,
        "worker_timeouts": timeouts,
        "expected_drain_rejections": expected_rejections,
        "completed_requests_per_second": completed / elapsed if elapsed else 0.0,
        "baseline": asdict(baseline),
        "post_cycle_resource_peaks": peaks,
        "lifecycle_counts": actions,
        "cleanup_runs": cleanup_runs,
    }
