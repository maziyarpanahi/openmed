"""Deterministic autoscaling guidance for aggregate service metrics.

This module contains no cluster client and performs no network access. It
turns the same queue-depth and in-flight gauges exported by
``PrometheusMetricsRegistry`` into a reproducible replica recommendation that
operators can use when choosing HorizontalPodAutoscaler targets.
"""

from __future__ import annotations

from dataclasses import dataclass

from .metrics import ADMISSION_QUEUE_DEPTH_NAME, INFLIGHT_NAME

QUEUE_DEPTH_METRIC = ADMISSION_QUEUE_DEPTH_NAME
INFLIGHT_REQUESTS_METRIC = INFLIGHT_NAME


@dataclass(frozen=True)
class ScalingTargets:
    """Per-pod metric targets and replica bounds for an OpenMed service."""

    queue_depth_per_pod: int = 4
    inflight_requests_per_pod: int = 8
    min_replicas: int = 2
    max_replicas: int = 10

    def __post_init__(self) -> None:
        """Reject ambiguous or unsafe scaling bounds."""

        for name in (
            "queue_depth_per_pod",
            "inflight_requests_per_pod",
            "min_replicas",
            "max_replicas",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_replicas > self.max_replicas:
            raise ValueError("min_replicas must not exceed max_replicas")


@dataclass(frozen=True)
class ScalingRecommendation:
    """Replica counts implied by each aggregate load signal."""

    queue_depth: int
    inflight_requests: int
    queue_replicas: int
    inflight_replicas: int
    recommended_replicas: int


def recommend_replicas(
    *,
    queue_depth: int,
    inflight_requests: int,
    targets: ScalingTargets | None = None,
) -> ScalingRecommendation:
    """Map aggregate queue and in-flight load to a bounded replica count.

    Kubernetes chooses the largest recommendation among configured HPA
    metrics. This helper mirrors that behavior for a documented load shape,
    then applies the configured minimum and maximum replica bounds.

    Args:
        queue_depth: Aggregate admitted queue depth across service pods.
        inflight_requests: Aggregate active HTTP requests across service pods.
        targets: Optional per-pod targets and replica bounds.

    Returns:
        A value-only recommendation containing both signal calculations.

    Raises:
        TypeError: If ``targets`` is not a ``ScalingTargets`` instance.
        ValueError: If either observed count is not a non-negative integer.
    """

    observed_queue = _non_negative_integer(queue_depth, "queue_depth")
    observed_inflight = _non_negative_integer(
        inflight_requests,
        "inflight_requests",
    )
    if targets is None:
        resolved = ScalingTargets()
    elif isinstance(targets, ScalingTargets):
        resolved = targets
    else:
        raise TypeError("targets must be a ScalingTargets instance")
    queue_replicas = _ceil_division(
        observed_queue,
        resolved.queue_depth_per_pod,
    )
    inflight_replicas = _ceil_division(
        observed_inflight,
        resolved.inflight_requests_per_pod,
    )
    recommended = min(
        max(
            resolved.min_replicas,
            queue_replicas,
            inflight_replicas,
        ),
        resolved.max_replicas,
    )
    return ScalingRecommendation(
        queue_depth=observed_queue,
        inflight_requests=observed_inflight,
        queue_replicas=queue_replicas,
        inflight_replicas=inflight_replicas,
        recommended_replicas=recommended,
    )


def _non_negative_integer(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _ceil_division(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


__all__ = [
    "INFLIGHT_REQUESTS_METRIC",
    "QUEUE_DEPTH_METRIC",
    "ScalingRecommendation",
    "ScalingTargets",
    "recommend_replicas",
]
