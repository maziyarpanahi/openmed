"""Low-cardinality gauges used to autoscale the OpenMed REST service."""

from __future__ import annotations

import threading
from dataclasses import dataclass

SCALING_QUEUE_DEPTH_NAME = "openmed_service_scaling_queue_depth"
SCALING_INFLIGHT_NAME = "openmed_service_scaling_inflight_requests"


@dataclass(frozen=True)
class ScalingMetricSnapshot:
    """Point-in-time autoscaling values for one service process.

    Attributes:
        queue_depth: Pending requests across all registered queue partitions.
        inflight_requests: Active model-backed requests in this process.
    """

    queue_depth: int
    inflight_requests: int


class ScalingMetrics:
    """Track aggregate model-work signals without exporting labels.

    Queue identifiers and priority classes are retained only as internal keys so
    multiple dynamic batchers can contribute to one per-pod queue depth. They
    are intentionally absent from the exported samples, which keeps the custom
    metrics surface bounded and prevents request-derived values from becoming
    metric labels.
    """

    def __init__(self) -> None:
        self._queue_depths: dict[tuple[str, str], int] = {}
        self._inflight_requests = 0
        self._lock = threading.RLock()

    def request_started(self) -> None:
        """Increment the active model-backed request count."""
        with self._lock:
            self._inflight_requests += 1

    def request_finished(self) -> None:
        """Decrement the active model-backed request count without underflow."""
        with self._lock:
            self._inflight_requests = max(self._inflight_requests - 1, 0)

    def record_queue_depth(
        self,
        *,
        queue: str,
        priority: str,
        depth: int,
    ) -> None:
        """Replace one internal queue partition's current pending-work count.

        Args:
            queue: Stable internal name for the contributing batch queue.
            priority: Stable internal priority partition name.
            depth: Current number of pending requests in the partition.
        """
        key = (str(queue), str(priority))
        with self._lock:
            self._queue_depths[key] = max(int(depth), 0)

    def snapshot(self) -> ScalingMetricSnapshot:
        """Return aggregate, label-free values suitable for pod metrics."""
        with self._lock:
            return ScalingMetricSnapshot(
                queue_depth=sum(self._queue_depths.values()),
                inflight_requests=self._inflight_requests,
            )
