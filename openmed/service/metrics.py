"""Dependency-free Prometheus metrics for the OpenMed REST service."""

from __future__ import annotations

import math
import os
import threading
from typing import Mapping

METRICS_ENABLED_ENV_VAR = "OPENMED_SERVICE_METRICS_ENABLED"
PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

REQUEST_TOTAL_NAME = "openmed_service_request_total"
REQUEST_DURATION_NAME = "openmed_service_request_duration_seconds"
INFLIGHT_NAME = "openmed_service_inflight_requests"
MODEL_LOAD_NAME = "openmed_service_model_load_total"
MODEL_EVICTION_NAME = "openmed_service_model_eviction_total"
PAGED_KV_CACHE_OCCUPANCY_NAME = "openmed_service_mlx_paged_kv_cache_occupancy_pages"
PAGED_KV_CACHE_CAPACITY_NAME = "openmed_service_mlx_paged_kv_cache_capacity_pages"
PAGED_KV_CACHE_PEAK_NAME = "openmed_service_mlx_paged_kv_cache_peak_pages"
PAGED_KV_CACHE_EVICTION_NAME = "openmed_service_mlx_paged_kv_cache_eviction_total"
PAGED_KV_CACHE_BUDGET_BYTES_NAME = "openmed_service_mlx_paged_kv_cache_budget_bytes"
SPECULATIVE_DECODE_NAME = "openmed_mlx_speculative_decode_total"
SPECULATIVE_DRAFT_TOKEN_NAME = "openmed_mlx_speculative_draft_token_total"
SPECULATIVE_ACCEPTED_TOKEN_NAME = "openmed_mlx_speculative_accepted_token_total"
SPECULATIVE_ROLLBACK_NAME = "openmed_mlx_speculative_rollback_total"
SPECULATIVE_FALLBACK_NAME = "openmed_mlx_speculative_fallback_total"
SPECULATIVE_ACCEPTANCE_RATE_NAME = "openmed_mlx_speculative_acceptance_rate"
SPECULATIVE_AVERAGE_DEPTH_NAME = "openmed_mlx_speculative_average_depth"

_ENABLED_VALUES = {"1", "true", "yes", "on", "enabled"}
_DISABLED_VALUES = {"0", "false", "no", "off", "disabled"}
_DEFAULT_DURATION_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
)


def parse_metrics_enabled(raw_value: str | None) -> bool:
    """Parse the metrics endpoint feature flag."""
    if raw_value is None:
        return False

    normalized = raw_value.strip().lower()
    if not normalized:
        return False
    if normalized in _ENABLED_VALUES:
        return True
    if normalized in _DISABLED_VALUES:
        return False
    raise ValueError(
        f"{METRICS_ENABLED_ENV_VAR} must be a boolean value like 'true' or 'false'"
    )


def metrics_enabled_from_env() -> bool:
    """Return whether the optional Prometheus metrics endpoint is enabled."""
    return parse_metrics_enabled(os.getenv(METRICS_ENABLED_ENV_VAR))


class PrometheusMetricsRegistry:
    """Small in-process Prometheus text-exposition registry.

    The registry deliberately exposes only aggregate counters, gauges, and
    durations. Label values are restricted to static route templates and status
    codes supplied by the service middleware.
    """

    def __init__(self, duration_buckets: tuple[float, ...] | None = None) -> None:
        buckets = (
            _DEFAULT_DURATION_BUCKETS if duration_buckets is None else duration_buckets
        )
        self._duration_buckets = tuple(sorted(float(bucket) for bucket in buckets))
        self._request_total: dict[tuple[str, str], int] = {}
        self._duration_bucket_counts: dict[str, list[int]] = {}
        self._duration_count: dict[str, int] = {}
        self._duration_sum: dict[str, float] = {}
        self._inflight_requests = 0
        self._model_load_total = 0
        self._model_eviction_total = 0
        self._paged_kv_cache_occupancy_pages = 0
        self._paged_kv_cache_capacity_pages = 0
        self._paged_kv_cache_peak_pages = 0
        self._paged_kv_cache_eviction_total = 0
        self._paged_kv_cache_budget_bytes = 0
        self._speculative_decode_total = 0
        self._speculative_draft_tokens = 0
        self._speculative_accepted_tokens = 0
        self._speculative_rollbacks = 0
        self._speculative_fallbacks = 0
        self._lock = threading.RLock()

    def request_started(self) -> None:
        """Increment the active request gauge."""
        with self._lock:
            self._inflight_requests += 1

    def request_finished(
        self,
        *,
        route: str,
        status_code: int,
        duration_seconds: float,
    ) -> None:
        """Record one completed HTTP request."""
        route_label = str(route)
        status_label = str(int(status_code))
        observed_duration = max(float(duration_seconds), 0.0)

        with self._lock:
            self._inflight_requests = max(self._inflight_requests - 1, 0)
            key = (route_label, status_label)
            self._request_total[key] = self._request_total.get(key, 0) + 1

            bucket_counts = self._duration_bucket_counts.get(route_label)
            if bucket_counts is None:
                bucket_counts = [0] * (len(self._duration_buckets) + 1)
                self._duration_bucket_counts[route_label] = bucket_counts

            for index, bucket in enumerate(self._duration_buckets):
                if observed_duration <= bucket:
                    bucket_counts[index] += 1
            bucket_counts[-1] += 1
            self._duration_count[route_label] = (
                self._duration_count.get(route_label, 0) + 1
            )
            self._duration_sum[route_label] = (
                self._duration_sum.get(route_label, 0.0) + observed_duration
            )

    def record_model_load(self, count: int = 1) -> None:
        """Record model resource loads performed by the warm-pool."""
        if count <= 0:
            return
        with self._lock:
            self._model_load_total += int(count)

    def record_model_eviction(self, count: int = 1) -> None:
        """Record model resource evictions performed by the warm-pool."""
        if count <= 0:
            return
        with self._lock:
            self._model_eviction_total += int(count)

    def record_mlx_paged_kv_cache(
        self,
        *,
        total_pages: int,
        resident_pages: int,
        evictions: int = 0,
        peak_pages: int | None = None,
        memory_budget_bytes: int = 0,
    ) -> None:
        """Record aggregate MLX-LM paged KV-cache occupancy."""
        capacity_pages = max(int(total_pages), 0)
        occupancy_pages = min(max(int(resident_pages), 0), capacity_pages)
        observed_peak = occupancy_pages if peak_pages is None else int(peak_pages)

        with self._lock:
            self._paged_kv_cache_capacity_pages = capacity_pages
            self._paged_kv_cache_occupancy_pages = occupancy_pages
            self._paged_kv_cache_peak_pages = max(
                self._paged_kv_cache_peak_pages,
                min(max(observed_peak, 0), capacity_pages),
            )
            self._paged_kv_cache_budget_bytes = max(int(memory_budget_bytes), 0)
            if evictions > 0:
                self._paged_kv_cache_eviction_total += int(evictions)

    def record_speculative_decode(self, metrics: Mapping[str, object] | object) -> None:
        """Record aggregate speculative decode counters.

        The payload must contain only non-PHI aggregate counts. Unknown keys are
        ignored so callers can pass ``SpeculativeDecodeMetrics.to_dict()``.
        """

        if hasattr(metrics, "to_dict"):
            metrics = metrics.to_dict()  # type: ignore[assignment]
        if not isinstance(metrics, Mapping):
            return

        drafted = _non_negative_int(metrics.get("drafted_tokens"))
        accepted = _non_negative_int(metrics.get("accepted_tokens"))
        rollbacks = _non_negative_int(metrics.get("rollback_count"))
        fallback_reason = metrics.get("fallback_reason")

        with self._lock:
            self._speculative_decode_total += 1
            self._speculative_draft_tokens += drafted
            self._speculative_accepted_tokens += accepted
            self._speculative_rollbacks += rollbacks
            if fallback_reason:
                self._speculative_fallbacks += 1

    def render(self) -> str:
        """Render metrics using the Prometheus 0.0.4 text format."""
        with self._lock:
            request_total = dict(self._request_total)
            duration_bucket_counts = {
                route: list(counts)
                for route, counts in self._duration_bucket_counts.items()
            }
            duration_count = dict(self._duration_count)
            duration_sum = dict(self._duration_sum)
            inflight_requests = self._inflight_requests
            model_load_total = self._model_load_total
            model_eviction_total = self._model_eviction_total
            paged_kv_cache_occupancy_pages = self._paged_kv_cache_occupancy_pages
            paged_kv_cache_capacity_pages = self._paged_kv_cache_capacity_pages
            paged_kv_cache_peak_pages = self._paged_kv_cache_peak_pages
            paged_kv_cache_eviction_total = self._paged_kv_cache_eviction_total
            paged_kv_cache_budget_bytes = self._paged_kv_cache_budget_bytes
            speculative_decode_total = self._speculative_decode_total
            speculative_draft_tokens = self._speculative_draft_tokens
            speculative_accepted_tokens = self._speculative_accepted_tokens
            speculative_rollbacks = self._speculative_rollbacks
            speculative_fallbacks = self._speculative_fallbacks

        lines: list[str] = []
        _append_family_header(
            lines,
            REQUEST_TOTAL_NAME,
            "Total REST requests by static route and HTTP status code.",
            "counter",
        )
        for (route, status_code), value in sorted(request_total.items()):
            labels = _label_suffix({"route": route, "status_code": status_code})
            lines.append(f"{REQUEST_TOTAL_NAME}{labels} {value}")

        _append_family_header(
            lines,
            REQUEST_DURATION_NAME,
            "REST request duration in seconds by static route.",
            "histogram",
        )
        for route in sorted(duration_bucket_counts):
            cumulative_counts = duration_bucket_counts[route]
            for bucket, value in zip(self._duration_buckets, cumulative_counts):
                labels = _label_suffix(
                    {"route": route, "le": _format_bucket_label(bucket)}
                )
                lines.append(f"{REQUEST_DURATION_NAME}_bucket{labels} {value}")
            labels = _label_suffix({"route": route, "le": "+Inf"})
            lines.append(
                f"{REQUEST_DURATION_NAME}_bucket{labels} {cumulative_counts[-1]}"
            )
            count_labels = _label_suffix({"route": route})
            lines.append(
                f"{REQUEST_DURATION_NAME}_count{count_labels} "
                f"{duration_count.get(route, 0)}"
            )
            lines.append(
                f"{REQUEST_DURATION_NAME}_sum{count_labels} "
                f"{_format_sample_value(duration_sum.get(route, 0.0))}"
            )

        _append_family_header(
            lines,
            INFLIGHT_NAME,
            "Active HTTP requests currently being handled.",
            "gauge",
        )
        lines.append(f"{INFLIGHT_NAME} {inflight_requests}")

        _append_family_header(
            lines,
            MODEL_LOAD_NAME,
            "Model resource loads performed by the service warm-pool.",
            "counter",
        )
        lines.append(f"{MODEL_LOAD_NAME} {model_load_total}")

        _append_family_header(
            lines,
            MODEL_EVICTION_NAME,
            "Model resource evictions performed by the service warm-pool.",
            "counter",
        )
        lines.append(f"{MODEL_EVICTION_NAME} {model_eviction_total}")

        _append_family_header(
            lines,
            PAGED_KV_CACHE_OCCUPANCY_NAME,
            "Current MLX-LM paged KV-cache resident pages.",
            "gauge",
        )
        lines.append(
            f"{PAGED_KV_CACHE_OCCUPANCY_NAME} {paged_kv_cache_occupancy_pages}"
        )

        _append_family_header(
            lines,
            PAGED_KV_CACHE_CAPACITY_NAME,
            "Configured MLX-LM paged KV-cache page capacity.",
            "gauge",
        )
        lines.append(f"{PAGED_KV_CACHE_CAPACITY_NAME} {paged_kv_cache_capacity_pages}")

        _append_family_header(
            lines,
            PAGED_KV_CACHE_PEAK_NAME,
            "Peak MLX-LM paged KV-cache resident pages observed in this process.",
            "gauge",
        )
        lines.append(f"{PAGED_KV_CACHE_PEAK_NAME} {paged_kv_cache_peak_pages}")

        _append_family_header(
            lines,
            PAGED_KV_CACHE_EVICTION_NAME,
            "Total MLX-LM paged KV-cache page evictions in this process.",
            "counter",
        )
        lines.append(f"{PAGED_KV_CACHE_EVICTION_NAME} {paged_kv_cache_eviction_total}")

        _append_family_header(
            lines,
            PAGED_KV_CACHE_BUDGET_BYTES_NAME,
            "Configured MLX-LM paged KV-cache memory budget in bytes.",
            "gauge",
        )
        lines.append(
            f"{PAGED_KV_CACHE_BUDGET_BYTES_NAME} {paged_kv_cache_budget_bytes}"
        )

        _append_family_header(
            lines,
            SPECULATIVE_DECODE_NAME,
            "MLX language-model speculative decode attempts.",
            "counter",
        )
        lines.append(f"{SPECULATIVE_DECODE_NAME} {speculative_decode_total}")

        _append_family_header(
            lines,
            SPECULATIVE_DRAFT_TOKEN_NAME,
            "Draft tokens proposed by MLX speculative decoding.",
            "counter",
        )
        lines.append(f"{SPECULATIVE_DRAFT_TOKEN_NAME} {speculative_draft_tokens}")

        _append_family_header(
            lines,
            SPECULATIVE_ACCEPTED_TOKEN_NAME,
            "Draft tokens accepted by target verification.",
            "counter",
        )
        lines.append(f"{SPECULATIVE_ACCEPTED_TOKEN_NAME} {speculative_accepted_tokens}")

        _append_family_header(
            lines,
            SPECULATIVE_ROLLBACK_NAME,
            "Speculative decode rollbacks after target verification.",
            "counter",
        )
        lines.append(f"{SPECULATIVE_ROLLBACK_NAME} {speculative_rollbacks}")

        _append_family_header(
            lines,
            SPECULATIVE_FALLBACK_NAME,
            "Speculative decode attempts that fell back to plain decoding.",
            "counter",
        )
        lines.append(f"{SPECULATIVE_FALLBACK_NAME} {speculative_fallbacks}")

        acceptance_rate = (
            speculative_accepted_tokens / speculative_draft_tokens
            if speculative_draft_tokens
            else 0.0
        )
        average_depth = (
            speculative_draft_tokens / speculative_decode_total
            if speculative_decode_total
            else 0.0
        )
        _append_family_header(
            lines,
            SPECULATIVE_ACCEPTANCE_RATE_NAME,
            "Aggregate accepted draft tokens divided by proposed draft tokens.",
            "gauge",
        )
        lines.append(
            f"{SPECULATIVE_ACCEPTANCE_RATE_NAME} "
            f"{_format_sample_value(acceptance_rate)}"
        )

        _append_family_header(
            lines,
            SPECULATIVE_AVERAGE_DEPTH_NAME,
            "Aggregate proposed draft tokens per speculative decode attempt.",
            "gauge",
        )
        lines.append(
            f"{SPECULATIVE_AVERAGE_DEPTH_NAME} {_format_sample_value(average_depth)}"
        )

        return "\n".join(lines) + "\n"


def _append_family_header(
    lines: list[str],
    name: str,
    help_text: str,
    metric_type: str,
) -> None:
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} {metric_type}")


def _label_suffix(labels: Mapping[str, str]) -> str:
    if not labels:
        return ""
    rendered = ",".join(
        f'{name}="{_escape_label_value(value)}"' for name, value in labels.items()
    )
    return "{" + rendered + "}"


def _escape_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _non_negative_int(value: object) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return max(parsed, 0)


def _format_bucket_label(value: float) -> str:
    if value == math.inf:
        return "+Inf"
    return _format_sample_value(value)


def _format_sample_value(value: float) -> str:
    if math.isfinite(value):
        return f"{value:.12g}"
    if value == math.inf:
        return "+Inf"
    if value == -math.inf:
        return "-Inf"
    return "NaN"
