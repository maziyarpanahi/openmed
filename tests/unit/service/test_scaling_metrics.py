"""Tests for REST service autoscaling metrics and the HPA manifest."""

from __future__ import annotations

import asyncio
from pathlib import Path

import yaml

from openmed.service.batcher import DynamicBatcher
from openmed.service.metrics import PrometheusMetricsRegistry
from openmed.service.scaling_metrics import (
    SCALING_INFLIGHT_NAME,
    SCALING_QUEUE_DEPTH_NAME,
    ScalingMetrics,
)

ROOT = Path(__file__).resolve().parents[3]
HPA_MANIFEST = ROOT / "deploy" / "k8s" / "hpa.yaml"


def test_scaling_metrics_aggregate_queue_partitions_and_clamp_values() -> None:
    metrics = ScalingMetrics()

    metrics.request_finished()
    metrics.request_started()
    metrics.request_started()
    metrics.request_finished()
    metrics.record_queue_depth(queue="analyze", priority="interactive", depth=3)
    metrics.record_queue_depth(queue="analyze", priority="bulk", depth=-1)
    metrics.record_queue_depth(queue="pii_extract", priority="bulk", depth=2)

    snapshot = metrics.snapshot()

    assert snapshot.queue_depth == 5
    assert snapshot.inflight_requests == 1


def test_synthetic_load_is_exposed_as_label_free_scaling_gauges() -> None:
    async def run_load() -> str:
        registry = PrometheusMetricsRegistry()
        analyze_batcher = DynamicBatcher(
            lambda items: list(items),
            max_batch_size=16,
            max_wait_ms=60_000,
            metrics=registry,
            metrics_queue="analyze",
        )
        pii_batcher = DynamicBatcher(
            lambda items: list(items),
            max_batch_size=16,
            max_wait_ms=60_000,
            metrics=registry,
            metrics_queue="pii_extract",
        )
        pending = [
            asyncio.create_task(analyze_batcher.submit("a", priority="interactive")),
            asyncio.create_task(analyze_batcher.submit("b", priority="bulk")),
            asyncio.create_task(pii_batcher.submit("c", priority="bulk")),
        ]
        await asyncio.sleep(0)
        for _ in range(4):
            registry.scaling_request_started()

        rendered = registry.render()

        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        return rendered

    rendered = asyncio.run(run_load())

    assert f"# TYPE {SCALING_QUEUE_DEPTH_NAME} gauge" in rendered
    assert f"{SCALING_QUEUE_DEPTH_NAME} 3" in rendered
    assert f"# TYPE {SCALING_INFLIGHT_NAME} gauge" in rendered
    assert f"{SCALING_INFLIGHT_NAME} 4" in rendered
    assert f"{SCALING_QUEUE_DEPTH_NAME}{{" not in rendered
    assert f"{SCALING_INFLIGHT_NAME}{{" not in rendered


def test_hpa_manifest_targets_cpu_and_both_custom_metrics() -> None:
    manifest = yaml.safe_load(HPA_MANIFEST.read_text(encoding="utf-8"))

    assert manifest["apiVersion"] == "autoscaling/v2"
    assert manifest["kind"] == "HorizontalPodAutoscaler"
    assert manifest["spec"]["scaleTargetRef"] == {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "name": "openmed-service",
    }
    assert manifest["spec"]["minReplicas"] == 2
    assert manifest["spec"]["maxReplicas"] == 20

    metrics = manifest["spec"]["metrics"]
    cpu = next(metric["resource"] for metric in metrics if metric["type"] == "Resource")
    pods = {
        metric["pods"]["metric"]["name"]: metric["pods"]["target"]
        for metric in metrics
        if metric["type"] == "Pods"
    }

    assert cpu == {
        "name": "cpu",
        "target": {"type": "Utilization", "averageUtilization": 60},
    }
    assert pods == {
        SCALING_QUEUE_DEPTH_NAME: {"type": "AverageValue", "averageValue": "8"},
        SCALING_INFLIGHT_NAME: {"type": "AverageValue", "averageValue": "4"},
    }
