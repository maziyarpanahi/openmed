"""Tests for scaling metrics and the reference HPA."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from openmed.service.metrics import PrometheusMetricsRegistry
from openmed.service.scaling_metrics import (
    INFLIGHT_REQUESTS_METRIC,
    QUEUE_DEPTH_METRIC,
    ScalingTargets,
    recommend_replicas,
)

ROOT = Path(__file__).resolve().parents[3]
HPA_PATH = ROOT / "deploy" / "k8s" / "hpa.yaml"
GUIDE_PATH = ROOT / "docs" / "deploy" / "autoscaling.md"


def test_scaling_gauges_are_present_under_synthetic_load() -> None:
    metrics = PrometheusMetricsRegistry()
    metrics.request_started()
    metrics.record_admission_queue_state(
        queue="analyze",
        depth=5,
        shedding=False,
    )

    rendered = metrics.render()

    assert f"# TYPE {QUEUE_DEPTH_METRIC} gauge" in rendered
    assert f'{QUEUE_DEPTH_METRIC}{{queue="analyze"}} 5' in rendered
    assert f"# TYPE {INFLIGHT_REQUESTS_METRIC} gauge" in rendered
    assert f"{INFLIGHT_REQUESTS_METRIC} 1" in rendered


def test_reference_hpa_targets_queue_inflight_and_cpu_metrics() -> None:
    manifest = yaml.safe_load(HPA_PATH.read_text(encoding="utf-8"))

    assert manifest["apiVersion"] == "autoscaling/v2"
    assert manifest["kind"] == "HorizontalPodAutoscaler"
    assert manifest["spec"]["scaleTargetRef"] == {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "name": "openmed-service",
    }
    assert manifest["spec"]["minReplicas"] == 2
    assert manifest["spec"]["maxReplicas"] == 10

    metrics = manifest["spec"]["metrics"]
    pod_metrics = {
        item["pods"]["metric"]["name"]: item["pods"]["target"]
        for item in metrics
        if item["type"] == "Pods"
    }
    assert pod_metrics == {
        QUEUE_DEPTH_METRIC: {"type": "AverageValue", "averageValue": "4"},
        INFLIGHT_REQUESTS_METRIC: {
            "type": "AverageValue",
            "averageValue": "8",
        },
    }
    cpu = next(item for item in metrics if item["type"] == "Resource")
    assert cpu["resource"]["name"] == "cpu"
    assert cpu["resource"]["target"]["averageUtilization"] == 65


@pytest.mark.parametrize(
    ("queue_depth", "inflight_requests", "expected"),
    [
        (0, 0, 2),
        (8, 8, 2),
        (9, 8, 3),
        (4, 33, 5),
        (1000, 1000, 10),
    ],
)
def test_load_shape_maps_thresholds_to_bounded_replicas(
    queue_depth: int,
    inflight_requests: int,
    expected: int,
) -> None:
    result = recommend_replicas(
        queue_depth=queue_depth,
        inflight_requests=inflight_requests,
    )

    assert result.recommended_replicas == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"queue_depth": -1, "inflight_requests": 0},
        {"queue_depth": True, "inflight_requests": 0},
        {"queue_depth": 0, "inflight_requests": -1},
    ],
)
def test_load_shape_rejects_invalid_observations(kwargs) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        recommend_replicas(**kwargs)


def test_scaling_targets_reject_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="min_replicas"):
        ScalingTargets(min_replicas=4, max_replicas=3)


def test_autoscaling_guide_documents_reproducible_wiring() -> None:
    guide = GUIDE_PATH.read_text(encoding="utf-8")

    assert QUEUE_DEPTH_METRIC in guide
    assert INFLIGHT_REQUESTS_METRIC in guide
    assert "prometheus-adapter" in guide
    assert "kubectl apply -f deploy/k8s/hpa.yaml" in guide
    assert "ceil(max(queue_depth / 4, in_flight / 8, 2))" in guide
