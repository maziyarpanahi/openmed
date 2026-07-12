"""Tests for the OpenMed service Helm chart."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
CHART_DIR = ROOT / "deploy" / "helm" / "openmed-service"
CI_VALUES = CHART_DIR / "ci-values.yaml"
HELM = shutil.which("helm")

pytestmark = pytest.mark.skipif(HELM is None, reason="helm is not installed")


def _render_chart(*args: str) -> list[dict]:
    command = [
        HELM or "helm",
        "template",
        "openmed-service",
        str(CHART_DIR),
        "--namespace",
        "openmed",
        *args,
    ]
    result = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return [
        manifest
        for manifest in yaml.safe_load_all(result.stdout)
        if isinstance(manifest, dict)
    ]


def _by_kind(manifests: list[dict], kind: str) -> dict:
    matches = [manifest for manifest in manifests if manifest.get("kind") == kind]
    assert len(matches) == 1
    return matches[0]


def test_default_render_wires_probes_and_model_cache_volume():
    manifests = _render_chart()
    deployment = _by_kind(manifests, "Deployment")
    pvc = _by_kind(manifests, "PersistentVolumeClaim")

    spec = deployment["spec"]["template"]["spec"]
    container = spec["containers"][0]

    assert container["livenessProbe"]["httpGet"]["path"] == "/livez"
    assert container["readinessProbe"]["httpGet"]["path"] == "/readyz"
    assert container["livenessProbe"]["httpGet"]["httpHeaders"] == [
        {"name": "Host", "value": "localhost"}
    ]
    assert container["readinessProbe"]["httpGet"]["httpHeaders"] == [
        {"name": "Host", "value": "localhost"}
    ]
    assert {
        "name": "model-cache",
        "mountPath": "/root/.cache/huggingface",
    } in container["volumeMounts"]
    assert spec["volumes"][0]["persistentVolumeClaim"]["claimName"].endswith(
        "-model-cache"
    )
    assert pvc["spec"]["resources"]["requests"]["storage"] == "20Gi"


def test_default_render_configures_service_environment():
    manifests = _render_chart()
    deployment = _by_kind(manifests, "Deployment")
    configmap = _by_kind(manifests, "ConfigMap")
    container = deployment["spec"]["template"]["spec"]["containers"][0]

    assert (
        container["envFrom"][0]["configMapRef"]["name"] == configmap["metadata"]["name"]
    )
    assert configmap["data"]["OPENMED_PROFILE"] == "prod"
    assert configmap["data"]["OPENMED_CACHE_DIR"] == "/root/.cache/huggingface/openmed"
    assert (
        "openmed-service.openmed.svc"
        in configmap["data"]["OPENMED_SERVICE_TRUSTED_HOSTS"]
    )


def test_synthetic_values_exercise_image_resources_and_secret_env():
    manifests = _render_chart("--values", str(CI_VALUES))
    deployment = _by_kind(manifests, "Deployment")
    configmap = _by_kind(manifests, "ConfigMap")
    container = deployment["spec"]["template"]["spec"]["containers"][0]

    assert deployment["spec"]["replicas"] == 2
    assert container["image"] == "ghcr.io/maziyarpanahi/openmed:v1.8.2"
    assert container["resources"]["limits"]["memory"] == "8Gi"
    assert container["env"] == [
        {
            "name": "HF_TOKEN",
            "valueFrom": {"secretKeyRef": {"name": "openmed-hf-token", "key": "token"}},
        }
    ]
    assert (
        configmap["data"]["OPENMED_SERVICE_PRELOAD_MODELS"]
        == "disease_detection_superclinical"
    )
    assert configmap["data"]["OPENMED_SERVICE_METRICS_ENABLED"] == "true"
