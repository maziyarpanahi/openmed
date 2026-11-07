"""Tests for the OpenMed service Helm chart."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from openmed.__about__ import __version__

ROOT = Path(__file__).resolve().parents[3]
CHART_DIR = ROOT / "deploy" / "helm" / "openmed-service"
CI_VALUES = CHART_DIR / "ci-values.yaml"
REFERENCE_HPA = ROOT / "deploy" / "k8s" / "hpa.yaml"
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


def _by_component(manifests: list[dict], kind: str, component: str) -> dict:
    matches = [
        manifest
        for manifest in manifests
        if manifest.get("kind") == kind
        and manifest.get("metadata", {})
        .get("labels", {})
        .get("app.kubernetes.io/component")
        == component
    ]
    assert len(matches) == 1
    return matches[0]


def test_default_render_wires_probes_and_model_cache_volume():
    manifests = _render_chart()
    deployment = _by_kind(manifests, "Deployment")
    pvc = _by_kind(manifests, "PersistentVolumeClaim")

    spec = deployment["spec"]["template"]["spec"]
    container = spec["containers"][0]

    assert (
        deployment["spec"]["selector"]["matchLabels"]["app.kubernetes.io/component"]
        == "api"
    )
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
    assert container["image"] == f"ghcr.io/maziyarpanahi/openmed:v{__version__}"
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
    assert configmap["data"]["OPENMED_SERVICE_BATCH_MAX_QUEUE_SIZE"] == "32"
    assert configmap["data"]["OPENMED_SERVICE_BATCH_HIGH_WATERMARK"] == "24"
    assert configmap["data"]["OPENMED_SERVICE_BATCH_LOW_WATERMARK"] == "8"
    assert configmap["data"]["OPENMED_SERVICE_BATCH_MAX_QUEUE_WAIT_MS"] == "250"


def test_journey_profile_renders_migration_worker_storage_and_default_deny():
    manifests = _render_chart(
        "--set",
        "journey.enabled=true",
        "--set",
        "journey.networkPolicy.databasePodSelector.app=postgres",
    )
    api = _by_component(manifests, "Deployment", "journey-worker")
    migration = _by_component(manifests, "Job", "journey-migration")
    artifacts = _by_component(manifests, "PersistentVolumeClaim", "journey-artifacts")
    policy = _by_kind(manifests, "NetworkPolicy")
    service_deployments = [
        item
        for item in manifests
        if item.get("kind") == "Deployment"
        and item.get("metadata", {})
        .get("labels", {})
        .get("app.kubernetes.io/component")
        != "journey-worker"
    ]
    assert len(service_deployments) == 1
    service_container = service_deployments[0]["spec"]["template"]["spec"][
        "containers"
    ][0]

    assert migration["metadata"]["annotations"]["helm.sh/hook"] == (
        "pre-install,pre-upgrade"
    )
    assert migration["spec"]["backoffLimit"] == 3
    assert (
        api["spec"]["template"]["spec"]["containers"][0]["readinessProbe"]["httpGet"][
            "path"
        ]
        == "/readyz"
    )
    assert artifacts["spec"]["resources"]["requests"]["storage"] == "20Gi"
    assert service_container["readinessProbe"]["exec"]["command"][-4:] == [
        "--worker-url",
        "http://openmed-service-journey-worker:8091/readyz",
        "--model-url",
        "http://127.0.0.1:8080/readyz",
    ]
    assert policy["spec"]["policyTypes"] == ["Ingress", "Egress"]
    assert all(set(rule) <= {"to", "ports"} for rule in policy["spec"]["egress"])


def _configmap_data(*args: str) -> dict[str, str]:
    return _by_kind(_render_chart(*args), "ConfigMap")["data"]


def _replica_report(monkeypatch: pytest.MonkeyPatch, data: dict[str, str]):
    from openmed.service.app import create_app, replica_state_report

    for name, value in data.items():
        monkeypatch.setenv(name, value)
    return replica_state_report(create_app())


def test_default_render_declares_single_replica_mode():
    data = _configmap_data()

    assert data["OPENMED_SERVICE_REPLICA_MODE"] == "single"
    assert data["OPENMED_SERVICE_REPLICA_ACCEPT_LOCAL_STATE"] == ""
    assert data["OPENMED_SERVICE_REPLICA_DISABLE_LOCAL_STATE"] == ""


def test_ci_values_render_multi_replica_mode_with_explicit_choices():
    data = _configmap_data("--values", str(CI_VALUES))

    assert data["OPENMED_SERVICE_REPLICA_MODE"] == "multi"
    assert data["OPENMED_SERVICE_REPLICA_ACCEPT_LOCAL_STATE"] == (
        "rate_limiter,concurrency_limiter"
    )
    assert data["OPENMED_SERVICE_REPLICA_DISABLE_LOCAL_STATE"] == (
        "async_jobs,fhir_bulk_jobs,smart_backend_jobs"
    )


@pytest.mark.parametrize(
    ("args", "mode"),
    [
        (("--set", "replicaCount=1"), "single"),
        (("--set", "replicaCount=3"), "multi"),
        (("--set", "autoscaling.enabled=true"), "multi"),
        (("--set", "replicaCount=0", "--set", "autoscaling.enabled=true"), "multi"),
    ],
)
def test_replica_count_and_autoscaling_derive_the_replica_mode(args, mode):
    assert _configmap_data(*args)["OPENMED_SERVICE_REPLICA_MODE"] == mode


def test_reference_autoscaling_example_renders_multi_replica_mode():
    hpa = yaml.safe_load(REFERENCE_HPA.read_text(encoding="utf-8"))
    min_replicas = hpa["spec"]["minReplicas"]

    assert min_replicas > 1
    for args in (
        ("--set", "autoscaling.enabled=true"),
        ("--set", f"replicaCount={min_replicas}"),
    ):
        assert _configmap_data(*args)["OPENMED_SERVICE_REPLICA_MODE"] == "multi"


def test_rendered_replica_policy_drives_service_readiness(
    monkeypatch: pytest.MonkeyPatch,
):
    ci_report = _replica_report(
        monkeypatch, _configmap_data("--values", str(CI_VALUES))
    )
    autoscaled_report = _replica_report(
        monkeypatch, _configmap_data("--set", "autoscaling.enabled=true")
    )

    assert ci_report.ready
    assert not autoscaled_report.ready
    assert autoscaled_report.reason_code == "replica_unsafe_state"
    assert autoscaled_report.unsafe_categories == (
        "async_jobs",
        "fhir_bulk_jobs",
        "smart_backend_jobs",
    )
