"""Tests for the OpenMed service Helm chart."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from openmed.__about__ import __version__
from openmed.service.secret_files import CREDENTIAL_SETTINGS

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
    for name in ("HF_HUB_DISABLE_TELEMETRY", "DISABLE_TELEMETRY", "DO_NOT_TRACK"):
        assert configmap["data"][name] == "1"
    assert (
        "openmed-service.openmed.svc"
        in configmap["data"]["OPENMED_SERVICE_TRUSTED_HOSTS"]
    )
    assert configmap["data"]["OPENMED_SERVICE_TRUSTED_PROXIES"] == ""
    assert configmap["data"]["OPENMED_SERVICE_TRUSTED_PROXY_HOPS"] == "1"
    assert configmap["data"]["OPENMED_SERVICE_THROTTLE_MAX_TRACKED_KEYS"] == "10000"


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
            "name": "OPENMED_SERVICE_AUTH_API_KEYS_FILE",
            "value": "/var/run/secrets/openmed/openmed-service-auth-api-keys/value",
        },
        {
            "name": "OPENMED_SERVICE_OTLP_HEADERS_FILE",
            "value": "/var/run/secrets/openmed/openmed-service-otlp-headers/value",
        },
        {
            "name": "HF_TOKEN",
            "valueFrom": {"secretKeyRef": {"name": "openmed-hf-token", "key": "token"}},
        },
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
    assert configmap["data"]["OPENMED_SERVICE_THROTTLE_KEY"] == "x-forwarded-for"
    assert (
        configmap["data"]["OPENMED_SERVICE_TRUSTED_PROXIES"]
        == "198.51.100.0/24,2001:db8:ffff::/48"
    )
    assert configmap["data"]["OPENMED_SERVICE_TRUSTED_PROXY_HOPS"] == "2"
    assert configmap["data"]["OPENMED_SERVICE_THROTTLE_MAX_TRACKED_KEYS"] == "5000"


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
    for workload in (api, migration):
        env = workload["spec"]["template"]["spec"]["containers"][0]["env"]
        assert {"name": "DISABLE_TELEMETRY", "value": "1"} in env
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


def test_journey_database_tls_secret_is_mounted_for_every_store_client():
    manifests = _render_chart(
        "--set",
        "journey.enabled=true",
        "--set",
        "journey.networkPolicy.databasePodSelector.app=postgres",
        "--set",
        "journey.database.tls.secretName=openmed-journey-postgres-tls",
        "--set",
        "journey.database.tls.clientCertKey=tls.crt",
        "--set",
        "journey.database.tls.clientKeyKey=tls.key",
    )
    worker = _by_component(manifests, "Deployment", "journey-worker")
    migration = _by_component(manifests, "Job", "journey-migration")
    api = next(
        item
        for item in manifests
        if item.get("kind") == "Deployment"
        and item["spec"]["selector"]["matchLabels"]["app.kubernetes.io/component"]
        == "api"
    )

    for workload in (worker, migration, api):
        pod = workload["spec"]["template"]["spec"]
        container = pod["containers"][0]
        env = {item["name"]: item.get("value") for item in container["env"]}
        assert env["OPENMED_JOURNEY_POSTGRES_SSLROOTCERT"] == (
            "/etc/openmed/postgres-tls/ca.crt"
        )
        assert env["OPENMED_JOURNEY_POSTGRES_SSLCERT"] == (
            "/etc/openmed/postgres-tls/tls.crt"
        )
        assert env["OPENMED_JOURNEY_POSTGRES_SSLKEY"] == (
            "/etc/openmed/postgres-tls/tls.key"
        )
        assert {
            "name": "journey-database-tls",
            "mountPath": "/etc/openmed/postgres-tls",
            "readOnly": True,
        } in container["volumeMounts"]
        assert {
            "name": "journey-database-tls",
            "secret": {"secretName": "openmed-journey-postgres-tls"},
        } in pod["volumes"]


def test_journey_database_tls_is_absent_until_a_secret_is_named():
    manifests = _render_chart(
        "--set",
        "journey.enabled=true",
        "--set",
        "journey.networkPolicy.databasePodSelector.app=postgres",
    )
    rendered = yaml.safe_dump_all(manifests)

    assert "OPENMED_JOURNEY_POSTGRES_SSLROOTCERT" not in rendered
    assert "journey-database-tls" not in rendered



def _containers(manifests: list[dict]) -> list[dict]:
    return [
        container
        for manifest in manifests
        if manifest.get("kind") in {"Deployment", "Job"}
        for container in manifest["spec"]["template"]["spec"]["containers"]
    ]


def _api_deployment(manifests: list[dict]) -> dict:
    matches = [
        manifest
        for manifest in manifests
        if manifest.get("kind") == "Deployment"
        and manifest["spec"]["selector"]["matchLabels"].get(
            "app.kubernetes.io/component"
        )
        == "api"
    ]
    assert len(matches) == 1
    return matches[0]


def _volume(manifest: dict, name: str) -> dict:
    volumes = manifest["spec"]["template"]["spec"]["volumes"]
    matches = [volume for volume in volumes if volume["name"] == name]
    assert len(matches) == 1
    return matches[0]


def test_journey_profile_mounts_database_secret_as_file_by_default():
    manifests = _render_chart("--set", "journey.enabled=true")
    workloads = [
        _api_deployment(manifests),
        _by_component(manifests, "Deployment", "journey-worker"),
        _by_component(manifests, "Job", "journey-migration"),
    ]
    dsn_file = "/var/run/secrets/openmed/journey-database/dsn"

    for workload in workloads:
        pod = workload["spec"]["template"]["spec"]
        container = pod["containers"][0]
        env = {item["name"]: item for item in container["env"]}
        assert "OPENMED_JOURNEY_POSTGRES_DSN" not in env
        assert env["OPENMED_JOURNEY_POSTGRES_DSN_FILE"] == {
            "name": "OPENMED_JOURNEY_POSTGRES_DSN_FILE",
            "value": dsn_file,
        }
        assert {
            "name": "journey-database",
            "mountPath": "/var/run/secrets/openmed/journey-database",
            "readOnly": True,
        } in container["volumeMounts"]
        assert _volume(workload, "journey-database")["secret"] == {
            "secretName": "openmed-journey-database",
            "defaultMode": 0o440,
            "items": [{"key": "dsn", "path": "dsn"}],
        }
        # Root-owned 0440 Secret files are readable through the pod fsGroup.
        assert pod["securityContext"]["fsGroup"] == 65532


def test_journey_database_env_source_keeps_secret_key_ref_mapping():
    manifests = _render_chart(
        "--set",
        "journey.enabled=true",
        "--set",
        "journey.database.source=env",
    )

    for container in _containers(manifests):
        env = {item["name"]: item for item in container.get("env", [])}
        assert "OPENMED_JOURNEY_POSTGRES_DSN_FILE" not in env
        assert env["OPENMED_JOURNEY_POSTGRES_DSN"]["valueFrom"] == {
            "secretKeyRef": {"name": "openmed-journey-database", "key": "dsn"}
        }
        mounts = container.get("volumeMounts", [])
        assert all(mount["name"] != "journey-database" for mount in mounts)


def test_secret_files_mount_credentials_without_environment_values():
    manifests = _render_chart(
        "--values",
        str(CI_VALUES),
        "--set",
        "journey.enabled=true",
    )
    api = _api_deployment(manifests)
    container = api["spec"]["template"]["spec"]["containers"][0]

    assert {
        "name": "secret-file-0",
        "mountPath": "/var/run/secrets/openmed/openmed-service-auth-api-keys",
        "readOnly": True,
    } in container["volumeMounts"]
    assert _volume(api, "secret-file-0")["secret"] == {
        "secretName": "openmed-service-auth",
        "defaultMode": 0o440,
        "items": [{"key": "api-keys.json", "path": "value"}],
    }
    assert _volume(api, "secret-file-1")["secret"]["secretName"] == "openmed-otlp"

    for rendered in _containers(manifests):
        names = {item["name"] for item in rendered.get("env", [])}
        assert names.isdisjoint(CREDENTIAL_SETTINGS)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("secretFiles[0].env=OPENMED_SERVICE_AUTH_API_KEYS_FILE", "_FILE suffix"),
        ("secretFiles[0].env=lower_case", "upper-case"),
        ("journey.database.source=inline", "journey.database.source"),
    ],
)
def test_secret_file_values_are_validated(override: str, message: str):
    command = [
        HELM or "helm",
        "template",
        "openmed-service",
        str(CHART_DIR),
        "--set",
        "journey.enabled=true",
        "--set",
        "secretFiles[0].env=OPENMED_SERVICE_AUTH_JWKS",
        "--set",
        "secretFiles[0].secretName=openmed-jwks",
        "--set",
        "secretFiles[0].secretKey=jwks.json",
        "--set",
        override,
    ]
    result = subprocess.run(command, text=True, capture_output=True)

    assert result.returncode != 0
    assert message in result.stderr
