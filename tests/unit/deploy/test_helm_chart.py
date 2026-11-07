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
HELM = shutil.which("helm")

pytestmark = pytest.mark.skipif(HELM is None, reason="helm is not installed")

NON_ROOT_ID = 65532
HOME_DIR = "/home/nonroot"
CACHE_DIR = f"{HOME_DIR}/.cache"
JOURNEY_ARGS = (
    "--set",
    "journey.enabled=true",
    "--set",
    "journey.networkPolicy.databasePodSelector.app=postgres",
)
HARDENED_POD_SECURITY_CONTEXT = {
    "runAsNonRoot": True,
    "runAsUser": NON_ROOT_ID,
    "runAsGroup": NON_ROOT_ID,
    "fsGroup": NON_ROOT_ID,
    "fsGroupChangePolicy": "OnRootMismatch",
    "seccompProfile": {"type": "RuntimeDefault"},
}
HARDENED_CONTAINER_SECURITY_CONTEXT = {
    "runAsNonRoot": True,
    "runAsUser": NON_ROOT_ID,
    "runAsGroup": NON_ROOT_ID,
    "privileged": False,
    "allowPrivilegeEscalation": False,
    "readOnlyRootFilesystem": True,
    "capabilities": {"drop": ["ALL"]},
    "seccompProfile": {"type": "RuntimeDefault"},
}
# Renders that must all carry the hardened posture: the defaults, the
# synthetic CI values, the production Journey profile and the emptyDir cache.
HARDENED_RENDERS = {
    "default": (),
    "ci-values": ("--values", str(CI_VALUES)),
    "journey": JOURNEY_ARGS,
    "ephemeral-cache": ("--set", "persistence.enabled=false"),
}


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


def _api_container(deployment: dict) -> dict:
    containers = deployment["spec"]["template"]["spec"]["containers"]
    matches = [item for item in containers if item["name"] == "openmed-service"]
    assert len(matches) == 1
    return matches[0]


def _volumes_by_name(deployment: dict) -> dict[str, dict]:
    volumes = deployment["spec"]["template"]["spec"]["volumes"]
    by_name = {volume["name"]: volume for volume in volumes}
    assert len(by_name) == len(volumes)
    return by_name


def _is_non_root_id(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _posture_findings(workload: dict, *, require_fixed_ids: bool = True) -> list[str]:
    """Return value-free codes for every hardening rule a workload breaks.

    Container settings are checked as Kubernetes applies them: a key set on
    the container wins over the pod-level key of the same name. Pod-level
    keys are also checked on their own so weakening either level is visible.
    """
    pod_spec = workload["spec"]["template"]["spec"]
    pod_context = pod_spec.get("securityContext") or {}
    findings = []

    for flag in ("hostNetwork", "hostPID", "hostIPC"):
        if pod_spec.get(flag):
            findings.append(f"pod.{flag}")
    if any("hostPath" in volume for volume in pod_spec.get("volumes", [])):
        findings.append("pod.volumes.hostPath")
    if pod_context.get("runAsNonRoot") is not True:
        findings.append("pod.runAsNonRoot")
    if (pod_context.get("seccompProfile") or {}).get("type") != "RuntimeDefault":
        findings.append("pod.seccompProfile")
    for key in ("runAsUser", "runAsGroup"):
        value = pod_context.get(key)
        if (require_fixed_ids and not _is_non_root_id(value)) or value == 0:
            findings.append(f"pod.{key}")

    containers = [*pod_spec.get("initContainers", []), *pod_spec["containers"]]
    for container in containers:
        prefix = f"container[{container['name']}]"
        context = container.get("securityContext") or {}
        effective = {**pod_context, **context}
        if effective.get("runAsNonRoot") is not True:
            findings.append(f"{prefix}.runAsNonRoot")
        for key in ("runAsUser", "runAsGroup"):
            value = effective.get(key)
            if (require_fixed_ids and not _is_non_root_id(value)) or value == 0:
                findings.append(f"{prefix}.{key}")
        if context.get("privileged") is True:
            findings.append(f"{prefix}.privileged")
        if context.get("allowPrivilegeEscalation") is not False:
            findings.append(f"{prefix}.allowPrivilegeEscalation")
        if context.get("readOnlyRootFilesystem") is not True:
            findings.append(f"{prefix}.readOnlyRootFilesystem")
        capabilities = context.get("capabilities") or {}
        if "ALL" not in (capabilities.get("drop") or []):
            findings.append(f"{prefix}.capabilities.drop")
        if capabilities.get("add"):
            findings.append(f"{prefix}.capabilities.add")
        seccomp = effective.get("seccompProfile") or {}
        if seccomp.get("type") != "RuntimeDefault":
            findings.append(f"{prefix}.seccompProfile")
        if any("hostPort" in port for port in container.get("ports", [])):
            findings.append(f"{prefix}.hostPort")
    return findings


def _walk(node: object):
    yield node
    if isinstance(node, dict):
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def test_default_render_wires_probes_and_model_cache_volume():
    manifests = _render_chart()
    deployment = _by_kind(manifests, "Deployment")
    pvc = _by_kind(manifests, "PersistentVolumeClaim")

    container = _api_container(deployment)
    volumes = _volumes_by_name(deployment)

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
        "mountPath": CACHE_DIR,
    } in container["volumeMounts"]
    assert volumes["model-cache"]["persistentVolumeClaim"]["claimName"].endswith(
        "-model-cache"
    )
    assert pvc["spec"]["resources"]["requests"]["storage"] == "20Gi"


def test_default_render_configures_service_environment():
    manifests = _render_chart()
    deployment = _by_kind(manifests, "Deployment")
    configmap = _by_kind(manifests, "ConfigMap")
    container = _api_container(deployment)

    assert (
        container["envFrom"][0]["configMapRef"]["name"] == configmap["metadata"]["name"]
    )
    assert configmap["data"]["OPENMED_PROFILE"] == "prod"
    assert configmap["data"]["OPENMED_CACHE_DIR"] == f"{CACHE_DIR}/openmed"
    assert configmap["data"]["HF_HOME"] == f"{CACHE_DIR}/huggingface"
    assert container["env"] == [
        {"name": "HOME", "value": HOME_DIR},
        {"name": "TMPDIR", "value": "/tmp"},
    ]
    assert (
        "openmed-service.openmed.svc"
        in configmap["data"]["OPENMED_SERVICE_TRUSTED_HOSTS"]
    )


def test_synthetic_values_exercise_image_resources_and_secret_env():
    manifests = _render_chart("--values", str(CI_VALUES))
    deployment = _by_kind(manifests, "Deployment")
    configmap = _by_kind(manifests, "ConfigMap")
    container = _api_container(deployment)

    assert deployment["spec"]["replicas"] == 2
    assert container["image"] == f"ghcr.io/maziyarpanahi/openmed:v{__version__}"
    assert container["resources"]["limits"]["memory"] == "8Gi"
    assert container["env"] == [
        {"name": "HOME", "value": HOME_DIR},
        {"name": "TMPDIR", "value": "/tmp"},
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


def test_journey_profile_renders_migration_worker_storage_and_default_deny():
    manifests = _render_chart(*JOURNEY_ARGS)
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


@pytest.mark.parametrize("render_args", HARDENED_RENDERS.values(), ids=HARDENED_RENDERS)
def test_api_deployment_runs_hardened_in_every_supported_render(render_args):
    manifests = _render_chart(*render_args)
    api = _api_deployment(manifests)
    pod_context = api["spec"]["template"]["spec"]["securityContext"]
    container_context = _api_container(api)["securityContext"]

    assert _posture_findings(api) == []
    # Exact equality pins the documented baseline, so dropping or loosening
    # any default key in values.yaml or the template fails here.
    assert pod_context == HARDENED_POD_SECURITY_CONTEXT
    assert container_context == HARDENED_CONTAINER_SECURITY_CONTEXT


def test_journey_worker_and_migration_job_stay_hardened():
    manifests = _render_chart(*JOURNEY_ARGS)
    worker = _by_component(manifests, "Deployment", "journey-worker")
    migration = _by_component(manifests, "Job", "journey-migration")

    # These workloads inherit the fixed UID/GID from the image user.
    assert _posture_findings(worker, require_fixed_ids=False) == []
    assert _posture_findings(migration, require_fixed_ids=False) == []


@pytest.mark.parametrize("render_args", HARDENED_RENDERS.values(), ids=HARDENED_RENDERS)
def test_read_only_root_filesystem_has_writable_runtime_paths(render_args):
    manifests = _render_chart(*render_args)
    api = _api_deployment(manifests)
    configmap = _by_kind(manifests, "ConfigMap")
    container = _api_container(api)
    volumes = _volumes_by_name(api)
    mounts = {mount["mountPath"]: mount["name"] for mount in container["volumeMounts"]}
    env = {item["name"]: item.get("value") for item in container["env"]}

    assert mounts[HOME_DIR] == "home"
    assert mounts["/tmp"] == "tmp"
    assert mounts[CACHE_DIR] == "model-cache"
    assert volumes["home"]["emptyDir"] == {"sizeLimit": "128Mi"}
    assert volumes["tmp"]["emptyDir"] == {"sizeLimit": "512Mi"}
    assert env["HOME"] == HOME_DIR
    assert env["TMPDIR"] == "/tmp"

    writable_paths = [
        env["HOME"],
        env["TMPDIR"],
        configmap["data"]["HF_HOME"],
        configmap["data"]["OPENMED_CACHE_DIR"],
    ]
    if "OPENMED_JOURNEY_ARTIFACT_ROOT" in configmap["data"]:
        writable_paths.append(configmap["data"]["OPENMED_JOURNEY_ARTIFACT_ROOT"])
    for path in writable_paths:
        assert any(path == root or path.startswith(f"{root}/") for root in mounts)
        assert not path.startswith("/root")


def test_disabled_persistence_backs_the_model_cache_with_a_bounded_emptydir():
    manifests = _render_chart("--set", "persistence.enabled=false")
    api = _api_deployment(manifests)

    assert not [
        item for item in manifests if item.get("kind") == "PersistentVolumeClaim"
    ]
    assert _volumes_by_name(api)["model-cache"] == {
        "name": "model-cache",
        "emptyDir": {"sizeLimit": "20Gi"},
    }

    unbounded = _render_chart(
        "--set",
        "persistence.enabled=false",
        "--set",
        "writableVolumes.modelCache.sizeLimit=",
    )
    unbounded_api = _api_deployment(unbounded)
    assert _volumes_by_name(unbounded_api)["model-cache"]["emptyDir"] == {}


def test_empty_context_overrides_keep_the_hardened_defaults(tmp_path):
    # Helm merges maps key by key, so an empty map cannot silently drop the
    # hardening: weakening always needs an explicit per-key value.
    overrides = tmp_path / "empty-contexts.yaml"
    overrides.write_text(
        "podSecurityContext: {}\nsecurityContext: {}\n", encoding="utf-8"
    )

    manifests = _render_chart("--values", str(overrides))
    api = _api_deployment(manifests)

    assert _posture_findings(api) == []
    assert api["spec"]["template"]["spec"]["securityContext"] == (
        HARDENED_POD_SECURITY_CONTEXT
    )


WEAKENING_OVERRIDES = {
    "read-write-root": (
        ("--set", "securityContext.readOnlyRootFilesystem=false"),
        ["container[openmed-service].readOnlyRootFilesystem"],
    ),
    "unset-read-only-root": (
        ("--set", "securityContext.readOnlyRootFilesystem=null"),
        ["container[openmed-service].readOnlyRootFilesystem"],
    ),
    "privilege-escalation": (
        ("--set", "securityContext.allowPrivilegeEscalation=true"),
        ["container[openmed-service].allowPrivilegeEscalation"],
    ),
    "privileged": (
        ("--set", "securityContext.privileged=true"),
        ["container[openmed-service].privileged"],
    ),
    "container-root-allowed": (
        ("--set", "securityContext.runAsNonRoot=false"),
        ["container[openmed-service].runAsNonRoot"],
    ),
    "pod-root-allowed": (
        ("--set", "podSecurityContext.runAsNonRoot=false"),
        ["pod.runAsNonRoot"],
    ),
    "container-root-uid": (
        ("--set", "securityContext.runAsUser=0"),
        ["container[openmed-service].runAsUser"],
    ),
    "pod-root-gid": (
        ("--set", "podSecurityContext.runAsGroup=0"),
        ["pod.runAsGroup"],
    ),
    "capabilities-kept": (
        ("--set-json", 'securityContext.capabilities={"drop":[],"add":["NET_RAW"]}'),
        [
            "container[openmed-service].capabilities.drop",
            "container[openmed-service].capabilities.add",
        ],
    ),
    "container-unconfined": (
        ("--set", "securityContext.seccompProfile.type=Unconfined"),
        ["container[openmed-service].seccompProfile"],
    ),
    "pod-unconfined": (
        ("--set", "podSecurityContext.seccompProfile.type=Unconfined"),
        ["pod.seccompProfile"],
    ),
    "journey-read-write-root": (
        (*JOURNEY_ARGS, "--set", "securityContext.readOnlyRootFilesystem=false"),
        ["container[openmed-service].readOnlyRootFilesystem"],
    ),
    # Removing the container context leaves only what the pod level can carry.
    "container-context-removed": (
        ("--set", "securityContext=null"),
        [
            "container[openmed-service].allowPrivilegeEscalation",
            "container[openmed-service].readOnlyRootFilesystem",
            "container[openmed-service].capabilities.drop",
        ],
    ),
}


@pytest.mark.parametrize(
    ("render_args", "expected"),
    WEAKENING_OVERRIDES.values(),
    ids=WEAKENING_OVERRIDES,
)
def test_explicit_weakening_is_flagged(render_args, expected):
    manifests = _render_chart(*render_args)
    api = _api_deployment(manifests)

    # Each explicit override is reported, and only the rule it weakens.
    assert sorted(_posture_findings(api)) == sorted(expected)


@pytest.mark.parametrize(
    "render_args",
    [*HARDENED_RENDERS.values(), (*JOURNEY_ARGS, "--values", str(CI_VALUES))],
    ids=[*HARDENED_RENDERS, "journey-ci-values"],
)
def test_rendered_manifests_contain_no_secrets_or_host_access(render_args):
    manifests = _render_chart(*render_args)

    assert not [item for item in manifests if item.get("kind") == "Secret"]
    for node in _walk(manifests):
        if not isinstance(node, dict):
            continue
        assert "hostPath" not in node
        assert "hostPort" not in node
        for flag in ("hostNetwork", "hostPID", "hostIPC"):
            assert not node.get(flag)
    for workload in manifests:
        if workload.get("kind") not in {"Deployment", "Job"}:
            continue
        pod_spec = workload["spec"]["template"]["spec"]
        for container in pod_spec["containers"]:
            for item in container.get("env", []):
                # Plain values are limited to non-secret runtime paths;
                # everything else must come from a referenced Secret.
                if "value" in item:
                    assert item["name"] in {
                        "HOME",
                        "TMPDIR",
                        "OPENMED_JOURNEY_SCHEMA",
                        "OPENMED_JOURNEY_ARTIFACT_ROOT",
                        "DISABLE_TELEMETRY",
                    }
                else:
                    assert set(item["valueFrom"]) == {"secretKeyRef"}
