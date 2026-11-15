"""Unit tests for the deploy posture contract on synthetic rendered manifests."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytest.importorskip("fastapi")

from openmed.deploy import posture as dp  # noqa: E402
from openmed.service import security_profile as sp  # noqa: E402
from openmed.service.auth import hash_api_key  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]

SENTINELS = {
    "secret": "sentinel-secret-name-4471",
    "key": "sentinel-secret-key-4471",
    "host": "sentinel-host-4471.example.test",
    "password": "sentinel-password-4471",
    "api_key": "sentinel-api-key-4471",
    "schema": "sentinel_schema_4471",
}

HARDENED_POD = {
    "runAsNonRoot": True,
    "runAsUser": 65532,
    "seccompProfile": {"type": "RuntimeDefault"},
}
HARDENED_CONTAINER = {
    "allowPrivilegeEscalation": False,
    "readOnlyRootFilesystem": True,
    "capabilities": {"drop": ["ALL"]},
}
LIMITS = {"limits": {"cpu": "1", "memory": "1Gi"}}
TELEMETRY_OFF = {"HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1"}


def _configmap(data: dict[str, str], name: str = "svc-config") -> dict[str, Any]:
    return {"kind": "ConfigMap", "metadata": {"name": name}, "data": data}


def _api_data(**overrides: str) -> dict[str, str]:
    data = {
        "OPENMED_PROFILE": "prod",
        "OPENMED_SERVICE_AUTH_ENABLED": "true",
        "OPENMED_SERVICE_TRUSTED_HOSTS": f"localhost,{SENTINELS['host']}",
        "OPENMED_SERVICE_SERVED_MODELS": "pii_detection_superclinical",
        "OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY": "2",
        **TELEMETRY_OFF,
    }
    data.update(overrides)
    return data


def _deployment(
    component: str,
    container: dict[str, Any],
    *,
    kind: str = "Deployment",
    pod: dict[str, Any] | None = None,
) -> dict[str, Any]:
    labels = {"app.kubernetes.io/name": "svc", "app.kubernetes.io/component": component}
    return {
        "kind": kind,
        "metadata": {"name": f"svc-{component}", "labels": labels},
        "spec": {
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "securityContext": copy.deepcopy(HARDENED_POD),
                    "containers": [container],
                    **(pod or {}),
                },
            }
        },
    }


def _api_container(**extra: Any) -> dict[str, Any]:
    container: dict[str, Any] = {
        "name": "service",
        "envFrom": [{"configMapRef": {"name": "svc-config"}}],
        "env": [
            {
                "name": "OPENMED_SERVICE_AUTH_API_KEYS",
                "valueFrom": {
                    "secretKeyRef": {"name": SENTINELS["secret"], "key": "api-keys"}
                },
            }
        ],
        "securityContext": copy.deepcopy(HARDENED_CONTAINER),
        "resources": copy.deepcopy(LIMITS),
        "readinessProbe": {"httpGet": {"path": "/readyz", "port": "http"}},
    }
    container.update(extra)
    return container


def _worker_container(**env: str) -> dict[str, Any]:
    entries = [
        {
            "name": "OPENMED_JOURNEY_POSTGRES_DSN",
            "valueFrom": {"secretKeyRef": {"name": "journey-db", "key": "dsn"}},
        },
        {"name": "OPENMED_JOURNEY_SCHEMA", "value": "openmed_journey"},
        {"name": "OPENMED_JOURNEY_ARTIFACT_ROOT", "value": "/artifacts"},
        *({"name": name, "value": value} for name, value in TELEMETRY_OFF.items()),
        *({"name": name, "value": value} for name, value in env.items()),
    ]
    return {
        "name": "worker",
        "command": ["/usr/bin/python3", "-m", "openmed.deploy.journey", "worker"],
        "env": entries,
        "securityContext": copy.deepcopy(HARDENED_CONTAINER),
        "resources": copy.deepcopy(LIMITS),
        "readinessProbe": {"httpGet": {"path": "/readyz", "port": "health"}},
    }


def _journey_api_container() -> dict[str, Any]:
    container = _api_container()
    container["env"].append(
        {
            "name": "OPENMED_JOURNEY_POSTGRES_DSN",
            "valueFrom": {"secretKeyRef": {"name": "journey-db", "key": "dsn"}},
        }
    )
    return container


def _gated(evaluation: dp.PostureEvaluation) -> set[tuple[str, str, str]]:
    return {(f.workload, f.check, f.reason) for f in evaluation.gated}


def _advisory(evaluation: dp.PostureEvaluation) -> set[tuple[str, str, str]]:
    return {(f.workload, f.check, f.reason) for f in evaluation.findings if not f.gated}


def _hardened_manifests(**data: str) -> list[dict[str, Any]]:
    return [_configmap(_api_data(**data)), _deployment("api", _api_container())]


# --- commands and listeners --------------------------------------------------


@pytest.mark.parametrize(
    ("command", "args", "role", "launcher"),
    [
        ([], [], dp.WorkloadRole.API, dp.IMAGE_DEFAULT_LISTENER_ARGS),
        ([], ["--host", "127.0.0.1"], dp.WorkloadRole.API, ("--host", "127.0.0.1")),
        (
            ["python", "-m", "openmed.service.logging", "--host", "0.0.0.0"],
            [],
            dp.WorkloadRole.API,
            ("--host", "0.0.0.0"),
        ),
        (
            ["python", "-m", "openmed.deploy.journey"],
            ["worker"],
            dp.WorkloadRole.WORKER,
            (),
        ),
        (["openmed-journey", "migrate"], [], dp.WorkloadRole.JOB, ()),
        (
            ["python", "-m", "openmed.mcp.server"],
            ["--transport", "x"],
            dp.WorkloadRole.MCP,
            (),
        ),
        (["/bin/sh", "-c", "sleep 1"], [], dp.WorkloadRole.UNCLASSIFIED, ()),
    ],
)
def test_launch_commands_classify_workloads(command, args, role, launcher) -> None:
    assert dp.classify_command(command, args) == (role, tuple(launcher))


def test_launcher_listener_reads_bind_host_and_tls() -> None:
    assert dp.launcher_listener(()) == ("127.0.0.1", False)
    assert dp.launcher_listener(("--host", "0.0.0.0", "--port", "8080")) == (
        "0.0.0.0",
        False,
    )
    assert dp.launcher_listener(("--host=::", "--ssl-certfile", "/tls/crt")) == (
        "::",
        True,
    )


def test_image_default_listener_matches_both_service_images() -> None:
    expected = ", ".join(f'"{item}"' for item in dp.IMAGE_DEFAULT_LISTENER_ARGS)
    for name in ("Dockerfile", "Dockerfile.distroless"):
        dockerfile = (ROOT / "deploy" / "docker" / name).read_text(encoding="utf-8")
        cmd = [line for line in dockerfile.splitlines() if line.startswith("CMD [")]
        assert cmd and cmd[-1].endswith(f"{expected}]")
        assert '"-m", "openmed.service.logging"' in dockerfile


# --- Compose interpolation ---------------------------------------------------


def test_compose_interpolation_uses_only_supplied_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "dev")
    assert dp.interpolate_compose("${OPENMED_PROFILE:-prod}", {}) == ("prod", False)
    assert dp.interpolate_compose("${A:-x}|${A-y}", {"A": ""}) == ("x|", False)
    assert dp.interpolate_compose("${A:+on}${B+set}", {"A": "1"}) == ("on", False)
    assert dp.interpolate_compose("$$HOME-$A-${B:-${A}}", {"A": "z"}) == (
        "$HOME-z-z",
        False,
    )
    text, operator = dp.interpolate_compose(
        "postgresql://u:${PASSWORD:?set PASSWORD}@db/x", {}
    )
    assert operator is True
    assert "PASSWORD" not in text
    assert dp.interpolate_compose("${PASSWORD:?set}", {"PASSWORD": "given"}) == (
        "given",
        False,
    )
    with pytest.raises(dp.PostureLaneError, match="compose_interpolation_invalid"):
        dp.interpolate_compose("${unterminated", {})
    with pytest.raises(dp.PostureLaneError, match="compose_interpolation_invalid"):
        dp.interpolate_compose("${A/b}", {})


# --- Kubernetes manifests ----------------------------------------------------


def test_hardened_synthetic_render_passes_with_runtime_checks() -> None:
    evaluation = dp.evaluate_helm_render(_hardened_manifests())

    assert [(w.workload_id, w.role) for w in evaluation.workloads] == [
        ("Deployment/api", dp.WorkloadRole.API)
    ]
    assert evaluation.gated == ()
    # The in-pod bind is reported, but platforms commonly terminate TLS.
    assert ("Deployment/api", "listener.transport", "plaintext_remote_listener") in (
        _advisory(evaluation)
    )


def test_rendered_environment_feeds_the_same_runtime_checks() -> None:
    manifests = _hardened_manifests(OPENMED_SERVICE_AUTH_ENABLED="false")
    workload = dp.workloads_from_manifests(manifests)[0]

    report, deferred, findings = dp.evaluate_service_workload(workload)
    environ = dp.evaluation_environment(workload)
    direct = sp.run_service_preflight(
        profile="prod", listen_host="0.0.0.0", listen_tls=False, environ=environ
    )

    assert report is not None and deferred == ()
    assert report.blocking_checks == direct.blocking_checks == ("auth.enabled",)
    assert report.digest == direct.digest
    assert {(f.check, f.reason) for f in findings if f.gated} == {
        ("auth.enabled", "auth_disabled")
    }


def test_environment_sources_and_contradictions_are_resolved() -> None:
    container = _api_container()
    container["envFrom"].append({"secretRef": {"name": SENTINELS["secret"]}})
    container["env"] += [
        {"name": "OPENMED_PROFILE", "value": "dev"},
        {
            "name": "OPENMED_SERVICE_MAX_TEXT_LENGTH",
            "valueFrom": {"configMapKeyRef": {"name": "svc-config", "key": "LENGTH"}},
        },
        {"name": "POD_IP", "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}}},
        {
            "name": "OPENMED_SERVICE_CORS_ORIGINS",
            "valueFrom": {"configMapKeyRef": {"name": "external", "key": "x"}},
        },
    ]
    data = _api_data(LENGTH="5000")
    workload = dp.workloads_from_manifests(
        [_configmap(data), _deployment("api", container)]
    )[0]

    assert workload.environment["OPENMED_SERVICE_MAX_TEXT_LENGTH"].value == "5000"
    assert workload.environment["POD_IP"].source is dp.SettingSource.RUNTIME
    assert workload.environment["OPENMED_SERVICE_AUTH_API_KEYS"].source is (
        dp.SettingSource.SECRET
    )
    assert workload.contradictions == ("OPENMED_PROFILE",)
    assert workload.unverified_sources == 2

    evaluation = dp.evaluate_workloads([workload])
    gated = _gated(evaluation)
    assert ("Deployment/api", "deploy.environment", "setting_contradictory") in gated
    assert (
        "Deployment/api",
        "deploy.environment",
        "environment_source_unverified",
    ) in (gated)
    assert ("Deployment/api", "deploy.profile", "development_profile_selected") in gated
    contradiction = next(
        f for f in evaluation.gated if f.reason == "setting_contradictory"
    )
    assert contradiction.setting == "OPENMED_PROFILE"


def test_file_settings_on_secret_mounts_count_as_secrets() -> None:
    container = _api_container(
        volumeMounts=[{"name": "auth", "mountPath": "/var/run/secrets/openmed"}]
    )
    container["env"] = [
        {
            "name": "OPENMED_SERVICE_AUTH_JWKS_FILE",
            "value": "/var/run/secrets/openmed/jwks.json",
        }
    ]
    pod = {"volumes": [{"name": "auth", "secret": {"secretName": SENTINELS["secret"]}}]}
    data = _api_data(
        OPENMED_SERVICE_AUTH_JWT_ISSUER="https://issuer.example.test",
        OPENMED_SERVICE_AUTH_JWT_AUDIENCE="openmed-api",
    )
    manifests = [_configmap(data), _deployment("api", container, pod=pod)]
    workload = dp.workloads_from_manifests(manifests)[0]

    assert "OPENMED_SERVICE_AUTH_JWKS_FILE" not in workload.environment
    assert workload.environment["OPENMED_SERVICE_AUTH_JWKS"].source is (
        dp.SettingSource.SECRET
    )
    assert dp.evaluate_workloads([workload]).gated == ()


def test_security_contexts_follow_kubernetes_precedence() -> None:
    bare = dp.workloads_from_manifests(
        [_deployment("api", {"name": "c"}, pod={"securityContext": {}})]
    )[0]
    assert set(bare.hardening) == {
        "runs_as_root",
        "privilege_escalation_allowed",
        "root_filesystem_writable",
        "capabilities_not_dropped",
        "seccomp_not_confined",
    }

    inherited = dp.workloads_from_manifests([_deployment("api", _api_container())])[0]
    assert inherited.hardening == ()

    root = _api_container()
    root["securityContext"].update(
        {
            "runAsUser": 0,
            "privileged": True,
            "capabilities": {"drop": ["ALL"], "add": ["SYS_ADMIN"]},
        }
    )
    root["ports"] = [{"containerPort": 8080, "hostPort": 8080}]
    pod = {"hostNetwork": True, "volumes": [{"name": "h", "hostPath": {"path": "/"}}]}
    weakened = dp.workloads_from_manifests([_deployment("api", root, pod=pod)])[0]
    assert set(weakened.hardening) == {
        "runs_as_root",
        "privilege_escalation_allowed",
        "capabilities_added",
        "privileged_container",
        "host_namespace_shared",
        "host_path_mounted",
    }


@pytest.mark.parametrize(
    ("probe", "reason"),
    [
        (None, "readiness_probe_missing"),
        ({"httpGet": {"path": "/livez"}}, "readiness_not_wired"),
        ({"tcpSocket": {"port": 8080}}, "readiness_not_wired"),
        (
            {"exec": {"command": ["probe", "--model-url", "http://127.0.0.1/readyz"]}},
            None,
        ),
    ],
)
def test_readiness_must_reach_readyz(probe, reason) -> None:
    container = _api_container(readinessProbe=probe)
    evaluation = dp.evaluate_helm_render(
        [_configmap(_api_data()), _deployment("api", container)]
    )
    readiness = {f.reason for f in evaluation.gated if f.check == "deploy.readiness"}
    assert readiness == ({reason} if reason else set())


def test_resource_limits_telemetry_offline_and_overrides() -> None:
    container = _api_container(resources={"requests": {"cpu": "1"}})
    data = _api_data(
        HF_HUB_DISABLE_TELEMETRY="1",
        DO_NOT_TRACK="0",
        OPENMED_OFFLINE="1",
        HF_HUB_OFFLINE="0",
        OPENMED_MCP_ALLOW_INSECURE_LOCALHOST="true",
    )
    evaluation = dp.evaluate_helm_render(
        [_configmap(data), _deployment("api", container)]
    )
    gated = {(f.check, f.reason) for f in evaluation.gated}
    assert {
        ("deploy.resources", "cpu_limit_missing"),
        ("deploy.resources", "memory_limit_missing"),
        ("deploy.egress", "telemetry_contradictory"),
        ("deploy.egress", "offline_mode_contradictory"),
        ("deploy.overrides", "insecure_override_enabled"),
    } <= gated

    silent = _api_data()
    for name in TELEMETRY_OFF:
        silent.pop(name)
    evaluation = dp.evaluate_helm_render(
        [_configmap(silent), _deployment("api", _api_container())]
    )
    assert {(f.check, f.reason) for f in evaluation.gated} == {
        ("deploy.egress", "telemetry_not_disabled"),
    }
    evaluation = dp.evaluate_helm_render(_hardened_manifests(), require_offline=True)
    assert {(f.check, f.reason) for f in evaluation.gated} == {
        ("deploy.egress", "offline_mode_required"),
    }


def test_exposed_service_without_tls_mirrors_the_listener_level() -> None:
    service = {
        "kind": "Service",
        "metadata": {"name": "svc"},
        "spec": {
            "type": "LoadBalancer",
            "selector": {"app.kubernetes.io/component": "api"},
        },
    }
    evaluation = dp.evaluate_helm_render([*_hardened_manifests(), service])
    finding = next(f for f in evaluation.findings if f.check == "deploy.listener")
    assert finding.reason == "exposed_without_tls"
    assert finding.level is sp.CheckLevel.ADVISORY
    assert evaluation.gated == ()

    tls = _api_container(args=["--host", "0.0.0.0", "--ssl-certfile", "/tls/crt"])
    manifests = [_configmap(_api_data()), _deployment("api", tls), service]
    evaluation = dp.evaluate_helm_render(manifests)
    assert not [f for f in evaluation.findings if f.check == "deploy.listener"]


def test_journey_store_tls_is_deferred_only_for_secret_dsns() -> None:
    manifests = [_configmap(_api_data()), _deployment("api", _journey_api_container())]
    evaluation = dp.evaluate_helm_render(manifests)
    assert [(d.check, d.reason) for d in evaluation.deferred] == [
        ("journey.store_tls", dp.DEFERRED_REASON)
    ]
    assert "journey.store_tls" not in {f.check for f in evaluation.findings}

    for query, reason in (
        ("", "postgres_tls_not_configured"),
        ("?sslmode=require", "postgres_tls_not_verified"),
        ("?sslmode=verify-full", None),
    ):
        dsn = f"postgresql://openmed:{SENTINELS['password']}@db/openmed{query}"
        data = _api_data(OPENMED_JOURNEY_POSTGRES_DSN=dsn)
        evaluation = dp.evaluate_helm_render(
            [_configmap(data), _deployment("api", _api_container())]
        )
        tls = {f.reason for f in evaluation.findings if f.check == "journey.store_tls"}
        assert tls == ({reason} if reason else set())
        assert evaluation.deferred == ()


def test_workers_and_jobs_inherit_the_api_profile_and_store() -> None:
    manifests = [
        _configmap(
            _api_data(
                OPENMED_JOURNEY_SCHEMA="openmed_journey",
                OPENMED_JOURNEY_ARTIFACT_ROOT="/artifacts",
            )
        ),
        _deployment("api", _journey_api_container()),
        _deployment("journey-worker", _worker_container()),
    ]
    evaluation = dp.evaluate_helm_render(manifests)
    assert not [f for f in evaluation.gated if f.check.startswith("deploy.")]

    worker = _worker_container(OPENMED_SERVICE_SECURITY_PROFILE="dev")
    worker["env"] = [
        item
        for item in worker["env"]
        if item["name"] != ("OPENMED_JOURNEY_ARTIFACT_ROOT")
    ]
    worker["env"][1]["value"] = SENTINELS["schema"]
    job = _worker_container()
    job["name"] = "migrate"
    job["command"] = ["openmed-journey", "migrate"]
    job["env"] = [
        item for item in job["env"] if item["name"] != ("OPENMED_JOURNEY_POSTGRES_DSN")
    ]
    manifests[2] = _deployment("journey-worker", worker)
    manifests.append(_deployment("journey-migration", job, kind="Job"))
    evaluation = dp.evaluate_helm_render(manifests)
    inheritance = {
        (f.workload, f.reason, f.setting)
        for f in evaluation.gated
        if f.check == "deploy.inheritance"
    }
    assert inheritance == {
        ("Deployment/journey-worker", "security_profile_mismatch", None),
        (
            "Deployment/journey-worker",
            "inherited_setting_mismatch",
            "OPENMED_JOURNEY_SCHEMA",
        ),
        (
            "Deployment/journey-worker",
            "inherited_setting_missing",
            "OPENMED_JOURNEY_ARTIFACT_ROOT",
        ),
        (
            "Job/journey-migration",
            "inherited_setting_missing",
            "OPENMED_JOURNEY_POSTGRES_DSN",
        ),
    }


def test_unparseable_service_configuration_fails_before_deployment() -> None:
    container = _api_container()
    container["env"] = [{"name": "OPENMED_SERVICE_AUTH_API_KEYS", "value": "{not-json"}]
    evaluation = dp.evaluate_helm_render(
        [_configmap(_api_data()), _deployment("api", container)]
    )
    assert _gated(evaluation) == {
        ("Deployment/api", "deploy.service_config", "service_config_invalid")
    }


def test_unknown_workloads_and_missing_service_are_findings() -> None:
    sidecar = dict(_worker_container(), command=["/bin/sh", "-c", "true"])
    evaluation = dp.evaluate_helm_render([_deployment("helper", sidecar)])
    assert _gated(evaluation) >= {
        ("deployment", "deploy.workload", "service_workload_missing"),
        ("Deployment/helper", "deploy.workload", "workload_unclassified"),
    }


def test_findings_and_reports_are_value_free() -> None:
    container = _journey_api_container()
    container["env"].append({"name": "OPENMED_PROFILE", "value": SENTINELS["key"]})
    data = _api_data(
        OPENMED_SERVICE_AUTH_API_KEYS=json.dumps(
            [
                {
                    "id": "k",
                    "key_hash": hash_api_key(SENTINELS["api_key"]),
                    "principal": "p",
                    "scopes": ["pii:read"],
                }
            ]
        ),
        OPENMED_SERVICE_CORS_ORIGINS=f"http://{SENTINELS['host']}",
        OPENMED_JOURNEY_SCHEMA=SENTINELS["schema"],
    )
    evaluation = dp.evaluate_helm_render(
        [_configmap(data), _deployment("api", container)]
    )
    payload = json.dumps(
        [f.to_dict() for f in evaluation.findings]
        + [d.to_dict() for d in evaluation.deferred]
    )
    assert evaluation.gated
    for value in (*SENTINELS.values(), hash_api_key(SENTINELS["api_key"])):
        assert value not in payload
    assert re.search(r"/(var|tmp|etc)/", payload) is None


# --- Compose -----------------------------------------------------------------


def _compose_service(**overrides: Any) -> dict[str, Any]:
    service: dict[str, Any] = {
        "image": "${OPENMED_IMAGE:-openmed:distroless}",
        "user": "65532:65532",
        "read_only": True,
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges:true"],
        "cpus": "1",
        "mem_limit": "1g",
        "environment": {
            "OPENMED_PROFILE": "${OPENMED_PROFILE:-prod}",
            "OPENMED_SERVICE_AUTH_ENABLED": "true",
            "OPENMED_SERVICE_AUTH_API_KEYS": "${API_KEYS:?set API_KEYS}",
            **TELEMETRY_OFF,
        },
        "healthcheck": {"test": ["CMD", "probe", "http://127.0.0.1:8080/readyz"]},
        "ports": ["127.0.0.1:8080:8080"],
    }
    service.update(overrides)
    return service


def test_compose_services_are_normalized_and_third_party_images_skipped() -> None:
    document = {
        "services": {
            "postgres": {"image": "postgres:17-alpine"},
            "openmed": _compose_service(),
        }
    }
    workloads, skipped = dp.workloads_from_compose(document)
    assert skipped == 1
    (service,) = workloads
    assert service.role is dp.WorkloadRole.API
    assert service.hardening == ()
    assert service.exposed is False
    assert service.environment["OPENMED_SERVICE_AUTH_API_KEYS"].source is (
        dp.SettingSource.OPERATOR
    )
    # A value the operator supplies at deploy time is evaluated as well formed.
    assert dp.evaluate_compose(document).gated == ()
    keys = json.dumps(
        [
            {
                "id": "k",
                "key_hash": hash_api_key("synthetic"),
                "principal": "p",
                "scopes": ["pii:read"],
            }
        ]
    )
    assert dp.evaluate_compose(document, environ={"API_KEYS": keys}).gated == ()
    evaluation = dp.evaluate_compose(document, environ={"API_KEYS": "{not-json"})
    assert _gated(evaluation) == {
        ("openmed", "deploy.service_config", "service_config_invalid")
    }


def test_compose_hardening_limits_exposure_and_contradictions() -> None:
    service = _compose_service(
        user="root",
        read_only=False,
        cap_drop=[],
        cap_add=["NET_ADMIN"],
        security_opt=["seccomp=unconfined"],
        network_mode="host",
        volumes=["/srv/data:/data", "./tls/ca.crt:/ca.crt:ro", "cache:/cache"],
        ports=["${BIND:-0.0.0.0}:8080:8080"],
        environment=["OPENMED_PROFILE=prod", "OPENMED_PROFILE=dev", "DO_NOT_TRACK=1"],
        env_file=[".env"],
        cpus=None,
        mem_limit=None,
        deploy={"resources": {"limits": {"memory": "2g"}}},
        healthcheck={"disable": True},
    )
    workloads, _ = dp.workloads_from_compose({"services": {"openmed": service}})
    (workload,) = workloads
    assert set(workload.hardening) == {
        "runs_as_root",
        "privilege_escalation_allowed",
        "root_filesystem_writable",
        "capabilities_not_dropped",
        "capabilities_added",
        "seccomp_not_confined",
        "host_namespace_shared",
        "host_path_mounted",
    }
    assert workload.exposed is True
    assert workload.contradictions == ("OPENMED_PROFILE",)
    assert (workload.cpu_limit, workload.memory_limit) == (False, True)
    assert workload.unverified_sources == 1
    assert workload.readiness_probe is False

    loopback = dp.workloads_from_compose(
        {
            "services": {
                "openmed": _compose_service(ports=["${BIND:-0.0.0.0}:8080:8080"])
            }
        },
        environ={"BIND": "127.0.0.1"},
    )[0][0]
    assert loopback.exposed is False


def test_shipped_compose_bundle_workloads_are_classified() -> None:
    document = yaml.safe_load(
        (ROOT / "deploy" / "openmed-compose.yaml").read_text(encoding="utf-8")
    )
    workloads, skipped = dp.workloads_from_compose(document)
    roles = {item.workload_id: item.role for item in workloads}
    assert skipped == 1
    assert roles == {
        "journey-migrate": dp.WorkloadRole.JOB,
        "journey-worker": dp.WorkloadRole.WORKER,
        "openmed": dp.WorkloadRole.API,
        "journey-golden": dp.WorkloadRole.JOB,
        "mcp": dp.WorkloadRole.MCP,
    }
    assert all(item.hardening == () for item in workloads)
    assert all(item.cpu_limit and item.memory_limit for item in workloads)


# --- lane --------------------------------------------------------------------


def _write_lane(tmp_path: Path, targets: list[dict[str, Any]]) -> Path:
    compose = {"services": {"openmed": _compose_service()}}
    (tmp_path / "compose.yaml").write_text(yaml.safe_dump(compose), encoding="utf-8")
    lane = {"schema_version": dp.POSTURE_LANE_SCHEMA_VERSION, "targets": targets}
    path = tmp_path / "lane.yaml"
    path.write_text(yaml.safe_dump(lane), encoding="utf-8")
    return path


def _compose_target(**overrides: Any) -> dict[str, Any]:
    target: dict[str, Any] = {
        "id": "compose-synthetic",
        "platform": "compose",
        "purpose": "shipped",
        "file": "compose.yaml",
        "environment": {"API_KEYS": "{not-json"},
        "expect": [
            {
                "workload": "openmed",
                "check": "deploy.service_config",
                "reason": "service_config_invalid",
                "why": "known",
                "owner_pr": 1,
            }
        ],
    }
    target.update(overrides)
    return target


@pytest.mark.parametrize(
    ("expectation", "purpose", "code"),
    [
        ({"why": "known"}, "shipped", "lane_known_finding_without_owner"),
        ({"why": "fail_closed"}, "shipped", "lane_fail_closed_not_runtime"),
        ({"why": "negative_control"}, "shipped", "lane_negative_control_misplaced"),
        ({"why": "unknown"}, "shipped", "lane_invalid"),
        ({"why": "known", "owner_pr": "4443"}, "shipped", "lane_invalid"),
    ],
)
def test_lane_definitions_are_validated(tmp_path, expectation, purpose, code) -> None:
    entry = {"workload": "openmed", "check": "deploy.profile", "reason": "r"}
    entry.update(expectation)
    path = _write_lane(tmp_path, [_compose_target(purpose=purpose, expect=[entry])])
    with pytest.raises(dp.PostureLaneError) as error:
        dp.load_lane(path)
    assert str(error.value) == code


def test_negative_control_targets_need_their_defect(tmp_path) -> None:
    path = _write_lane(tmp_path, [_compose_target(purpose="negative_control")])
    with pytest.raises(dp.PostureLaneError) as error:
        dp.load_lane(path)
    assert str(error.value) == "lane_negative_control_without_expectation"


def test_lane_fails_on_new_and_stale_findings(tmp_path, capsys) -> None:
    passing = _write_lane(tmp_path, [_compose_target()])
    assert dp.main([str(passing)]) == dp.EXIT_LANE_PASSED
    assert "compose-synthetic [compose, shipped]" in capsys.readouterr().out

    path = _write_lane(tmp_path, [_compose_target(environment={})])
    report = dp.run_lane(dp.load_lane(path))
    assert report.passed is False
    assert [e.reason for e in report.results[0].stale] == ["service_config_invalid"]
    assert dp.main([str(path), "--json"]) == dp.EXIT_LANE_FAILED
    body = json.loads(capsys.readouterr().out)
    assert body["schema_version"] == dp.POSTURE_REPORT_SCHEMA_VERSION
    assert body["summary"]["stale"] == 1
    assert body["digest"].startswith("sha256:")

    new = _compose_target(
        environment={"API_KEYS": "{not-json", "OPENMED_PROFILE": "dev"}
    )
    path = _write_lane(tmp_path, [new])
    report = dp.run_lane(dp.load_lane(path))
    assert [f.reason for f in report.results[0].unexpected] == [
        "development_profile_selected"
    ]
    assert "NEW" in report.to_text()


def test_lane_cli_reports_unavailable_inputs(tmp_path, capsys) -> None:
    assert dp.main([str(tmp_path / "missing.yaml")]) == dp.EXIT_LANE_UNAVAILABLE
    assert "lane_unreadable" in capsys.readouterr().err
    path = _write_lane(
        tmp_path,
        [{"id": "helm-x", "platform": "helm", "purpose": "shipped", "expect": []}],
    )
    assert dp.main([str(path), "--helm", str(tmp_path / "no-helm")]) == (
        dp.EXIT_LANE_UNAVAILABLE
    )
    assert "helm_unavailable" in capsys.readouterr().err


def test_contract_digest_tracks_check_levels() -> None:
    contract = dp.posture_contract()
    assert contract["id"] == dp.POSTURE_CONTRACT_ID
    assert contract["security_profile"] == sp.PROD_PROFILE.identifier
    assert contract["deploy_checks"]["deploy.listener"] == "advisory"
    assert set(contract["runtime_checks"]) == {
        check.check_id for check in sp.registered_security_checks()
    }
    assert contract["digest"] == dp.posture_contract()["digest"]
