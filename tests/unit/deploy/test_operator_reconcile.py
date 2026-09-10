"""Synthetic Kubernetes API tests for the OpenMed model operator."""

from __future__ import annotations

import copy
import json
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator, Mapping
from urllib.request import Request, urlopen

import pytest
import yaml
from jsonschema import Draft202012Validator

from deploy.operator.openmed_operator import (
    MANIFEST_DATA_KEY,
    MANIFEST_HASH_ANNOTATION,
    MODEL_RESOURCE_ANNOTATION,
    PHASE_READY,
    PHASE_ROLLED_BACK,
    PHASE_ROLLING_OUT,
    PRELOAD_ENV_NAME,
    DesiredModel,
    KubernetesAPIClient,
    SpecValidationError,
    decommission_openmed_model,
    reconcile_openmed_model,
)

ROOT = Path(__file__).resolve().parents[3]
OPERATOR_DIR = ROOT / "deploy" / "operator"
CRD_PATH = OPERATOR_DIR / "crd" / "openmedmodel.yaml"

NOW = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)


def _resource(*, version: str = "OpenMed/synthetic-pii-v1", generation: int = 1):
    return {
        "apiVersion": "openmed.ai/v1alpha1",
        "kind": "OpenMedModel",
        "metadata": {
            "name": "openmed-service",
            "namespace": "synthetic",
            "uid": "00000000-0000-4000-8000-000000000830",
            "generation": generation,
        },
        "spec": {
            "family": "PII",
            "version": version,
            "tier": "Small",
            "replicas": 2,
            "targetRef": {
                "name": "openmed-service",
                "containerName": "openmed-service",
            },
            "rolloutStrategy": {
                "type": "RollingUpdate",
                "maxUnavailable": 0,
                "maxSurge": 1,
                "rollbackOnFailure": True,
                "progressDeadlineSeconds": 60,
            },
        },
    }


def _deployment() -> dict[str, Any]:
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": "openmed-service",
            "namespace": "synthetic",
            "generation": 1,
            "annotations": {},
        },
        "spec": {
            "replicas": 1,
            "progressDeadlineSeconds": 600,
            "strategy": {
                "type": "RollingUpdate",
                "rollingUpdate": {"maxUnavailable": "25%", "maxSurge": "25%"},
            },
            "template": {
                "metadata": {"annotations": {}},
                "spec": {
                    "containers": [
                        {
                            "name": "openmed-service",
                            "image": "openmed:test",
                            "env": [],
                        }
                    ]
                },
            },
        },
        "status": {
            "observedGeneration": 1,
            "updatedReplicas": 1,
            "readyReplicas": 1,
            "availableReplicas": 1,
            "unavailableReplicas": 0,
            "conditions": [
                {
                    "type": "Progressing",
                    "status": "True",
                    "reason": "NewReplicaSetAvailable",
                }
            ],
        },
    }


class _ClusterState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.deployments = {("synthetic", "openmed-service"): _deployment()}
        self.config_maps: dict[tuple[str, str], dict[str, Any]] = {}
        self.events: dict[tuple[str, str], dict[str, Any]] = {}
        self.resources: dict[tuple[str, str], dict[str, Any]] = {}

    def mark_ready(self, namespace: str, name: str) -> None:
        with self.lock:
            deployment = self.deployments[(namespace, name)]
            replicas = int(deployment["spec"]["replicas"])
            generation = int(deployment["metadata"]["generation"])
            deployment["status"] = {
                "observedGeneration": generation,
                "updatedReplicas": replicas,
                "readyReplicas": replicas,
                "availableReplicas": replicas,
                "unavailableReplicas": 0,
                "conditions": [
                    {
                        "type": "Progressing",
                        "status": "True",
                        "reason": "NewReplicaSetAvailable",
                    }
                ],
            }

    def mark_failed(self, namespace: str, name: str) -> None:
        with self.lock:
            deployment = self.deployments[(namespace, name)]
            generation = int(deployment["metadata"]["generation"])
            deployment["status"] = {
                "observedGeneration": generation,
                "updatedReplicas": 1,
                "readyReplicas": 1,
                "availableReplicas": 1,
                "unavailableReplicas": 1,
                "conditions": [
                    {
                        "type": "Progressing",
                        "status": "False",
                        "reason": "ProgressDeadlineExceeded",
                    }
                ],
            }


class _FakeKubernetesServer(ThreadingHTTPServer):
    def __init__(self) -> None:
        self.state = _ClusterState()
        super().__init__(("127.0.0.1", 0), _FakeKubernetesHandler)


class _FakeKubernetesHandler(BaseHTTPRequestHandler):
    server: _FakeKubernetesServer

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parts = self.path.strip("/").split("/")
        with self.server.state.lock:
            if _matches(
                parts, "apis", "apps", "v1", "namespaces", "*", "deployments", "*"
            ):
                self._item(self.server.state.deployments, (parts[4], parts[6]))
                return
            if _matches(parts, "api", "v1", "namespaces", "*", "configmaps", "*"):
                self._item(self.server.state.config_maps, (parts[3], parts[5]))
                return
        self._write(404, {"kind": "Status", "reason": "NotFound"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parts = self.path.strip("/").split("/")
        payload = self._payload()
        with self.server.state.lock:
            if _matches(parts, "api", "v1", "namespaces", "*", "configmaps"):
                key = (parts[3], payload["metadata"]["name"])
                if key in self.server.state.config_maps:
                    self._write(409, {"kind": "Status", "reason": "AlreadyExists"})
                    return
                self.server.state.config_maps[key] = copy.deepcopy(payload)
                self._write(201, payload)
                return
            if _matches(parts, "api", "v1", "namespaces", "*", "events"):
                key = (parts[3], payload["metadata"]["name"])
                if key in self.server.state.events:
                    self._write(409, {"kind": "Status", "reason": "AlreadyExists"})
                    return
                self.server.state.events[key] = copy.deepcopy(payload)
                self._write(201, payload)
                return
            if _matches(
                parts,
                "apis",
                "openmed.ai",
                "v1alpha1",
                "namespaces",
                "*",
                "openmedmodels",
            ):
                key = (parts[4], payload["metadata"]["name"])
                self.server.state.resources[key] = copy.deepcopy(payload)
                self._write(201, payload)
                return
        self._write(404, {"kind": "Status", "reason": "NotFound"})

    def do_PATCH(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parts = self.path.strip("/").split("/")
        payload = self._payload()
        with self.server.state.lock:
            if _matches(
                parts, "apis", "apps", "v1", "namespaces", "*", "deployments", "*"
            ):
                key = (parts[4], parts[6])
                current = self.server.state.deployments.get(key)
                if current is None:
                    self._write(404, {"kind": "Status", "reason": "NotFound"})
                    return
                merged = _strategic_merge(current, payload)
                merged["metadata"]["generation"] = (
                    int(current["metadata"]["generation"]) + 1
                )
                self.server.state.deployments[key] = merged
                self._write(200, merged)
                return
            if _matches(parts, "api", "v1", "namespaces", "*", "configmaps", "*"):
                key = (parts[3], parts[5])
                current = self.server.state.config_maps.get(key)
                if current is None:
                    self._write(404, {"kind": "Status", "reason": "NotFound"})
                    return
                merged = _strategic_merge(current, payload)
                self.server.state.config_maps[key] = merged
                self._write(200, merged)
                return
        self._write(404, {"kind": "Status", "reason": "NotFound"})

    def do_DELETE(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parts = self.path.strip("/").split("/")
        with self.server.state.lock:
            if _matches(parts, "api", "v1", "namespaces", "*", "configmaps", "*"):
                key = (parts[3], parts[5])
                if self.server.state.config_maps.pop(key, None) is None:
                    self._write(404, {"kind": "Status", "reason": "NotFound"})
                    return
                self._write(200, {"kind": "Status", "status": "Success"})
                return
        self._write(404, {"kind": "Status", "reason": "NotFound"})

    def log_message(self, _format: str, *args: Any) -> None:
        del args

    def _item(
        self,
        store: Mapping[tuple[str, str], dict[str, Any]],
        key: tuple[str, str],
    ) -> None:
        item = store.get(key)
        if item is None:
            self._write(404, {"kind": "Status", "reason": "NotFound"})
        else:
            self._write(200, item)

    def _payload(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        assert isinstance(payload, dict)
        return payload

    def _write(self, status: int, payload: Mapping[str, Any]) -> None:
        rendered = json.dumps(payload, sort_keys=True).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(rendered)))
        self.end_headers()
        self.wfile.write(rendered)


def _matches(parts: list[str], *pattern: str) -> bool:
    return len(parts) == len(pattern) and all(
        expected == "*" or actual == expected
        for actual, expected in zip(parts, pattern)
    )


_DELETE = object()


def _strategic_merge(current: Any, patch: Any) -> Any:
    if patch is None:
        return _DELETE
    if isinstance(current, dict) and isinstance(patch, dict):
        merged = copy.deepcopy(current)
        for key, value in patch.items():
            updated = _strategic_merge(merged.get(key), value)
            if updated is _DELETE:
                merged.pop(key, None)
            else:
                merged[key] = updated
        return merged
    if isinstance(current, list) and isinstance(patch, list):
        if all(isinstance(item, dict) and "name" in item for item in patch):
            merged = copy.deepcopy(current)
            indices = {
                item.get("name"): index
                for index, item in enumerate(merged)
                if isinstance(item, dict)
            }
            for item in patch:
                name = item["name"]
                if name in indices:
                    merged[indices[name]] = _strategic_merge(
                        merged[indices[name]], item
                    )
                else:
                    merged.append(copy.deepcopy(item))
            return merged
        return copy.deepcopy(patch)
    return copy.deepcopy(patch)


@contextmanager
def _fake_cluster() -> Iterator[tuple[_FakeKubernetesServer, KubernetesAPIClient]]:
    server = _FakeKubernetesServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield server, KubernetesAPIClient(f"http://{host}:{port}", token="synthetic")
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def _apply_resource(
    server: _FakeKubernetesServer, body: Mapping[str, Any]
) -> dict[str, Any]:
    host, port = server.server_address
    namespace = body["metadata"]["namespace"]
    request = Request(
        f"http://{host}:{port}/apis/openmed.ai/v1alpha1/namespaces/"
        f"{namespace}/openmedmodels",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=2) as response:  # noqa: S310 - local fake server
        payload = json.loads(response.read())
    assert isinstance(payload, dict)
    return payload


def _condition(status: Mapping[str, Any], condition_type: str) -> Mapping[str, Any]:
    return next(item for item in status["conditions"] if item["type"] == condition_type)


def _container(server: _FakeKubernetesServer) -> Mapping[str, Any]:
    deployment = server.state.deployments[("synthetic", "openmed-service")]
    return deployment["spec"]["template"]["spec"]["containers"][0]


def _converge_initial_version(
    server: _FakeKubernetesServer,
    client: KubernetesAPIClient,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resource = _apply_resource(server, _resource())
    started = reconcile_openmed_model(resource, client, clock=lambda: NOW)
    resource["status"] = started.status
    server.state.mark_ready("synthetic", "openmed-service")
    ready = reconcile_openmed_model(resource, client, clock=lambda: NOW)
    resource["status"] = ready.status
    return resource, ready.status


def test_crd_requires_lifecycle_fields_and_rejects_unknown_tier() -> None:
    crd = yaml.safe_load(CRD_PATH.read_text(encoding="utf-8"))
    schema = crd["spec"]["versions"][0]["schema"]["openAPIV3Schema"]
    validator = Draft202012Validator(schema)
    valid = _resource()

    assert set(schema["properties"]["spec"]["required"]) == {
        "family",
        "version",
        "tier",
        "replicas",
        "rolloutStrategy",
    }
    assert list(validator.iter_errors(valid)) == []

    unknown_tier = copy.deepcopy(valid)
    unknown_tier["spec"]["tier"] = "Unlimited"
    errors = list(validator.iter_errors(unknown_tier))
    assert any(list(error.path) == ["spec", "tier"] for error in errors)

    missing_family = copy.deepcopy(valid)
    del missing_family["spec"]["family"]
    errors = list(validator.iter_errors(missing_family))
    assert any("family" in error.message for error in errors)

    unknown_field = copy.deepcopy(valid)
    unknown_field["spec"]["trainingJob"] = "out-of-scope"
    errors = list(validator.iter_errors(unknown_field))
    assert any("Additional properties" in error.message for error in errors)

    blocked_rollout = copy.deepcopy(valid)
    blocked_rollout["spec"]["rolloutStrategy"]["maxUnavailable"] = "0%"
    blocked_rollout["spec"]["rolloutStrategy"]["maxSurge"] = 0
    with pytest.raises(SpecValidationError, match="both maxUnavailable"):
        DesiredModel.from_resource(blocked_rollout)


@pytest.mark.parametrize(
    "base_url",
    [
        "file:///var/run/secrets/kubernetes.io/serviceaccount/token",
        "https://operator@example.test",
        "http://kubernetes.default.svc",
        "https://example.test/api/v1",
    ],
)
def test_kubernetes_client_rejects_unsafe_api_origins(base_url: str) -> None:
    with pytest.raises(ValueError, match="Kubernetes API|unencrypted"):
        KubernetesAPIClient(base_url)


def test_synthetic_apply_drives_manifest_rollout_and_ready_conditions() -> None:
    with _fake_cluster() as (server, client):
        resource = _apply_resource(server, _resource())

        started = reconcile_openmed_model(resource, client, clock=lambda: NOW)

        assert started.changed is True
        assert started.status["phase"] == PHASE_ROLLING_OUT
        assert _condition(started.status, "Progressing")["status"] == "True"
        config_map = server.state.config_maps[
            ("synthetic", "openmed-service-model-manifest")
        ]
        manifest = json.loads(config_map["data"][MANIFEST_DATA_KEY])
        assert manifest["models"] == [
            {
                "family": "PII",
                "tier": "Small",
                "version": "OpenMed/synthetic-pii-v1",
            }
        ]
        deployment = server.state.deployments[("synthetic", "openmed-service")]
        assert deployment["spec"]["replicas"] == 2
        assert deployment["metadata"]["annotations"][MODEL_RESOURCE_ANNOTATION] == (
            "synthetic/openmed-service"
        )
        env = next(
            item
            for item in _container(server)["env"]
            if item["name"] == PRELOAD_ENV_NAME
        )
        assert env["valueFrom"]["configMapKeyRef"] == {
            "name": "openmed-service-model-manifest",
            "key": PRELOAD_ENV_NAME,
        }
        assert (
            MANIFEST_HASH_ANNOTATION
            in deployment["spec"]["template"]["metadata"]["annotations"]
        )

        resource["status"] = started.status
        server.state.mark_ready("synthetic", "openmed-service")
        ready = reconcile_openmed_model(resource, client, clock=lambda: NOW)

        assert ready.status["phase"] == PHASE_READY
        assert ready.status["activeVersion"] == "OpenMed/synthetic-pii-v1"
        assert _condition(ready.status, "Ready") == {
            "type": "Ready",
            "status": "True",
            "reason": "RolloutSucceeded",
            "message": "Desired model version is available.",
            "observedGeneration": 1,
            "lastTransitionTime": "2026-08-18T12:00:00Z",
        }
        assert {event["reason"] for event in server.state.events.values()} == {
            "RolloutStarted",
            "RolloutSucceeded",
        }

        resource["status"] = ready.status
        deployment_generation = deployment["metadata"]["generation"]
        repeated = reconcile_openmed_model(resource, client, clock=lambda: NOW)
        assert repeated.changed is False
        assert deployment["metadata"]["generation"] == deployment_generation
        assert len(server.state.events) == 2

        config_map["data"][PRELOAD_ENV_NAME] = "OpenMed/tampered"
        corrected = reconcile_openmed_model(resource, client, clock=lambda: NOW)
        assert corrected.status["phase"] == PHASE_ROLLING_OUT
        corrected_config_map = server.state.config_maps[
            ("synthetic", "openmed-service-model-manifest")
        ]
        assert corrected_config_map["data"][PRELOAD_ENV_NAME] == (
            "OpenMed/synthetic-pii-v1"
        )


def test_version_change_rolls_back_to_last_successful_manifest_on_failure() -> None:
    with _fake_cluster() as (server, client):
        resource, ready_status = _converge_initial_version(server, client)
        resource["metadata"]["generation"] = 2
        resource["spec"]["version"] = "OpenMed/synthetic-pii-v2"
        resource["status"] = ready_status

        rollout = reconcile_openmed_model(resource, client, clock=lambda: NOW)

        assert rollout.status["phase"] == PHASE_ROLLING_OUT
        assert rollout.status["activeVersion"] == "OpenMed/synthetic-pii-v1"
        config_map = server.state.config_maps[
            ("synthetic", "openmed-service-model-manifest")
        ]
        assert config_map["data"][PRELOAD_ENV_NAME] == "OpenMed/synthetic-pii-v2"

        resource["status"] = rollout.status
        server.state.mark_failed("synthetic", "openmed-service")
        rolled_back = reconcile_openmed_model(resource, client, clock=lambda: NOW)

        assert rolled_back.status["phase"] == PHASE_ROLLED_BACK
        assert rolled_back.status["desiredVersion"] == "OpenMed/synthetic-pii-v2"
        assert rolled_back.status["activeVersion"] == "OpenMed/synthetic-pii-v1"
        assert rolled_back.status["rollbackCount"] == 1
        assert _condition(rolled_back.status, "Degraded")["status"] == "True"
        config_map = server.state.config_maps[
            ("synthetic", "openmed-service-model-manifest")
        ]
        assert config_map["data"][PRELOAD_ENV_NAME] == "OpenMed/synthetic-pii-v1"
        assert {event["reason"] for event in server.state.events.values()} >= {
            "RolloutFailed",
            "RollbackStarted",
        }

        resource["status"] = rolled_back.status
        server.state.mark_ready("synthetic", "openmed-service")
        restored = reconcile_openmed_model(resource, client, clock=lambda: NOW)

        assert restored.status["phase"] == PHASE_ROLLED_BACK
        assert _condition(restored.status, "RolledBack")["status"] == "True"
        assert _condition(restored.status, "RolledBack")["reason"] == (
            "RollbackSucceeded"
        )


def test_failed_rollback_reports_a_terminal_condition() -> None:
    with _fake_cluster() as (server, client):
        resource, ready_status = _converge_initial_version(server, client)
        resource["metadata"]["generation"] = 2
        resource["spec"]["version"] = "OpenMed/synthetic-pii-v2"
        resource["status"] = ready_status
        rollout = reconcile_openmed_model(resource, client, clock=lambda: NOW)
        resource["status"] = rollout.status
        server.state.mark_failed("synthetic", "openmed-service")
        rollback = reconcile_openmed_model(resource, client, clock=lambda: NOW)

        resource["status"] = rollback.status
        server.state.mark_failed("synthetic", "openmed-service")
        failed = reconcile_openmed_model(resource, client, clock=lambda: NOW)

        assert failed.status["phase"] == "Failed"
        assert _condition(failed.status, "RolledBack")["reason"] == "RollbackFailed"
        assert any(
            event["reason"] == "RollbackFailed"
            for event in server.state.events.values()
        )


def test_manual_rollback_accepts_only_a_retained_successful_version() -> None:
    with _fake_cluster() as (server, client):
        resource, ready_v1 = _converge_initial_version(server, client)
        resource["metadata"]["generation"] = 2
        resource["spec"]["version"] = "OpenMed/synthetic-pii-v2"
        resource["status"] = ready_v1
        rollout_v2 = reconcile_openmed_model(resource, client, clock=lambda: NOW)
        resource["status"] = rollout_v2.status
        server.state.mark_ready("synthetic", "openmed-service")
        ready_v2 = reconcile_openmed_model(resource, client, clock=lambda: NOW)

        assert ready_v2.status["previousSuccessfulSpec"]["version"] == (
            "OpenMed/synthetic-pii-v1"
        )
        resource["status"] = ready_v2.status
        resource["metadata"]["annotations"] = {
            "openmed.ai/rollback-to": "OpenMed/synthetic-pii-v1"
        }
        manual = reconcile_openmed_model(resource, client, clock=lambda: NOW)

        assert manual.status["phase"] == PHASE_ROLLED_BACK
        assert manual.status["rollbackRequest"] == ("manual:OpenMed/synthetic-pii-v1")
        config_map = server.state.config_maps[
            ("synthetic", "openmed-service-model-manifest")
        ]
        assert config_map["data"][PRELOAD_ENV_NAME] == "OpenMed/synthetic-pii-v1"

        resource["status"] = manual.status
        server.state.mark_ready("synthetic", "openmed-service")
        resource["spec"]["version"] = "OpenMed/synthetic-pii-v1"
        converged = reconcile_openmed_model(resource, client, clock=lambda: NOW)
        resource["status"] = converged.status
        del resource["metadata"]["annotations"]["openmed.ai/rollback-to"]
        resumed = reconcile_openmed_model(resource, client, clock=lambda: NOW)
        assert resumed.status["phase"] == PHASE_ROLLING_OUT

        resource["metadata"]["annotations"]["openmed.ai/rollback-to"] = (
            "OpenMed/not-retained"
        )
        resource["status"] = ready_v2.status
        rejected = reconcile_openmed_model(resource, client, clock=lambda: NOW)
        assert _condition(rejected.status, "RolledBack")["reason"] == (
            "RollbackRejected"
        )


def test_delete_removes_pointer_and_restarts_target_without_preload() -> None:
    with _fake_cluster() as (server, client):
        resource, _ = _converge_initial_version(server, client)

        decommission_openmed_model(resource, client, clock=lambda: NOW)

        assert ("synthetic", "openmed-service-model-manifest") not in (
            server.state.config_maps
        )
        env = next(
            item
            for item in _container(server)["env"]
            if item["name"] == PRELOAD_ENV_NAME
        )
        assert env == {"name": PRELOAD_ENV_NAME, "value": ""}
        deployment = server.state.deployments[("synthetic", "openmed-service")]
        assert MODEL_RESOURCE_ANNOTATION not in deployment["metadata"]["annotations"]
        assert any(
            event["reason"] == "ModelRemoved" for event in server.state.events.values()
        )


def test_delete_does_not_mutate_resources_owned_by_another_controller() -> None:
    with _fake_cluster() as (server, client):
        resource, _ = _converge_initial_version(server, client)
        deployment = server.state.deployments[("synthetic", "openmed-service")]
        deployment["metadata"]["annotations"][MODEL_RESOURCE_ANNOTATION] = (
            "synthetic/other"
        )
        deployment["spec"]["template"]["metadata"]["annotations"][
            MODEL_RESOURCE_ANNOTATION
        ] = "synthetic/other"
        config_map_key = ("synthetic", "openmed-service-model-manifest")
        config_map = server.state.config_maps[config_map_key]
        config_map["metadata"]["annotations"][MODEL_RESOURCE_ANNOTATION] = (
            "synthetic/other"
        )
        config_map["metadata"]["ownerReferences"][0]["uid"] = (
            "00000000-0000-4000-8000-000000000999"
        )

        decommission_openmed_model(resource, client, clock=lambda: NOW)

        assert config_map_key in server.state.config_maps
        env = next(
            item
            for item in _container(server)["env"]
            if item["name"] == PRELOAD_ENV_NAME
        )
        assert "valueFrom" in env
        assert any(
            event["reason"] == "ModelRemovalSkipped"
            for event in server.state.events.values()
        )


def test_manifests_wire_least_privilege_rbac_and_single_operator_replica() -> None:
    cluster_role = yaml.safe_load(
        (OPERATOR_DIR / "rbac.yaml").read_text(encoding="utf-8")
    )
    rules = cluster_role["rules"]
    assert any(
        rule["apiGroups"] == ["openmed.ai"]
        and "openmedmodels/status" in rule["resources"]
        and {"patch", "update"} <= set(rule["verbs"])
        for rule in rules
    )
    deployment_rule = next(
        rule
        for rule in rules
        if rule["apiGroups"] == ["apps"] and rule["resources"] == ["deployments"]
    )
    assert set(deployment_rule["verbs"]) == {"get", "patch"}
    config_map_rule = next(
        rule for rule in rules if rule["resources"] == ["configmaps"]
    )
    assert set(config_map_rule["verbs"]) == {"get", "create", "patch", "delete"}
    event_rule = next(rule for rule in rules if rule["resources"] == ["events"])
    assert event_rule["verbs"] == ["create"]
    assert all("secrets" not in rule["resources"] for rule in rules)

    service_account = yaml.safe_load(
        (OPERATOR_DIR / "service-account.yaml").read_text(encoding="utf-8")
    )
    role_binding = yaml.safe_load(
        (OPERATOR_DIR / "role-binding.yaml").read_text(encoding="utf-8")
    )
    assert service_account["metadata"]["name"] == "openmed-operator"
    assert role_binding["subjects"][0]["name"] == "openmed-operator"

    deployment = yaml.safe_load(
        (OPERATOR_DIR / "deployment.yaml").read_text(encoding="utf-8")
    )
    assert deployment["spec"]["replicas"] == 1
    assert deployment["spec"]["strategy"]["type"] == "Recreate"
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert "--standalone" in container["args"]
    assert "--all-namespaces" in container["args"]
    assert container["securityContext"]["readOnlyRootFilesystem"] is True
    assert container["securityContext"]["capabilities"]["drop"] == ["ALL"]

    kustomization = yaml.safe_load(
        (OPERATOR_DIR / "kustomization.yaml").read_text(encoding="utf-8")
    )
    assert set(kustomization["resources"]) == {
        "crd/openmedmodel.yaml",
        "namespace.yaml",
        "service-account.yaml",
        "role-binding.yaml",
        "deployment.yaml",
        "rbac.yaml",
    }
    dockerfile = (OPERATOR_DIR / "Dockerfile").read_text(encoding="utf-8")
    assert '"kopf==${KOPF_VERSION}"' in dockerfile
    assert "USER 65532:65532" in dockerfile
