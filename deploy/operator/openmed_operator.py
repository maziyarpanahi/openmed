"""Kopf operator for declarative OpenMed model lifecycle management.

The reconciliation core is deliberately independent from Kopf so it can be
exercised against a synthetic Kubernetes API server without a live cluster.
The production handlers registered at the bottom of this module adapt Kopf
events to the same deterministic reconciliation function.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import re
import ssl
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

try:
    import kopf
except ModuleNotFoundError as exc:  # pragma: no cover - exercised without extra
    if exc.name != "kopf":
        raise
    kopf = None  # type: ignore[assignment]


API_GROUP = "openmed.ai"
API_VERSION = "v1alpha1"
API_PLURAL = "openmedmodels"
API_KIND = "OpenMedModel"

MANIFEST_DATA_KEY = "manifest.json"
PRELOAD_DATA_KEY = "OPENMED_SERVICE_PRELOAD_MODELS"
PRELOAD_ENV_NAME = PRELOAD_DATA_KEY
MANIFEST_HASH_ANNOTATION = "openmed.ai/model-manifest-hash"
MODEL_RESOURCE_ANNOTATION = "openmed.ai/model-resource"
ROLLBACK_ANNOTATION = "openmed.ai/rollback-to"

DEFAULT_CONTAINER_NAME = "openmed-service"
DEFAULT_PROGRESS_DEADLINE_SECONDS = 600
DEFAULT_MAX_UNAVAILABLE: int | str = 0
DEFAULT_MAX_SURGE: int | str = 1

ALLOWED_TIERS = frozenset(
    {"Tiny", "Small", "Medium", "Base", "Large", "XLarge", "Accurate-XLarge"}
)
PHASE_PENDING = "Pending"
PHASE_ROLLING_OUT = "RollingOut"
PHASE_READY = "Ready"
PHASE_FAILED = "Failed"
PHASE_ROLLED_BACK = "RolledBack"

_FAMILY_RE = re.compile(r"^[A-Za-z](?:[A-Za-z0-9._-]{0,61}[A-Za-z0-9])?$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/+~-]{0,252}$")
_DNS_LABEL_RE = re.compile(r"^[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?$")
_DNS_SUBDOMAIN_RE = re.compile(
    r"^[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?"
    r"(?:\.[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?)*$"
)
_PERCENT_RE = re.compile(r"^(?:0|[1-9][0-9]{0,2})%$")

Clock = Callable[[], datetime]


class SpecValidationError(ValueError):
    """Raised when a resource bypasses or violates CRD validation."""


class KubernetesAPIError(RuntimeError):
    """A sanitized Kubernetes API failure safe for operator status/logs."""

    def __init__(self, operation: str, status_code: int | None = None) -> None:
        self.operation = operation
        self.status_code = status_code
        suffix = "" if status_code is None else f" (HTTP {status_code})"
        super().__init__(f"Kubernetes API request failed: {operation}{suffix}")


@dataclass(frozen=True)
class DesiredModel:
    """Normalized desired state for one ``OpenMedModel`` resource."""

    family: str
    version: str
    tier: str
    replicas: int
    rollout_type: str
    max_unavailable: int | str
    max_surge: int | str
    rollback_on_failure: bool
    progress_deadline_seconds: int
    deployment_name: str
    container_name: str
    manifest_config_map_name: str

    @classmethod
    def from_resource(cls, body: Mapping[str, Any]) -> "DesiredModel":
        """Validate and normalize a Kubernetes custom-resource body."""

        metadata = _mapping(body.get("metadata"), field="metadata")
        spec = _mapping(body.get("spec"), field="spec")
        resource_name = _required_string(metadata, "name", field="metadata.name")

        family = _required_string(spec, "family", field="spec.family")
        if _FAMILY_RE.fullmatch(family) is None:
            raise SpecValidationError(
                "spec.family must start with a letter, end with a letter or digit, "
                "and contain only letters, digits, '.', '_' or '-'"
            )

        version = _required_string(spec, "version", field="spec.version")
        if _VERSION_RE.fullmatch(version) is None:
            raise SpecValidationError(
                "spec.version must be a non-empty local path or model pointer "
                "without whitespace"
            )

        tier = _required_string(spec, "tier", field="spec.tier")
        if tier not in ALLOWED_TIERS:
            raise SpecValidationError(
                f"spec.tier must be one of {', '.join(sorted(ALLOWED_TIERS))}"
            )

        replicas = _required_int(spec, "replicas", field="spec.replicas")
        if replicas < 1 or replicas > 1000:
            raise SpecValidationError("spec.replicas must be between 1 and 1000")

        strategy = _mapping(spec.get("rolloutStrategy"), field="spec.rolloutStrategy")
        rollout_type = _required_string(
            strategy, "type", field="spec.rolloutStrategy.type"
        )
        if rollout_type not in {"RollingUpdate", "Recreate"}:
            raise SpecValidationError(
                "spec.rolloutStrategy.type must be RollingUpdate or Recreate"
            )
        max_unavailable = _int_or_percent(
            strategy.get("maxUnavailable", DEFAULT_MAX_UNAVAILABLE),
            field="spec.rolloutStrategy.maxUnavailable",
        )
        max_surge = _int_or_percent(
            strategy.get("maxSurge", DEFAULT_MAX_SURGE),
            field="spec.rolloutStrategy.maxSurge",
        )
        if (
            rollout_type == "RollingUpdate"
            and _is_zero(max_unavailable)
            and _is_zero(max_surge)
        ):
            raise SpecValidationError(
                "RollingUpdate cannot set both maxUnavailable and maxSurge to zero"
            )
        rollback_on_failure = strategy.get("rollbackOnFailure", True)
        if not isinstance(rollback_on_failure, bool):
            raise SpecValidationError(
                "spec.rolloutStrategy.rollbackOnFailure must be a boolean"
            )
        deadline = strategy.get(
            "progressDeadlineSeconds", DEFAULT_PROGRESS_DEADLINE_SECONDS
        )
        if isinstance(deadline, bool) or not isinstance(deadline, int) or deadline < 1:
            raise SpecValidationError(
                "spec.rolloutStrategy.progressDeadlineSeconds must be a positive "
                "integer"
            )

        target_ref = spec.get("targetRef", {})
        target = _mapping(target_ref, field="spec.targetRef")
        deployment_name = str(target.get("name") or resource_name)
        container_name = str(target.get("containerName") or DEFAULT_CONTAINER_NAME)
        config_map_name = str(
            spec.get("manifestConfigMapName")
            or _name_with_suffix(resource_name, "model-manifest")
        )
        for value, field_name in (
            (deployment_name, "spec.targetRef.name"),
            (config_map_name, "spec.manifestConfigMapName"),
        ):
            _validate_dns_subdomain(value, field=field_name)
        _validate_dns_label(container_name, field="spec.targetRef.containerName")

        return cls(
            family=family,
            version=version,
            tier=tier,
            replicas=replicas,
            rollout_type=rollout_type,
            max_unavailable=max_unavailable,
            max_surge=max_surge,
            rollback_on_failure=rollback_on_failure,
            progress_deadline_seconds=deadline,
            deployment_name=deployment_name,
            container_name=container_name,
            manifest_config_map_name=config_map_name,
        )

    @classmethod
    def from_status(cls, value: Mapping[str, Any]) -> "DesiredModel":
        """Restore a previously successful desired state from CR status."""

        rollout = _mapping(value.get("rolloutStrategy"), field="status rollout")
        target = _mapping(value.get("targetRef"), field="status targetRef")
        return cls(
            family=_required_string(value, "family", field="status family"),
            version=_required_string(value, "version", field="status version"),
            tier=_required_string(value, "tier", field="status tier"),
            replicas=_required_int(value, "replicas", field="status replicas"),
            rollout_type=_required_string(
                rollout, "type", field="status rolloutStrategy.type"
            ),
            max_unavailable=rollout.get("maxUnavailable", DEFAULT_MAX_UNAVAILABLE),
            max_surge=rollout.get("maxSurge", DEFAULT_MAX_SURGE),
            rollback_on_failure=bool(rollout.get("rollbackOnFailure", True)),
            progress_deadline_seconds=int(
                rollout.get(
                    "progressDeadlineSeconds", DEFAULT_PROGRESS_DEADLINE_SECONDS
                )
            ),
            deployment_name=_required_string(
                target, "name", field="status targetRef.name"
            ),
            container_name=_required_string(
                target, "containerName", field="status targetRef.containerName"
            ),
            manifest_config_map_name=_required_string(
                value,
                "manifestConfigMapName",
                field="status manifestConfigMapName",
            ),
        )

    def to_status(self) -> dict[str, Any]:
        """Return the stable status representation used for rollback."""

        return {
            "family": self.family,
            "version": self.version,
            "tier": self.tier,
            "replicas": self.replicas,
            "rolloutStrategy": {
                "type": self.rollout_type,
                "maxUnavailable": self.max_unavailable,
                "maxSurge": self.max_surge,
                "rollbackOnFailure": self.rollback_on_failure,
                "progressDeadlineSeconds": self.progress_deadline_seconds,
            },
            "targetRef": {
                "name": self.deployment_name,
                "containerName": self.container_name,
            },
            "manifestConfigMapName": self.manifest_config_map_name,
        }

    @property
    def spec_hash(self) -> str:
        """Return a deterministic, value-free digest of desired state."""

        return _digest(self.to_status())

    def manifest(self) -> dict[str, Any]:
        """Return the deterministic served-model manifest pointer payload."""

        return {
            "apiVersion": f"{API_GROUP}/{API_VERSION}",
            "kind": "OpenMedModelManifest",
            "models": [
                {
                    "family": self.family,
                    "tier": self.tier,
                    "version": self.version,
                }
            ],
        }

    @property
    def manifest_hash(self) -> str:
        """Return the pod-template hash that triggers warm-pool replacement."""

        return _digest(self.manifest())


@dataclass(frozen=True)
class ReconcileResult:
    """Result returned by a single level-based reconciliation pass."""

    status: dict[str, Any]
    changed: bool


class KubernetesAPIClient:
    """Small in-cluster Kubernetes JSON client with no telemetry or SDK state."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        ssl_context: ssl.SSLContext | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Kubernetes API base URL must be an HTTP(S) origin")
        if parsed.scheme == "http" and parsed.hostname not in {
            "127.0.0.1",
            "::1",
            "localhost",
        }:
            raise ValueError("unencrypted Kubernetes API access is limited to loopback")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.ssl_context = ssl_context
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_service_account(cls) -> "KubernetesAPIClient":
        """Build a client from the mounted Kubernetes service account."""

        host = os.getenv("KUBERNETES_SERVICE_HOST")
        port = os.getenv("KUBERNETES_SERVICE_PORT_HTTPS", "443")
        if not host:
            raise KubernetesAPIError("discover in-cluster API endpoint")
        rendered_host = (
            f"[{host}]" if ":" in host and not host.startswith("[") else host
        )
        token_path = Path(
            os.getenv(
                "OPENMED_OPERATOR_TOKEN_PATH",
                "/var/run/secrets/kubernetes.io/serviceaccount/token",
            )
        )
        ca_path = Path(
            os.getenv(
                "OPENMED_OPERATOR_CA_PATH",
                "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt",
            )
        )
        try:
            token = token_path.read_text(encoding="utf-8").strip()
            context = ssl.create_default_context(cafile=str(ca_path))
        except OSError as exc:
            raise KubernetesAPIError("read service-account credentials") from exc
        return cls(f"https://{rendered_host}:{port}", token=token, ssl_context=context)

    def get_deployment(self, namespace: str, name: str) -> dict[str, Any]:
        """Read a namespaced Deployment."""

        return self._request(
            "GET",
            f"/apis/apps/v1/namespaces/{_url(namespace)}/deployments/{_url(name)}",
        )

    def get_config_map(self, namespace: str, name: str) -> dict[str, Any]:
        """Read a namespaced ConfigMap."""

        return self._request(
            "GET",
            f"/api/v1/namespaces/{_url(namespace)}/configmaps/{_url(name)}",
        )

    def patch_deployment(
        self, namespace: str, name: str, patch: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Strategic-merge patch a namespaced Deployment."""

        return self._request(
            "PATCH",
            f"/apis/apps/v1/namespaces/{_url(namespace)}/deployments/{_url(name)}",
            payload=patch,
            content_type="application/strategic-merge-patch+json",
        )

    def upsert_config_map(
        self, namespace: str, name: str, body: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Create or merge-patch a namespaced ConfigMap."""

        item_path = f"/api/v1/namespaces/{_url(namespace)}/configmaps/{_url(name)}"
        try:
            existing = self._request("GET", item_path)
        except KubernetesAPIError as exc:
            if exc.status_code != 404:
                raise
            return self._request(
                "POST",
                f"/api/v1/namespaces/{_url(namespace)}/configmaps",
                payload=body,
            )
        expected_metadata = _mapping_or_empty(body.get("metadata"))
        existing_metadata = _mapping_or_empty(existing.get("metadata"))
        expected_owner = _mapping_or_empty(expected_metadata.get("annotations")).get(
            MODEL_RESOURCE_ANNOTATION
        )
        existing_owner = _mapping_or_empty(existing_metadata.get("annotations")).get(
            MODEL_RESOURCE_ANNOTATION
        )
        expected_uids = {
            str(reference.get("uid"))
            for reference in expected_metadata.get("ownerReferences") or []
            if isinstance(reference, Mapping) and reference.get("controller") is True
        }
        existing_uids = {
            str(reference.get("uid"))
            for reference in existing_metadata.get("ownerReferences") or []
            if isinstance(reference, Mapping) and reference.get("controller") is True
        }
        if existing_owner != expected_owner and not (expected_uids & existing_uids):
            raise KubernetesAPIError(
                f"claim ConfigMap {_url(namespace)}/{_url(name)}", status_code=409
            )
        return self._request(
            "PATCH",
            item_path,
            payload=body,
            content_type="application/merge-patch+json",
        )

    def delete_config_map(self, namespace: str, name: str) -> None:
        """Delete a namespaced ConfigMap if it exists."""

        try:
            self._request(
                "DELETE",
                f"/api/v1/namespaces/{_url(namespace)}/configmaps/{_url(name)}",
                payload={"kind": "DeleteOptions", "apiVersion": "v1"},
            )
        except KubernetesAPIError as exc:
            if exc.status_code != 404:
                raise

    def create_event(self, namespace: str, body: Mapping[str, Any]) -> None:
        """Create one idempotently named core/v1 Event."""

        try:
            self._request(
                "POST",
                f"/api/v1/namespaces/{_url(namespace)}/events",
                payload=body,
            )
        except KubernetesAPIError as exc:
            if exc.status_code != 409:
                raise

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
        content_type: str = "application/json",
    ) -> dict[str, Any]:
        if not path.startswith("/") or path.startswith("//"):
            raise KubernetesAPIError("validate Kubernetes API path")
        data = None
        headers = {
            "Accept": "application/json",
            "User-Agent": "openmed-operator/1",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if payload is not None:
            data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            headers["Content-Type"] = content_type
        request = Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        operation = f"{method} {path}"
        try:
            # The constructor and path guard constrain dispatch to a safe HTTP(S) origin.
            with urlopen(  # nosec B310
                request,
                timeout=self.timeout_seconds,
                context=self.ssl_context,
            ) as response:
                raw = response.read()
        except HTTPError as exc:
            raise KubernetesAPIError(operation, exc.code) from exc
        except (OSError, URLError) as exc:
            raise KubernetesAPIError(operation) from exc
        if not raw:
            return {}
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise KubernetesAPIError(f"decode response for {operation}") from exc
        if not isinstance(decoded, dict):
            raise KubernetesAPIError(f"validate response for {operation}")
        return decoded


def reconcile_openmed_model(
    body: Mapping[str, Any],
    api: KubernetesAPIClient,
    *,
    clock: Clock | None = None,
) -> ReconcileResult:
    """Converge one OpenMedModel resource and return its complete status.

    Reconciliation is level-based and idempotent. A changed model pointer first
    updates its owned ConfigMap, then patches the target Deployment's preload
    environment and pod-template digest. Kubernetes performs the rollout while
    later passes observe readiness or invoke the retained rollback state.
    """

    desired = DesiredModel.from_resource(body)
    metadata = _mapping(body.get("metadata"), field="metadata")
    status = _mapping_or_empty(body.get("status"))
    namespace = str(metadata.get("namespace") or "default")
    name = _required_string(metadata, "name", field="metadata.name")
    uid = _required_string(metadata, "uid", field="metadata.uid")
    generation = _positive_int(metadata.get("generation", 1), "metadata.generation")
    now = _utcnow() if clock is None else _as_utc(clock())
    owner = f"{namespace}/{name}"

    try:
        deployment = api.get_deployment(namespace, desired.deployment_name)
    except KubernetesAPIError as exc:
        if exc.status_code != 404:
            raise
        result_status = _phase_status(
            status,
            desired=desired,
            generation=generation,
            phase=PHASE_PENDING,
            now=now,
            active_version=_optional_string(status.get("activeVersion")),
            conditions=(
                (
                    "Ready",
                    "False",
                    "TargetNotFound",
                    "Target Deployment was not found.",
                ),
                (
                    "Progressing",
                    "False",
                    "Blocked",
                    "Rollout is waiting for its target.",
                ),
                (
                    "Degraded",
                    "True",
                    "TargetNotFound",
                    "Target Deployment was not found.",
                ),
                ("RolledBack", "False", "NotRequested", "No rollback is active."),
            ),
        )
        _safe_event(
            api,
            body,
            reason="TargetNotFound",
            event_type="Warning",
            message="Target Deployment was not found; reconciliation will retry.",
            generation=generation,
            now=now,
        )
        return ReconcileResult(result_status, changed=False)

    container = _find_container(deployment, desired.container_name)
    if container is None:
        result_status = _phase_status(
            status,
            desired=desired,
            generation=generation,
            phase=PHASE_FAILED,
            now=now,
            active_version=_optional_string(status.get("activeVersion")),
            failed_spec_hash=desired.spec_hash,
            conditions=(
                (
                    "Ready",
                    "False",
                    "ContainerNotFound",
                    "Target container was not found.",
                ),
                (
                    "Progressing",
                    "False",
                    "Blocked",
                    "Rollout cannot update the target.",
                ),
                (
                    "Degraded",
                    "True",
                    "ContainerNotFound",
                    "Target container was not found.",
                ),
                ("RolledBack", "False", "NotRequested", "No rollback is active."),
            ),
        )
        _safe_event(
            api,
            body,
            reason="ContainerNotFound",
            event_type="Warning",
            message="Target Deployment does not contain the configured container.",
            generation=generation,
            now=now,
        )
        return ReconcileResult(result_status, changed=False)

    deployment_owner = _deployment_annotation(deployment, MODEL_RESOURCE_ANNOTATION)
    if deployment_owner not in {None, owner}:
        result_status = _phase_status(
            status,
            desired=desired,
            generation=generation,
            phase=PHASE_FAILED,
            now=now,
            active_version=_optional_string(status.get("activeVersion")),
            failed_spec_hash=desired.spec_hash,
            conditions=(
                (
                    "Ready",
                    "False",
                    "TargetConflict",
                    "Target is owned by another model resource.",
                ),
                ("Progressing", "False", "Blocked", "Rollout cannot claim the target."),
                (
                    "Degraded",
                    "True",
                    "TargetConflict",
                    "Target is owned by another model resource.",
                ),
                ("RolledBack", "False", "NotRequested", "No rollback is active."),
            ),
        )
        _safe_event(
            api,
            body,
            reason="TargetConflict",
            event_type="Warning",
            message="Target Deployment is already managed by another OpenMedModel.",
            generation=generation,
            now=now,
        )
        return ReconcileResult(result_status, changed=False)

    rollback_target = _rollback_target(metadata)
    if rollback_target:
        return _reconcile_manual_rollback(
            api=api,
            body=body,
            deployment=deployment,
            desired=desired,
            status=status,
            namespace=namespace,
            name=name,
            uid=uid,
            generation=generation,
            rollback_target=rollback_target,
            now=now,
        )

    if (
        status.get("phase") in {PHASE_FAILED, PHASE_ROLLED_BACK}
        and status.get("failedSpecHash") == desired.spec_hash
    ):
        if status.get("phase") == PHASE_ROLLED_BACK:
            return _monitor_rollback(
                api=api,
                body=body,
                deployment=deployment,
                desired=desired,
                status=status,
                generation=generation,
                now=now,
            )
        return ReconcileResult(copy.deepcopy(dict(status)), changed=False)

    desired_changed = status.get("desiredSpecHash") != desired.spec_hash
    phase = str(status.get("phase") or "")
    if desired_changed or phase not in {PHASE_ROLLING_OUT, PHASE_READY}:
        return _start_rollout(
            api=api,
            body=body,
            deployment=deployment,
            desired=desired,
            status=status,
            namespace=namespace,
            name=name,
            uid=uid,
            generation=generation,
            now=now,
            reason="SpecChanged" if desired_changed else "InitialRollout",
        )

    target_generation = int(status.get("targetDeploymentGeneration") or 0)
    if _deployment_failed(deployment, target_generation):
        return _handle_rollout_failure(
            api=api,
            body=body,
            deployment=deployment,
            desired=desired,
            status=status,
            namespace=namespace,
            name=name,
            uid=uid,
            generation=generation,
            now=now,
        )

    if not _config_map_matches(api, namespace, desired, owner, uid):
        return _start_rollout(
            api=api,
            body=body,
            deployment=deployment,
            desired=desired,
            status=status,
            namespace=namespace,
            name=name,
            uid=uid,
            generation=generation,
            now=now,
            reason="DriftCorrected",
        )

    if _deployment_ready(
        deployment, desired.replicas, target_generation
    ) and _deployment_matches(deployment, desired, owner):
        if phase == PHASE_READY:
            return ReconcileResult(copy.deepcopy(dict(status)), changed=False)
        last_success = _mapping_or_none(status.get("lastSuccessfulSpec"))
        previous_success = _mapping_or_none(status.get("previousSuccessfulSpec"))
        if last_success is not None and _digest(last_success) != desired.spec_hash:
            previous_success = copy.deepcopy(dict(last_success))
        result_status = _phase_status(
            status,
            desired=desired,
            generation=generation,
            phase=PHASE_READY,
            now=now,
            active_version=desired.version,
            applied_spec_hash=desired.spec_hash,
            target_deployment_generation=target_generation,
            last_successful_spec=desired.to_status(),
            previous_successful_spec=previous_success,
            failed_spec_hash=None,
            rollback_spec=None,
            rollback_request=None,
            conditions=(
                (
                    "Ready",
                    "True",
                    "RolloutSucceeded",
                    "Desired model version is available.",
                ),
                (
                    "Progressing",
                    "False",
                    "RolloutComplete",
                    "Deployment rollout completed.",
                ),
                ("Degraded", "False", "Healthy", "No rollout failure is active."),
                ("RolledBack", "False", "NotRequested", "No rollback is active."),
            ),
        )
        _safe_event(
            api,
            body,
            reason="RolloutSucceeded",
            event_type="Normal",
            message="Desired model version is available and warm.",
            generation=generation,
            now=now,
        )
        return ReconcileResult(result_status, changed=True)

    if not _deployment_matches(deployment, desired, owner):
        return _start_rollout(
            api=api,
            body=body,
            deployment=deployment,
            desired=desired,
            status=status,
            namespace=namespace,
            name=name,
            uid=uid,
            generation=generation,
            now=now,
            reason="DriftCorrected",
        )

    result_status = _phase_status(
        status,
        desired=desired,
        generation=generation,
        phase=PHASE_ROLLING_OUT,
        now=now,
        active_version=_optional_string(status.get("activeVersion")),
        applied_spec_hash=desired.spec_hash,
        target_deployment_generation=target_generation,
        conditions=(
            (
                "Ready",
                "False",
                "RolloutInProgress",
                "Desired model version is not available yet.",
            ),
            (
                "Progressing",
                "True",
                "DeploymentProgressing",
                "Kubernetes is progressing the rollout.",
            ),
            ("Degraded", "False", "Healthy", "No rollout failure is active."),
            ("RolledBack", "False", "NotRequested", "No rollback is active."),
        ),
    )
    return ReconcileResult(result_status, changed=False)


def decommission_openmed_model(
    body: Mapping[str, Any],
    api: KubernetesAPIClient,
    *,
    clock: Clock | None = None,
) -> None:
    """Remove the manifest pointer and trigger a cold target rollout on delete."""

    desired = DesiredModel.from_resource(body)
    metadata = _mapping(body.get("metadata"), field="metadata")
    namespace = str(metadata.get("namespace") or "default")
    name = _required_string(metadata, "name", field="metadata.name")
    uid = _required_string(metadata, "uid", field="metadata.uid")
    generation = _positive_int(metadata.get("generation", 1), "metadata.generation")
    owner = f"{namespace}/{name}"
    foreign_ownership = False
    try:
        deployment = api.get_deployment(namespace, desired.deployment_name)
    except KubernetesAPIError as exc:
        if exc.status_code != 404:
            raise
    else:
        if owner in {
            _deployment_annotation(deployment, MODEL_RESOURCE_ANNOTATION),
            _deployment_template_annotation(deployment, MODEL_RESOURCE_ANNOTATION),
        }:
            empty_hash = _digest(
                {
                    "apiVersion": f"{API_GROUP}/{API_VERSION}",
                    "kind": "OpenMedModelManifest",
                    "models": [],
                }
            )
            api.patch_deployment(
                namespace,
                desired.deployment_name,
                {
                    "metadata": {"annotations": {MODEL_RESOURCE_ANNOTATION: None}},
                    "spec": {
                        "template": {
                            "metadata": {
                                "annotations": {
                                    MANIFEST_HASH_ANNOTATION: empty_hash,
                                    MODEL_RESOURCE_ANNOTATION: None,
                                }
                            },
                            "spec": {
                                "containers": [
                                    {
                                        "name": desired.container_name,
                                        "env": [
                                            {
                                                "name": PRELOAD_ENV_NAME,
                                                "value": "",
                                                "valueFrom": None,
                                            }
                                        ],
                                    }
                                ]
                            },
                        }
                    },
                },
            )
        else:
            foreign_ownership = True
    try:
        config_map = api.get_config_map(namespace, desired.manifest_config_map_name)
    except KubernetesAPIError as exc:
        if exc.status_code != 404:
            raise
    else:
        if _config_map_owned_by(config_map, owner, uid):
            api.delete_config_map(namespace, desired.manifest_config_map_name)
        else:
            foreign_ownership = True
    _safe_event(
        api,
        body,
        reason="ModelRemovalSkipped" if foreign_ownership else "ModelRemoved",
        event_type="Warning" if foreign_ownership else "Normal",
        message=(
            "Resources owned by another controller were left unchanged."
            if foreign_ownership
            else "Model manifest pointer was removed from the target Deployment."
        ),
        generation=generation,
        now=_utcnow() if clock is None else _as_utc(clock()),
    )


def _start_rollout(
    *,
    api: KubernetesAPIClient,
    body: Mapping[str, Any],
    deployment: Mapping[str, Any],
    desired: DesiredModel,
    status: Mapping[str, Any],
    namespace: str,
    name: str,
    uid: str,
    generation: int,
    now: datetime,
    reason: str,
) -> ReconcileResult:
    patched = _apply_desired(
        api,
        deployment=deployment,
        desired=desired,
        namespace=namespace,
        name=name,
        uid=uid,
    )
    target_generation = _deployment_generation(patched)
    result_status = _phase_status(
        status,
        desired=desired,
        generation=generation,
        phase=PHASE_ROLLING_OUT,
        now=now,
        active_version=_optional_string(status.get("activeVersion")),
        applied_spec_hash=desired.spec_hash,
        target_deployment_generation=target_generation,
        failed_spec_hash=None,
        rollback_spec=None,
        rollback_request=None,
        conditions=(
            (
                "Ready",
                "False",
                "RolloutStarted",
                "Desired model version is not available yet.",
            ),
            (
                "Progressing",
                "True",
                reason,
                "Kubernetes Deployment rollout was started.",
            ),
            ("Degraded", "False", "Healthy", "No rollout failure is active."),
            ("RolledBack", "False", "NotRequested", "No rollback is active."),
        ),
    )
    _safe_event(
        api,
        body,
        reason="RolloutStarted",
        event_type="Normal",
        message="Model manifest pointer changed; warm-pool rollout started.",
        generation=generation,
        now=now,
    )
    return ReconcileResult(result_status, changed=True)


def _handle_rollout_failure(
    *,
    api: KubernetesAPIClient,
    body: Mapping[str, Any],
    deployment: Mapping[str, Any],
    desired: DesiredModel,
    status: Mapping[str, Any],
    namespace: str,
    name: str,
    uid: str,
    generation: int,
    now: datetime,
) -> ReconcileResult:
    _safe_event(
        api,
        body,
        reason="RolloutFailed",
        event_type="Warning",
        message="Kubernetes reported that the model rollout exceeded its deadline.",
        generation=generation,
        now=now,
    )
    last_success = _mapping_or_none(status.get("lastSuccessfulSpec"))
    if desired.rollback_on_failure and last_success is not None:
        rollback = DesiredModel.from_status(last_success)
        patched = _apply_desired(
            api,
            deployment=deployment,
            desired=rollback,
            namespace=namespace,
            name=name,
            uid=uid,
        )
        result_status = _phase_status(
            status,
            desired=desired,
            generation=generation,
            phase=PHASE_ROLLED_BACK,
            now=now,
            active_version=rollback.version,
            applied_spec_hash=rollback.spec_hash,
            target_deployment_generation=_deployment_generation(patched),
            failed_spec_hash=desired.spec_hash,
            rollback_spec=rollback.to_status(),
            rollback_request=f"automatic:{desired.version}",
            rollback_count=int(status.get("rollbackCount") or 0) + 1,
            conditions=(
                (
                    "Ready",
                    "False",
                    "RolledBack",
                    "Desired model version failed and was rolled back.",
                ),
                (
                    "Progressing",
                    "True",
                    "RollbackInProgress",
                    "The last successful version is being restored.",
                ),
                (
                    "Degraded",
                    "True",
                    "RolloutFailed",
                    "Desired model version did not become available.",
                ),
                (
                    "RolledBack",
                    "False",
                    "RollbackInProgress",
                    "The last successful version is being restored.",
                ),
            ),
        )
        _safe_event(
            api,
            body,
            reason="RollbackStarted",
            event_type="Normal",
            message="Automatic rollback to the last successful version started.",
            generation=generation,
            now=now,
        )
        return ReconcileResult(result_status, changed=True)

    result_status = _phase_status(
        status,
        desired=desired,
        generation=generation,
        phase=PHASE_FAILED,
        now=now,
        active_version=_optional_string(status.get("activeVersion")),
        failed_spec_hash=desired.spec_hash,
        conditions=(
            (
                "Ready",
                "False",
                "RolloutFailed",
                "Desired model version did not become available.",
            ),
            (
                "Progressing",
                "False",
                "ProgressDeadlineExceeded",
                "Kubernetes stopped progressing the rollout.",
            ),
            (
                "Degraded",
                "True",
                "RolloutFailed",
                "Desired model version did not become available.",
            ),
            (
                "RolledBack",
                "False",
                "RollbackDisabled",
                "Automatic rollback is disabled or unavailable.",
            ),
        ),
    )
    return ReconcileResult(result_status, changed=True)


def _reconcile_manual_rollback(
    *,
    api: KubernetesAPIClient,
    body: Mapping[str, Any],
    deployment: Mapping[str, Any],
    desired: DesiredModel,
    status: Mapping[str, Any],
    namespace: str,
    name: str,
    uid: str,
    generation: int,
    rollback_target: str,
    now: datetime,
) -> ReconcileResult:
    request_key = f"manual:{rollback_target}"
    if (
        status.get("phase") == PHASE_ROLLED_BACK
        and status.get("rollbackRequest") == request_key
    ):
        return _monitor_rollback(
            api=api,
            body=body,
            deployment=deployment,
            desired=desired,
            status=status,
            generation=generation,
            now=now,
        )

    candidates = (
        _mapping_or_none(status.get("lastSuccessfulSpec")),
        _mapping_or_none(status.get("previousSuccessfulSpec")),
    )
    rollback: DesiredModel | None = None
    for candidate in candidates:
        if candidate is None:
            continue
        parsed = DesiredModel.from_status(candidate)
        if parsed.version == rollback_target:
            rollback = parsed
            break
    if rollback is None:
        result_status = _phase_status(
            status,
            desired=desired,
            generation=generation,
            phase=PHASE_FAILED,
            now=now,
            active_version=_optional_string(status.get("activeVersion")),
            failed_spec_hash=desired.spec_hash,
            rollback_request=request_key,
            conditions=(
                (
                    "Ready",
                    "False",
                    "RollbackTargetUnavailable",
                    "Requested rollback target is not retained.",
                ),
                ("Progressing", "False", "Blocked", "Manual rollback cannot start."),
                (
                    "Degraded",
                    "True",
                    "RollbackTargetUnavailable",
                    "Requested rollback target is not retained.",
                ),
                (
                    "RolledBack",
                    "False",
                    "RollbackRejected",
                    "Manual rollback target was rejected.",
                ),
            ),
        )
        _safe_event(
            api,
            body,
            reason="RollbackRejected",
            event_type="Warning",
            message="Requested rollback target is not one of the retained successful versions.",
            generation=generation,
            now=now,
        )
        return ReconcileResult(result_status, changed=False)

    patched = _apply_desired(
        api,
        deployment=deployment,
        desired=rollback,
        namespace=namespace,
        name=name,
        uid=uid,
    )
    result_status = _phase_status(
        status,
        desired=desired,
        generation=generation,
        phase=PHASE_ROLLED_BACK,
        now=now,
        active_version=rollback.version,
        applied_spec_hash=rollback.spec_hash,
        target_deployment_generation=_deployment_generation(patched),
        failed_spec_hash=desired.spec_hash,
        rollback_spec=rollback.to_status(),
        rollback_request=request_key,
        rollback_count=int(status.get("rollbackCount") or 0) + 1,
        conditions=(
            ("Ready", "False", "ManualRollback", "Manual rollback is in progress."),
            (
                "Progressing",
                "True",
                "RollbackInProgress",
                "Requested successful version is being restored.",
            ),
            (
                "Degraded",
                "True",
                "ManualRollback",
                "Desired spec is held at a retained version.",
            ),
            (
                "RolledBack",
                "False",
                "RollbackInProgress",
                "Requested successful version is being restored.",
            ),
        ),
    )
    _safe_event(
        api,
        body,
        reason="RollbackStarted",
        event_type="Normal",
        message="Manual rollback to a retained successful version started.",
        generation=generation,
        now=now,
    )
    return ReconcileResult(result_status, changed=True)


def _monitor_rollback(
    *,
    api: KubernetesAPIClient,
    body: Mapping[str, Any],
    deployment: Mapping[str, Any],
    desired: DesiredModel,
    status: Mapping[str, Any],
    generation: int,
    now: datetime,
) -> ReconcileResult:
    rollback_value = _mapping_or_none(status.get("rollbackSpec"))
    if rollback_value is None:
        return ReconcileResult(copy.deepcopy(dict(status)), changed=False)
    rollback = DesiredModel.from_status(rollback_value)
    target_generation = int(status.get("targetDeploymentGeneration") or 0)
    if _deployment_failed(deployment, target_generation):
        result_status = _phase_status(
            status,
            desired=desired,
            generation=generation,
            phase=PHASE_FAILED,
            now=now,
            active_version=_optional_string(status.get("activeVersion")),
            applied_spec_hash=_optional_string(status.get("appliedSpecHash")),
            target_deployment_generation=target_generation,
            failed_spec_hash=_optional_string(status.get("failedSpecHash")),
            rollback_spec=rollback.to_status(),
            rollback_request=_optional_string(status.get("rollbackRequest")),
            rollback_count=int(status.get("rollbackCount") or 0),
            conditions=(
                ("Ready", "False", "RollbackFailed", "No retained version is ready."),
                (
                    "Progressing",
                    "False",
                    "ProgressDeadlineExceeded",
                    "Kubernetes stopped progressing the rollback.",
                ),
                (
                    "Degraded",
                    "True",
                    "RollbackFailed",
                    "The retained version could not be restored.",
                ),
                (
                    "RolledBack",
                    "False",
                    "RollbackFailed",
                    "The retained version could not be restored.",
                ),
            ),
        )
        _safe_event(
            api,
            body,
            reason="RollbackFailed",
            event_type="Warning",
            message="Kubernetes reported that the rollback exceeded its deadline.",
            generation=generation,
            now=now,
        )
        return ReconcileResult(result_status, changed=True)
    succeeded = _deployment_ready(
        deployment, rollback.replicas, target_generation
    ) and _deployment_matches(
        deployment,
        rollback,
        _resource_owner(body),
    )
    conditions: Sequence[tuple[str, str, str, str]]
    if succeeded:
        conditions = (
            (
                "Ready",
                "False",
                "RolledBack",
                "A retained version is available; desired spec remains unmet.",
            ),
            (
                "Progressing",
                "False",
                "RollbackComplete",
                "Rollback Deployment rollout completed.",
            ),
            (
                "Degraded",
                "True",
                "DesiredVersionFailed",
                "Desired spec is held at a retained version.",
            ),
            (
                "RolledBack",
                "True",
                "RollbackSucceeded",
                "Retained successful version was restored.",
            ),
        )
        _safe_event(
            api,
            body,
            reason="RollbackSucceeded",
            event_type="Normal",
            message="Retained successful model version is available.",
            generation=generation,
            now=now,
        )
    else:
        conditions = (
            ("Ready", "False", "RolledBack", "Desired model version is not active."),
            (
                "Progressing",
                "True",
                "RollbackInProgress",
                "Retained successful version is being restored.",
            ),
            (
                "Degraded",
                "True",
                "DesiredVersionFailed",
                "Desired spec is held at a retained version.",
            ),
            (
                "RolledBack",
                "False",
                "RollbackInProgress",
                "Retained successful version is being restored.",
            ),
        )
    result_status = _phase_status(
        status,
        desired=desired,
        generation=generation,
        phase=PHASE_ROLLED_BACK,
        now=now,
        active_version=rollback.version,
        applied_spec_hash=rollback.spec_hash,
        target_deployment_generation=target_generation,
        failed_spec_hash=_optional_string(status.get("failedSpecHash")),
        rollback_spec=rollback.to_status(),
        rollback_request=_optional_string(status.get("rollbackRequest")),
        rollback_count=int(status.get("rollbackCount") or 0),
        conditions=conditions,
    )
    return ReconcileResult(result_status, changed=succeeded)


def _apply_desired(
    api: KubernetesAPIClient,
    *,
    deployment: Mapping[str, Any],
    desired: DesiredModel,
    namespace: str,
    name: str,
    uid: str,
) -> dict[str, Any]:
    owner = f"{namespace}/{name}"
    config_map = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": desired.manifest_config_map_name,
            "namespace": namespace,
            "annotations": {MODEL_RESOURCE_ANNOTATION: owner},
            "labels": {
                "app.kubernetes.io/managed-by": "openmed-operator",
                "openmed.ai/model-family": desired.family,
            },
            "ownerReferences": [
                {
                    "apiVersion": f"{API_GROUP}/{API_VERSION}",
                    "kind": API_KIND,
                    "name": name,
                    "uid": uid,
                    "controller": True,
                    "blockOwnerDeletion": True,
                }
            ],
        },
        "data": {
            MANIFEST_DATA_KEY: json.dumps(
                desired.manifest(), sort_keys=True, separators=(",", ":")
            )
            + "\n",
            PRELOAD_DATA_KEY: desired.version,
        },
    }
    api.upsert_config_map(namespace, desired.manifest_config_map_name, config_map)

    strategy: dict[str, Any]
    if desired.rollout_type == "Recreate":
        strategy = {"type": "Recreate", "rollingUpdate": None}
    else:
        strategy = {
            "type": "RollingUpdate",
            "rollingUpdate": {
                "maxUnavailable": desired.max_unavailable,
                "maxSurge": desired.max_surge,
            },
        }
    patch = {
        "metadata": {
            "annotations": {MODEL_RESOURCE_ANNOTATION: owner},
        },
        "spec": {
            "replicas": desired.replicas,
            "progressDeadlineSeconds": desired.progress_deadline_seconds,
            "strategy": strategy,
            "template": {
                "metadata": {
                    "annotations": {
                        MANIFEST_HASH_ANNOTATION: desired.manifest_hash,
                        MODEL_RESOURCE_ANNOTATION: owner,
                    }
                },
                "spec": {
                    "containers": [
                        {
                            "name": desired.container_name,
                            "env": [
                                {
                                    "name": PRELOAD_ENV_NAME,
                                    "value": None,
                                    "valueFrom": {
                                        "configMapKeyRef": {
                                            "name": desired.manifest_config_map_name,
                                            "key": PRELOAD_DATA_KEY,
                                        }
                                    },
                                }
                            ],
                        }
                    ]
                },
            },
        },
    }
    patched = api.patch_deployment(namespace, desired.deployment_name, patch)
    if not patched:
        patched = copy.deepcopy(dict(deployment))
        patched_metadata = patched.setdefault("metadata", {})
        patched_metadata["generation"] = _deployment_generation(deployment) + 1
    return patched


def _phase_status(
    existing: Mapping[str, Any],
    *,
    desired: DesiredModel,
    generation: int,
    phase: str,
    now: datetime,
    active_version: str | None,
    conditions: Sequence[tuple[str, str, str, str]],
    applied_spec_hash: str | None | object = ...,  # type: ignore[assignment]
    target_deployment_generation: int | None | object = ...,  # type: ignore[assignment]
    failed_spec_hash: str | None | object = ...,  # type: ignore[assignment]
    last_successful_spec: Mapping[str, Any] | None | object = ...,  # type: ignore[assignment]
    previous_successful_spec: Mapping[str, Any] | None | object = ...,  # type: ignore[assignment]
    rollback_spec: Mapping[str, Any] | None | object = ...,  # type: ignore[assignment]
    rollback_request: str | None | object = ...,  # type: ignore[assignment]
    rollback_count: int | None = None,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(existing))
    result.update(
        {
            "observedGeneration": generation,
            "phase": phase,
            "desiredVersion": desired.version,
            "activeVersion": active_version,
            "desiredSpecHash": desired.spec_hash,
            "manifestHash": desired.manifest_hash,
        }
    )
    optional_updates = {
        "appliedSpecHash": applied_spec_hash,
        "targetDeploymentGeneration": target_deployment_generation,
        "failedSpecHash": failed_spec_hash,
        "lastSuccessfulSpec": last_successful_spec,
        "previousSuccessfulSpec": previous_successful_spec,
        "rollbackSpec": rollback_spec,
        "rollbackRequest": rollback_request,
    }
    for key, value in optional_updates.items():
        if value is not Ellipsis:
            result[key] = copy.deepcopy(value)
    if rollback_count is not None:
        result["rollbackCount"] = rollback_count
    else:
        result.setdefault("rollbackCount", 0)
    result["conditions"] = _merge_conditions(
        existing.get("conditions"), conditions, generation=generation, now=now
    )
    return result


def _merge_conditions(
    current: Any,
    desired: Sequence[tuple[str, str, str, str]],
    *,
    generation: int,
    now: datetime,
) -> list[dict[str, Any]]:
    existing = {
        str(item.get("type")): item
        for item in current or []
        if isinstance(item, Mapping) and item.get("type")
    }
    rendered: list[dict[str, Any]] = []
    timestamp = _timestamp(now)
    for condition_type, status, reason, message in desired:
        previous = existing.get(condition_type)
        transition_time = timestamp
        if (
            previous is not None
            and previous.get("status") == status
            and previous.get("reason") == reason
            and previous.get("message") == message
        ):
            transition_time = str(previous.get("lastTransitionTime") or timestamp)
        rendered.append(
            {
                "type": condition_type,
                "status": status,
                "reason": reason,
                "message": message,
                "observedGeneration": generation,
                "lastTransitionTime": transition_time,
            }
        )
    return rendered


def _config_map_matches(
    api: KubernetesAPIClient,
    namespace: str,
    desired: DesiredModel,
    owner: str,
    uid: str,
) -> bool:
    try:
        config_map = api.get_config_map(namespace, desired.manifest_config_map_name)
    except KubernetesAPIError as exc:
        if exc.status_code == 404:
            return False
        raise
    data = _mapping_or_empty(config_map.get("data"))
    rendered_manifest = (
        json.dumps(desired.manifest(), sort_keys=True, separators=(",", ":")) + "\n"
    )
    return (
        _config_map_owned_by(config_map, owner, uid)
        and data.get(MANIFEST_DATA_KEY) == rendered_manifest
        and data.get(PRELOAD_DATA_KEY) == desired.version
    )


def _config_map_owned_by(config_map: Mapping[str, Any], owner: str, uid: str) -> bool:
    metadata = _mapping_or_empty(config_map.get("metadata"))
    annotations = _mapping_or_empty(metadata.get("annotations"))
    owner_references = metadata.get("ownerReferences") or []
    owns_config_map = any(
        isinstance(reference, Mapping)
        and reference.get("controller") is True
        and reference.get("uid") == uid
        for reference in owner_references
    )
    return annotations.get(MODEL_RESOURCE_ANNOTATION) == owner and owns_config_map


def _deployment_matches(
    deployment: Mapping[str, Any], desired: DesiredModel, owner: str
) -> bool:
    spec = _mapping_or_empty(deployment.get("spec"))
    if int(spec.get("replicas") or 0) != desired.replicas:
        return False
    if (
        int(spec.get("progressDeadlineSeconds") or 0)
        != desired.progress_deadline_seconds
    ):
        return False
    strategy = _mapping_or_empty(spec.get("strategy"))
    if strategy.get("type") != desired.rollout_type:
        return False
    if desired.rollout_type == "RollingUpdate":
        rolling = _mapping_or_empty(strategy.get("rollingUpdate"))
        if rolling.get("maxUnavailable") != desired.max_unavailable:
            return False
        if rolling.get("maxSurge") != desired.max_surge:
            return False
    if _deployment_annotation(deployment, MODEL_RESOURCE_ANNOTATION) != owner:
        return False
    template = _mapping_or_empty(spec.get("template"))
    template_metadata = _mapping_or_empty(template.get("metadata"))
    annotations = _mapping_or_empty(template_metadata.get("annotations"))
    if annotations.get(MANIFEST_HASH_ANNOTATION) != desired.manifest_hash:
        return False
    container = _find_container(deployment, desired.container_name)
    if container is None:
        return False
    for env in container.get("env") or []:
        if not isinstance(env, Mapping) or env.get("name") != PRELOAD_ENV_NAME:
            continue
        value_from = _mapping_or_empty(env.get("valueFrom"))
        config_ref = _mapping_or_empty(value_from.get("configMapKeyRef"))
        return (
            config_ref.get("name") == desired.manifest_config_map_name
            and config_ref.get("key") == PRELOAD_DATA_KEY
        )
    return False


def _deployment_ready(
    deployment: Mapping[str, Any], replicas: int, target_generation: int
) -> bool:
    metadata = _mapping_or_empty(deployment.get("metadata"))
    status = _mapping_or_empty(deployment.get("status"))
    generation = int(metadata.get("generation") or 0)
    observed = int(status.get("observedGeneration") or 0)
    return (
        generation >= target_generation
        and observed >= target_generation
        and int(status.get("updatedReplicas") or 0) >= replicas
        and int(status.get("readyReplicas") or 0) >= replicas
        and int(status.get("availableReplicas") or 0) >= replicas
        and int(status.get("unavailableReplicas") or 0) == 0
    )


def _deployment_failed(deployment: Mapping[str, Any], target_generation: int) -> bool:
    status = _mapping_or_empty(deployment.get("status"))
    if int(status.get("observedGeneration") or 0) < target_generation:
        return False
    for condition in status.get("conditions") or []:
        if not isinstance(condition, Mapping):
            continue
        condition_type = condition.get("type")
        condition_status = condition.get("status")
        reason = condition.get("reason")
        if (
            condition_type == "Progressing"
            and condition_status == "False"
            and reason == "ProgressDeadlineExceeded"
        ):
            return True
        if condition_type == "ReplicaFailure" and condition_status == "True":
            return True
    return False


def _find_container(
    deployment: Mapping[str, Any], name: str
) -> Mapping[str, Any] | None:
    spec = _mapping_or_empty(deployment.get("spec"))
    template = _mapping_or_empty(spec.get("template"))
    pod_spec = _mapping_or_empty(template.get("spec"))
    for container in pod_spec.get("containers") or []:
        if isinstance(container, Mapping) and container.get("name") == name:
            return container
    return None


def _deployment_annotation(deployment: Mapping[str, Any], key: str) -> str | None:
    metadata = _mapping_or_empty(deployment.get("metadata"))
    annotations = _mapping_or_empty(metadata.get("annotations"))
    value = annotations.get(key)
    return None if value is None else str(value)


def _deployment_template_annotation(
    deployment: Mapping[str, Any], key: str
) -> str | None:
    spec = _mapping_or_empty(deployment.get("spec"))
    template = _mapping_or_empty(spec.get("template"))
    metadata = _mapping_or_empty(template.get("metadata"))
    annotations = _mapping_or_empty(metadata.get("annotations"))
    value = annotations.get(key)
    return None if value is None else str(value)


def _deployment_generation(deployment: Mapping[str, Any]) -> int:
    metadata = _mapping_or_empty(deployment.get("metadata"))
    return int(metadata.get("generation") or 1)


def _resource_owner(body: Mapping[str, Any]) -> str:
    metadata = _mapping_or_empty(body.get("metadata"))
    return f"{metadata.get('namespace') or 'default'}/{metadata.get('name')}"


def _rollback_target(metadata: Mapping[str, Any]) -> str | None:
    annotations = _mapping_or_empty(metadata.get("annotations"))
    value = annotations.get(ROLLBACK_ANNOTATION)
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped or None


def _safe_event(
    api: KubernetesAPIClient,
    body: Mapping[str, Any],
    *,
    reason: str,
    event_type: str,
    message: str,
    generation: int,
    now: datetime,
) -> None:
    metadata = _mapping_or_empty(body.get("metadata"))
    namespace = str(metadata.get("namespace") or "default")
    name = str(metadata.get("name") or "openmedmodel")
    uid = str(metadata.get("uid") or "")
    event_name = _event_name(name, reason, generation)
    event = {
        "apiVersion": "v1",
        "kind": "Event",
        "metadata": {"name": event_name, "namespace": namespace},
        "involvedObject": {
            "apiVersion": f"{API_GROUP}/{API_VERSION}",
            "kind": API_KIND,
            "namespace": namespace,
            "name": name,
            "uid": uid,
        },
        "reason": reason,
        "message": message,
        "type": event_type,
        "source": {"component": "openmed-operator"},
        "firstTimestamp": _timestamp(now),
        "lastTimestamp": _timestamp(now),
        "count": 1,
    }
    try:
        api.create_event(namespace, event)
    except KubernetesAPIError:
        # Events are observability aids and must not make model convergence fail.
        return


def _event_name(resource_name: str, reason: str, generation: int) -> str:
    base = re.sub(r"[^a-z0-9-]+", "-", f"{resource_name}-{reason}".lower())
    base = base.strip("-")[:220]
    digest = hashlib.sha256(
        f"{resource_name}:{reason}:{generation}".encode()
    ).hexdigest()[:10]
    return f"{base}-{generation}-{digest}"[:253].rstrip("-")


def _name_with_suffix(name: str, suffix: str) -> str:
    candidate = f"{name}-{suffix}"
    if len(candidate) <= 253 and _DNS_SUBDOMAIN_RE.fullmatch(candidate) is not None:
        return candidate
    digest = hashlib.sha256(candidate.encode()).hexdigest()[:10]
    prefix_length = 63 - len(suffix) - len(digest) - 2
    prefix = name.replace(".", "-")[:prefix_length].rstrip("-")
    return f"{prefix}-{suffix}-{digest}"


def _validate_dns_subdomain(value: str, *, field: str) -> None:
    if len(value) > 253 or _DNS_SUBDOMAIN_RE.fullmatch(value) is None:
        raise SpecValidationError(f"{field} must be a valid DNS subdomain")


def _validate_dns_label(value: str, *, field: str) -> None:
    if len(value) > 63 or _DNS_LABEL_RE.fullmatch(value) is None:
        raise SpecValidationError(f"{field} must be a valid DNS label")


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpecValidationError(f"{field} must be an object")
    return value


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mapping_or_none(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _required_string(value: Mapping[str, Any], key: str, *, field: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise SpecValidationError(f"{field} must be a non-empty string")
    return raw.strip()


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _required_int(value: Mapping[str, Any], key: str, *, field: str) -> int:
    raw = value.get(key)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise SpecValidationError(f"{field} must be an integer")
    return raw


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SpecValidationError(f"{field} must be a positive integer")
    return value


def _int_or_percent(value: Any, *, field: str) -> int | str:
    if isinstance(value, bool):
        raise SpecValidationError(f"{field} must be an integer or percentage")
    if isinstance(value, int):
        if value < 0:
            raise SpecValidationError(f"{field} must not be negative")
        return value
    if isinstance(value, str) and _PERCENT_RE.fullmatch(value):
        percentage = int(value[:-1])
        if percentage <= 100:
            return value
    raise SpecValidationError(
        f"{field} must be a non-negative integer or a percentage from 0% to 100%"
    )


def _is_zero(value: int | str) -> bool:
    return value == 0 or value == "0%"


def _digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _url(value: str) -> str:
    return quote(value, safe="")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _timestamp(value: datetime) -> str:
    return _as_utc(value).isoformat(timespec="seconds").replace("+00:00", "Z")


def _invalid_status(
    body: Mapping[str, Any], message: str, *, now: datetime
) -> dict[str, Any]:
    metadata = _mapping_or_empty(body.get("metadata"))
    generation = int(metadata.get("generation") or 1)
    existing = _mapping_or_empty(body.get("status"))
    result = copy.deepcopy(dict(existing))
    result.update(
        {
            "observedGeneration": generation,
            "phase": PHASE_FAILED,
            "conditions": _merge_conditions(
                existing.get("conditions"),
                (
                    ("Ready", "False", "InvalidSpec", message),
                    ("Progressing", "False", "Blocked", "Reconciliation is blocked."),
                    ("Degraded", "True", "InvalidSpec", message),
                    ("RolledBack", "False", "NotRequested", "No rollback is active."),
                ),
                generation=generation,
                now=now,
            ),
        }
    )
    return result


async def _kopf_reconcile_handler(
    *,
    body: Mapping[str, Any],
    patch: MutableMapping[str, Any],
    logger: Any,
    **_: Any,
) -> None:
    try:
        result = await asyncio.to_thread(
            reconcile_openmed_model,
            body,
            KubernetesAPIClient.from_service_account(),
        )
    except SpecValidationError as exc:
        patch["status"] = _invalid_status(body, str(exc), now=_utcnow())
        logger.warning("OpenMedModel spec validation failed")
        return
    except KubernetesAPIError as exc:
        if kopf is None:  # pragma: no cover - defensive import boundary
            raise
        raise kopf.TemporaryError(str(exc), delay=15) from exc
    patch["status"] = result.status
    metadata = _mapping_or_empty(body.get("metadata"))
    logger.info(
        "Reconciled OpenMedModel %s/%s to phase %s",
        metadata.get("namespace") or "default",
        metadata.get("name"),
        result.status.get("phase"),
    )


async def _kopf_delete_handler(
    *, body: Mapping[str, Any], logger: Any, **_: Any
) -> None:
    try:
        await asyncio.to_thread(
            decommission_openmed_model,
            body,
            KubernetesAPIClient.from_service_account(),
        )
    except KubernetesAPIError as exc:
        if kopf is None:  # pragma: no cover - defensive import boundary
            raise
        raise kopf.TemporaryError(str(exc), delay=15) from exc
    metadata = _mapping_or_empty(body.get("metadata"))
    logger.info(
        "Decommissioned OpenMedModel %s/%s",
        metadata.get("namespace") or "default",
        metadata.get("name"),
    )


if kopf is not None:  # pragma: no branch - registration only in operator extra
    kopf.on.create(API_GROUP, API_VERSION, API_PLURAL)(_kopf_reconcile_handler)
    kopf.on.update(API_GROUP, API_VERSION, API_PLURAL)(_kopf_reconcile_handler)
    kopf.on.resume(API_GROUP, API_VERSION, API_PLURAL)(_kopf_reconcile_handler)
    kopf.timer(
        API_GROUP,
        API_VERSION,
        API_PLURAL,
        interval=15.0,
        initial_delay=5.0,
        sharp=True,
    )(_kopf_reconcile_handler)
    kopf.on.delete(API_GROUP, API_VERSION, API_PLURAL)(_kopf_delete_handler)


__all__ = [
    "ALLOWED_TIERS",
    "API_GROUP",
    "API_KIND",
    "API_PLURAL",
    "API_VERSION",
    "DesiredModel",
    "KubernetesAPIClient",
    "KubernetesAPIError",
    "MANIFEST_DATA_KEY",
    "MANIFEST_HASH_ANNOTATION",
    "MODEL_RESOURCE_ANNOTATION",
    "PHASE_FAILED",
    "PHASE_PENDING",
    "PHASE_READY",
    "PHASE_ROLLED_BACK",
    "PHASE_ROLLING_OUT",
    "PRELOAD_DATA_KEY",
    "PRELOAD_ENV_NAME",
    "ROLLBACK_ANNOTATION",
    "ReconcileResult",
    "SpecValidationError",
    "decommission_openmed_model",
    "reconcile_openmed_model",
]
