"""Deployment security-posture contract for rendered Helm and Compose files.

The production service security profile (:mod:`openmed.service.security_profile`)
checks a running configuration. This module checks what a deployment would hand
that service before anything is deployed: it reads rendered Kubernetes
manifests (``helm template`` output) and Compose files, turns every OpenMed
workload into a normalized description, and evaluates it against one contract.

Two kinds of rules apply:

* The REST service workload's rendered environment is fed into the *same*
  registered security-profile checks the service runs at startup, under the
  ``prod`` profile, through :func:`run_service_preflight`. Chart values and the
  runtime self-check therefore cannot drift apart.
* Deployment-only rules (``deploy.*``) cover what the service cannot see from
  inside: container security contexts, resource limits, readiness wiring,
  telemetry and offline flags, development overrides, contradictory settings,
  network exposure, and whether workers and jobs inherit the API's security
  profile and Journey store settings.

Settings sourced from Secrets are never read. Checks that would need a secret
value (for example the PostgreSQL ``sslmode`` inside a DSN Secret) are reported
as deferred to the runtime self-check, which the readiness probe enforces.

Findings and reports contain target ids, workload ids, check ids, reason codes,
levels, issue or pull-request numbers and digests only: never setting values,
secret names, hosts, paths or file contents.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import secrets
import shlex
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final

import yaml

from openmed.service.security_profile import (
    CONFIG_PROFILE_ENV_VAR,
    JOURNEY_POSTGRES_DSN_ENV_VAR,
    PROD_PROFILE,
    SECURITY_PROFILE_ENV_VAR,
    CheckLevel,
    SecurityProfileName,
    SecurityProfileReport,
    is_loopback_host,
    registered_security_checks,
    resolve_security_profile,
    run_service_preflight,
)

POSTURE_CONTRACT_ID: Final = "openmed.deploy.posture_contract.v1"
POSTURE_REPORT_SCHEMA_VERSION: Final = "openmed.deploy.posture_report.v1"
POSTURE_LANE_SCHEMA_VERSION: Final = "openmed.deploy.posture_lane.v1"

EXIT_LANE_PASSED: Final = 0
EXIT_LANE_FAILED: Final = 1
EXIT_LANE_UNAVAILABLE: Final = 2

SERVICE_LAUNCHER_MODULE: Final = "openmed.service.logging"
JOURNEY_MODULE: Final = "openmed.deploy.journey"
MCP_MODULE: Final = "openmed.mcp.server"
# Arguments the OpenMed service images pass to the reference launcher when a
# deployment does not override the command. Both Dockerfiles must match.
IMAGE_DEFAULT_LISTENER_ARGS: Final = ("--host", "0.0.0.0", "--port", "8080")
LAUNCHER_DEFAULT_HOST: Final = "127.0.0.1"

TELEMETRY_SETTINGS: Final = (
    "HF_HUB_DISABLE_TELEMETRY",
    "DISABLE_TELEMETRY",
    "DO_NOT_TRACK",
)
OFFLINE_SETTINGS: Final = ("OPENMED_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
# Settings that enable development or insecure behavior when truthy.
INSECURE_OVERRIDE_SETTINGS: Final = (
    "OPENMED_MCP_ALLOW_INSECURE_LOCALHOST",
    "OPENMED_OPENHIM_ALLOW_INSECURE_HTTP",
    "OPENMED_SKIP_MODEL_VERIFY",
)
JOURNEY_STORE_SETTINGS: Final = (
    JOURNEY_POSTGRES_DSN_ENV_VAR,
    "OPENMED_JOURNEY_SCHEMA",
    "OPENMED_JOURNEY_POSTGRES_SSLROOTCERT",
    "OPENMED_JOURNEY_POSTGRES_SSLCERT",
    "OPENMED_JOURNEY_POSTGRES_SSLKEY",
)
JOURNEY_ARTIFACT_ROOT_SETTING: Final = "OPENMED_JOURNEY_ARTIFACT_ROOT"

# Runtime checks whose inputs cannot be judged when the setting comes from a
# Secret or a mounted file. The service re-runs them on the real value at
# startup and keeps readiness closed when they fail.
SECRET_DEFERRED_CHECKS: Final[Mapping[str, tuple[str, ...]]] = {
    "journey.store_tls": (JOURNEY_POSTGRES_DSN_ENV_VAR,),
}
DEFERRED_REASON: Final = "secret_sourced_runtime_check"

_AUTH_API_KEYS: Final = "OPENMED_SERVICE_AUTH_API_KEYS"
_AUTH_JWKS: Final = "OPENMED_SERVICE_AUTH_JWKS"
_TRUE: Final = frozenset({"1", "true", "yes", "on", "enabled"})
_FALSE: Final = frozenset({"0", "false", "no", "off", "disabled"})
_OPERATOR_PLACEHOLDER: Final = "posture-lane-operator-value"
_GENERIC_STAND_IN: Final = "posture-lane-stand-in"
_REPORTABLE_SETTINGS: Final = frozenset(
    (*TELEMETRY_SETTINGS, *OFFLINE_SETTINGS, "HF_DATASETS_OFFLINE")
)
_TLS_LISTENER_REASONS: Final = frozenset(
    {"tls_listener", "tls_terminated_by_trusted_proxy", "loopback_listener"}
)


class PostureLaneError(RuntimeError):
    """A lane definition or render failed. The message is a fixed code."""


class PostureLaneUnavailable(PostureLaneError):
    """A required tool, such as ``helm``, is not available."""


class Platform(str, Enum):
    """Deployment format a workload was read from."""

    HELM = "helm"
    COMPOSE = "compose"


class WorkloadRole(str, Enum):
    """What an OpenMed workload runs, derived from its launch command."""

    API = "api"
    WORKER = "worker"
    JOB = "job"
    MCP = "mcp"
    UNCLASSIFIED = "unclassified"


class SettingSource(str, Enum):
    """Where one environment setting's value comes from."""

    LITERAL = "literal"
    SECRET = "secret"
    FILE = "file"
    OPERATOR = "operator"
    RUNTIME = "runtime"


class FindingSource(str, Enum):
    """Which rule set produced a finding."""

    SECURITY_PROFILE = "security_profile"
    DEPLOY_CONTRACT = "deploy_contract"


class Disposition(str, Enum):
    """Why a lane expects a finding."""

    FAIL_CLOSED = "fail_closed"
    KNOWN = "known"
    NEGATIVE_CONTROL = "negative_control"


class TargetPurpose(str, Enum):
    """What a lane target represents."""

    SHIPPED = "shipped"
    EXAMPLE = "example"
    NEGATIVE_CONTROL = "negative_control"


@dataclass(frozen=True, slots=True)
class Setting:
    """One rendered environment setting.

    Args:
        source: Where the value comes from.
        value: Literal or interpolated value; empty for secrets.
        identity: Opaque identity used to compare settings across workloads.
    """

    source: SettingSource
    value: str = field(default="", repr=False)
    identity: str = field(default="", repr=False)

    @classmethod
    def literal(cls, value: Any) -> Setting:
        """Return a literal setting."""
        text = "" if value is None else _scalar_text(value)
        return cls(SettingSource.LITERAL, text, f"literal:{text}")

    @property
    def known(self) -> bool:
        """Return whether the value is visible at render time."""
        return self.source is SettingSource.LITERAL


@dataclass(frozen=True, slots=True)
class DeployWorkload:
    """A normalized OpenMed container read from a manifest or Compose file.

    Args:
        workload_id: Stable id such as ``Deployment/api`` or ``openmed``.
        platform: Deployment format.
        role: What the container runs.
        environment: Effective environment settings by name.
        launcher_args: Arguments passed to the reference service launcher.
        hardening: Security-context reason codes the container breaks.
        cpu_limit: Whether a CPU limit is set.
        memory_limit: Whether a memory limit is set.
        readiness_probe: Whether a readiness probe or health check is set.
        readiness_readyz: Whether that probe reaches a ``/readyz`` endpoint.
        exposed: Whether the platform publishes the container beyond the
            host or cluster network.
        contradictions: Setting names defined more than once with different
            values.
        unverified_sources: Count of environment sources the lane cannot read.
    """

    workload_id: str
    platform: Platform
    role: WorkloadRole
    environment: Mapping[str, Setting] = field(repr=False, default_factory=dict)
    launcher_args: tuple[str, ...] = field(default=(), repr=False)
    hardening: tuple[str, ...] = ()
    cpu_limit: bool = False
    memory_limit: bool = False
    readiness_probe: bool = False
    readiness_readyz: bool = False
    exposed: bool = False
    contradictions: tuple[str, ...] = field(default=(), repr=False)
    unverified_sources: int = 0

    def literal(self, name: str) -> str | None:
        """Return a setting's visible value, or ``None`` when absent or hidden."""
        setting = self.environment.get(name)
        if setting is None or not setting.known:
            return None
        return setting.value.strip()


@dataclass(frozen=True, slots=True)
class PostureFinding:
    """One value-free contract finding for a workload."""

    workload: str
    check: str
    reason: str
    level: CheckLevel
    source: FindingSource
    tracking_issue: int | None = None
    setting: str | None = None

    @property
    def gated(self) -> bool:
        """Return whether the finding blocks a production deployment."""
        return self.level is CheckLevel.REQUIRED

    @property
    def key(self) -> tuple[str, str, str, str | None]:
        """Return the (workload, check, reason, setting) key lanes compare."""
        return (self.workload, self.check, self.reason, self.setting)

    def to_dict(self) -> dict[str, Any]:
        """Return the value-free JSON form."""
        payload: dict[str, Any] = {
            "workload": self.workload,
            "check": self.check,
            "reason": self.reason,
            "level": self.level.value,
            "source": self.source.value,
        }
        if self.tracking_issue is not None:
            payload["tracking_issue"] = self.tracking_issue
        if self.setting is not None:
            payload["setting"] = self.setting
        return payload


@dataclass(frozen=True, slots=True)
class DeferredCheck:
    """A runtime check the render cannot judge because its input is secret."""

    workload: str
    check: str
    reason: str = DEFERRED_REASON

    def to_dict(self) -> dict[str, str]:
        """Return the value-free JSON form."""
        return {"workload": self.workload, "check": self.check, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class PostureEvaluation:
    """Findings for every workload of one rendered deployment."""

    workloads: tuple[DeployWorkload, ...]
    findings: tuple[PostureFinding, ...]
    deferred: tuple[DeferredCheck, ...] = ()
    skipped_workloads: int = 0

    @property
    def gated(self) -> tuple[PostureFinding, ...]:
        """Return the findings that block a production deployment."""
        return tuple(finding for finding in self.findings if finding.gated)


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

_REQUIRED = CheckLevel.REQUIRED

DEPLOY_CHECKS: Final[Mapping[str, CheckLevel | str]] = {
    "deploy.workload": _REQUIRED,
    "deploy.service_config": _REQUIRED,
    "deploy.environment": _REQUIRED,
    "deploy.profile": _REQUIRED,
    "deploy.overrides": _REQUIRED,
    "deploy.security_context": _REQUIRED,
    "deploy.resources": _REQUIRED,
    "deploy.readiness": _REQUIRED,
    "deploy.egress": _REQUIRED,
    "deploy.inheritance": _REQUIRED,
    # Network exposure without TLS mirrors the runtime listener check, so it
    # becomes blocking in the profile version that makes that check required.
    "deploy.listener": "listener.transport",
}


def _deploy_level(check_id: str) -> CheckLevel:
    level = DEPLOY_CHECKS[check_id]
    if isinstance(level, CheckLevel):
        return level
    for check in registered_security_checks():
        if check.check_id == level:
            return check.level_for(PROD_PROFILE)
    return CheckLevel.REQUIRED


def posture_contract() -> dict[str, Any]:
    """Return the value-free contract: deploy rules and runtime check levels.

    Returns:
        The contract id, the production security profile id, every deploy rule
        and runtime check with its production level, and a digest over them.
    """
    body: dict[str, Any] = {
        "id": POSTURE_CONTRACT_ID,
        "security_profile": PROD_PROFILE.identifier,
        "deploy_checks": {
            check_id: _deploy_level(check_id).value for check_id in DEPLOY_CHECKS
        },
        "runtime_checks": {
            check.check_id: check.level_for(PROD_PROFILE).value
            for check in registered_security_checks()
        },
    }
    body["digest"] = _digest(body)
    return body


def _digest(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in _TRUE


def _falsy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in _FALSE


def _scalar_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _reportable(name: str) -> str | None:
    if name.startswith("OPENMED_") or name in _REPORTABLE_SETTINGS:
        return name
    return None


# ---------------------------------------------------------------------------
# Launch commands
# ---------------------------------------------------------------------------


def _tokens(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return shlex.split(value)
    return [_scalar_text(item) for item in value]


def _after(tokens: Sequence[str], marker: str) -> list[str]:
    index = tokens.index(marker)
    return list(tokens[index + 1 :])


def classify_command(
    command: Sequence[str], args: Sequence[str] = ()
) -> tuple[WorkloadRole, tuple[str, ...]]:
    """Classify an OpenMed container from its entrypoint and arguments.

    Args:
        command: Entrypoint override (Kubernetes ``command``, Compose
            ``entrypoint``). Empty means the image entrypoint.
        args: Arguments (Kubernetes ``args``, Compose ``command``).

    Returns:
        The role and, for the REST service, the reference launcher arguments.
    """
    command = list(command)
    args = list(args)
    if not command and not args:
        return WorkloadRole.API, IMAGE_DEFAULT_LISTENER_ARGS
    if not command and args[0].startswith("-"):
        return WorkloadRole.API, tuple(args)
    tokens = command + args
    if JOURNEY_MODULE in tokens or "openmed-journey" in tokens:
        marker = JOURNEY_MODULE if JOURNEY_MODULE in tokens else "openmed-journey"
        rest = _after(tokens, marker)
        sub = next((token for token in rest if not token.startswith("-")), "")
        return (WorkloadRole.WORKER if sub == "worker" else WorkloadRole.JOB), ()
    if MCP_MODULE in tokens or "openmed-mcp" in tokens:
        return WorkloadRole.MCP, ()
    if SERVICE_LAUNCHER_MODULE in tokens:
        return WorkloadRole.API, tuple(_after(tokens, SERVICE_LAUNCHER_MODULE))
    return WorkloadRole.UNCLASSIFIED, ()


def launcher_listener(args: Sequence[str]) -> tuple[str, bool]:
    """Return the bind host and TLS flag the reference launcher would use.

    Args:
        args: Reference launcher arguments.

    Returns:
        The bind host and whether ``--ssl-certfile`` is set.
    """
    host = LAUNCHER_DEFAULT_HOST
    tls = False
    items = list(args)
    for index, token in enumerate(items):
        name, _, inline = token.partition("=")
        value = (
            inline if inline else (items[index + 1] if index + 1 < len(items) else "")
        )
        if name == "--host" and value:
            host = value
        elif name == "--ssl-certfile" and value:
            tls = True
    return host, tls


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def _merge(
    entries: Iterable[tuple[str, Setting]],
) -> tuple[dict[str, Setting], tuple[str, ...]]:
    merged: dict[str, Setting] = {}
    contradictions: list[str] = []
    for name, setting in entries:
        previous = merged.get(name)
        if previous is not None and previous.identity != setting.identity:
            if name not in contradictions:
                contradictions.append(name)
        merged[name] = setting
    return merged, tuple(contradictions)


def _resolve_file_settings(
    environment: dict[str, Setting], secret_paths: Sequence[str]
) -> tuple[dict[str, Setting], tuple[str, ...]]:
    """Fold ``OPENMED_*_FILE`` indirections into their base setting."""
    resolved = dict(environment)
    contradictions: list[str] = []
    for name in [key for key in environment if key.endswith("_FILE")]:
        base = name[: -len("_FILE")]
        if not base.startswith("OPENMED_"):
            continue
        setting = environment[name]
        if setting.source is not SettingSource.LITERAL:
            continue
        path = setting.value
        source = (
            SettingSource.SECRET
            if any(
                path == root or path.startswith(root.rstrip("/") + "/")
                for root in secret_paths
            )
            else SettingSource.FILE
        )
        if base in environment:
            contradictions.append(base)
        resolved.pop(name)
        resolved[base] = Setting(source, "", f"file:{path}")
    return resolved, tuple(contradictions)


def _stand_in(name: str) -> str:
    """Return a synthetic, well-formed value for a secret-sourced setting."""
    if name == _AUTH_API_KEYS:
        from openmed.service.auth import hash_api_key

        return json.dumps(
            [
                {
                    "id": "posture-lane",
                    "key_hash": hash_api_key(secrets.token_urlsafe(24)),
                    "principal": "posture-lane",
                    "scopes": ["models:read"],
                }
            ]
        )
    if name == _AUTH_JWKS:
        key = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=")
        return json.dumps(
            {"keys": [{"kty": "oct", "kid": "posture-lane", "k": key.decode("ascii")}]}
        )
    if name == JOURNEY_POSTGRES_DSN_ENV_VAR:
        return "postgresql://posture-lane.invalid/openmed"
    return _GENERIC_STAND_IN


def evaluation_environment(workload: DeployWorkload) -> dict[str, str]:
    """Return the service environment the security profile is evaluated on.

    Literal and interpolated settings keep their rendered value. Secret,
    file, runtime and operator-supplied settings are replaced by synthetic
    stand-ins that exist only in memory for the evaluation.

    Args:
        workload: The rendered REST service workload.

    Returns:
        Environment mapping for :func:`run_service_preflight`.
    """
    return {
        name: setting.value if setting.known else _stand_in(name)
        for name, setting in workload.environment.items()
    }


# ---------------------------------------------------------------------------
# Kubernetes manifests
# ---------------------------------------------------------------------------

_POD_KINDS: Final = ("Deployment", "StatefulSet", "DaemonSet", "Job", "ReplicaSet")


def _pod_template(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], dict] | None:
    kind = manifest.get("kind")
    spec = manifest.get("spec") or {}
    if kind == "Pod":
        return dict((manifest.get("metadata") or {}).get("labels") or {}), spec
    if kind == "CronJob":
        spec = ((spec.get("jobTemplate") or {}).get("spec")) or {}
    elif kind not in _POD_KINDS:
        return None
    template = spec.get("template") or {}
    labels = dict((template.get("metadata") or {}).get("labels") or {})
    return labels, template.get("spec") or {}


def _k8s_environment(
    container: Mapping[str, Any], configmaps: Mapping[str, Mapping[str, Any]]
) -> tuple[list[tuple[str, Setting]], int]:
    entries: list[tuple[str, Setting]] = []
    unverified = 0
    for source in container.get("envFrom") or []:
        prefix = str(source.get("prefix") or "")
        reference = source.get("configMapRef")
        data = configmaps.get(str((reference or {}).get("name")))
        if reference is None or data is None:
            unverified += 1
            continue
        for key, value in data.items():
            entries.append((prefix + str(key), Setting.literal(value)))
    for item in container.get("env") or []:
        name = str(item.get("name") or "")
        value_from = item.get("valueFrom")
        if not value_from:
            entries.append((name, Setting.literal(item.get("value"))))
            continue
        if "secretKeyRef" in value_from:
            ref = value_from["secretKeyRef"] or {}
            identity = f"secret:{ref.get('name')}/{ref.get('key')}"
            entries.append((name, Setting(SettingSource.SECRET, "", identity)))
        elif "configMapKeyRef" in value_from:
            ref = value_from["configMapKeyRef"] or {}
            data = configmaps.get(str(ref.get("name")))
            if data is None or ref.get("key") not in data:
                unverified += 1
                continue
            entries.append((name, Setting.literal(data[ref.get("key")])))
        else:
            identity = f"runtime:{json.dumps(value_from, sort_keys=True)}"
            entries.append((name, Setting(SettingSource.RUNTIME, "", identity)))
    return entries, unverified


def _k8s_hardening(pod: Mapping[str, Any], container: Mapping[str, Any]) -> list[str]:
    pod_context = pod.get("securityContext") or {}
    context = container.get("securityContext") or {}
    effective = {
        key: pod_context[key]
        for key in ("runAsUser", "runAsGroup", "runAsNonRoot", "seccompProfile")
        if key in pod_context
    }
    effective.update(context)
    reasons: list[str] = []
    user = effective.get("runAsUser")
    if user == 0 or (
        effective.get("runAsNonRoot") is not True
        and not (isinstance(user, int) and user > 0)
    ):
        reasons.append("runs_as_root")
    if context.get("allowPrivilegeEscalation") is not False or context.get(
        "privileged"
    ):
        reasons.append("privilege_escalation_allowed")
    if context.get("readOnlyRootFilesystem") is not True:
        reasons.append("root_filesystem_writable")
    capabilities = context.get("capabilities") or {}
    if "ALL" not in (capabilities.get("drop") or []):
        reasons.append("capabilities_not_dropped")
    if capabilities.get("add"):
        reasons.append("capabilities_added")
    if context.get("privileged"):
        reasons.append("privileged_container")
    seccomp = (effective.get("seccompProfile") or {}).get("type")
    if seccomp not in {"RuntimeDefault", "Localhost"}:
        reasons.append("seccomp_not_confined")
    if any(pod.get(flag) for flag in ("hostNetwork", "hostPID", "hostIPC")) or any(
        "hostPort" in port for port in container.get("ports") or []
    ):
        reasons.append("host_namespace_shared")
    if any("hostPath" in volume for volume in pod.get("volumes") or []):
        reasons.append("host_path_mounted")
    return reasons


def _probe_reaches_readyz(probe: Mapping[str, Any] | None) -> tuple[bool, bool]:
    if not probe:
        return False, False
    http = probe.get("httpGet") or {}
    if http:
        path = str(http.get("path") or "")
        return True, path == "/readyz" or path.startswith("/readyz?")
    command = (probe.get("exec") or {}).get("command") or []
    return True, any("/readyz" in str(token) for token in command)


def _selector_matches(selector: Mapping[str, Any], labels: Mapping[str, Any]) -> bool:
    return bool(selector) and all(labels.get(k) == v for k, v in selector.items())


def _secret_mount_paths(pod: Mapping[str, Any], container: Mapping[str, Any]) -> list:
    secret_volumes = {
        volume.get("name")
        for volume in pod.get("volumes") or []
        if "secret" in volume
        or any(
            "secret" in item
            for item in (volume.get("projected") or {}).get("sources", [])
        )
    }
    return [
        str(mount.get("mountPath"))
        for mount in container.get("volumeMounts") or []
        if mount.get("name") in secret_volumes
    ]


def workloads_from_manifests(
    manifests: Sequence[Mapping[str, Any]],
) -> tuple[DeployWorkload, ...]:
    """Normalize every container of rendered Kubernetes manifests.

    Args:
        manifests: Parsed documents, for example ``helm template`` output.

    Returns:
        One workload per container, in manifest order.
    """
    configmaps = {
        str((item.get("metadata") or {}).get("name")): item.get("data") or {}
        for item in manifests
        if item.get("kind") == "ConfigMap"
    }
    exposing_services = [
        (item.get("spec") or {}).get("selector") or {}
        for item in manifests
        if item.get("kind") == "Service"
        and (item.get("spec") or {}).get("type") in {"NodePort", "LoadBalancer"}
    ]
    workloads: list[DeployWorkload] = []
    for manifest in manifests:
        template = _pod_template(manifest)
        if template is None:
            continue
        labels, pod = template
        metadata = manifest.get("metadata") or {}
        component = labels.get("app.kubernetes.io/component") or (
            metadata.get("labels") or {}
        ).get("app.kubernetes.io/component")
        base_id = f"{manifest.get('kind')}/{component or metadata.get('name')}"
        containers = [
            *(pod.get("initContainers") or []),
            *(pod.get("containers") or []),
        ]
        exposed = any(_selector_matches(sel, labels) for sel in exposing_services)
        for container in containers:
            workload_id = (
                base_id
                if len(containers) == 1
                else f"{base_id}/{container.get('name')}"
            )
            role, launcher_args = classify_command(
                _tokens(container.get("command")), _tokens(container.get("args"))
            )
            entries, unverified = _k8s_environment(container, configmaps)
            environment, contradictions = _merge(entries)
            environment, file_conflicts = _resolve_file_settings(
                environment, _secret_mount_paths(pod, container)
            )
            limits = (container.get("resources") or {}).get("limits") or {}
            probe, readyz = _probe_reaches_readyz(container.get("readinessProbe"))
            workloads.append(
                DeployWorkload(
                    workload_id=workload_id,
                    platform=Platform.HELM,
                    role=role,
                    environment=environment,
                    launcher_args=launcher_args,
                    hardening=tuple(_k8s_hardening(pod, container)),
                    cpu_limit=bool(limits.get("cpu")),
                    memory_limit=bool(limits.get("memory")),
                    readiness_probe=probe,
                    readiness_readyz=readyz,
                    exposed=exposed,
                    contradictions=contradictions + file_conflicts,
                    unverified_sources=unverified,
                )
            )
    return tuple(workloads)


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


@dataclass
class _Interpolation:
    environ: Mapping[str, str]
    operator: bool = False


def _closing_brace(text: str, start: int) -> int:
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    raise PostureLaneError("compose_interpolation_invalid")


def _interpolate(text: str, state: _Interpolation) -> str:
    out: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char != "$":
            out.append(char)
            index += 1
            continue
        following = text[index + 1 : index + 2]
        if following == "$":
            out.append("$")
            index += 2
        elif following == "{":
            end = _closing_brace(text, index + 1)
            out.append(_expand(text[index + 2 : end], state))
            index = end + 1
        else:
            end = index + 1
            while end < len(text) and (text[end].isalnum() or text[end] == "_"):
                end += 1
            if end == index + 1:
                out.append("$")
                index += 1
                continue
            out.append(state.environ.get(text[index + 1 : end], ""))
            index = end
    return "".join(out)


def _expand(expression: str, state: _Interpolation) -> str:
    end = 0
    while end < len(expression) and (
        expression[end].isalnum() or expression[end] == "_"
    ):
        end += 1
    if end == 0:
        raise PostureLaneError("compose_interpolation_invalid")
    name, rest = expression[:end], expression[end:]
    value = state.environ.get(name)
    if not rest:
        return value or ""
    for operator in (":-", ":?", ":+", "-", "?", "+"):
        if not rest.startswith(operator):
            continue
        argument = rest[len(operator) :]
        present = value is not None and (value != "" or ":" not in operator)
        if operator.endswith("-"):
            return (
                value
                if present and value is not None
                else _interpolate(argument, state)
            )
        if operator.endswith("?"):
            if present and value is not None:
                return value
            state.operator = True
            return _OPERATOR_PLACEHOLDER
        return _interpolate(argument, state) if present else ""
    raise PostureLaneError("compose_interpolation_invalid")


def interpolate_compose(text: str, environ: Mapping[str, str]) -> tuple[str, bool]:
    """Apply Compose variable interpolation without reading the process env.

    Supports ``$$``, ``$VAR``, ``${VAR}``, ``${VAR:-default}``,
    ``${VAR-default}``, ``${VAR:?message}``, ``${VAR?message}``,
    ``${VAR:+alt}`` and ``${VAR+alt}``. A required variable that ``environ``
    does not supply is replaced by an opaque operator placeholder.

    Args:
        text: Raw Compose string.
        environ: Operator variables (the ``.env`` a deployment would use).

    Returns:
        The interpolated text and whether an operator-supplied value was used.

    Raises:
        PostureLaneError: If the expression is malformed.
    """
    state = _Interpolation(environ)
    return _interpolate(text, state), state.operator


def _compose_setting(value: Any, environ: Mapping[str, str]) -> Setting:
    if value is None:
        return Setting.literal("")
    raw = _scalar_text(value)
    text, operator = interpolate_compose(raw, environ)
    if operator and text.strip() == _OPERATOR_PLACEHOLDER:
        # The whole value is supplied by the operator at deploy time.
        return Setting(SettingSource.OPERATOR, "", f"operator:{raw}")
    # A partly supplied value (for example a DSN password) keeps its visible
    # structure, so options such as sslmode are still checked.
    return Setting.literal(text)


def _compose_environment(
    service: Mapping[str, Any], environ: Mapping[str, str]
) -> tuple[list[tuple[str, Setting]], int]:
    raw = service.get("environment") or {}
    entries: list[tuple[str, Setting]] = []
    if isinstance(raw, Mapping):
        for name, value in raw.items():
            if value is None:
                if str(name) in environ:
                    entries.append((str(name), Setting.literal(environ[str(name)])))
                continue
            entries.append((str(name), _compose_setting(value, environ)))
    else:
        for item in raw:
            name, separator, value = str(item).partition("=")
            if not separator:
                if name in environ:
                    entries.append((name, Setting.literal(environ[name])))
                continue
            entries.append((name, _compose_setting(value, environ)))
    unverified = 1 if service.get("env_file") else 0
    return entries, unverified


def _text(value: Any, environ: Mapping[str, str]) -> str:
    if value is None:
        return ""
    return interpolate_compose(_scalar_text(value), environ)[0].strip()


def _compose_hardening(service: Mapping[str, Any], environ: Mapping[str, str]) -> list:
    reasons: list[str] = []
    user = _text(service.get("user"), environ)
    if not user or user.split(":", 1)[0] in {"0", "root"}:
        reasons.append("runs_as_root")
    options = [
        _text(item, environ).replace("=", ":")
        for item in service.get("security_opt") or []
    ]
    if not any(
        option in {"no-new-privileges", "no-new-privileges:true"} for option in options
    ) or service.get("privileged"):
        reasons.append("privilege_escalation_allowed")
    if service.get("read_only") is not True:
        reasons.append("root_filesystem_writable")
    if "ALL" not in [
        _text(item, environ).upper() for item in service.get("cap_drop") or []
    ]:
        reasons.append("capabilities_not_dropped")
    if service.get("cap_add"):
        reasons.append("capabilities_added")
    if service.get("privileged"):
        reasons.append("privileged_container")
    if "seccomp:unconfined" in options:
        reasons.append("seccomp_not_confined")
    if any(
        _text(service.get(key), environ) == "host"
        for key in ("network_mode", "pid", "ipc", "userns_mode")
    ):
        reasons.append("host_namespace_shared")
    for volume in service.get("volumes") or []:
        if isinstance(volume, Mapping):
            bind = volume.get("type") == "bind" and volume.get("read_only") is not True
        else:
            parts = _text(volume, environ).split(":")
            source = parts[0] if len(parts) > 1 else ""
            modes = parts[2].split(",") if len(parts) > 2 else []
            bind = (
                bool(source)
                and ("/" in source or source.startswith((".", "~")))
                and "ro" not in modes
            )
        if bind:
            reasons.append("host_path_mounted")
            break
    return reasons


def _compose_published(service: Mapping[str, Any], environ: Mapping[str, str]) -> bool:
    for port in service.get("ports") or []:
        if isinstance(port, Mapping):
            host = _text(port.get("host_ip"), environ)
            if not port.get("published") and not host:
                continue
        else:
            parts = _text(port, environ).rsplit(":", 2)
            host = parts[0] if len(parts) == 3 else ""
        if not host or not is_loopback_host(host):
            return True
    return False


def _compose_readiness(service: Mapping[str, Any]) -> tuple[bool, bool]:
    check = service.get("healthcheck") or {}
    if not check or check.get("disable"):
        return False, False
    test = check.get("test") or []
    tokens = [test] if isinstance(test, str) else [str(item) for item in test]
    if tokens and tokens[0] == "NONE":
        return False, False
    return True, any("/readyz" in token for token in tokens)


def _compose_limit(
    service: Mapping[str, Any], key: str, legacy: str, environ: Mapping[str, str]
) -> bool:
    limits = (
        ((service.get("deploy") or {}).get("resources") or {}).get("limits")
    ) or {}
    value = _text(service.get(legacy), environ) or _text(limits.get(key), environ)
    return bool(value) and value not in {"0", "0.0"}


def _is_openmed_service(service: Mapping[str, Any], environ: Mapping[str, str]) -> bool:
    if service.get("build"):
        return True
    image = _text(service.get("image"), environ)
    name = image.rsplit("/", 1)[-1].split("@", 1)[0].split(":", 1)[0]
    return name.startswith("openmed")


def workloads_from_compose(
    document: Mapping[str, Any], *, environ: Mapping[str, str] | None = None
) -> tuple[tuple[DeployWorkload, ...], int]:
    """Normalize every OpenMed service of a Compose document.

    Services that run a third-party image (for example PostgreSQL) are not
    OpenMed workloads; they are counted and skipped.

    Args:
        document: Parsed Compose file.
        environ: Operator variables used for interpolation. The process
            environment is never read.

    Returns:
        The workloads and the number of skipped third-party services.
    """
    env = dict(environ or {})
    workloads: list[DeployWorkload] = []
    skipped = 0
    for name, service in (document.get("services") or {}).items():
        if not isinstance(service, Mapping) or not _is_openmed_service(service, env):
            skipped += 1
            continue
        role, launcher_args = classify_command(
            _tokens(service.get("entrypoint")), _tokens(service.get("command"))
        )
        entries, unverified = _compose_environment(service, env)
        environment, contradictions = _merge(entries)
        environment, file_conflicts = _resolve_file_settings(
            environment, ("/run/secrets",)
        )
        probe, readyz = _compose_readiness(service)
        workloads.append(
            DeployWorkload(
                workload_id=str(name),
                platform=Platform.COMPOSE,
                role=role,
                environment=environment,
                launcher_args=launcher_args,
                hardening=tuple(_compose_hardening(service, env)),
                cpu_limit=_compose_limit(service, "cpus", "cpus", env),
                memory_limit=_compose_limit(service, "memory", "mem_limit", env),
                readiness_probe=probe,
                readiness_readyz=readyz,
                exposed=_compose_published(service, env),
                contradictions=contradictions + file_conflicts,
                unverified_sources=unverified,
            )
        )
    return tuple(workloads), skipped


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _deploy(
    workload: str, check: str, reason: str, *, setting: str | None = None
) -> PostureFinding:
    return PostureFinding(
        workload=workload,
        check=check,
        reason=reason,
        level=_deploy_level(check),
        source=FindingSource.DEPLOY_CONTRACT,
        setting=setting,
    )


def _resolved_profile(workload: DeployWorkload) -> str:
    """Return the resolved profile name, or a reason code when unresolvable."""
    for name in (SECURITY_PROFILE_ENV_VAR, CONFIG_PROFILE_ENV_VAR):
        setting = workload.environment.get(name)
        if setting is not None and not setting.known:
            return "security_profile_unverified"
    environ = {
        name: workload.literal(name) or ""
        for name in (SECURITY_PROFILE_ENV_VAR, CONFIG_PROFILE_ENV_VAR)
    }
    try:
        return resolve_security_profile(environ).name.value
    except ValueError:
        return "security_profile_value_invalid"


def _profile_findings(workload: DeployWorkload) -> list[PostureFinding]:
    resolved = _resolved_profile(workload)
    if resolved == SecurityProfileName.PROD.value:
        return []
    if resolved == SecurityProfileName.DEV.value:
        resolved = "development_profile_selected"
    return [_deploy(workload.workload_id, "deploy.profile", resolved)]


def _egress_findings(
    workload: DeployWorkload, *, require_offline: bool
) -> list[PostureFinding]:
    findings: list[PostureFinding] = []
    wid = workload.workload_id
    telemetry = [workload.literal(name) for name in TELEMETRY_SETTINGS]
    if not any(_truthy(value) for value in telemetry):
        findings.append(_deploy(wid, "deploy.egress", "telemetry_not_disabled"))
    elif any(_falsy(value) for value in telemetry):
        findings.append(_deploy(wid, "deploy.egress", "telemetry_contradictory"))
    offline = [
        workload.literal(name) for name in (*OFFLINE_SETTINGS, "HF_DATASETS_OFFLINE")
    ]
    if any(_truthy(value) for value in offline) and any(
        _falsy(value) for value in offline
    ):
        findings.append(_deploy(wid, "deploy.egress", "offline_mode_contradictory"))
    elif require_offline and not all(
        _truthy(workload.literal(name)) for name in OFFLINE_SETTINGS
    ):
        findings.append(_deploy(wid, "deploy.egress", "offline_mode_required"))
    return findings


def _container_findings(
    workload: DeployWorkload, *, require_offline: bool
) -> list[PostureFinding]:
    wid = workload.workload_id
    findings: list[PostureFinding] = []
    if workload.role is WorkloadRole.UNCLASSIFIED:
        findings.append(_deploy(wid, "deploy.workload", "workload_unclassified"))
    if workload.unverified_sources:
        findings.append(
            _deploy(wid, "deploy.environment", "environment_source_unverified")
        )
    for name in workload.contradictions:
        findings.append(
            _deploy(
                wid,
                "deploy.environment",
                "setting_contradictory",
                setting=_reportable(name),
            )
        )
    findings.extend(_profile_findings(workload))
    for name in INSECURE_OVERRIDE_SETTINGS:
        if _truthy(workload.literal(name)):
            findings.append(
                _deploy(
                    wid, "deploy.overrides", "insecure_override_enabled", setting=name
                )
            )
    findings.extend(
        _deploy(wid, "deploy.security_context", r) for r in workload.hardening
    )
    if not workload.cpu_limit:
        findings.append(_deploy(wid, "deploy.resources", "cpu_limit_missing"))
    if not workload.memory_limit:
        findings.append(_deploy(wid, "deploy.resources", "memory_limit_missing"))
    if workload.role in {WorkloadRole.API, WorkloadRole.WORKER}:
        if not workload.readiness_probe:
            findings.append(_deploy(wid, "deploy.readiness", "readiness_probe_missing"))
        elif not workload.readiness_readyz:
            findings.append(_deploy(wid, "deploy.readiness", "readiness_not_wired"))
    findings.extend(_egress_findings(workload, require_offline=require_offline))
    return findings


def evaluate_service_workload(
    workload: DeployWorkload,
) -> tuple[SecurityProfileReport | None, tuple[DeferredCheck, ...], list]:
    """Run the production security-profile checks on a rendered service.

    The rendered environment (with in-memory stand-ins for secrets) is passed
    to :func:`run_service_preflight` under the ``prod`` profile, with the
    listener the launch command binds.

    Args:
        workload: A workload with the ``api`` role.

    Returns:
        The runtime report (``None`` when the configuration cannot be parsed),
        the checks deferred to runtime, and the resulting findings.
    """
    host, tls = launcher_listener(workload.launcher_args)
    try:
        report = run_service_preflight(
            profile=SecurityProfileName.PROD.value,
            listen_host=host,
            listen_tls=tls,
            environ=evaluation_environment(workload),
        )
    except ValueError:
        finding = _deploy(
            workload.workload_id, "deploy.service_config", "service_config_invalid"
        )
        return None, (), [finding]
    deferred: list[DeferredCheck] = []
    findings: list[PostureFinding] = []
    for result in report.results:
        inputs = SECRET_DEFERRED_CHECKS.get(result.check_id, ())
        hidden = any(
            name in workload.environment and not workload.environment[name].known
            for name in inputs
        )
        if hidden:
            deferred.append(DeferredCheck(workload.workload_id, result.check_id))
            continue
        if result.is_finding:
            findings.append(
                PostureFinding(
                    workload=workload.workload_id,
                    check=result.check_id,
                    reason=result.reason,
                    level=result.level,
                    source=FindingSource.SECURITY_PROFILE,
                    tracking_issue=result.tracking_issue,
                )
            )
    return report, tuple(deferred), findings


def _listener_is_protected(
    workload: DeployWorkload, report: SecurityProfileReport | None
) -> bool:
    if report is not None:
        for result in report.results:
            if result.check_id == "listener.transport":
                return result.reason in _TLS_LISTENER_REASONS
    return launcher_listener(workload.launcher_args)[1]


def _inheritance_findings(
    workload: DeployWorkload, api: DeployWorkload
) -> list[PostureFinding]:
    findings: list[PostureFinding] = []
    wid = workload.workload_id
    if _resolved_profile(workload) != _resolved_profile(api):
        findings.append(_deploy(wid, "deploy.inheritance", "security_profile_mismatch"))
    names: tuple[str, ...] = ()
    if workload.role is WorkloadRole.WORKER:
        names = (*JOURNEY_STORE_SETTINGS, JOURNEY_ARTIFACT_ROOT_SETTING)
    elif workload.role is WorkloadRole.JOB:
        names = JOURNEY_STORE_SETTINGS
    for name in names:
        mine = workload.environment.get(name)
        theirs = api.environment.get(name)
        if mine is None and theirs is None:
            continue
        if mine is None or theirs is None:
            findings.append(
                _deploy(
                    wid, "deploy.inheritance", "inherited_setting_missing", setting=name
                )
            )
        elif mine.identity != theirs.identity:
            findings.append(
                _deploy(
                    wid,
                    "deploy.inheritance",
                    "inherited_setting_mismatch",
                    setting=name,
                )
            )
    return findings


def evaluate_workloads(
    workloads: Sequence[DeployWorkload],
    *,
    require_offline: bool = False,
    skipped_workloads: int = 0,
) -> PostureEvaluation:
    """Evaluate one rendered deployment against the posture contract.

    Args:
        workloads: Every OpenMed workload of the deployment.
        require_offline: Whether the deployment promises offline operation, so
            every workload must set the offline flags.
        skipped_workloads: Third-party workloads that were not evaluated.

    Returns:
        Every finding, gated or advisory, plus the deferred runtime checks.
    """
    findings: list[PostureFinding] = []
    deferred: list[DeferredCheck] = []
    apis = [item for item in workloads if item.role is WorkloadRole.API]
    if not apis:
        findings.append(
            _deploy("deployment", "deploy.workload", "service_workload_missing")
        )
    for workload in workloads:
        findings.extend(_container_findings(workload, require_offline=require_offline))
        report = None
        if workload.role is WorkloadRole.API:
            report, waived, runtime = evaluate_service_workload(workload)
            deferred.extend(waived)
            findings.extend(runtime)
        if workload.exposed and not _listener_is_protected(workload, report):
            findings.append(
                _deploy(workload.workload_id, "deploy.listener", "exposed_without_tls")
            )
        if apis and workload.role is not WorkloadRole.API:
            findings.extend(_inheritance_findings(workload, apis[0]))
    return PostureEvaluation(
        workloads=tuple(workloads),
        findings=tuple(findings),
        deferred=tuple(deferred),
        skipped_workloads=skipped_workloads,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_helm_chart(
    chart: Path,
    *,
    values: Sequence[Path] = (),
    release: str = "openmed-service",
    namespace: str = "openmed",
    helm: str | None = None,
) -> list[dict[str, Any]]:
    """Render a chart offline with ``helm template``.

    Args:
        chart: Chart directory.
        values: Values files, applied in order.
        release: Release name.
        namespace: Release namespace.
        helm: Path to the ``helm`` binary. Defaults to ``PATH`` lookup.

    Returns:
        The rendered documents.

    Raises:
        PostureLaneUnavailable: If ``helm`` is not installed.
        PostureLaneError: If rendering fails. Helm output is not attached.
    """
    binary = helm or shutil.which("helm")
    if binary is None:
        raise PostureLaneUnavailable("helm_unavailable")
    command = [binary, "template", release, str(chart), "--namespace", namespace]
    for path in values:
        command.extend(["--values", str(path)])
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=120
        )
    except (OSError, subprocess.SubprocessError):
        raise PostureLaneUnavailable("helm_unavailable") from None
    if completed.returncode != 0:
        raise PostureLaneError("helm_render_failed")
    return [
        document
        for document in yaml.safe_load_all(completed.stdout)
        if isinstance(document, dict)
    ]


def evaluate_helm_render(
    manifests: Sequence[Mapping[str, Any]], *, require_offline: bool = False
) -> PostureEvaluation:
    """Evaluate rendered Kubernetes manifests against the contract.

    Args:
        manifests: Parsed ``helm template`` documents.
        require_offline: Whether the deployment promises offline operation.

    Returns:
        The evaluation.
    """
    return evaluate_workloads(
        workloads_from_manifests(manifests), require_offline=require_offline
    )


def evaluate_compose(
    document: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
    require_offline: bool = False,
) -> PostureEvaluation:
    """Evaluate a parsed Compose file against the contract.

    Args:
        document: Parsed Compose file.
        environ: Operator variables used for interpolation.
        require_offline: Whether the deployment promises offline operation.

    Returns:
        The evaluation.
    """
    workloads, skipped = workloads_from_compose(document, environ=environ)
    return evaluate_workloads(
        workloads, require_offline=require_offline, skipped_workloads=skipped
    )


# ---------------------------------------------------------------------------
# Lane
# ---------------------------------------------------------------------------

_ID: Final = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-")


@dataclass(frozen=True, slots=True)
class Expectation:
    """A finding a lane target is expected to produce."""

    workload: str
    check: str
    reason: str
    disposition: Disposition
    setting: str | None = None
    owner_pr: int | None = None
    issue: int | None = None

    @property
    def key(self) -> tuple[str, str, str, str | None]:
        """Return the (workload, check, reason, setting) key."""
        return (self.workload, self.check, self.reason, self.setting)

    def to_dict(self) -> dict[str, Any]:
        """Return the value-free JSON form."""
        payload: dict[str, Any] = {
            "workload": self.workload,
            "check": self.check,
            "reason": self.reason,
            "why": self.disposition.value,
        }
        if self.setting is not None:
            payload["setting"] = self.setting
        if self.owner_pr is not None:
            payload["owner_pr"] = self.owner_pr
        if self.issue is not None:
            payload["issue"] = self.issue
        return payload


@dataclass(frozen=True, slots=True)
class LaneTarget:
    """One deployment rendering the lane evaluates."""

    target_id: str
    platform: Platform
    purpose: TargetPurpose
    source: Path = field(repr=False)
    values: tuple[Path, ...] = field(default=(), repr=False)
    environ: Mapping[str, str] = field(default_factory=dict, repr=False)
    require_offline: bool = False
    expectations: tuple[Expectation, ...] = ()


@dataclass(frozen=True, slots=True)
class PostureLane:
    """A parsed lane definition."""

    targets: tuple[LaneTarget, ...]
    release: str = "openmed-service"
    namespace: str = "openmed"


def _positive_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PostureLaneError("lane_invalid")
    return value


def _expectation(raw: Any, purpose: TargetPurpose) -> Expectation:
    if not isinstance(raw, Mapping):
        raise PostureLaneError("lane_invalid")
    try:
        disposition = Disposition(raw.get("why"))
        expectation = Expectation(
            workload=str(raw["workload"]),
            check=str(raw["check"]),
            reason=str(raw["reason"]),
            disposition=disposition,
            setting=str(raw["setting"]) if raw.get("setting") is not None else None,
            owner_pr=_positive_int(raw.get("owner_pr")),
            issue=_positive_int(raw.get("issue")),
        )
    except (KeyError, ValueError):
        raise PostureLaneError("lane_invalid") from None
    if disposition is Disposition.KNOWN and expectation.owner_pr is None:
        raise PostureLaneError("lane_known_finding_without_owner")
    if disposition is Disposition.FAIL_CLOSED and expectation.check.startswith(
        "deploy."
    ):
        raise PostureLaneError("lane_fail_closed_not_runtime")
    if (
        disposition is Disposition.NEGATIVE_CONTROL
        and purpose is not TargetPurpose.NEGATIVE_CONTROL
    ):
        raise PostureLaneError("lane_negative_control_misplaced")
    return expectation


def load_lane(path: Path) -> PostureLane:
    """Load and validate a lane definition.

    Paths inside the lane are relative to the lane file.

    Args:
        path: Lane YAML file.

    Returns:
        The parsed lane.

    Raises:
        PostureLaneError: If the definition is invalid.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        raise PostureLaneError("lane_unreadable") from None
    if not isinstance(raw, Mapping) or raw.get("schema_version") != (
        POSTURE_LANE_SCHEMA_VERSION
    ):
        raise PostureLaneError("lane_invalid")
    base = path.parent
    helm = raw.get("helm") or {}
    targets: list[LaneTarget] = []
    seen: set[str] = set()
    for item in raw.get("targets") or []:
        if not isinstance(item, Mapping):
            raise PostureLaneError("lane_invalid")
        try:
            target_id = str(item["id"])
            platform = Platform(item["platform"])
            purpose = TargetPurpose(item["purpose"])
        except (KeyError, ValueError):
            raise PostureLaneError("lane_invalid") from None
        if target_id in seen or not target_id or set(target_id) - _ID:
            raise PostureLaneError("lane_invalid")
        seen.add(target_id)
        expectations = tuple(
            _expectation(entry, purpose) for entry in item.get("expect") or []
        )
        if purpose is TargetPurpose.NEGATIVE_CONTROL and not any(
            entry.disposition is Disposition.NEGATIVE_CONTROL for entry in expectations
        ):
            raise PostureLaneError("lane_negative_control_without_expectation")
        if platform is Platform.HELM:
            source = base / str(helm.get("chart") or "")
        else:
            source = base / str(item.get("file") or "")
        environ = {
            str(k): _scalar_text(v) for k, v in (item.get("environment") or {}).items()
        }
        targets.append(
            LaneTarget(
                target_id=target_id,
                platform=platform,
                purpose=purpose,
                source=source,
                values=tuple(base / str(entry) for entry in item.get("values") or []),
                environ=environ,
                require_offline=bool(item.get("require_offline", False)),
                expectations=expectations,
            )
        )
    if not targets:
        raise PostureLaneError("lane_invalid")
    return PostureLane(
        targets=tuple(targets),
        release=str(helm.get("release") or "openmed-service"),
        namespace=str(helm.get("namespace") or "openmed"),
    )


@dataclass(frozen=True, slots=True)
class TargetResult:
    """The evaluation of one lane target compared with its expectations."""

    target: LaneTarget
    evaluation: PostureEvaluation
    unexpected: tuple[PostureFinding, ...]
    stale: tuple[Expectation, ...]

    @property
    def passed(self) -> bool:
        """Return whether the target produced exactly its expected findings."""
        return not self.unexpected and not self.stale

    def to_dict(self) -> dict[str, Any]:
        """Return the value-free JSON form."""
        expected = {entry.key: entry for entry in self.target.expectations}
        findings = []
        for finding in self.evaluation.findings:
            payload = finding.to_dict()
            entry = expected.get(finding.key)
            if finding.gated:
                payload["expected"] = entry.to_dict() if entry is not None else None
            findings.append(payload)
        return {
            "id": self.target.target_id,
            "platform": self.target.platform.value,
            "purpose": self.target.purpose.value,
            "status": "pass" if self.passed else "fail",
            "workloads": [
                {"id": item.workload_id, "role": item.role.value}
                for item in self.evaluation.workloads
            ],
            "skipped_workloads": self.evaluation.skipped_workloads,
            "findings": findings,
            "deferred": [item.to_dict() for item in self.evaluation.deferred],
            "unexpected": [item.to_dict() for item in self.unexpected],
            "stale": [item.to_dict() for item in self.stale],
        }


@dataclass(frozen=True, slots=True)
class LaneReport:
    """Value-free result of one lane run."""

    results: tuple[TargetResult, ...]

    @property
    def passed(self) -> bool:
        """Return whether every target matched its expectations."""
        return all(result.passed for result in self.results)

    @property
    def exit_code(self) -> int:
        """Return the stable CLI exit code."""
        return EXIT_LANE_PASSED if self.passed else EXIT_LANE_FAILED

    def _body(self) -> dict[str, Any]:
        gated = [f for r in self.results for f in r.evaluation.findings if f.gated]
        return {
            "schema_version": POSTURE_REPORT_SCHEMA_VERSION,
            "contract": posture_contract(),
            "status": "pass" if self.passed else "fail",
            "summary": {
                "targets": len(self.results),
                "workloads": sum(len(r.evaluation.workloads) for r in self.results),
                "gated_findings": len(gated),
                "advisory_findings": sum(
                    1
                    for r in self.results
                    for f in r.evaluation.findings
                    if not f.gated
                ),
                "deferred_checks": sum(
                    len(r.evaluation.deferred) for r in self.results
                ),
                "unexpected": sum(len(r.unexpected) for r in self.results),
                "stale": sum(len(r.stale) for r in self.results),
            },
            "targets": [result.to_dict() for result in self.results],
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned JSON report with its digest."""
        body = self._body()
        body["digest"] = _digest(body)
        return body

    def to_json(self) -> str:
        """Return the report as canonical JSON text."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    def to_text(self) -> str:
        """Return the human report."""
        body = self.to_dict()
        contract = body["contract"]
        lines = [
            f"OpenMed deploy posture lane: {body['status']} "
            f"(contract {contract['id']}, {contract['security_profile']})"
        ]
        for result in self.results:
            target = result.target
            gated = [item for item in result.evaluation.findings if item.gated]
            lines.append(
                f"{target.target_id} [{target.platform.value}, {target.purpose.value}]: "
                f"{len(result.evaluation.workloads)} workloads, {len(gated)} gated "
                f"findings, {len(result.unexpected)} unexpected, "
                f"{len(result.stale)} stale"
            )
            expected = {entry.key: entry for entry in target.expectations}
            for finding in gated:
                entry = expected.get(finding.key)
                if entry is None:
                    label = "NEW"
                elif entry.owner_pr is not None:
                    label = f"{entry.disposition.value} (#{entry.owner_pr})"
                else:
                    label = entry.disposition.value
                lines.append(
                    f"  {label:<22} {finding.workload}  {finding.check}  "
                    f"{finding.reason}{_setting_suffix(finding.setting)}"
                )
            for entry in result.stale:
                lines.append(
                    f"  {'STALE':<22} {entry.workload}  {entry.check}  "
                    f"{entry.reason}{_setting_suffix(entry.setting)}"
                )
        lines.append(f"Lane digest: {body['digest']}")
        return "\n".join(lines)


def _setting_suffix(setting: str | None) -> str:
    return f" ({setting})" if setting else ""


def evaluate_target(
    target: LaneTarget,
    *,
    release: str = "openmed-service",
    namespace: str = "openmed",
    helm: str | None = None,
) -> TargetResult:
    """Render one lane target, evaluate it and compare with expectations.

    Args:
        target: The lane target.
        release: Helm release name.
        namespace: Helm release namespace.
        helm: Optional path to the ``helm`` binary.

    Returns:
        The target result.
    """
    if target.platform is Platform.HELM:
        manifests = render_helm_chart(
            target.source,
            values=target.values,
            release=release,
            namespace=namespace,
            helm=helm,
        )
        evaluation = evaluate_helm_render(
            manifests, require_offline=target.require_offline
        )
    else:
        try:
            document = yaml.safe_load(target.source.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            raise PostureLaneError("compose_unreadable") from None
        if not isinstance(document, Mapping):
            raise PostureLaneError("compose_unreadable")
        evaluation = evaluate_compose(
            document, environ=target.environ, require_offline=target.require_offline
        )
    expected = {entry.key for entry in target.expectations}
    actual = {finding.key for finding in evaluation.gated}
    unexpected = tuple(f for f in evaluation.gated if f.key not in expected)
    stale = tuple(e for e in target.expectations if e.key not in actual)
    return TargetResult(target, evaluation, unexpected, stale)


def run_lane(
    lane: PostureLane,
    *,
    helm: str | None = None,
    platforms: Iterable[Platform] | None = None,
) -> LaneReport:
    """Evaluate every lane target.

    Args:
        lane: The lane definition.
        helm: Optional path to the ``helm`` binary.
        platforms: Optional subset of platforms to evaluate.

    Returns:
        The lane report.
    """
    selected = set(platforms) if platforms is not None else set(Platform)
    return LaneReport(
        tuple(
            evaluate_target(
                target, release=lane.release, namespace=lane.namespace, helm=helm
            )
            for target in lane.targets
            if target.platform in selected
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deploy posture lane from the command line.

    Args:
        argv: Command-line arguments.

    Returns:
        ``0`` when every target matches its expected findings, ``1`` on any
        new or stale finding, ``2`` when the lane or a tool is unavailable.
    """
    parser = argparse.ArgumentParser(
        prog="python -m openmed.deploy.posture",
        description="Check rendered Helm and Compose deployments against the "
        "production security posture contract.",
    )
    parser.add_argument("lane", nargs="?", default="deploy/posture/lane.yaml")
    parser.add_argument("--json", action="store_true", help="Print the JSON report.")
    parser.add_argument("--helm", default=None, help="Path to the helm binary.")
    parser.add_argument(
        "--platform",
        action="append",
        choices=[item.value for item in Platform],
        help="Only evaluate targets of this platform. Repeatable.",
    )
    options = parser.parse_args(argv)
    platforms = (
        [Platform(item) for item in options.platform] if options.platform else None
    )
    try:
        lane = load_lane(Path(options.lane))
        report = run_lane(lane, helm=options.helm, platforms=platforms)
    except ImportError:
        print("deploy_posture_unavailable: service_extra_missing", file=sys.stderr)
        return EXIT_LANE_UNAVAILABLE
    except PostureLaneError as error:
        print(f"deploy_posture_unavailable: {error}", file=sys.stderr)
        return EXIT_LANE_UNAVAILABLE
    print(report.to_json() if options.json else report.to_text())
    return report.exit_code


__all__ = [
    "DEFERRED_REASON",
    "DEPLOY_CHECKS",
    "EXIT_LANE_FAILED",
    "EXIT_LANE_PASSED",
    "EXIT_LANE_UNAVAILABLE",
    "IMAGE_DEFAULT_LISTENER_ARGS",
    "INSECURE_OVERRIDE_SETTINGS",
    "JOURNEY_STORE_SETTINGS",
    "OFFLINE_SETTINGS",
    "POSTURE_CONTRACT_ID",
    "POSTURE_LANE_SCHEMA_VERSION",
    "POSTURE_REPORT_SCHEMA_VERSION",
    "SECRET_DEFERRED_CHECKS",
    "TELEMETRY_SETTINGS",
    "DeferredCheck",
    "DeployWorkload",
    "Disposition",
    "Expectation",
    "FindingSource",
    "LaneReport",
    "LaneTarget",
    "Platform",
    "PostureEvaluation",
    "PostureFinding",
    "PostureLane",
    "PostureLaneError",
    "PostureLaneUnavailable",
    "Setting",
    "SettingSource",
    "TargetPurpose",
    "TargetResult",
    "WorkloadRole",
    "classify_command",
    "evaluate_compose",
    "evaluate_helm_render",
    "evaluate_service_workload",
    "evaluate_target",
    "evaluate_workloads",
    "evaluation_environment",
    "interpolate_compose",
    "launcher_listener",
    "load_lane",
    "main",
    "posture_contract",
    "render_helm_chart",
    "run_lane",
    "workloads_from_compose",
    "workloads_from_manifests",
]


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
