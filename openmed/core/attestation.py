"""PHI-free attestations: data residency and OS-enforced no-egress evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .audit import AuditReport, stable_hash
from .offline import (
    HF_OFFLINE_ENV_VARS,
    OFFLINE_ENV_VAR,
)
from .policy import load_policy

ATTESTATION_SCHEMA_VERSION = 1
DEFAULT_ATTESTATION_TEMPLATE = "africa-data-residency.json"
_TEMPLATE_PACKAGE = "openmed.core.attestation_templates"
_TEMPLATE_FIELDS = (
    "id",
    "jurisdiction",
    "country_code",
    "legal_reference",
    "source_url",
    "policy_profile",
    "attestation_title",
    "residency_statement",
    "host_local_statement",
    "offline_enabled_statement",
    "offline_disabled_statement",
    "guide_field",
)


class AttestationTemplateError(ValueError):
    """Raised when an attestation wording template is invalid."""


@dataclass(frozen=True)
class AttestationReport:
    """Machine-readable attestation with a human-readable Markdown renderer."""

    _data: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_data", copy.deepcopy(dict(self._data)))

    def to_dict(self) -> dict[str, Any]:
        """Return an isolated JSON-compatible attestation payload."""

        return copy.deepcopy(dict(self._data))

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize the attestation as deterministic JSON."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            separators=(",", ":") if indent is None else None,
            sort_keys=True,
        )

    def to_markdown(self) -> str:
        """Render the attestation as PHI-free Markdown."""

        payload = self.to_dict()
        jurisdiction = payload["jurisdiction"]
        policy = payload["policy"]
        execution = payload["execution"]
        run = payload["run"]
        lines = [
            f"# {jurisdiction['attestation_title']}",
            "",
            f"> {payload['disclaimer']}",
            "",
            "## Jurisdiction decision support",
            "",
            f"- Jurisdiction: {jurisdiction['name']} ({jurisdiction['country_code']})",
            f"- Legal reference: {jurisdiction['legal_reference']}",
            f"- Attestation profile: `{payload['profile_id']}`",
            f"- Template field: `{jurisdiction['guide_field']}`",
            "",
            jurisdiction["residency_statement"],
            "",
            "## Recorded execution posture",
            "",
            execution["host_local_statement"],
            "",
            execution["offline_statement"],
            "",
            f"- Generated at: `{payload['generated_at']}`",
            f"- Host-local execution: `{str(execution['host_local']).lower()}`",
            "- Offline assertion: "
            f"`{str(execution['offline_assertion']['asserted']).lower()}`",
            f"- Offline assertion source: `{execution['offline_assertion']['source']}`",
            "",
            "## Policy",
            "",
            f"- Name: `{policy['name']}`",
            f"- Posture: `{policy['posture']}`",
            f"- Policy schema version: `{policy['schema_version']}`",
            f"- Default action: `{policy['default_action']}`",
            "",
            "## Model artifacts",
            "",
            "| Name | Path | Content checksum | Kind | Files |",
            "|---|---|---|---|---:|",
        ]
        for model in payload["models"]:
            lines.append(
                "| "
                + " | ".join(
                    _markdown_cell(model[field])
                    for field in ("name", "path", "checksum", "kind", "file_count")
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "## PHI-free run evidence",
                "",
                f"- Document length: `{run['document_length']}`",
                f"- Redacted span count: `{run['span_count']}`",
                f"- Input hash: `{run['input_hash']}`",
                f"- De-identified output hash: `{run['deidentified_text_hash']}`",
                f"- Audit repro hash: `{run['audit_repro_hash']}`",
                f"- Model manifest hash: `{run['manifest_hash']}`",
                f"- Attestation repro hash: `{payload['repro_hash']}`",
                f"- Attestation integrity hash: `{payload['integrity_hash']}`",
                "",
            ]
        )
        return "\n".join(lines)

    def integrity_hash_matches(self) -> bool:
        """Return whether the full payload matches its integrity hash."""

        payload = self.to_dict()
        expected = payload.pop("integrity_hash", "")
        return expected == stable_hash(payload)

    def repro_hash_matches(self) -> bool:
        """Return whether stable run evidence matches its repro hash."""

        payload = self.to_dict()
        expected = payload.pop("repro_hash", "")
        payload.pop("integrity_hash", None)
        payload.pop("generated_at", None)
        return expected == stable_hash(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AttestationReport":
        """Restore an attestation payload for offline integrity checks."""

        return cls(data)

    @classmethod
    def from_json(cls, data: str | bytes) -> "AttestationReport":
        """Restore an attestation from a JSON object."""

        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON for AttestationReport: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("AttestationReport JSON must contain an object")
        return cls(payload)


def load_attestation_template(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate a jurisdiction wording template."""

    if path is None:
        resource = resources.files(_TEMPLATE_PACKAGE).joinpath(
            DEFAULT_ATTESTATION_TEMPLATE
        )
        with resource.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        source = str(resource)
    else:
        source_path = Path(path)
        try:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise AttestationTemplateError(
                f"Could not read attestation template {source_path}: {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise AttestationTemplateError(
                f"Invalid JSON in attestation template {source_path}: {exc}"
            ) from exc
        source = str(source_path)
    return _validate_template(payload, source=source)


def list_attestation_profiles(
    template_path: str | Path | None = None,
) -> tuple[str, ...]:
    """Return the profile identifiers in a validated template."""

    template = load_attestation_template(template_path)
    return tuple(str(profile["id"]) for profile in template["profiles"])


def generate_attestation(
    audit_report: AuditReport,
    profile: str,
    *,
    model_artifacts: Mapping[str, str | Path],
    generated_at: datetime | str | None = None,
    template_path: str | Path | None = None,
) -> AttestationReport:
    """Generate JSON and Markdown residency evidence for an audit report.

    Raw source text, redacted text, span text, surrogates, and mappings are not
    copied. The output contains only hashes, counts, policy metadata, offline
    posture metadata, jurisdiction wording, and model artifact provenance.
    """

    if not isinstance(audit_report, AuditReport):
        raise TypeError("audit_report must be an AuditReport")
    if not audit_report.repro_hash_matches():
        raise ValueError(
            "Audit report repro hash does not match its canonical PHI-free payload"
        )
    template = load_attestation_template(template_path)
    jurisdiction_profile = _profile(template, profile)
    try:
        policy = load_policy(audit_report.policy)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Attestation policy profile {audit_report.policy!r} could not be loaded: "
            f"{exc}"
        ) from exc
    expected_policy = jurisdiction_profile["policy_profile"]
    if policy.name != expected_policy:
        raise ValueError(
            f"Attestation profile {profile!r} requires policy {expected_policy!r}; "
            f"the audit report records {policy.name!r}"
        )

    offline = _safe_offline_assertion(
        audit_report.resolved_profile.get("offline_assertion")
    )
    offline_statement_key = (
        "offline_enabled_statement"
        if offline["asserted"]
        else "offline_disabled_statement"
    )
    models = _model_artifact_records(model_artifacts)
    payload: dict[str, Any] = {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "template": {
            "id": template["template_id"],
            "version": template["template_version"],
            "hash": stable_hash(template),
        },
        "profile_id": jurisdiction_profile["id"],
        "generated_at": _iso_timestamp(generated_at),
        "disclaimer": template["disclaimer"],
        "jurisdiction": {
            "name": jurisdiction_profile["jurisdiction"],
            "country_code": jurisdiction_profile["country_code"],
            "legal_reference": jurisdiction_profile["legal_reference"],
            "source_url": jurisdiction_profile["source_url"],
            "attestation_title": jurisdiction_profile["attestation_title"],
            "residency_statement": jurisdiction_profile["residency_statement"],
            "guide_field": jurisdiction_profile["guide_field"],
        },
        "policy": {
            "name": policy.name,
            "posture": policy.posture,
            "schema_version": policy.schema_version,
            "default_action": policy.default_action,
        },
        "execution": {
            "host_local": True,
            "host_local_statement": jurisdiction_profile["host_local_statement"],
            "offline_statement": jurisdiction_profile[offline_statement_key],
            "offline_assertion": offline,
        },
        "models": models,
        "run": {
            "document_length": audit_report.document_length,
            "span_count": len(audit_report.spans),
            "input_hash": audit_report.input_hash,
            "deidentified_text_hash": audit_report.deidentified_text_hash,
            "audit_repro_hash": audit_report.repro_hash,
            "manifest_hash": audit_report.manifest_hash,
            "openmed_version": audit_report.openmed_version,
        },
    }
    repro_payload = copy.deepcopy(payload)
    repro_payload.pop("generated_at")
    payload["repro_hash"] = stable_hash(repro_payload)
    payload["integrity_hash"] = stable_hash(payload)
    return AttestationReport(payload)


def _validate_template(payload: Any, *, source: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise AttestationTemplateError(
            f"Attestation template {source} must be an object"
        )
    template = copy.deepcopy(dict(payload))
    if template.get("template_version") != ATTESTATION_SCHEMA_VERSION:
        raise AttestationTemplateError(
            f"Attestation template {source} must use template_version "
            f"{ATTESTATION_SCHEMA_VERSION}"
        )
    for field in ("template_id", "disclaimer"):
        if not isinstance(template.get(field), str) or not template[field].strip():
            raise AttestationTemplateError(
                f"Attestation template {source} field {field!r} must be non-empty"
            )
    disclaimer = template["disclaimer"].lower()
    if "decision-support" not in disclaimer or "not legal advice" not in disclaimer:
        raise AttestationTemplateError(
            f"Attestation template {source} disclaimer must state decision-support "
            "and not legal advice"
        )
    profiles = template.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise AttestationTemplateError(
            f"Attestation template {source} profiles must be a non-empty list"
        )
    identifiers: set[str] = set()
    for index, profile in enumerate(profiles):
        if not isinstance(profile, Mapping):
            raise AttestationTemplateError(
                f"Attestation template {source} profile {index} must be an object"
            )
        for field in _TEMPLATE_FIELDS:
            value = profile.get(field)
            if not isinstance(value, str) or not value.strip():
                raise AttestationTemplateError(
                    f"Attestation template {source} profile {index} field "
                    f"{field!r} must be non-empty"
                )
        identifier = str(profile["id"])
        if identifier in identifiers:
            raise AttestationTemplateError(
                f"Attestation template {source} repeats profile {identifier!r}"
            )
        identifiers.add(identifier)
        try:
            loaded = load_policy(str(profile["policy_profile"]))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise AttestationTemplateError(
                f"Attestation template {source} profile {identifier!r} names an "
                f"invalid policy: {exc}"
            ) from exc
        if loaded.name != profile["policy_profile"]:
            raise AttestationTemplateError(
                f"Attestation template {source} profile {identifier!r} must use "
                f"canonical policy name {loaded.name!r}"
            )
    return template


def _profile(template: Mapping[str, Any], identifier: str) -> dict[str, str]:
    for profile in template["profiles"]:
        if profile["id"] == identifier:
            return dict(profile)
    available = ", ".join(str(item["id"]) for item in template["profiles"])
    raise ValueError(
        f"Unknown attestation profile {identifier!r}; available profiles: {available}"
    )


def _safe_offline_assertion(value: Any) -> dict[str, Any]:
    # Attest only to evidence captured with the audited run. Falling back to
    # the environment at attestation-generation time could incorrectly upgrade
    # a legacy or incomplete audit report to an offline run.
    source = value if isinstance(value, Mapping) else {}
    environment = source.get("environment_flags")
    environment_flags = environment if isinstance(environment, Mapping) else {}
    allowed_flags = (OFFLINE_ENV_VAR, *HF_OFFLINE_ENV_VARS)
    normalized_flags = {
        name: bool(environment_flags.get(name, False)) for name in allowed_flags
    }
    source_name = str(source.get("source", "none"))
    if source_name not in {"none", "config", "environment", "config+environment"}:
        source_name = "none"
    raw_local_only = bool(source.get("local_only", False))
    source_is_consistent = (
        (source_name == "config" and raw_local_only)
        or (
            source_name == "environment"
            and raw_local_only
            and normalized_flags[OFFLINE_ENV_VAR]
        )
        or (
            source_name == "config+environment"
            and raw_local_only
            and normalized_flags[OFFLINE_ENV_VAR]
        )
    )
    if not source_is_consistent:
        source_name = "none"
    local_only = bool(raw_local_only and source_name != "none")
    network_guard_requested = bool(
        source.get("network_guard_requested", False) and local_only
    )
    dependency_flags_enabled = all(
        normalized_flags[name] for name in HF_OFFLINE_ENV_VARS
    )
    asserted = bool(
        source.get("asserted", False)
        and local_only
        and network_guard_requested
        and dependency_flags_enabled
    )
    return {
        "asserted": asserted,
        "local_only": local_only,
        "source": source_name,
        "network_guard_requested": network_guard_requested,
        "dependency_flags_enabled": dependency_flags_enabled,
        "environment_flags": normalized_flags,
    }


def _model_artifact_records(
    model_artifacts: Mapping[str, str | Path],
) -> list[dict[str, Any]]:
    if not isinstance(model_artifacts, Mapping) or not model_artifacts:
        raise ValueError("model_artifacts must be a non-empty mapping")
    records: list[dict[str, Any]] = []
    for name, raw_path in sorted(
        model_artifacts.items(), key=lambda item: str(item[0])
    ):
        model_name = str(name).strip()
        if not model_name:
            raise ValueError("model_artifacts names must be non-empty")
        path = Path(raw_path).expanduser()
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise ValueError(
                f"Model artifact {model_name!r} does not exist at {path}"
            ) from exc
        checksum, kind, file_count = _hash_artifact(resolved)
        records.append(
            {
                "name": model_name,
                "path": str(resolved),
                "checksum": checksum,
                "kind": kind,
                "file_count": file_count,
            }
        )
    return records


def _hash_artifact(path: Path) -> tuple[str, str, int]:
    if path.is_file():
        return _hash_file(path), "file", 1
    if not path.is_dir():
        raise ValueError(f"Model artifact path is not a file or directory: {path}")

    records: list[dict[str, str]] = []
    for item in sorted(path.rglob("*"), key=lambda candidate: candidate.as_posix()):
        if item.is_file():
            records.append(
                {
                    "path": item.relative_to(path).as_posix(),
                    "checksum": _hash_file(item),
                }
            )
    return stable_hash(records), "directory", len(records)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _iso_timestamp(value: datetime | str | None) -> str:
    if value is None:
        parsed = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(
                f"generated_at must be an ISO-8601 timestamp: {exc}"
            ) from exc
    else:
        raise TypeError("generated_at must be datetime, str, or None")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("generated_at must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


# ---------------------------------------------------------------------------
# OS-enforced no-egress attestation
# ---------------------------------------------------------------------------

NO_EGRESS_ATTESTATION_SCHEMA_VERSION = 1
NO_EGRESS_ATTESTATION_TYPE = "openmed.no_egress"
NO_EGRESS_SCHEMA_NAME = "no_egress_attestation.schema.json"
NO_EGRESS_VOLATILE_FIELDS = ("generated_at", "integrity_hash")

NO_EGRESS_STATUS_PASSED = "passed"
NO_EGRESS_STATUS_FAILED = "failed"
NO_EGRESS_STATUS_SKIPPED = "skipped"
NO_EGRESS_EXIT_CODES: Mapping[str, int] = MappingProxyType(
    {
        NO_EGRESS_STATUS_PASSED: 0,
        NO_EGRESS_STATUS_FAILED: 1,
        NO_EGRESS_STATUS_SKIPPED: 3,
    }
)
"""Process exit code per status. Only ``passed`` exits zero."""

NO_EGRESS_BOUNDARY_MECHANISMS: Mapping[str, str] = MappingProxyType(
    {
        "netns": "linux_user_network_namespace",
        "container": "container_network_none",
        "none": "none",
    }
)
"""Boundary kind to the OS mechanism that enforces it."""

NO_EGRESS_SKIP_REASONS = frozenset({"unsupported_host", "boundary_unavailable"})
NO_EGRESS_RUNNER_FAILURE_REASONS = frozenset(
    {"runner_failed", "runner_timeout", "runner_result_invalid"}
)

# Case identifier -> (transport, family, executor). The attestation carries only
# these identifiers, so destinations never enter the evidence.
NO_EGRESS_NEGATIVE_CONTROLS: Mapping[str, tuple[str, str, str]] = MappingProxyType(
    {
        "ipv4_tcp_socket": ("tcp", "ipv4", "python_socket"),
        "ipv6_tcp_socket": ("tcp", "ipv6", "python_socket"),
        "ipv4_udp_socket": ("udp", "ipv4", "python_socket"),
        "ipv6_udp_socket": ("udp", "ipv6", "python_socket"),
        "dns_system_resolver": ("dns", "any", "system_resolver"),
        "dns_udp_query": ("dns", "ipv4", "python_socket"),
        "subprocess_ipv4_tcp": ("tcp", "ipv4", "subprocess"),
        "subprocess_ipv6_tcp": ("tcp", "ipv6", "subprocess"),
        "subprocess_dns": ("dns", "any", "subprocess"),
        "native_ipv4_tcp": ("tcp", "ipv4", "native_libc"),
        "native_ipv6_tcp": ("tcp", "ipv6", "native_libc"),
        "native_ipv4_udp": ("udp", "ipv4", "native_libc"),
    }
)
"""External-egress probes. Every one must be blocked for a pass."""

NO_EGRESS_POSITIVE_CONTROLS: Mapping[str, tuple[str, str, str]] = MappingProxyType(
    {
        "loopback_ipv4_tcp_socket": ("tcp", "ipv4", "python_socket"),
        "loopback_ipv4_udp_socket": ("udp", "ipv4", "python_socket"),
        "loopback_dns_udp_query": ("dns", "ipv4", "python_socket"),
        "loopback_system_resolver": ("dns", "any", "system_resolver"),
        "loopback_subprocess_ipv4_tcp": ("tcp", "ipv4", "subprocess"),
        "loopback_native_ipv4_tcp": ("tcp", "ipv4", "native_libc"),
    }
)
"""Loopback probes proving each probe executor works when a route exists."""

NO_EGRESS_WORKFLOW_CASES: Mapping[str, str] = MappingProxyType(
    {
        "service_start": "lifecycle",
        "rest_analyze": "rest",
        "rest_extract": "rest",
        "rest_deidentify": "rest",
        "rest_job": "rest",
        "rest_webhook_egress_denied": "rest",
        "rest_stream_cancel": "rest",
        "graphql_analyze": "graphql",
        "graphql_deidentify": "graphql",
        "grpc_analyze": "grpc",
        "grpc_extract": "grpc",
        "grpc_deidentify": "grpc",
        "grpc_stream_cancel": "grpc",
        "journey_persist": "journey",
        "service_restart": "lifecycle",
        "journey_reopen": "journey",
        "service_shutdown": "lifecycle",
    }
)
"""Workflow case identifier -> service surface. Every one must pass."""

NO_EGRESS_FIXTURE_IDS = frozenset({"service_workflow_fixture", "no_egress_gate_runner"})

_NEGATIVE_STATUSES = ("blocked", "not_blocked", "inconclusive", "not_run")
_POSITIVE_STATUSES = ("reachable", "unreachable", "not_run")
_WORKFLOW_STATUSES = ("passed", "failed", "not_run")
_PROBE_REASONS = frozenset(
    {
        "network_unreachable",
        "address_unavailable",
        "family_unsupported",
        "permission_denied",
        "resolver_unreachable",
        "loopback_reachable",
        "connection_established",
        "datagram_sent",
        "peer_refused",
        "peer_reset",
        "name_resolved",
        "resolver_answered",
        "host_unreachable",
        "timeout",
        "probe_error",
        "executor_unavailable",
        "dependency_failed",
    }
)
_BLOCKED_REASONS = frozenset(
    {
        "network_unreachable",
        "address_unavailable",
        "family_unsupported",
        "permission_denied",
        "resolver_unreachable",
    }
)
_WORKFLOW_REASONS = frozenset(
    {
        "ok",
        "unexpected_status",
        "unexpected_result",
        "transport_error",
        "timeout",
        "server_exit",
        "not_cancelled",
        "persistence_mismatch",
        "dependency_unavailable",
        "dependency_failed",
    }
)
_HOST_KINDS = frozenset({"linux", "darwin", "windows", "other"})
_REQUESTED_BOUNDARIES = frozenset({"auto", "netns", "container"})
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_VERSION_RE = re.compile(r"^[0-9][0-9A-Za-z.+-]{0,63}$")


class NoEgressAttestationError(ValueError):
    """Raised with a controlled code when no-egress evidence is invalid.

    The message is always one fixed code; probe destinations, paths, and
    exception text are never attached.
    """

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class NoEgressAttestation(AttestationReport):
    """Value-free attestation for an OS-enforced no-egress gate run.

    Shares the residency attestation envelope: canonical JSON, a ``repro_hash``
    over every field except the declared volatile fields, and an
    ``integrity_hash`` over the full payload.
    """

    @property
    def status(self) -> str:
        """Return ``passed``, ``failed``, or ``skipped``."""

        return str(self._data["status"])

    @property
    def reason_code(self) -> str:
        """Return the controlled top-level reason code."""

        return str(self._data["reason_code"])

    @property
    def passed(self) -> bool:
        """Return whether the boundary held and the workflow succeeded."""

        return self.status == NO_EGRESS_STATUS_PASSED

    @property
    def exit_code(self) -> int:
        """Return the gate exit code; only a pass exits zero."""

        return NO_EGRESS_EXIT_CODES.get(self.status, 1)

    def to_markdown(self) -> str:
        """Render the attestation as a value-free Markdown summary."""

        payload = self.to_dict()
        boundary = payload["boundary"]
        lines = [
            "# OpenMed no-egress attestation",
            "",
            f"- Status: `{payload['status']}` (`{payload['reason_code']}`)",
            f"- Commit: `{payload['commit']}`",
            f"- OpenMed version: `{payload['openmed_version']}`",
            f"- Host kind: `{payload['host']['kind']}`",
            f"- Boundary: `{boundary['kind']}` (`{boundary['mechanism']}`)",
            f"- Boundary isolated: `{str(boundary['isolated']).lower()}`",
            f"- Generated at: `{payload['generated_at']}`",
            f"- Repro hash: `{payload['repro_hash']}`",
            f"- Integrity hash: `{payload['integrity_hash']}`",
            "",
        ]
        for title, key in (
            ("Negative controls", "negative_controls"),
            ("Positive controls", "positive_controls"),
        ):
            lines.extend(
                [
                    f"## {title}",
                    "",
                    "| Case | Transport | Family | Executor | Status | Reason |",
                    "|---|---|---|---|---|---|",
                ]
            )
            for case in payload[key]:
                lines.append(
                    "| "
                    + " | ".join(
                        f"`{case[field]}`"
                        for field in (
                            "id",
                            "transport",
                            "family",
                            "executor",
                            "status",
                            "reason_code",
                        )
                    )
                    + " |"
                )
            lines.append("")
        lines.extend(
            [
                "## Workflow",
                "",
                "| Case | Surface | Status | Reason |",
                "|---|---|---|---|",
            ]
        )
        for case in payload["workflow"]:
            lines.append(
                "| "
                + " | ".join(
                    f"`{case[field]}`"
                    for field in ("id", "surface", "status", "reason_code")
                )
                + " |"
            )
        lines.extend(["", "## Fixtures", ""])
        for fixture in payload["fixtures"]:
            lines.append(f"- `{fixture['id']}`: `{fixture['digest']}`")
        lines.append("")
        return "\n".join(lines)


def load_no_egress_attestation_schema() -> dict[str, Any]:
    """Return the bundled JSON Schema for no-egress attestations.

    Returns:
        The Draft 2020-12 schema as a dictionary.
    """

    resource = resources.files("openmed.core.schemas.json").joinpath(
        NO_EGRESS_SCHEMA_NAME
    )
    with resource.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    if not isinstance(schema, dict):  # pragma: no cover - bundled resource
        raise NoEgressAttestationError("schema_unavailable")
    return schema


def build_no_egress_attestation(
    *,
    host_kind: str,
    requested_boundary: str,
    boundary_kind: str,
    fixtures: Mapping[str, str],
    kernel: str = "unknown",
    interfaces: Mapping[str, int] | None = None,
    routes: Mapping[str, int] | None = None,
    negative_controls: Sequence[Mapping[str, Any]] = (),
    positive_controls: Sequence[Mapping[str, Any]] = (),
    workflow: Sequence[Mapping[str, Any]] = (),
    skip_reason: str | None = None,
    runner_failure: str | None = None,
    commit: str | None = None,
    openmed_version: str | None = None,
    generated_at: datetime | str | None = None,
) -> NoEgressAttestation:
    """Build a schema-valid, value-free no-egress attestation.

    The status is derived from the evidence, never supplied by the caller: a
    run passes only when the boundary is an OS mechanism, the namespace has no
    non-loopback interface or route, every external negative control is
    blocked, every loopback positive control is reachable, and every workflow
    case passed. A host without a supported mechanism produces ``skipped``,
    which exits non-zero and is never a pass.

    Args:
        host_kind: ``linux``, ``darwin``, ``windows``, or ``other``.
        requested_boundary: ``auto``, ``netns``, or ``container``.
        boundary_kind: The boundary that ran: ``netns``, ``container``, or
            ``none`` when no mechanism was available.
        fixtures: Fixture identifier to ``sha256:`` digest.
        kernel: Kernel observed inside the boundary (``linux`` or ``unknown``).
        interfaces: ``total``, ``loopback_up``, and ``non_loopback_up`` counts
            observed inside the boundary.
        routes: ``ipv4_non_loopback`` and ``ipv6_non_loopback`` route counts.
        negative_controls: Probe results with ``id``, ``status``, and
            ``reason_code``; transport, family, and executor come from
            :data:`NO_EGRESS_NEGATIVE_CONTROLS`.
        positive_controls: Loopback probe results in the same shape.
        workflow: Workflow results with ``id``, ``status``, ``reason_code``,
            and an optional ``result_digest``.
        skip_reason: A controlled skip reason when no boundary ran.
        runner_failure: A controlled runner failure when the in-boundary
            runner crashed, timed out, or returned an invalid result.
        commit: Git commit identifier; anything else is recorded as
            ``unknown``.
        openmed_version: Package version; defaults to the installed version.
        generated_at: Declared timestamp, the only volatile evidence field.

    Returns:
        The validated attestation.

    Raises:
        NoEgressAttestationError: If any input is outside the controlled
            vocabulary or the evidence is inconsistent.
    """

    if host_kind not in _HOST_KINDS:
        raise NoEgressAttestationError("host_kind_invalid")
    if requested_boundary not in _REQUESTED_BOUNDARIES:
        raise NoEgressAttestationError("requested_boundary_invalid")
    if boundary_kind not in NO_EGRESS_BOUNDARY_MECHANISMS:
        raise NoEgressAttestationError("boundary_kind_invalid")
    if kernel not in {"linux", "unknown"}:
        raise NoEgressAttestationError("kernel_invalid")
    if skip_reason is not None and skip_reason not in NO_EGRESS_SKIP_REASONS:
        raise NoEgressAttestationError("skip_reason_invalid")
    if (
        runner_failure is not None
        and runner_failure not in NO_EGRESS_RUNNER_FAILURE_REASONS
    ):
        raise NoEgressAttestationError("runner_failure_invalid")
    if skip_reason is not None and runner_failure is not None:
        raise NoEgressAttestationError("skip_and_failure_conflict")
    if skip_reason is not None:
        if boundary_kind != "none":
            raise NoEgressAttestationError("skip_requires_no_boundary")
        if negative_controls or positive_controls or workflow:
            raise NoEgressAttestationError("skip_with_evidence")
    elif boundary_kind == "none":
        # Without an OS boundary there is nothing to attest; refusing here
        # keeps an unsupported host from ever being recorded as a pass.
        raise NoEgressAttestationError("boundary_required")

    interface_counts = _count_mapping(
        interfaces, ("total", "loopback_up", "non_loopback_up")
    )
    route_counts = _count_mapping(routes, ("ipv4_non_loopback", "ipv6_non_loopback"))
    isolated = bool(
        boundary_kind != "none"
        and kernel == "linux"
        and interface_counts["loopback_up"] >= 1
        and interface_counts["non_loopback_up"] == 0
        and route_counts["ipv4_non_loopback"] == 0
        and route_counts["ipv6_non_loopback"] == 0
    )
    negatives = _probe_records(
        negative_controls, NO_EGRESS_NEGATIVE_CONTROLS, _NEGATIVE_STATUSES
    )
    positives = _probe_records(
        positive_controls, NO_EGRESS_POSITIVE_CONTROLS, _POSITIVE_STATUSES
    )
    for record in negatives:
        if record["status"] == "blocked" and (
            record["reason_code"] not in _BLOCKED_REASONS
        ):
            raise NoEgressAttestationError("blocked_reason_invalid")
    cases = _workflow_records(workflow)
    fixture_records = _fixture_records(fixtures)
    counts = {
        "negative_controls": _status_counts(negatives, _NEGATIVE_STATUSES),
        "positive_controls": _status_counts(positives, _POSITIVE_STATUSES),
        "workflow": _status_counts(cases, _WORKFLOW_STATUSES),
    }
    status, reason = _derive_no_egress_status(
        isolated=isolated,
        negatives=negatives,
        positives=positives,
        cases=cases,
        skip_reason=skip_reason,
        runner_failure=runner_failure,
    )
    payload: dict[str, Any] = {
        "schema_version": NO_EGRESS_ATTESTATION_SCHEMA_VERSION,
        "attestation_type": NO_EGRESS_ATTESTATION_TYPE,
        "generated_at": _iso_timestamp(generated_at),
        "volatile_fields": list(NO_EGRESS_VOLATILE_FIELDS),
        "commit": _safe_commit(commit),
        "openmed_version": _safe_version(openmed_version),
        "status": status,
        "reason_code": reason,
        "host": {"kind": host_kind},
        "boundary": {
            "requested": requested_boundary,
            "kind": boundary_kind,
            "mechanism": NO_EGRESS_BOUNDARY_MECHANISMS[boundary_kind],
            "kernel": kernel,
            "isolated": isolated,
            "interfaces": interface_counts,
            "routes": route_counts,
        },
        "negative_controls": negatives,
        "positive_controls": positives,
        "workflow": cases,
        "counts": counts,
        "fixtures": fixture_records,
    }
    repro_payload = copy.deepcopy(payload)
    repro_payload.pop("generated_at")
    payload["repro_hash"] = stable_hash(repro_payload)
    payload["integrity_hash"] = stable_hash(payload)
    attestation = NoEgressAttestation(payload)
    validate_no_egress_attestation(attestation.to_dict())
    return attestation


def validate_no_egress_attestation(
    payload: Mapping[str, Any],
) -> NoEgressAttestation:
    """Validate a no-egress attestation against its schema and evidence rules.

    Beyond the JSON Schema this recomputes the counts, the derived status, both
    hashes, and the per-case vocabulary, so an edited or hand-written
    ``passed`` attestation does not validate.

    Args:
        payload: A decoded attestation object.

    Returns:
        The attestation wrapped in :class:`NoEgressAttestation`.

    Raises:
        NoEgressAttestationError: With a controlled code when the payload is
            invalid.
    """

    from jsonschema import Draft202012Validator

    if not isinstance(payload, Mapping):
        raise NoEgressAttestationError("attestation_not_object")
    data = copy.deepcopy(dict(payload))
    validator = Draft202012Validator(load_no_egress_attestation_schema())
    if next(validator.iter_errors(data), None) is not None:
        raise NoEgressAttestationError("schema_invalid")

    for key, vocabulary in (
        ("negative_controls", NO_EGRESS_NEGATIVE_CONTROLS),
        ("positive_controls", NO_EGRESS_POSITIVE_CONTROLS),
    ):
        seen: set[str] = set()
        for record in data[key]:
            if record["id"] in seen:
                raise NoEgressAttestationError("duplicate_case")
            seen.add(record["id"])
            expected = vocabulary[record["id"]]
            if (record["transport"], record["family"], record["executor"]) != (
                expected
            ):
                raise NoEgressAttestationError("case_vocabulary_mismatch")
    seen_cases: set[str] = set()
    for record in data["workflow"]:
        if record["id"] in seen_cases:
            raise NoEgressAttestationError("duplicate_case")
        seen_cases.add(record["id"])
        if record["surface"] != NO_EGRESS_WORKFLOW_CASES[record["id"]]:
            raise NoEgressAttestationError("case_vocabulary_mismatch")
    fixture_ids = [fixture["id"] for fixture in data["fixtures"]]
    if len(set(fixture_ids)) != len(fixture_ids):
        raise NoEgressAttestationError("duplicate_fixture")

    boundary = data["boundary"]
    if boundary["mechanism"] != NO_EGRESS_BOUNDARY_MECHANISMS[boundary["kind"]]:
        raise NoEgressAttestationError("boundary_mechanism_mismatch")
    interfaces = boundary["interfaces"]
    routes = boundary["routes"]
    isolated = bool(
        boundary["kind"] != "none"
        and boundary["kernel"] == "linux"
        and interfaces["loopback_up"] >= 1
        and interfaces["non_loopback_up"] == 0
        and routes["ipv4_non_loopback"] == 0
        and routes["ipv6_non_loopback"] == 0
    )
    if boundary["isolated"] is not isolated:
        raise NoEgressAttestationError("boundary_isolation_mismatch")
    for record in data["negative_controls"]:
        if record["status"] == "blocked" and (
            record["reason_code"] not in _BLOCKED_REASONS
        ):
            raise NoEgressAttestationError("blocked_reason_invalid")

    expected_counts = {
        "negative_controls": _status_counts(
            data["negative_controls"], _NEGATIVE_STATUSES
        ),
        "positive_controls": _status_counts(
            data["positive_controls"], _POSITIVE_STATUSES
        ),
        "workflow": _status_counts(data["workflow"], _WORKFLOW_STATUSES),
    }
    if data["counts"] != expected_counts:
        raise NoEgressAttestationError("counts_mismatch")

    status = data["status"]
    reason = data["reason_code"]
    if status == NO_EGRESS_STATUS_SKIPPED:
        expected_status, expected_reason = status, reason
    elif reason in NO_EGRESS_RUNNER_FAILURE_REASONS:
        expected_status, expected_reason = NO_EGRESS_STATUS_FAILED, reason
    else:
        expected_status, expected_reason = _derive_no_egress_status(
            isolated=isolated,
            negatives=data["negative_controls"],
            positives=data["positive_controls"],
            cases=data["workflow"],
            skip_reason=None,
            runner_failure=None,
        )
    if (status, reason) != (expected_status, expected_reason):
        raise NoEgressAttestationError("status_mismatch")

    attestation = NoEgressAttestation(data)
    if not attestation.repro_hash_matches():
        raise NoEgressAttestationError("repro_hash_mismatch")
    if not attestation.integrity_hash_matches():
        raise NoEgressAttestationError("integrity_hash_mismatch")
    return attestation


def _derive_no_egress_status(
    *,
    isolated: bool,
    negatives: Sequence[Mapping[str, Any]],
    positives: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
    skip_reason: str | None,
    runner_failure: str | None,
) -> tuple[str, str]:
    if skip_reason is not None:
        return NO_EGRESS_STATUS_SKIPPED, skip_reason
    if runner_failure is not None:
        return NO_EGRESS_STATUS_FAILED, runner_failure
    negative_status = {record["id"]: record["status"] for record in negatives}
    if any(value == "not_blocked" for value in negative_status.values()):
        return NO_EGRESS_STATUS_FAILED, "egress_not_blocked"
    if set(negative_status) != set(NO_EGRESS_NEGATIVE_CONTROLS) or any(
        value != "blocked" for value in negative_status.values()
    ):
        return NO_EGRESS_STATUS_FAILED, "negative_control_incomplete"
    if not isolated:
        return NO_EGRESS_STATUS_FAILED, "boundary_not_isolated"
    positive_status = {record["id"]: record["status"] for record in positives}
    if set(positive_status) != set(NO_EGRESS_POSITIVE_CONTROLS) or any(
        value != "reachable" for value in positive_status.values()
    ):
        return NO_EGRESS_STATUS_FAILED, "positive_control_failed"
    case_status = {record["id"]: record["status"] for record in cases}
    if set(case_status) != set(NO_EGRESS_WORKFLOW_CASES) or any(
        value != "passed" for value in case_status.values()
    ):
        return NO_EGRESS_STATUS_FAILED, "workflow_failed"
    return NO_EGRESS_STATUS_PASSED, "all_controls_blocked"


def _count_mapping(
    value: Mapping[str, int] | None, keys: tuple[str, ...]
) -> dict[str, int]:
    source = value if isinstance(value, Mapping) else {}
    counts: dict[str, int] = {}
    for key in keys:
        raw = source.get(key, 0)
        if type(raw) is not int or raw < 0:
            raise NoEgressAttestationError("count_invalid")
        counts[key] = raw
    return counts


def _probe_records(
    records: Sequence[Mapping[str, Any]],
    vocabulary: Mapping[str, tuple[str, str, str]],
    statuses: tuple[str, ...],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise NoEgressAttestationError("case_invalid")
        identifier = record.get("id")
        if not isinstance(identifier, str) or identifier not in vocabulary:
            raise NoEgressAttestationError("case_unknown")
        if identifier in seen:
            raise NoEgressAttestationError("duplicate_case")
        seen.add(identifier)
        status = record.get("status")
        reason = record.get("reason_code")
        if status not in statuses:
            raise NoEgressAttestationError("case_status_invalid")
        if reason not in _PROBE_REASONS:
            raise NoEgressAttestationError("case_reason_invalid")
        transport, family, executor = vocabulary[identifier]
        result.append(
            {
                "id": identifier,
                "transport": transport,
                "family": family,
                "executor": executor,
                "status": status,
                "reason_code": reason,
            }
        )
    order = list(vocabulary)
    return sorted(result, key=lambda item: order.index(item["id"]))


def _workflow_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise NoEgressAttestationError("case_invalid")
        identifier = record.get("id")
        if not isinstance(identifier, str) or identifier not in (
            NO_EGRESS_WORKFLOW_CASES
        ):
            raise NoEgressAttestationError("case_unknown")
        if identifier in seen:
            raise NoEgressAttestationError("duplicate_case")
        seen.add(identifier)
        status = record.get("status")
        reason = record.get("reason_code")
        digest = record.get("result_digest")
        if status not in _WORKFLOW_STATUSES:
            raise NoEgressAttestationError("case_status_invalid")
        if reason not in _WORKFLOW_REASONS:
            raise NoEgressAttestationError("case_reason_invalid")
        if (status == "passed") != (reason == "ok"):
            raise NoEgressAttestationError("case_reason_invalid")
        if digest is not None and (
            not isinstance(digest, str) or not _DIGEST_RE.match(digest)
        ):
            raise NoEgressAttestationError("case_digest_invalid")
        result.append(
            {
                "id": identifier,
                "surface": NO_EGRESS_WORKFLOW_CASES[identifier],
                "status": status,
                "reason_code": reason,
                "result_digest": digest,
            }
        )
    order = list(NO_EGRESS_WORKFLOW_CASES)
    return sorted(result, key=lambda item: order.index(item["id"]))


def _fixture_records(fixtures: Mapping[str, str]) -> list[dict[str, str]]:
    if not isinstance(fixtures, Mapping) or not fixtures:
        raise NoEgressAttestationError("fixtures_required")
    records: list[dict[str, str]] = []
    for identifier, digest in sorted(fixtures.items()):
        if identifier not in NO_EGRESS_FIXTURE_IDS:
            raise NoEgressAttestationError("fixture_unknown")
        if not isinstance(digest, str) or not _DIGEST_RE.match(digest):
            raise NoEgressAttestationError("fixture_digest_invalid")
        records.append({"id": identifier, "digest": digest})
    return records


def _status_counts(
    records: Sequence[Mapping[str, Any]], statuses: tuple[str, ...]
) -> dict[str, int]:
    counts = {"total": len(records)}
    for status in statuses:
        counts[status] = sum(1 for record in records if record["status"] == status)
    return counts


def _safe_commit(value: str | None) -> str:
    if isinstance(value, str) and _COMMIT_RE.match(value):
        return value
    return "unknown"


def _safe_version(value: str | None) -> str:
    if value is None:
        from ..__about__ import __version__

        value = __version__
    return value if isinstance(value, str) and _VERSION_RE.match(value) else "unknown"


__all__ = [
    "ATTESTATION_SCHEMA_VERSION",
    "AttestationReport",
    "AttestationTemplateError",
    "NO_EGRESS_ATTESTATION_SCHEMA_VERSION",
    "NO_EGRESS_ATTESTATION_TYPE",
    "NO_EGRESS_BOUNDARY_MECHANISMS",
    "NO_EGRESS_EXIT_CODES",
    "NO_EGRESS_NEGATIVE_CONTROLS",
    "NO_EGRESS_POSITIVE_CONTROLS",
    "NO_EGRESS_WORKFLOW_CASES",
    "NoEgressAttestation",
    "NoEgressAttestationError",
    "build_no_egress_attestation",
    "generate_attestation",
    "list_attestation_profiles",
    "load_attestation_template",
    "load_no_egress_attestation_schema",
    "validate_no_egress_attestation",
]
