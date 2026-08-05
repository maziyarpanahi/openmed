"""PHI-free, template-driven data-residency attestations."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
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


__all__ = [
    "ATTESTATION_SCHEMA_VERSION",
    "AttestationReport",
    "AttestationTemplateError",
    "generate_attestation",
    "list_attestation_profiles",
    "load_attestation_template",
]
