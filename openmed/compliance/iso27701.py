"""Generate an ISO 27701/27001 control-evidence pack for OpenMed.

The pack is a deterministic technical crosswalk. It maps product capabilities
to a focused set of ISO/IEC 27701:2025 and ISO/IEC 27001:2022 controls, while
making the boundary between OpenMed evidence and organization-owned controls
explicit. Generation uses only bundled metadata and local filesystem writes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "openmed.iso_control_evidence.v1"
PACK_ID = "openmed-iso-27701-27001-control-evidence"
MANIFEST_FILENAME = "iso-27701-control-evidence.json"
MARKDOWN_FILENAME = "iso-27701-control-evidence.md"

CONTROL_STATUSES = ("covered", "partial", "out-of-scope")
EVIDENCE_TYPES = (
    "config-flag",
    "documentation",
    "guard-test",
    "implementation",
    "report-artifact",
)
CAPABILITIES = (
    "audit-chain",
    "key-lifecycle",
    "no-phi-logging",
    "offline-mode",
    "release-gates",
)

PACK_SCOPE = (
    "Technical controls implemented by the OpenMed library and the bundled "
    "evidence that demonstrates those controls."
)
PACK_DISCLAIMER = (
    "This pack is a product-level technical evidence crosswalk, not an ISO "
    "certification, audit opinion, Statement of Applicability, or "
    "organization-specific policy. The adopting organization remains "
    "responsible for its ISMS/PIMS scope, risk treatment, operating evidence, "
    "legal basis, retention, access control, and independent audit."
)


@dataclass(frozen=True)
class EvidencePointer:
    """One machine-readable pointer to local control evidence."""

    evidence_type: str
    reference: str
    description: str

    def __post_init__(self) -> None:
        if self.evidence_type not in EVIDENCE_TYPES:
            raise ValueError(f"unsupported evidence type: {self.evidence_type!r}")
        if not self.reference.strip():
            raise ValueError("evidence reference must be non-empty")
        if not self.description.strip():
            raise ValueError("evidence description must be non-empty")

    def to_dict(self) -> dict[str, str]:
        """Serialize the pointer to JSON-compatible fields."""

        return {
            "type": self.evidence_type,
            "reference": self.reference,
            "description": self.description,
        }


@dataclass(frozen=True)
class ControlEvidence:
    """OpenMed evidence and scope rationale for one standard control."""

    framework: str
    control_id: str
    title: str
    status: str
    rationale: str
    capabilities: tuple[str, ...] = ()
    evidence: tuple[EvidencePointer, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("framework", "control_id", "title", "rationale"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be non-empty")
        if self.status not in CONTROL_STATUSES:
            raise ValueError(f"unsupported control status: {self.status!r}")
        unknown = set(self.capabilities).difference(CAPABILITIES)
        if unknown:
            raise ValueError(f"unsupported capabilities: {sorted(unknown)!r}")
        if self.status == "out-of-scope":
            if self.capabilities or self.evidence:
                raise ValueError(
                    "out-of-scope controls cannot claim capabilities or evidence"
                )
        elif not self.capabilities or not self.evidence:
            raise ValueError(
                "covered and partial controls require capabilities and evidence"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the control mapping to JSON-compatible fields."""

        return {
            "framework": self.framework,
            "control_id": self.control_id,
            "title": self.title,
            "status": self.status,
            "rationale": self.rationale,
            "capabilities": list(self.capabilities),
            "evidence": [pointer.to_dict() for pointer in self.evidence],
        }


@dataclass(frozen=True)
class ControlEvidencePack:
    """Deterministic ISO control-evidence pack."""

    controls: tuple[ControlEvidence, ...]
    schema_version: str = SCHEMA_VERSION
    pack_id: str = PACK_ID
    title: str = "OpenMed ISO 27701/27001 control-evidence pack"
    scope: str = PACK_SCOPE
    disclaimer: str = PACK_DISCLAIMER

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
        if not self.controls:
            raise ValueError("control-evidence pack must contain controls")
        identities = [
            (control.framework, control.control_id) for control in self.controls
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("control framework/id pairs must be unique")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete pack to its manifest representation."""

        status_counts = {
            status: sum(control.status == status for control in self.controls)
            for status in CONTROL_STATUSES
        }
        return {
            "schema_version": self.schema_version,
            "pack_id": self.pack_id,
            "title": self.title,
            "frameworks": sorted({control.framework for control in self.controls}),
            "scope": self.scope,
            "disclaimer": self.disclaimer,
            "summary": {
                "total": len(self.controls),
                **status_counts,
            },
            "controls": [control.to_dict() for control in self.controls],
        }


@dataclass(frozen=True)
class ControlEvidencePackResult:
    """Paths and in-memory content produced by pack generation."""

    output_dir: Path
    manifest_path: Path
    markdown_path: Path
    pack: ControlEvidencePack
    manifest: Mapping[str, Any]
    markdown: str


def _pointer(
    evidence_type: str,
    reference: str,
    description: str,
) -> EvidencePointer:
    return EvidencePointer(evidence_type, reference, description)


def _default_controls() -> tuple[ControlEvidence, ...]:
    no_phi_guard = _pointer(
        "guard-test",
        "tests/unit/test_no_raw_text_logging.py::"
        "test_pii_processing_and_service_paths_do_not_log_raw_phi",
        "Exercises core, batch, and REST paths and rejects raw PHI in logs.",
    )
    audit_chain = _pointer(
        "implementation",
        "openmed/compliance/audit_chain.py::HashChainAuditLog",
        "Links privacy-safe audit records so later mutation is detectable.",
    )
    audit_chain_guard = _pointer(
        "guard-test",
        "tests/unit/compliance/test_audit_chain.py::test_records_form_a_hash_chain",
        "Verifies record linkage and tamper detection.",
    )
    offline_flag = _pointer(
        "config-flag",
        "OPENMED_OFFLINE / OpenMedConfig.local_only",
        "Selects cache-only execution and fail-closed network blocking.",
    )
    offline_guard = _pointer(
        "guard-test",
        "tests/unit/test_offline_mode.py::"
        "test_pii_deidentification_path_runs_with_sockets_blocked",
        "Runs PII extraction and de-identification while socket use is blocked.",
    )
    key_lifecycle = _pointer(
        "implementation",
        "openmed/core/surrogate_vault.py::SurrogateVault.rotate",
        "Rotates locally derived vault epochs and migrates encrypted entries.",
    )
    key_revocation_guard = _pointer(
        "guard-test",
        "tests/unit/core/test_surrogate_vault.py::"
        "test_revoke_current_epoch_reencrypts_forward_and_retires_old_key",
        "Verifies forward re-encryption and retirement of revoked key material.",
    )
    gate_bundler = _pointer(
        "implementation",
        "openmed/eval/evidence_bundle.py::bundle_gate_evidence",
        "Collects release-gate artifacts into a deterministic hashed manifest.",
    )
    gate_artifact = _pointer(
        "report-artifact",
        "gate-evidence/manifest.json",
        "Machine-readable release-gate decisions, artifact hashes, and gaps.",
    )
    audit_artifact = _pointer(
        "report-artifact",
        "AuditReport.to_dict()",
        "Machine-readable span provenance, risk results, and optional signature.",
    )

    return (
        ControlEvidence(
            framework="ISO/IEC 27701:2025",
            control_id="A.1.4.5",
            title="PII minimization objectives",
            status="partial",
            rationale=(
                "OpenMed provides de-identification, no-PHI logging, and measurable "
                "release gates, but each organization must define and approve its "
                "own minimization objectives."
            ),
            capabilities=("no-phi-logging", "release-gates"),
            evidence=(no_phi_guard, gate_bundler, gate_artifact),
        ),
        ControlEvidence(
            framework="ISO/IEC 27701:2025",
            control_id="A.1.4.6",
            title="PII de-identification and deletion at end of processing",
            status="partial",
            rationale=(
                "OpenMed implements local de-identification and privacy-safe audit "
                "evidence. Retention triggers and deletion of source systems remain "
                "deployment responsibilities."
            ),
            capabilities=("audit-chain", "no-phi-logging", "offline-mode"),
            evidence=(no_phi_guard, audit_chain, offline_guard, audit_artifact),
        ),
        ControlEvidence(
            framework="ISO/IEC 27701:2025",
            control_id="A.3.25",
            title="Logging",
            status="covered",
            rationale=(
                "Within the OpenMed product boundary, logs are guarded against raw "
                "PHI and compliance events can be recorded in a tamper-evident, "
                "privacy-safe chain."
            ),
            capabilities=("audit-chain", "no-phi-logging"),
            evidence=(no_phi_guard, audit_chain, audit_chain_guard, audit_artifact),
        ),
        ControlEvidence(
            framework="ISO/IEC 27701:2025",
            control_id="A.3.26",
            title="Use of cryptography",
            status="partial",
            rationale=(
                "OpenMed supplies HMAC-derived identifiers, encrypted vault entries, "
                "rotation, and revocation. Root-secret custody, rotation schedules, "
                "and enterprise key-management policy remain organization-owned."
            ),
            capabilities=("key-lifecycle",),
            evidence=(key_lifecycle, key_revocation_guard),
        ),
        ControlEvidence(
            framework="ISO/IEC 27701:2025",
            control_id="9.1",
            title="Monitoring, measurement, analysis and evaluation",
            status="partial",
            rationale=(
                "Release gates and deterministic evidence bundles support product "
                "measurement. They do not replace an organization's PIMS metrics, "
                "management review, or internal-audit programme."
            ),
            capabilities=("release-gates",),
            evidence=(gate_bundler, gate_artifact),
        ),
        ControlEvidence(
            framework="ISO/IEC 27701:2025",
            control_id="A.1.2.3",
            title="Identify lawful basis",
            status="out-of-scope",
            rationale=(
                "Lawful basis depends on the adopting organization's purposes, "
                "jurisdiction, relationships, and legal analysis; OpenMed cannot "
                "determine it from technical processing."
            ),
        ),
        ControlEvidence(
            framework="ISO/IEC 27701:2025",
            control_id="A.1.2.7",
            title="Contracts with PII processors",
            status="out-of-scope",
            rationale=(
                "Supplier selection, data-processing agreements, and contractual "
                "oversight are organizational governance activities outside the "
                "OpenMed library."
            ),
        ),
        ControlEvidence(
            framework="ISO/IEC 27001:2022",
            control_id="A.5.34",
            title="Privacy and protection of PII",
            status="partial",
            rationale=(
                "OpenMed contributes de-identification, local-only execution, "
                "privacy-safe logging, and audit evidence. The complete control also "
                "depends on organization-wide legal and operational measures."
            ),
            capabilities=(
                "audit-chain",
                "no-phi-logging",
                "offline-mode",
                "release-gates",
            ),
            evidence=(
                no_phi_guard,
                offline_flag,
                offline_guard,
                audit_chain,
                gate_artifact,
            ),
        ),
        ControlEvidence(
            framework="ISO/IEC 27001:2022",
            control_id="A.8.9",
            title="Configuration management",
            status="partial",
            rationale=(
                "OpenMed exposes explicit offline configuration and tests fail-closed "
                "behavior, while deployment baselines, approvals, and configuration "
                "inventory remain organization-owned."
            ),
            capabilities=("offline-mode",),
            evidence=(offline_flag, offline_guard),
        ),
        ControlEvidence(
            framework="ISO/IEC 27001:2022",
            control_id="A.8.15",
            title="Logging",
            status="covered",
            rationale=(
                "OpenMed's product responsibilities are evidenced by the no-raw-PHI "
                "logging guard and tamper-evident audit records. Deployment log "
                "retention and access policy remain part of the adopter's ISMS."
            ),
            capabilities=("audit-chain", "no-phi-logging"),
            evidence=(no_phi_guard, audit_chain, audit_chain_guard, audit_artifact),
        ),
        ControlEvidence(
            framework="ISO/IEC 27001:2022",
            control_id="A.8.24",
            title="Use of cryptography",
            status="partial",
            rationale=(
                "The surrogate vault implements cryptographic protection and key "
                "epoch lifecycle operations, but enterprise key custody and policy "
                "are outside the library boundary."
            ),
            capabilities=("key-lifecycle",),
            evidence=(key_lifecycle, key_revocation_guard),
        ),
        ControlEvidence(
            framework="ISO/IEC 27001:2022",
            control_id="A.8.25",
            title="Secure development life cycle",
            status="partial",
            rationale=(
                "Fail-closed release gates and hashed evidence bundles support "
                "software release decisions. Organization-wide development policy, "
                "approvals, and change governance remain outside OpenMed."
            ),
            capabilities=("release-gates",),
            evidence=(gate_bundler, gate_artifact),
        ),
        ControlEvidence(
            framework="ISO/IEC 27001:2022",
            control_id="A.7.1",
            title="Physical security perimeters",
            status="out-of-scope",
            rationale=(
                "OpenMed is a software library and cannot implement or evidence the "
                "physical boundaries of facilities, devices, or hosting locations."
            ),
        ),
    )


def build_control_evidence_pack() -> ControlEvidencePack:
    """Build the bundled ISO control-evidence mapping without external calls.

    Returns:
        The deterministic control-evidence pack.
    """

    return ControlEvidencePack(controls=_default_controls())


def render_control_evidence_markdown(pack: ControlEvidencePack) -> str:
    """Render a pack as deterministic auditor-oriented Markdown.

    Args:
        pack: Structured control-evidence pack to render.

    Returns:
        Markdown containing the pack metadata and every control mapping.
    """

    manifest = pack.to_dict()
    summary = manifest["summary"]
    lines = [
        f"# {pack.title}",
        "",
        f"> {pack.disclaimer}",
        "",
        "## Pack metadata",
        "",
        f"- Schema version: `{pack.schema_version}`",
        f"- Pack ID: `{pack.pack_id}`",
        f"- Frameworks: {', '.join(manifest['frameworks'])}",
        f"- Scope: {pack.scope}",
        "- Generation: deterministic and offline; no external calls are made",
        "",
        "## Status summary",
        "",
        f"- Covered: {summary['covered']}",
        f"- Partial: {summary['partial']}",
        f"- Out of scope: {summary['out-of-scope']}",
        f"- Total: {summary['total']}",
        "",
        "`covered` applies only to OpenMed's product responsibility. `partial` "
        "requires adopter controls in addition to OpenMed evidence. "
        "`out-of-scope` identifies an explicit product boundary.",
        "",
        "## Control mapping",
    ]

    for control in pack.controls:
        lines.extend(
            [
                "",
                f"### {control.framework} {control.control_id} — {control.title}",
                "",
                f"- Status: **{control.status}**",
                f"- Rationale: {control.rationale}",
            ]
        )
        if control.capabilities:
            lines.append(
                "- OpenMed capabilities: "
                + ", ".join(f"`{item}`" for item in control.capabilities)
            )
        if control.evidence:
            lines.extend(["- Evidence:", ""])
            lines.extend(
                "  - "
                f"`{pointer.evidence_type}` — `{pointer.reference}`: "
                f"{pointer.description}"
                for pointer in control.evidence
            )
        else:
            lines.append("- Evidence: none; see the explicit scope rationale above")

    return "\n".join(lines) + "\n"


def load_control_evidence_schema() -> dict[str, Any]:
    """Load the committed JSON Schema for control-evidence manifests.

    Returns:
        Parsed Draft 2020-12 JSON Schema.
    """

    resource = resources.files("openmed.compliance").joinpath(
        "schemas/control_evidence.schema.json"
    )
    with resource.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    if not isinstance(schema, dict):
        raise ValueError("control-evidence schema must be a JSON object")
    return schema


def generate_control_evidence_pack(
    output_dir: str | Path,
) -> ControlEvidencePackResult:
    """Write the ISO control-evidence JSON manifest and Markdown rendering.

    The generator does not inspect the environment, read user data, or perform
    network calls. Repeated runs against the same package version produce the
    same manifest and Markdown bytes.

    Args:
        output_dir: Local destination for the JSON and Markdown files.

    Returns:
        Paths and in-memory representations of both generated artifacts.
    """

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    pack = build_control_evidence_pack()
    manifest = pack.to_dict()
    markdown = render_control_evidence_markdown(pack)
    manifest_path = destination / MANIFEST_FILENAME
    markdown_path = destination / MARKDOWN_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(markdown, encoding="utf-8")

    return ControlEvidencePackResult(
        output_dir=destination,
        manifest_path=manifest_path,
        markdown_path=markdown_path,
        pack=pack,
        manifest=manifest,
        markdown=markdown,
    )


__all__ = [
    "CAPABILITIES",
    "CONTROL_STATUSES",
    "EVIDENCE_TYPES",
    "MANIFEST_FILENAME",
    "MARKDOWN_FILENAME",
    "PACK_ID",
    "SCHEMA_VERSION",
    "ControlEvidence",
    "ControlEvidencePack",
    "ControlEvidencePackResult",
    "EvidencePointer",
    "build_control_evidence_pack",
    "generate_control_evidence_pack",
    "load_control_evidence_schema",
    "render_control_evidence_markdown",
]
