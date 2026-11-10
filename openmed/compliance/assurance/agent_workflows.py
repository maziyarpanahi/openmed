"""Deterministic deployment-assurance packs for v3.1 agent workflows.

The builder accepts only an exact source revision, an artifact digest, and the
aggregate release-gate report. Narrative content is selected from a closed
registry, so prompts, clinical outputs, credentials, and reviewer identities
cannot enter the generated JSON or Markdown through this API.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Final

from openmed.eval.suites.agent_release import (
    APPROVAL_BYPASS,
    CLINICIAN_REVIEW_AGREEMENT,
    DEFAULT_AGENT_RELEASE_GATES,
    EVIDENCE_COMPLETENESS,
    NOT_READY,
    READY,
    RECOVERY_CORRECTNESS,
    REFERENCE_SERVER_COVERAGE,
    REPLAY_EQUIVALENCE,
    UNAUTHORIZED_ACTION_ESCAPE,
    UNSAFE_SIDE_EFFECT,
    WORKFLOW_LATENCY,
    AgentReleaseGateReport,
)

ASSURANCE_PACK_SCHEMA_VERSION = "openmed.compliance.agent_assurance_pack.v1"
ASSURANCE_PACK_VERSION = "3.1"
VALIDATED_CLAIM = "validated_claim"
UNVALIDATED_LIMITATION = "unvalidated_limitation"

_DIGEST_RE: Final = re.compile(r"sha256:[0-9a-f]{64}")
_SOURCE_REVISION_RE: Final = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
_PUBLIC_REF_RE: Final = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}")
_GATE_DOCUMENTATION = "../evaluation/v3.1-agent-gates.md#pre-registered-thresholds"


class AgentAssurancePackError(ValueError):
    """Raised when assurance-pack evidence violates the closed schema."""


@dataclass(frozen=True, slots=True)
class ValidationReference:
    """Content-free reference to one v3.1 release-gate result."""

    metric: str
    passed: bool
    reason_code: str
    evidence_digest: str | None
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        metrics = {spec.metric for spec in DEFAULT_AGENT_RELEASE_GATES}
        if self.metric not in metrics:
            raise AgentAssurancePackError("metric: unknown_release_gate")
        if type(self.passed) is not bool:
            raise AgentAssurancePackError("passed: invalid_boolean")
        if _PUBLIC_REF_RE.fullmatch(self.reason_code) is None:
            raise AgentAssurancePackError("reason_code: invalid_reference")
        if self.evidence_digest is not None and (
            _DIGEST_RE.fullmatch(self.evidence_digest) is None
        ):
            raise AgentAssurancePackError("evidence_digest: invalid_digest")
        if self.passed and self.evidence_digest is None:
            raise AgentAssurancePackError("evidence_digest: required_for_passed_gate")
        limitations = tuple(self.limitations)
        if any(_PUBLIC_REF_RE.fullmatch(item) is None for item in limitations):
            raise AgentAssurancePackError("limitations: invalid_reference")
        object.__setattr__(self, "limitations", limitations)

    def to_dict(self) -> dict[str, Any]:
        """Return the stable validation-reference schema."""
        return {
            "documentation": _GATE_DOCUMENTATION,
            "evidence_digest": self.evidence_digest,
            "limitations": list(self.limitations),
            "metric": self.metric,
            "passed": self.passed,
            "reason_code": self.reason_code,
        }


_CLAIMS: Final = {
    UNAUTHORIZED_ACTION_ESCAPE: (
        "authority_boundary",
        "The evaluated workflow profile prevented unauthorized actions.",
        "Unauthorized-action containment is not validated for this artifact.",
    ),
    APPROVAL_BYPASS: (
        "human_approval_enforcement",
        "The evaluated workflow profile enforced required human approval.",
        "Human-approval enforcement is not validated for this artifact.",
    ),
    UNSAFE_SIDE_EFFECT: (
        "side_effect_containment",
        "The evaluated workflow profile contained unsafe side effects.",
        "Unsafe-side-effect containment is not validated for this artifact.",
    ),
    REPLAY_EQUIVALENCE: (
        "deterministic_replay",
        "The evaluated workflow profile met the replay-equivalence gate.",
        "Replay equivalence is not validated for this artifact.",
    ),
    RECOVERY_CORRECTNESS: (
        "failure_recovery",
        "The evaluated workflow profile met the recovery-correctness gate.",
        "Failure recovery is not validated for this artifact.",
    ),
    EVIDENCE_COMPLETENESS: (
        "incident_evidence_completeness",
        "The evaluated workflow profile met the evidence-completeness gate.",
        "Incident-evidence completeness is not validated for this artifact.",
    ),
    REFERENCE_SERVER_COVERAGE: (
        "reference_server_coverage",
        "The evaluated workflow profile met its declared server-coverage gate.",
        "Reference-server coverage is not validated for this artifact.",
    ),
    WORKFLOW_LATENCY: (
        "workflow_latency",
        "The evaluated workflow profile met the workflow-latency gate.",
        "Workflow latency is not validated for this artifact.",
    ),
    CLINICIAN_REVIEW_AGREEMENT: (
        "clinician_review_agreement",
        "The evaluated workflow profile met the clinician-agreement gate.",
        "Clinician-review agreement is not validated for this artifact.",
    ),
}

_DEPLOYMENT_PROFILE: Final = {
    "authority_model": "explicit_permissioned_tools",
    "clinical_decision_mode": "assistive_clinician_review_only",
    "deployment_mode": "local_first",
    "network_requirement_after_setup": "none_for_pack_generation",
    "production_writes": "explicit_enablement_and_approval_required",
    "profile_id": "v3.1_permissioned_clinical_workflows",
}

_THREAT_ROWS: Final = (
    {
        "claim_id": "authority_boundary",
        "threat": "tool_authority_escape",
    },
    {
        "claim_id": "human_approval_enforcement",
        "threat": "approval_bypass",
    },
    {
        "claim_id": "side_effect_containment",
        "threat": "unsafe_side_effect",
    },
    {
        "claim_id": "deterministic_replay",
        "threat": "non_equivalent_replay",
    },
    {
        "claim_id": "failure_recovery",
        "threat": "incorrect_failure_recovery",
    },
    {
        "classification": UNVALIDATED_LIMITATION,
        "limitation": "benchmark_slices_do_not_establish_site_specific_safety",
        "threat": "out_of_distribution_workflow",
    },
    {
        "classification": UNVALIDATED_LIMITATION,
        "limitation": "jurisdictional_compliance_requires_site_review",
        "threat": "unsupported_legal_or_regulatory_interpretation",
    },
)

_APPROVAL_STATES: Final = (
    {
        "required_evidence": "content_free_action_and_artifact_references",
        "state": "proposal",
    },
    {
        "required_evidence": "side_effect_preview_reference",
        "state": "preview",
    },
    {
        "required_evidence": "bound_approval_receipt_reference",
        "state": "approve_decline_or_escalate",
    },
    {
        "required_evidence": "approved_action_and_outcome_references",
        "state": "execute_or_abstain",
    },
    {
        "required_evidence": "reviewer_handoff_reference",
        "state": "handoff",
    },
)

_INCIDENT_CHECKLIST: Final = (
    "exact_source_revision",
    "artifact_digest",
    "candidate_manifest_digest",
    "release_gate_report_digest",
    "policy_and_tool_catalog_digests",
    "side_effect_preview_digest",
    "approval_receipt_reference",
    "workflow_event_and_outcome_references",
    "recovery_or_replay_evidence_digests",
    "clinician_review_and_handoff_references",
    "known_limitation_codes",
)

_JURISDICTIONS: Final = (
    {
        "classification": UNVALIDATED_LIMITATION,
        "jurisdiction": "global_baseline",
        "status": "technical_controls_only",
        "support_boundary": "site_governance_and_legal_review_required",
    },
    {
        "classification": UNVALIDATED_LIMITATION,
        "jurisdiction": "eu_eea",
        "status": "not_a_ce_mark_or_eu_ai_act_conformity_assessment",
        "support_boundary": "site_governance_and_legal_review_required",
    },
    {
        "classification": UNVALIDATED_LIMITATION,
        "jurisdiction": "united_states",
        "status": "not_hipaa_expert_determination_or_fda_clearance",
        "support_boundary": "site_governance_and_legal_review_required",
    },
    {
        "classification": UNVALIDATED_LIMITATION,
        "jurisdiction": "other_or_multi_jurisdiction",
        "status": "not_evaluated",
        "support_boundary": "site_governance_and_legal_review_required",
    },
)

_CLINICIAN_REVIEW_STEPS: Final = (
    {
        "evidence": "source_and_artifact_binding",
        "instruction": "Confirm the preview belongs to the reviewed artifact.",
        "step": "bind_candidate",
    },
    {
        "classification": UNVALIDATED_LIMITATION,
        "evidence": "side_effect_preview_reference",
        "instruction": "Inspect intended writes, targets, and abstentions.",
        "step": "inspect_consequences",
    },
    {
        "claim_id": "human_approval_enforcement",
        "evidence": "approval_receipt_reference",
        "instruction": "Approve, decline, or escalate before execution.",
        "step": "record_decision",
    },
    {
        "claim_id": "clinician_review_agreement",
        "evidence": "evidence_digest_and_limitation_codes",
        "instruction": "Review clinical meaning and unresolved limitations.",
        "step": "review_clinical_meaning",
    },
    {
        "classification": UNVALIDATED_LIMITATION,
        "evidence": "reviewer_handoff_reference",
        "instruction": "Transfer content-free references with accountability.",
        "step": "handoff_or_escalate",
    },
)


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest(value: dict[str, Any]) -> str:
    encoded = _canonical_json(value).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True, slots=True)
class AgentWorkflowAssurancePack:
    """Content-addressed deployment and clinician-review assurance pack."""

    source_revision: str
    artifact_digest: str
    candidate_digest: str
    gate_report_digest: str
    release_decision: str
    validations: tuple[ValidationReference, ...]
    pack_digest: str = ""
    schema_version: str = ASSURANCE_PACK_SCHEMA_VERSION
    pack_version: str = ASSURANCE_PACK_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ASSURANCE_PACK_SCHEMA_VERSION:
            raise AgentAssurancePackError("schema_version: unsupported_version")
        if self.pack_version != ASSURANCE_PACK_VERSION:
            raise AgentAssurancePackError("pack_version: unsupported_version")
        if _SOURCE_REVISION_RE.fullmatch(self.source_revision) is None:
            raise AgentAssurancePackError("source_revision: invalid_revision")
        for name, value in (
            ("artifact_digest", self.artifact_digest),
            ("candidate_digest", self.candidate_digest),
            ("gate_report_digest", self.gate_report_digest),
        ):
            if _DIGEST_RE.fullmatch(value) is None:
                raise AgentAssurancePackError(f"{name}: invalid_digest")

        validations = tuple(self.validations)
        expected_metrics = tuple(spec.metric for spec in DEFAULT_AGENT_RELEASE_GATES)
        if tuple(item.metric for item in validations) != expected_metrics:
            raise AgentAssurancePackError("validations: invalid_gate_set")
        expected_decision = (
            READY if all(item.passed for item in validations) else NOT_READY
        )
        if self.release_decision != expected_decision:
            raise AgentAssurancePackError("release_decision: inconsistent_validations")
        object.__setattr__(self, "validations", validations)

        calculated = _digest(self._payload(include_digest=False))
        if self.pack_digest and self.pack_digest != calculated:
            raise AgentAssurancePackError("pack_digest: payload_mismatch")
        object.__setattr__(self, "pack_digest", calculated)

    def _claims(self) -> list[dict[str, Any]]:
        claims: list[dict[str, Any]] = []
        for validation in self.validations:
            claim_id, validated, limitation = _CLAIMS[validation.metric]
            if validation.passed:
                claims.append(
                    {
                        "claim_id": claim_id,
                        "classification": VALIDATED_CLAIM,
                        "evidence_digest": validation.evidence_digest,
                        "gate_documentation": _GATE_DOCUMENTATION,
                        "release_gate": validation.metric,
                        "statement": validated,
                    }
                )
            else:
                claims.append(
                    {
                        "claim_id": claim_id,
                        "classification": UNVALIDATED_LIMITATION,
                        "evidence_digest": None,
                        "gate_documentation": None,
                        "release_gate": None,
                        "statement": limitation,
                    }
                )
        return claims

    def _payload(self, *, include_digest: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "approval_semantics": {
                "claim_id": "human_approval_enforcement",
                "states": list(_APPROVAL_STATES),
            },
            "artifact_digest": self.artifact_digest,
            "candidate_digest": self.candidate_digest,
            "change_log": [
                {
                    "change": "initial_v3.1_deployment_and_review_contract",
                    "pack_version": ASSURANCE_PACK_VERSION,
                }
            ],
            "claims": self._claims(),
            "clinician_review_protocol": list(_CLINICIAN_REVIEW_STEPS),
            "deployment_profile": dict(_DEPLOYMENT_PROFILE),
            "disclaimers": [
                "openmed_does_not_provide_certification",
                "openmed_does_not_make_autonomous_clinical_decisions",
                "not_legal_advice",
                "does_not_replace_site_specific_clinical_governance",
            ],
            "gate_report_digest": self.gate_report_digest,
            "incident_evidence_checklist": list(_INCIDENT_CHECKLIST),
            "jurisdiction_support_matrix": list(_JURISDICTIONS),
            "pack_version": self.pack_version,
            "release_decision": self.release_decision,
            "schema_version": self.schema_version,
            "source_revision": self.source_revision,
            "threat_and_limitation_summary": list(_THREAT_ROWS),
            "validation_references": [item.to_dict() for item in self.validations],
        }
        if include_digest:
            payload["pack_digest"] = self.pack_digest
        return payload

    def to_dict(self) -> dict[str, Any]:
        """Return the closed deterministic pack schema."""
        return self._payload(include_digest=True)

    def to_json(self) -> str:
        """Return canonical JSON without raw workflow or clinical content."""
        return _canonical_json(self.to_dict())

    def to_markdown(self) -> str:
        """Render a deterministic operator and clinician handoff document."""
        claims = self._claims()
        lines = [
            "# OpenMed v3.1 agent workflow assurance pack",
            "",
            "> OpenMed does not provide certification or make autonomous clinical",
            "> decisions. This pack is technical evidence, not legal advice, and it",
            "> does not replace site-specific clinical governance.",
            "",
            "## Exact evidence binding",
            "",
            f"- Pack version: `{self.pack_version}`",
            f"- Source revision: `{self.source_revision}`",
            f"- Artifact digest: `{self.artifact_digest}`",
            f"- Candidate digest: `{self.candidate_digest}`",
            f"- Release-gate report digest: `{self.gate_report_digest}`",
            f"- Release decision: `{self.release_decision}`",
            f"- Pack digest: `{self.pack_digest}`",
            "",
            "## Deployment profile",
            "",
        ]
        lines.extend(
            f"- {key.replace('_', ' ').title()}: `{value}`"
            for key, value in _DEPLOYMENT_PROFILE.items()
        )
        lines.extend(
            [
                "",
                "## Claims and unvalidated limitations",
                "",
                "| Claim | Classification | Statement | Evidence |",
                "| --- | --- | --- | --- |",
            ]
        )
        for claim in claims:
            if claim["classification"] == VALIDATED_CLAIM:
                evidence = (
                    f"[`{claim['release_gate']}`]({_GATE_DOCUMENTATION}); "
                    f"`{claim['evidence_digest']}`"
                )
            else:
                evidence = "unvalidated limitation"
            lines.append(
                f"| `{claim['claim_id']}` | `{claim['classification']}` | "
                f"{claim['statement']} | {evidence} |"
            )
        lines.extend(
            [
                "",
                "## Human-approval semantics",
                "",
                "Consequential workflows proceed through these explicit states:",
                "",
            ]
        )
        lines.extend(
            f"{index}. `{row['state']}` — require `{row['required_evidence']}`."
            for index, row in enumerate(_APPROVAL_STATES, start=1)
        )
        lines.extend(
            [
                "",
                "## Incident evidence checklist",
                "",
                *[f"- [ ] `{item}`" for item in _INCIDENT_CHECKLIST],
                "",
                "## Jurisdiction support matrix",
                "",
                "| Jurisdiction | Status | Boundary | Classification |",
                "| --- | --- | --- | --- |",
            ]
        )
        lines.extend(
            f"| `{row['jurisdiction']}` | `{row['status']}` | "
            f"`{row['support_boundary']}` | `{row['classification']}` |"
            for row in _JURISDICTIONS
        )
        lines.extend(
            [
                "",
                "## Clinician-review protocol",
                "",
            ]
        )
        lines.extend(
            f"{index}. **{row['step'].replace('_', ' ').title()}.** "
            f"{row['instruction']} Evidence: `{row['evidence']}`."
            for index, row in enumerate(_CLINICIAN_REVIEW_STEPS, start=1)
        )
        lines.extend(
            [
                "",
                "Missing, mismatched, or failed evidence requires decline, "
                "abstention, or escalation; it never permits autonomous execution.",
                "",
                "## Change log",
                "",
                "- `3.1`: initial deployment-assurance and clinician-review pack.",
                "",
            ]
        )
        return "\n".join(lines)


def build_agent_assurance_pack(
    *,
    source_revision: str,
    artifact_digest: str,
    gate_report: AgentReleaseGateReport,
) -> AgentWorkflowAssurancePack:
    """Build a PHI-safe pack bound to exact source, artifact, and gate evidence.

    Args:
        source_revision: Exact 40- or 64-character hexadecimal source revision.
        artifact_digest: SHA-256 digest of the deployment artifact or manifest.
        gate_report: Validated v3.1 aggregate agent release-gate report.

    Returns:
        A deterministic deployment-assurance and clinician-review pack.

    Raises:
        AgentAssurancePackError: If an input is malformed or the gate report is
            not the v3.1 aggregate report type.
    """
    if not isinstance(gate_report, AgentReleaseGateReport):
        raise AgentAssurancePackError("gate_report: invalid_report_type")
    validations = tuple(
        ValidationReference(
            metric=result.metric,
            passed=result.passed,
            reason_code=result.reason_code,
            evidence_digest=result.evidence_digest,
            limitations=result.limitations,
        )
        for result in gate_report.gate_results
    )
    return AgentWorkflowAssurancePack(
        source_revision=source_revision,
        artifact_digest=artifact_digest,
        candidate_digest=gate_report.candidate_digest,
        gate_report_digest=gate_report.report_digest,
        release_decision=gate_report.decision,
        validations=validations,
    )


# Keep the workflow-specific name discoverable without maintaining two contracts.
build_agent_workflow_assurance_pack = build_agent_assurance_pack


__all__ = [
    "ASSURANCE_PACK_SCHEMA_VERSION",
    "ASSURANCE_PACK_VERSION",
    "UNVALIDATED_LIMITATION",
    "VALIDATED_CLAIM",
    "AgentAssurancePackError",
    "AgentWorkflowAssurancePack",
    "ValidationReference",
    "build_agent_assurance_pack",
    "build_agent_workflow_assurance_pack",
]
