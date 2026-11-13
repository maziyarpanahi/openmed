"""Fail-closed v4.0 secure-service release gate and signed evidence packet.

The secure exchange and recoverable service scope first planned as v3.4 was
folded into v4.0 (release plan #4253, workstream #3543), so the gate, schemas
and documentation carry the v4.0 name. The gate binds one exact candidate
commit to the service release evidence:

* the production security-profile report (#3552);
* the OS-enforced no-egress attestation (#3551);
* the reference recovery-drill report
  (:mod:`openmed.eval.service_recovery_drill`);
* the durable-schema migration rehearsal matrix (#3578);
* the REST, GraphQL and gRPC protocol map (#3548);
* synthetic fixture digests;
* the controlled failure-code inventory;
* the P0 status snapshot; and
* the service threat model.

Every P0 issue in the committed checklist
(``openmed/eval/data/service_release_p0.json``) names the evidence kinds it
requires. The gate fails when any P0 is open, lacks one of those kinds, or is
moved out of the release, and when a moved P1 or P2 item would narrow a
protected invariant.

Evidence is read through loaders keyed by kind. Kinds whose producers are not
on master yet default to a loader that blocks with
``typed_loader_unavailable``; callers inject a typed loader once the producer
lands. A required kind with no entry is reported as ``not_present``.

The packet reuses the Journey gate's canonical JSON, caller-keyed
HMAC-SHA256 and atomic writer (:mod:`openmed.eval.journey_release`). It holds
controlled codes, counts, issue references and digests only: never paths,
file contents, hostnames, identifiers or key material. A sentinel scan checks
every retained evidence file and the packet itself for prohibited value
classes. Building or verifying a packet makes no network call, and a
``READY`` packet does not tag, publish or deploy anything.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from openmed.core.audit import AuditSignature, stable_hash
from openmed.eval.journey_release import (
    JOURNEY_RELEASE_COMPATIBILITY_POLICY,
    JOURNEY_RELEASE_SIGNATURE_ALGORITHM,
    JourneyReleaseError,
    _canonical_json,
    _git_output,
    _key_bytes,
    _parse_timestamp,
    _reject_duplicate_keys,
    _reject_non_finite,
    _release_hmac,
    _resolve_beneath,
    _write_release_json,
)
from openmed.eval.service_recovery_drill import (
    DRILL_CATEGORIES,
    DRILL_CHECK_CODES,
    DRILL_COMPLETE,
    DRILL_ERROR_CODES,
    DRILL_REPORT_CODES,
    RECOVERY_DRILL_SCHEMA,
    validate_recovery_drill_report,
)

SERVICE_RELEASE_SCHEMA_VERSION: Final = "1.0.0"
_SUPPORTED_SCHEMA_VERSIONS: Final = frozenset({SERVICE_RELEASE_SCHEMA_VERSION})
SERVICE_RELEASE_COMPATIBILITY_POLICY: Final = JOURNEY_RELEASE_COMPATIBILITY_POLICY
SERVICE_RELEASE_SIGNATURE_ALGORITHM: Final = JOURNEY_RELEASE_SIGNATURE_ALGORITHM
SERVICE_RELEASE_READY: Final = "READY"
SERVICE_RELEASE_NOT_READY: Final = "NOT_READY"
SERVICE_RELEASE_TARGET: Final = "v4.0"
SERVICE_RELEASE_SIGNING_ENV: Final = "OPENMED_SERVICE_RELEASE_KEY"

P0_CHECKLIST_SCHEMA: Final = "openmed.service_release.p0_checklist.v1"
P0_STATUS_SCHEMA: Final = "openmed.service_release.p0_status.v1"
FAILURE_CODE_INVENTORY_SCHEMA: Final = "openmed.service_release.failure_codes.v1"

SECURITY_PROFILE_REPORT: Final = "security_profile_report"
NO_EGRESS_ATTESTATION: Final = "no_egress_attestation"
RECOVERY_DRILL_REPORT: Final = "recovery_drill_report"
MIGRATION_REHEARSAL_MATRIX: Final = "migration_rehearsal_matrix"
PROTOCOL_MAP: Final = "protocol_map"
FIXTURE: Final = "fixture"
FAILURE_CODE_INVENTORY: Final = "failure_code_inventory"
P0_STATUS: Final = "p0_status"
THREAT_MODEL: Final = "threat_model"
# Canonical order: packet evidence is sorted by this order, then by id.
EVIDENCE_KINDS: Final = (
    SECURITY_PROFILE_REPORT,
    NO_EGRESS_ATTESTATION,
    RECOVERY_DRILL_REPORT,
    MIGRATION_REHEARSAL_MATRIX,
    PROTOCOL_MAP,
    FIXTURE,
    FAILURE_CODE_INVENTORY,
    P0_STATUS,
    THREAT_MODEL,
)
# Every kind is release-blocking: a kind without an entry is ``not_present``.
REQUIRED_EVIDENCE_KINDS: Final = EVIDENCE_KINDS
SINGLE_ENTRY_KINDS: Final = frozenset(set(EVIDENCE_KINDS) - {FIXTURE})
# The issue that owns each kind's producer.
EVIDENCE_PRODUCERS: Final[Mapping[str, str]] = MappingProxyType(
    {
        SECURITY_PROFILE_REPORT: "#3552",
        NO_EGRESS_ATTESTATION: "#3551",
        RECOVERY_DRILL_REPORT: "#3603",
        MIGRATION_REHEARSAL_MATRIX: "#3578",
        PROTOCOL_MAP: "#3548",
        FIXTURE: "#3603",
        FAILURE_CODE_INVENTORY: "#3603",
        P0_STATUS: "#3603",
        THREAT_MODEL: "#3603",
    }
)
# Kinds bound by digest alone; every other kind needs a typed loader.
DIGEST_ONLY_KINDS: Final = frozenset({FIXTURE, THREAT_MODEL})

EVIDENCE_VERIFIED: Final = "verified"
EVIDENCE_BLOCKED: Final = "blocked"
KIND_PRESENT: Final = "present"
KIND_NOT_PRESENT: Final = "not_present"
KIND_BLOCKED: Final = "blocked"
LOADER_TYPED: Final = "typed"
LOADER_DIGEST_ONLY: Final = "digest_only"
LOADER_UNAVAILABLE: Final = "unavailable"
P0_COMPLETE: Final = "complete"
P0_INCOMPLETE: Final = "incomplete"
ISSUE_CLOSED: Final = "closed"
ISSUE_OPEN: Final = "open"
ISSUE_UNKNOWN: Final = "unknown"

MOVED_ITEM_PRIORITIES: Final = ("P0", "P1", "P2")
MOVED_ITEM_RATIONALES: Final = (
    "blocked_by_dependency",
    "evidence_incomplete",
    "hardware_unavailable",
    "out_of_scope",
    "risk_not_accepted",
    "toolchain_unavailable",
    "upstream_unreleased",
)
SERVICE_RELEASE_LIMITATIONS: Final = (
    "no_autonomous_clinical_action",
    "no_certification_claim",
    "no_publication_authorization",
    "not_a_medical_device",
    "not_clinically_validated",
    "requires_local_validation",
)
FAILURE_CODE_SURFACES: Final = (
    "recovery_drill",
    "release_gate",
    "release_verifier",
    "service_error",
)

# Value classes the sentinel scan rejects anywhere in evidence or the packet.
SENTINEL_PATTERNS: Final[Mapping[str, re.Pattern[str]]] = MappingProxyType(
    {
        "absolute_user_path": re.compile(
            r"(?:/Users/|/home/|/root/|/private/var/|[A-Za-z]:\\\\Users\\\\)"
        ),
        "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "bearer_token": re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}"),
        "email_address": re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b"
        ),
        "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
        "private_key_block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        "us_ssn": re.compile(r"\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b"),
    }
)
PLANTED_SENTINEL: Final = "planted_sentinel"

# Root reason codes the gate and verifier can emit. Emitted codes may carry a
# ``:<kind>``, ``:<issue>`` or ``:<entry id>`` suffix.
SERVICE_RELEASE_REASON_CODES: Final = (
    "bundled_schema_invalid",
    "candidate_commit_invalid",
    "candidate_mismatch",
    "checkout_commit_mismatch",
    "checkout_unresolved",
    "compatibility_policy_unsupported",
    "decision_inconsistent",
    "decision_not_ready",
    "drill_category_missing",
    "drill_digest_mismatch",
    "drill_incomplete",
    "drill_objective_missed",
    "drill_objective_not_measured",
    "drill_report_inconsistent",
    "drill_report_invalid",
    "drill_step_failed",
    "drill_step_not_available",
    "evaluated_at_invalid",
    "evidence_candidate_invalid",
    "evidence_count_mismatch",
    "evidence_digest_invalid",
    "evidence_entry_invalid",
    "evidence_expired",
    "evidence_from_future",
    "evidence_generated_at_invalid",
    "evidence_id_duplicate",
    "evidence_id_invalid",
    "evidence_invalid",
    "evidence_kind_missing",
    "evidence_kind_repeated",
    "evidence_kind_unknown",
    "evidence_loader_failed",
    "evidence_missing",
    "evidence_not_present",
    "evidence_order_invalid",
    "evidence_path_invalid",
    "evidence_too_large",
    "evidence_unreadable",
    "expected_candidate_invalid",
    "failure_code_inventory_incomplete",
    "failure_code_inventory_invalid",
    "fixture_path_not_synthetic",
    "key_id_invalid",
    "loader_code_invalid",
    "loader_kind_unknown",
    "loader_summary_invalid",
    "manifest_fields_invalid",
    "manifest_not_object",
    "manifest_too_large",
    "manifest_unreadable",
    "moved_item_claims_invalid",
    "moved_item_destination_invalid",
    "moved_item_invalid",
    "moved_item_issue_duplicate",
    "moved_item_issue_invalid",
    "moved_item_priority_invalid",
    "moved_item_rationale_invalid",
    "moved_item_rationale_missing",
    "moved_item_residual_risk_invalid",
    "moved_item_residual_risk_missing",
    "moved_item_weakens_invariant",
    "moved_items_invalid",
    "output_exists",
    "output_unwritable",
    "p0_checklist_invalid",
    "p0_checklist_mismatch",
    "p0_checklist_release_mismatch",
    "p0_evidence_missing",
    "p0_moved",
    "p0_open",
    "p0_status_invalid",
    "p0_status_missing",
    "p0_status_unverifiable",
    "packet_digest_mismatch",
    "packet_kind_unknown",
    "packet_malformed",
    "packet_too_large",
    "packet_unreadable",
    "policy_invalid",
    "release_fields_invalid",
    "release_mismatch",
    "release_target_invalid",
    "release_target_unsupported",
    "release_version_invalid",
    "sentinel_detected",
    "sentinel_invalid",
    "service_release_invalid",
    "signature_algorithm_unsupported",
    "signature_invalid",
    "signature_missing",
    "signing_key_invalid",
    "signing_key_missing",
    "signing_key_unreadable",
    "source_date_epoch_invalid",
    "stale_digest",
    "typed_loader_unavailable",
    "unknown_schema",
    "wrong_candidate",
)

MAX_EVIDENCE_ENTRIES: Final = 512
MAX_MOVED_ITEMS: Final = 512
MAX_TYPED_EVIDENCE_BYTES: Final = 16 * 1024 * 1024
MAX_MANIFEST_BYTES: Final = 4 * 1024 * 1024
MAX_PACKET_BYTES: Final = 16 * 1024 * 1024
_MAX_AGE_SECONDS: Final = 31_536_000
_MAX_SUMMARY_INT: Final = 2**53

_REPRODUCTION_COMMAND: Final = (
    "python scripts/release/service_release_gate.py --manifest "
    "service-release-manifest.json --output service-release-packet.json"
)
_SCHEMA_PACKAGE: Final = "openmed.core.schemas.json"
_P0_CHECKLIST_RESOURCE: Final = ("data", "service_release_p0.json")
_ZERO_COMMIT: Final = "0" * 40
_ZERO_DIGEST: Final = "sha256:" + "0" * 64
_SYNTHETIC_FIXTURE_ROOTS: Final = ("tests/fixtures/", "fixtures/")
_COMMIT_RE: Final = re.compile(r"[0-9a-f]{40}\Z")
_DIGEST_RE: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ID_RE: Final = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\Z")
_CODE_RE: Final = re.compile(r"[a-z0-9][a-z0-9_.:#-]{0,159}\Z")
_PLAIN_CODE_RE: Final = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_ISSUE_RE: Final = re.compile(r"#[1-9][0-9]{0,6}\Z")
_MILESTONE_RE: Final = re.compile(r"v([0-9]{1,3})\.([0-9]{1,3})(?:\.([0-9]{1,3}))?\Z")
_SEMVER_RE: Final = re.compile(
    r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z.-]{1,32})?\Z"
)
_PORTABLE_PATH_RE: Final = re.compile(r"[A-Za-z0-9_.][A-Za-z0-9_./+@-]{0,511}\Z")
_SUMMARY_KEY_RE: Final = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_SUMMARY_TEXT_RE: Final = re.compile(r"[a-z0-9#][a-z0-9_.:#-]{0,63}\Z")

_MANIFEST_FIELDS: Final = frozenset(
    {
        "compatibility_policy",
        "evidence",
        "moved_items",
        "policy",
        "release",
        "schema_version",
    }
)
_RELEASE_FIELDS: Final = frozenset(
    {"candidate_commit", "evaluated_at", "source_date_epoch", "target", "version"}
)
_POLICY_FIELDS: Final = frozenset({"max_evidence_age_seconds"})
_ENTRY_FIELDS: Final = frozenset(
    {"candidate_commit", "generated_at", "id", "kind", "path", "sha256"}
)
_MOVED_FIELDS: Final = frozenset(
    {
        "destination",
        "issue",
        "narrowed_claims",
        "priority",
        "rationale",
        "residual_risk",
    }
)
_CHECKLIST_FIELDS: Final = frozenset(
    {
        "epic",
        "items",
        "protected_invariants",
        "release",
        "release_plan",
        "retargeted_from",
        "schema",
    }
)
_CHECKLIST_ITEM_FIELDS: Final = frozenset({"control", "issue", "required_evidence"})
_PACKET_FIELDS: Final = frozenset(
    {
        "blocking_codes",
        "compatibility_policy",
        "decision",
        "evidence",
        "evidence_kinds",
        "limitations",
        "manifest_digest",
        "moved_items",
        "p0_checklist",
        "p0_items",
        "packet_digest",
        "release",
        "repository_binding",
        "reproduction",
        "schema_version",
        "sentinel_scan",
        "signature",
    }
)


class ServiceReleaseError(ValueError):
    """Raised when service release input cannot produce a safe decision.

    Attributes:
        codes: Value-free reason codes. They name fields, evidence kinds and
            controlled identifiers only, never submitted values or paths.
    """

    def __init__(self, *codes: str) -> None:
        self.codes = tuple(codes) or ("service_release_invalid",)
        super().__init__("; ".join(self.codes))


@dataclass(frozen=True)
class ServiceEvidenceSource:
    """One digest-checked evidence file handed to an evidence loader.

    The source never exposes the evidence path. ``data`` holds the exact bytes
    that were digested when the file is at most
    :data:`MAX_TYPED_EVIDENCE_BYTES`; larger files are digested in chunks and
    their bytes are not retained.

    Attributes:
        entry_id: The manifest evidence identifier.
        kind: The evidence kind.
        digest: Prefixed SHA-256 digest of the file.
        byte_size: File size in bytes.
        release: The manifest release: ``version``, ``target`` and
            ``candidate_commit``.
        data: The digested bytes, or ``None`` for oversized files.
    """

    entry_id: str
    kind: str
    digest: str
    byte_size: int
    release: Mapping[str, str]
    data: bytes | None = field(default=None, repr=False)

    def read_json(self) -> Any:
        """Decode the evidence as strict JSON.

        Returns:
            The decoded JSON value.

        Raises:
            ServiceReleaseError: ``evidence_too_large`` when the bytes were not
                retained, or ``evidence_unreadable`` when they are not strict
                UTF-8 JSON without duplicate keys or non-finite numbers.
        """

        if self.data is None:
            raise ServiceReleaseError("evidence_too_large")
        return _decode_json(self.data, "evidence_unreadable")


@dataclass(frozen=True)
class ServiceEvidenceResult:
    """Value-free outcome of one evidence loader.

    Attributes:
        blocking_codes: Bare reason codes; the gate suffixes the evidence id.
        summary: Code-keyed counts, booleans, codes or digests for the packet.
        document: Parsed document kept in memory for cross-checks only. It
            never enters the packet.
        loader: ``typed``, ``digest_only`` or ``unavailable``.
    """

    blocking_codes: tuple[str, ...] = ()
    summary: Mapping[str, str | int | bool] = field(default_factory=dict)
    document: Any = field(default=None, repr=False, compare=False)
    loader: str = LOADER_TYPED


ServiceEvidenceLoader = Callable[[ServiceEvidenceSource], ServiceEvidenceResult]


@dataclass(frozen=True)
class ServiceReleaseVerification:
    """Offline verification outcome for a signed service release packet.

    Attributes:
        valid: True when the packet is authentic and internally consistent
            and no requested expectation failed.
        authentic: True when the packet digest and HMAC signature verify.
        decision: The packet's recorded decision, or ``NOT_READY``.
        reason_codes: Sorted, value-free failure codes.
    """

    valid: bool
    authentic: bool
    decision: str
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic JSON representation."""

        return {
            "authentic": self.authentic,
            "decision": self.decision,
            "reason_codes": list(self.reason_codes),
            "valid": self.valid,
        }


@dataclass(frozen=True)
class ServiceReleasePacket:
    """Signed, reproducible and value-free v4.0 service release decision."""

    release: Mapping[str, Any]
    decision: str
    repository_binding: Mapping[str, Any]
    evidence: tuple[Mapping[str, Any], ...]
    evidence_kinds: tuple[Mapping[str, Any], ...]
    p0_checklist: Mapping[str, Any]
    p0_items: tuple[Mapping[str, Any], ...]
    moved_items: tuple[Mapping[str, Any], ...]
    sentinel_scan: Mapping[str, Any]
    blocking_codes: tuple[str, ...]
    manifest_digest: str
    reproduction: Mapping[str, Any]
    limitations: tuple[str, ...] = SERVICE_RELEASE_LIMITATIONS
    schema_version: str = SERVICE_RELEASE_SCHEMA_VERSION
    compatibility_policy: str = SERVICE_RELEASE_COMPATIBILITY_POLICY
    packet_digest: str = ""
    signature: AuditSignature | None = None

    def _unsigned_payload(self) -> dict[str, Any]:
        return {
            "blocking_codes": list(self.blocking_codes),
            "compatibility_policy": self.compatibility_policy,
            "decision": self.decision,
            "evidence": [_plain(item) for item in self.evidence],
            "evidence_kinds": [_plain(item) for item in self.evidence_kinds],
            "limitations": list(self.limitations),
            "manifest_digest": self.manifest_digest,
            "moved_items": [_plain(item) for item in self.moved_items],
            "p0_checklist": _plain(self.p0_checklist),
            "p0_items": [_plain(item) for item in self.p0_items],
            "release": _plain(self.release),
            "repository_binding": _plain(self.repository_binding),
            "reproduction": _plain(self.reproduction),
            "schema_version": self.schema_version,
            "sentinel_scan": _plain(self.sentinel_scan),
        }

    def recompute_packet_digest(self) -> str:
        """Recompute the canonical digest without trusting the stored value."""

        return stable_hash(self._unsigned_payload())

    def sign(self, key: bytes | str, *, key_id: str) -> ServiceReleasePacket:
        """Return a copy signed over every packet field and its digest.

        Args:
            key: Caller-supplied HMAC key of at least 32 bytes.
            key_id: Bounded key identifier recorded instead of the key.

        Returns:
            The signed packet.

        Raises:
            ServiceReleaseError: If the key is too short or the key id is not a
                bounded code.
        """

        if not isinstance(key_id, str) or not _CODE_RE.fullmatch(key_id):
            raise ServiceReleaseError("key_id_invalid")
        packet_digest = self.recompute_packet_digest()
        signed = {**self._unsigned_payload(), "packet_digest": packet_digest}
        try:
            value = _release_hmac(key, signed)
        except JourneyReleaseError:
            raise ServiceReleaseError("signing_key_invalid") from None
        return replace(
            self,
            packet_digest=packet_digest,
            signature=AuditSignature(
                key_id=key_id,
                algorithm=SERVICE_RELEASE_SIGNATURE_ALGORITHM,
                value=value,
            ),
        )

    def verify(self, key: bytes | str) -> bool:
        """Return whether the packet is authentic and internally consistent.

        A signed ``NOT_READY`` packet verifies; use
        :func:`verify_service_release_packet` with ``require_ready=True`` to
        also require a ``READY`` decision.
        """

        return verify_service_release_packet(self.to_dict(), key).valid

    def to_dict(self) -> dict[str, Any]:
        """Return the complete deterministic JSON representation."""

        return {
            **self._unsigned_payload(),
            "packet_digest": self.packet_digest,
            "signature": self.signature.to_dict() if self.signature else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ServiceReleasePacket:
        """Restore a packet for independent offline verification.

        Raises:
            ServiceReleaseError: ``packet_malformed`` when a field is missing
                or has the wrong JSON type.
        """

        if not isinstance(payload, Mapping) or set(payload) != _PACKET_FIELDS:
            raise ServiceReleaseError("packet_malformed")
        signature = payload["signature"]
        if signature is not None and not isinstance(signature, Mapping):
            raise ServiceReleaseError("packet_malformed")
        return cls(
            release=_packet_mapping(payload["release"]),
            decision=_packet_str(payload["decision"]),
            repository_binding=_packet_mapping(payload["repository_binding"]),
            evidence=_packet_rows(payload["evidence"]),
            evidence_kinds=_packet_rows(payload["evidence_kinds"]),
            p0_checklist=_packet_mapping(payload["p0_checklist"]),
            p0_items=_packet_rows(payload["p0_items"]),
            moved_items=_packet_rows(payload["moved_items"]),
            sentinel_scan=_packet_mapping(payload["sentinel_scan"]),
            blocking_codes=_packet_codes(payload["blocking_codes"]),
            manifest_digest=_packet_str(payload["manifest_digest"]),
            reproduction=_packet_mapping(payload["reproduction"]),
            limitations=_packet_codes(payload["limitations"]),
            schema_version=_packet_str(payload["schema_version"]),
            compatibility_policy=_packet_str(payload["compatibility_policy"]),
            packet_digest=_packet_str(payload["packet_digest"]),
            signature=(
                AuditSignature.from_dict(signature) if signature is not None else None
            ),
        )


# --- P0 checklist ------------------------------------------------------------


def validate_service_p0_checklist(checklist: Any) -> dict[str, Any]:
    """Validate and normalize a P0 checklist.

    Args:
        checklist: A decoded checklist (``openmed.service_release.p0_checklist.v1``).

    Returns:
        The normalized checklist with items sorted by issue number.

    Raises:
        ServiceReleaseError: ``unknown_schema`` or ``p0_checklist_invalid``.
    """

    if not isinstance(checklist, Mapping):
        raise ServiceReleaseError("p0_checklist_invalid")
    if checklist.get("schema") != P0_CHECKLIST_SCHEMA:
        raise ServiceReleaseError("unknown_schema")
    if set(checklist) != _CHECKLIST_FIELDS:
        raise ServiceReleaseError("p0_checklist_invalid")
    release = checklist["release"]
    if not _is_milestone(release) or not _is_milestone(checklist["retargeted_from"]):
        raise ServiceReleaseError("p0_checklist_invalid")
    if not _is_issue(checklist["epic"]) or not _is_issue(checklist["release_plan"]):
        raise ServiceReleaseError("p0_checklist_invalid")
    invariants = checklist["protected_invariants"]
    if (
        not isinstance(invariants, list)
        or not invariants
        or not all(_is_plain_code(item) for item in invariants)
        or len(set(invariants)) != len(invariants)
    ):
        raise ServiceReleaseError("p0_checklist_invalid")
    raw_items = checklist["items"]
    if not isinstance(raw_items, list) or not raw_items:
        raise ServiceReleaseError("p0_checklist_invalid")
    items: list[dict[str, Any]] = []
    issues: set[str] = set()
    for item in raw_items:
        if not isinstance(item, Mapping) or set(item) != _CHECKLIST_ITEM_FIELDS:
            raise ServiceReleaseError("p0_checklist_invalid")
        issue, control, kinds = (
            item["issue"],
            item["control"],
            item["required_evidence"],
        )
        if not _is_issue(issue) or issue in issues or not _is_plain_code(control):
            raise ServiceReleaseError("p0_checklist_invalid")
        if (
            not isinstance(kinds, list)
            or not kinds
            or not all(
                isinstance(kind, str) and kind in EVIDENCE_KINDS for kind in kinds
            )
            or len(set(kinds)) != len(kinds)
        ):
            raise ServiceReleaseError("p0_checklist_invalid")
        issues.add(issue)
        items.append(
            {
                "control": control,
                "issue": issue,
                "required_evidence": sorted(kinds, key=EVIDENCE_KINDS.index),
            }
        )
    return {
        "epic": checklist["epic"],
        "items": sorted(items, key=lambda row: _issue_number(row["issue"])),
        "protected_invariants": sorted(invariants),
        "release": release,
        "release_plan": checklist["release_plan"],
        "retargeted_from": checklist["retargeted_from"],
        "schema": P0_CHECKLIST_SCHEMA,
    }


def load_service_p0_checklist(path: str | Path | None = None) -> dict[str, Any]:
    """Load the committed P0 checklist, or one at *path*.

    Args:
        path: Optional checklist file. Defaults to the packaged
            ``openmed/eval/data/service_release_p0.json``.

    Returns:
        The normalized checklist.

    Raises:
        ServiceReleaseError: ``p0_checklist_invalid`` or ``unknown_schema``.
    """

    try:
        if path is None:
            resource = resources.files("openmed.eval").joinpath(*_P0_CHECKLIST_RESOURCE)
            data = resource.read_bytes()
        else:
            data = Path(path).read_bytes()
    except OSError:
        raise ServiceReleaseError("p0_checklist_invalid") from None
    return validate_service_p0_checklist(_decode_json(data, "p0_checklist_invalid"))


def service_p0_checklist_digest(checklist: Mapping[str, Any]) -> str:
    """Return the canonical digest of a normalized P0 checklist."""

    return stable_hash(validate_service_p0_checklist(checklist))


# --- evidence producers available on master ----------------------------------


def required_failure_codes() -> tuple[str, ...]:
    """Return every controlled code the inventory must declare.

    Returns:
        Sorted ``surface:code`` pairs for the release gate, its verifier, the
        recovery drill and the REST service's stable error categories.
    """

    from openmed.service.api_compatibility import STABLE_ERROR_CATEGORIES

    pairs = {f"release_gate:{code}" for code in SERVICE_RELEASE_REASON_CODES}
    pairs.update(f"release_verifier:{code}" for code in _VERIFIER_CODES)
    pairs.update(
        f"recovery_drill:{code}"
        for code in (*DRILL_CHECK_CODES, *DRILL_REPORT_CODES, *DRILL_ERROR_CODES)
    )
    pairs.update(f"service_error:{code}" for code in STABLE_ERROR_CATEGORIES)
    return tuple(sorted(pairs))


def build_failure_code_inventory(candidate_commit: str) -> dict[str, Any]:
    """Build the controlled failure-code inventory for a candidate.

    Args:
        candidate_commit: The full candidate commit.

    Returns:
        An ``openmed.service_release.failure_codes.v1`` document.

    Raises:
        ServiceReleaseError: ``candidate_commit_invalid``.
    """

    if not isinstance(candidate_commit, str) or not _COMMIT_RE.fullmatch(
        candidate_commit
    ):
        raise ServiceReleaseError("candidate_commit_invalid")
    codes = []
    for pair in required_failure_codes():
        surface, code = pair.split(":", 1)
        codes.append({"code": code, "surface": surface})
    return {
        "candidate_commit": candidate_commit,
        "codes": codes,
        "release": SERVICE_RELEASE_TARGET,
        "schema": FAILURE_CODE_INVENTORY_SCHEMA,
    }


def build_p0_status(
    states: Mapping[str, str],
    *,
    candidate_commit: str,
    release: str = SERVICE_RELEASE_TARGET,
) -> dict[str, Any]:
    """Build a P0 status snapshot from issue states captured offline.

    The release operator captures issue states (for example with the GitHub
    CLI) outside the gate; this function only normalizes them.

    Args:
        states: Issue reference (``#3552``) to ``closed`` or ``open``.
        candidate_commit: The full candidate commit.
        release: The release target.

    Returns:
        An ``openmed.service_release.p0_status.v1`` document.

    Raises:
        ServiceReleaseError: ``p0_status_invalid``.
    """

    if not isinstance(candidate_commit, str) or not _COMMIT_RE.fullmatch(
        candidate_commit
    ):
        raise ServiceReleaseError("candidate_commit_invalid")
    issues = []
    for issue, state in states.items():
        if not _is_issue(issue) or state not in (ISSUE_CLOSED, ISSUE_OPEN):
            raise ServiceReleaseError("p0_status_invalid")
        issues.append({"issue": issue, "state": state})
    return {
        "candidate_commit": candidate_commit,
        "issues": sorted(issues, key=lambda row: _issue_number(row["issue"])),
        "release": release,
        "schema": P0_STATUS_SCHEMA,
    }


# --- evidence loaders --------------------------------------------------------


def load_recovery_drill_evidence(
    source: ServiceEvidenceSource,
) -> ServiceEvidenceResult:
    """Load a recovery-drill report and evaluate every phase and objective.

    Args:
        source: The digest-checked report from
            :func:`openmed.eval.service_recovery_drill.run_recovery_drill`.

    Returns:
        A typed result. Codes: ``evidence_unreadable``, ``unknown_schema``,
        ``drill_report_invalid``, ``drill_digest_mismatch``,
        ``drill_category_missing:<category>``, ``wrong_candidate``,
        ``drill_step_failed``, ``drill_step_not_available``,
        ``drill_objective_missed``, ``drill_objective_not_measured`` and
        ``drill_incomplete``.
    """

    try:
        document = source.read_json()
    except ServiceReleaseError as exc:
        return ServiceEvidenceResult(exc.codes)
    problems = validate_recovery_drill_report(document)
    if problems:
        return ServiceEvidenceResult(problems)
    codes: list[str] = []
    if document["candidate_commit"] != source.release["candidate_commit"]:
        codes.append("wrong_candidate")
    summary_block = document["summary"]
    if summary_block["steps_failed"]:
        codes.append("drill_step_failed")
    if summary_block["steps_not_available"]:
        codes.append("drill_step_not_available")
    if summary_block["objectives_missed"]:
        codes.append("drill_objective_missed")
    if summary_block["objectives_not_measured"]:
        codes.append("drill_objective_not_measured")
    if document["state"] != DRILL_COMPLETE:
        codes.append("drill_incomplete")
    summary: dict[str, str | int | bool] = {
        "categories": len(DRILL_CATEGORIES),
        "objectives_met": summary_block["objectives_met"],
        "objectives_missed": summary_block["objectives_missed"],
        "objectives_not_measured": summary_block["objectives_not_measured"],
        "profile": document["profile"],
        "report_digest": document["report_digest"],
        "state": document["state"],
        "steps_failed": summary_block["steps_failed"],
        "steps_not_available": summary_block["steps_not_available"],
        "steps_passed": summary_block["steps_passed"],
    }
    return ServiceEvidenceResult(tuple(codes), summary, document=dict(document))


def load_failure_code_inventory_evidence(
    source: ServiceEvidenceSource,
) -> ServiceEvidenceResult:
    """Load the failure-code inventory and check it is complete and controlled.

    Args:
        source: The digest-checked inventory.

    Returns:
        A typed result. Codes: ``evidence_unreadable``, ``unknown_schema``,
        ``failure_code_inventory_invalid``, ``wrong_candidate``,
        ``release_mismatch`` and ``failure_code_inventory_incomplete``.
    """

    try:
        document = source.read_json()
    except ServiceReleaseError as exc:
        return ServiceEvidenceResult(exc.codes)
    if (
        not isinstance(document, Mapping)
        or document.get("schema") != FAILURE_CODE_INVENTORY_SCHEMA
    ):
        return ServiceEvidenceResult(("unknown_schema",))
    rows = document.get("codes")
    if (
        set(document) != {"candidate_commit", "codes", "release", "schema"}
        or not isinstance(rows, list)
        or not rows
        or not all(
            isinstance(row, Mapping)
            and set(row) == {"code", "surface"}
            and row["surface"] in FAILURE_CODE_SURFACES
            and _is_plain_code(row["code"])
            for row in rows
        )
    ):
        return ServiceEvidenceResult(("failure_code_inventory_invalid",))
    pairs = [f"{row['surface']}:{row['code']}" for row in rows]
    codes: list[str] = []
    if pairs != sorted(set(pairs)):
        codes.append("failure_code_inventory_invalid")
    if document["candidate_commit"] != source.release["candidate_commit"]:
        codes.append("wrong_candidate")
    if document["release"] != source.release["target"]:
        codes.append("release_mismatch")
    missing = set(required_failure_codes()) - set(pairs)
    if missing:
        codes.append("failure_code_inventory_incomplete")
    by_surface = {surface: 0 for surface in FAILURE_CODE_SURFACES}
    for row in rows:
        by_surface[row["surface"]] += 1
    summary: dict[str, str | int | bool] = {
        "code_count": len(rows),
        "missing_code_count": len(missing),
        **{f"{surface}_codes": count for surface, count in by_surface.items()},
    }
    return ServiceEvidenceResult(tuple(codes), summary, document=dict(document))


def load_p0_status_evidence(source: ServiceEvidenceSource) -> ServiceEvidenceResult:
    """Load a P0 status snapshot for the checklist cross-check.

    Args:
        source: The digest-checked snapshot.

    Returns:
        A typed result. Codes: ``evidence_unreadable``, ``unknown_schema``,
        ``p0_status_invalid``, ``wrong_candidate`` and ``release_mismatch``.
    """

    try:
        document = source.read_json()
    except ServiceReleaseError as exc:
        return ServiceEvidenceResult(exc.codes)
    if not isinstance(document, Mapping) or document.get("schema") != P0_STATUS_SCHEMA:
        return ServiceEvidenceResult(("unknown_schema",))
    rows = document.get("issues")
    if (
        set(document) != {"candidate_commit", "issues", "release", "schema"}
        or not isinstance(rows, list)
        or not all(
            isinstance(row, Mapping)
            and set(row) == {"issue", "state"}
            and _is_issue(row["issue"])
            and row["state"] in (ISSUE_CLOSED, ISSUE_OPEN)
            for row in rows
        )
        or len({row["issue"] for row in rows}) != len(rows)
    ):
        return ServiceEvidenceResult(("p0_status_invalid",))
    codes: list[str] = []
    if document["candidate_commit"] != source.release["candidate_commit"]:
        codes.append("wrong_candidate")
    if document["release"] != source.release["target"]:
        codes.append("release_mismatch")
    states = [row["state"] for row in rows]
    summary: dict[str, str | int | bool] = {
        "closed_count": states.count(ISSUE_CLOSED),
        "issue_count": len(rows),
        "open_count": states.count(ISSUE_OPEN),
    }
    return ServiceEvidenceResult(tuple(codes), summary, document=dict(document))


def load_digest_only_evidence(source: ServiceEvidenceSource) -> ServiceEvidenceResult:
    """Bind evidence by digest alone, without parsing it.

    This is the default for synthetic fixtures and the threat model.

    Args:
        source: The digest-checked evidence file.

    Returns:
        A digest-only result without codes.
    """

    del source
    return ServiceEvidenceResult(loader=LOADER_DIGEST_ONLY)


def load_unavailable_evidence(source: ServiceEvidenceSource) -> ServiceEvidenceResult:
    """Block evidence whose typed loader has not landed on this checkout.

    The security-profile report, no-egress attestation, migration rehearsal
    matrix and protocol map are produced by open work (#3552, #3551, #3578
    and #3548). Their digests are bound, but the kind stays blocked until a
    typed loader is injected for it.

    Args:
        source: The digest-checked evidence file.

    Returns:
        A result with ``typed_loader_unavailable``.
    """

    del source
    return ServiceEvidenceResult(
        ("typed_loader_unavailable",), loader=LOADER_UNAVAILABLE
    )


DEFAULT_EVIDENCE_LOADERS: Final[Mapping[str, ServiceEvidenceLoader]] = MappingProxyType(
    {
        SECURITY_PROFILE_REPORT: load_unavailable_evidence,
        NO_EGRESS_ATTESTATION: load_unavailable_evidence,
        RECOVERY_DRILL_REPORT: load_recovery_drill_evidence,
        MIGRATION_REHEARSAL_MATRIX: load_unavailable_evidence,
        PROTOCOL_MAP: load_unavailable_evidence,
        FIXTURE: load_digest_only_evidence,
        FAILURE_CODE_INVENTORY: load_failure_code_inventory_evidence,
        P0_STATUS: load_p0_status_evidence,
        THREAT_MODEL: load_digest_only_evidence,
    }
)


# --- sentinel scan -------------------------------------------------------------


def scan_for_sentinels(
    data: bytes | str,
    *,
    sentinels: Sequence[str] = (),
) -> tuple[str, ...]:
    """Return the prohibited value classes found in *data*.

    Args:
        data: Text or bytes to scan. Bytes are decoded as UTF-8, replacing
            undecodable sequences.
        sentinels: Planted synthetic sentinel strings that must never appear.

    Returns:
        Sorted class names from :data:`SENTINEL_PATTERNS`, plus
        ``planted_sentinel`` when a planted value occurs. Matched values are
        never returned.
    """

    text = data.decode("utf-8", "replace") if isinstance(data, bytes) else data
    found = {
        name for name, pattern in SENTINEL_PATTERNS.items() if pattern.search(text)
    }
    if any(value and value in text for value in sentinels):
        found.add(PLANTED_SENTINEL)
    return tuple(sorted(found))


# --- manifest ----------------------------------------------------------------


def validate_service_release_manifest(manifest: Any) -> dict[str, Any]:
    """Validate and normalize a service release manifest.

    Structural problems raise; release-readiness problems (missing evidence,
    stale digests, wrong candidates, open or moved P0 items) are left for
    :func:`evaluate_service_release` to report as blocking codes.

    Args:
        manifest: A decoded manifest.

    Returns:
        The normalized manifest with evidence sorted canonically.

    Raises:
        ServiceReleaseError: With a value-free code such as ``unknown_schema``.
    """

    if not isinstance(manifest, Mapping):
        raise ServiceReleaseError("manifest_not_object")
    if manifest.get("schema_version") not in _SUPPORTED_SCHEMA_VERSIONS:
        raise ServiceReleaseError("unknown_schema")
    if set(manifest) != _MANIFEST_FIELDS:
        raise ServiceReleaseError("manifest_fields_invalid")
    if manifest["compatibility_policy"] != SERVICE_RELEASE_COMPATIBILITY_POLICY:
        raise ServiceReleaseError("compatibility_policy_unsupported")
    return {
        "compatibility_policy": SERVICE_RELEASE_COMPATIBILITY_POLICY,
        "evidence": _validate_evidence(manifest["evidence"]),
        "moved_items": _validate_moved_items(manifest["moved_items"]),
        "policy": _validate_policy(manifest["policy"]),
        "release": _validate_release(manifest["release"]),
        "schema_version": SERVICE_RELEASE_SCHEMA_VERSION,
    }


def load_service_release_manifest(path: str | Path) -> dict[str, Any]:
    """Read and validate a manifest, rejecting duplicate keys.

    Args:
        path: Manifest file.

    Returns:
        The normalized manifest.

    Raises:
        ServiceReleaseError: ``manifest_unreadable`` or a validation code.
    """

    source = Path(path)
    try:
        if source.stat().st_size > MAX_MANIFEST_BYTES:
            raise ServiceReleaseError("manifest_too_large")
        data = source.read_bytes()
    except OSError:
        raise ServiceReleaseError("manifest_unreadable") from None
    return validate_service_release_manifest(_decode_json(data, "manifest_unreadable"))


def load_service_release_manifest_schema() -> dict[str, Any]:
    """Load the bundled strict manifest JSON Schema."""

    return _load_bundled_schema("service_release_manifest.schema.json")


def load_service_release_packet_schema() -> dict[str, Any]:
    """Load the bundled signed-packet JSON Schema."""

    return _load_bundled_schema("service_release_packet.schema.json")


def load_service_recovery_drill_schema() -> dict[str, Any]:
    """Load the bundled recovery-drill report JSON Schema."""

    return _load_bundled_schema("service_recovery_drill.schema.json")


def load_service_release_packet(path: str | Path) -> dict[str, Any]:
    """Read a packet file as strict JSON for offline verification.

    Args:
        path: Packet file.

    Returns:
        The decoded packet payload.

    Raises:
        ServiceReleaseError: ``packet_unreadable``, ``packet_too_large`` or
            ``packet_malformed``.
    """

    source = Path(path)
    try:
        if source.stat().st_size > MAX_PACKET_BYTES:
            raise ServiceReleaseError("packet_too_large")
        data = source.read_bytes()
    except OSError:
        raise ServiceReleaseError("packet_unreadable") from None
    payload = _decode_json(data, "packet_unreadable")
    if not isinstance(payload, dict):
        raise ServiceReleaseError("packet_malformed")
    return payload


# --- evaluation --------------------------------------------------------------


def evaluate_service_release(
    manifest: Mapping[str, Any],
    *,
    repo_root: str | Path,
    signing_key: bytes | str,
    key_id: str = "service-release",
    loaders: Mapping[str, ServiceEvidenceLoader] | None = None,
    p0_checklist: Mapping[str, Any] | None = None,
    sentinels: Sequence[str] = (),
    resolve_head: Callable[[Path], str] | None = None,
) -> ServiceReleasePacket:
    """Evaluate and sign one v4.0 service release manifest.

    Every evidence path is read beneath ``repo_root`` without following
    symlinks, digested, scanned for sentinels and handed to the loader for its
    kind. The checkout must be at the manifest's candidate commit.

    Args:
        manifest: A decoded manifest (see the bundled manifest schema).
        repo_root: The candidate checkout.
        signing_key: Caller-supplied HMAC key of at least 32 bytes.
        key_id: Bounded key identifier recorded in the signature.
        loaders: Optional loaders by evidence kind; they replace the defaults
            in :data:`DEFAULT_EVIDENCE_LOADERS` for those kinds.
        p0_checklist: Optional checklist; defaults to the committed one.
        sentinels: Planted synthetic sentinel strings; only their count and
            digest enter the packet.
        resolve_head: Optional resolver for the checkout's ``HEAD`` commit.
            Defaults to ``git rev-parse HEAD``.

    Returns:
        The signed packet. Its decision is ``READY`` only when every required
        kind is present, every P0 is closed with its evidence, and every check
        passes.

    Raises:
        ServiceReleaseError: If the manifest, checklist, loaders or key are
            unusable.
    """

    normalized = validate_service_release_manifest(manifest)
    checklist = (
        load_service_p0_checklist()
        if p0_checklist is None
        else validate_service_p0_checklist(p0_checklist)
    )
    release = normalized["release"]
    if checklist["release"] != release["target"]:
        raise ServiceReleaseError("p0_checklist_release_mismatch")
    _key_or_raise(signing_key)
    planted = tuple(_validated_sentinels(sentinels))
    registry = _loader_registry(loaders)
    root = Path(repo_root).resolve()
    evaluated_at = _timestamp(release["evaluated_at"], "evaluated_at_invalid")
    binding = _repository_binding(root, release["candidate_commit"], resolve_head)
    loader_release = {
        key: release[key] for key in ("candidate_commit", "target", "version")
    }

    rows: list[dict[str, Any]] = []
    results: dict[str, ServiceEvidenceResult] = {}
    for entry in normalized["evidence"]:
        row, result = _evaluate_entry(
            root,
            entry,
            loader_release,
            max_age=normalized["policy"]["max_evidence_age_seconds"],
            evaluated_at=evaluated_at,
            registry=registry,
            sentinels=planted,
        )
        rows.append(row)
        if result is not None:
            results[entry["id"]] = result

    kinds = _kind_rows(rows)
    status = _single_document(rows, results, P0_STATUS)
    p0_rows = _p0_rows(checklist, kinds, status)
    moved = _moved_rows(normalized["moved_items"], checklist, release["target"])

    codes: set[str] = set(binding["blocking_codes"])
    for row in rows:
        codes.update(f"{code}:{row['id']}" for code in row["blocking_codes"])
    for table in (kinds, p0_rows, moved):
        for item in table:
            codes.update(item["blocking_codes"])
    entry_findings = sum(
        1
        for row in rows
        for code in row["blocking_codes"]
        if code.startswith("sentinel")
    )
    checklist_summary = {
        "digest": stable_hash(checklist),
        "item_count": len(checklist["items"]),
        "protected_invariants": list(checklist["protected_invariants"]),
        "release": checklist["release"],
        "retargeted_from": checklist["retargeted_from"],
    }
    packet = ServiceReleasePacket(
        release=dict(release),
        decision=SERVICE_RELEASE_NOT_READY,
        repository_binding=binding,
        evidence=tuple(rows),
        evidence_kinds=tuple(kinds),
        p0_checklist=checklist_summary,
        p0_items=tuple(p0_rows),
        moved_items=tuple(moved),
        sentinel_scan={},
        blocking_codes=(),
        manifest_digest=stable_hash(normalized),
        reproduction={
            "command": _REPRODUCTION_COMMAND,
            "evidence_count": len(rows),
            "moved_item_count": len(moved),
        },
    )
    # The final packet is scanned too; matches are recorded by class only.
    packet_classes = scan_for_sentinels(
        _canonical_json(packet._unsigned_payload()), sentinels=planted
    )
    codes.update(f"sentinel_detected:{name}:packet" for name in packet_classes)
    blocking = tuple(sorted(codes))
    packet = replace(
        packet,
        decision=SERVICE_RELEASE_NOT_READY if blocking else SERVICE_RELEASE_READY,
        blocking_codes=blocking,
        sentinel_scan={
            "finding_count": entry_findings + len(packet_classes),
            "pattern_classes": sorted((*SENTINEL_PATTERNS, PLANTED_SENTINEL)),
            "planted_sentinel_count": len(planted),
            "planted_sentinel_digest": stable_hash(sorted(planted)),
            "scanned_entry_count": sum(1 for row in rows if row["scanned"]),
        },
    )
    return packet.sign(signing_key, key_id=key_id)


def verify_service_release_packet(
    payload: Mapping[str, Any],
    key: bytes | str,
    *,
    expected_candidate: str | None = None,
    expected_release: str | None = None,
    expected_p0_checklist_digest: str | None = None,
    require_ready: bool = False,
) -> ServiceReleaseVerification:
    """Verify a packet offline and report stable reason codes.

    The verifier recomputes the canonical digest and HMAC, then checks the
    packet's own structure: canonical evidence order, every required kind,
    per-kind entry counts, candidate binding, P0 rows and decision
    consistency.

    Args:
        payload: A decoded packet.
        key: The HMAC key the packet was signed with.
        expected_candidate: Optional commit the packet must be bound to.
        expected_release: Optional release version the packet must name.
        expected_p0_checklist_digest: Optional digest of the committed P0
            checklist the packet must have been evaluated against.
        require_ready: Also require a ``READY`` decision.

    Returns:
        The verification outcome. Codes include ``unknown_schema``,
        ``packet_malformed``, ``signature_missing``, ``signature_invalid``,
        ``packet_digest_mismatch``, ``evidence_order_invalid``,
        ``evidence_kind_missing:<kind>``, ``evidence_count_mismatch:<kind>``,
        ``wrong_candidate:<id>``, ``candidate_mismatch``,
        ``release_mismatch``, ``release_target_unsupported``,
        ``p0_checklist_mismatch``, ``decision_inconsistent`` and
        ``decision_not_ready``.

    Raises:
        ServiceReleaseError: ``signing_key_invalid`` when the key is too short.
    """

    key_bytes = _key_or_raise(key)
    if not isinstance(payload, Mapping):
        return _verification(("packet_malformed",), authentic=False)
    if (
        payload.get("schema_version") not in _SUPPORTED_SCHEMA_VERSIONS
        or payload.get("compatibility_policy") != SERVICE_RELEASE_COMPATIBILITY_POLICY
    ):
        return _verification(("unknown_schema",), authentic=False)
    try:
        packet = ServiceReleasePacket.from_dict(payload)
        codes = _structural_codes(packet)
        authentic = _authentic(packet, key_bytes, codes)
    except (ServiceReleaseError, KeyError, TypeError, ValueError, AttributeError):
        return _verification(("packet_malformed",), authentic=False)
    if (
        expected_candidate is not None
        and packet.release.get("candidate_commit") != expected_candidate
    ):
        codes.add("candidate_mismatch")
    if expected_release is not None and packet.release.get("version") != (
        expected_release
    ):
        codes.add("release_mismatch")
    if (
        expected_p0_checklist_digest is not None
        and packet.p0_checklist.get("digest") != expected_p0_checklist_digest
    ):
        codes.add("p0_checklist_mismatch")
    if require_ready and packet.decision != SERVICE_RELEASE_READY:
        codes.add("decision_not_ready")
    decision = (
        packet.decision
        if packet.decision in (SERVICE_RELEASE_READY, SERVICE_RELEASE_NOT_READY)
        else SERVICE_RELEASE_NOT_READY
    )
    return ServiceReleaseVerification(
        valid=not codes,
        authentic=authentic,
        decision=decision,
        reason_codes=tuple(sorted(codes)),
    )


def write_service_release_packet(
    packet: ServiceReleasePacket,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write a packet, refusing replacement unless requested.

    Args:
        packet: The signed packet.
        path: Output file.
        overwrite: Replace an existing file.

    Returns:
        The written path.

    Raises:
        ServiceReleaseError: ``output_exists`` or ``output_unwritable``.
    """

    return _write_json(packet.to_dict(), path, overwrite=overwrite)


# --- command line ------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the service release gate command-line parser."""

    parser = argparse.ArgumentParser(
        description="Build or verify a signed, fail-closed v4.0 service release packet."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--manifest", type=Path, help="Build from this manifest.")
    mode.add_argument("--verify", type=Path, help="Verify this packet offline.")
    mode.add_argument(
        "--write-failure-codes",
        type=Path,
        help="Write the controlled failure-code inventory for --candidate.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, help="Packet output for --manifest.")
    parser.add_argument("--signing-key-file", type=Path)
    parser.add_argument("--key-id", default="service-release")
    parser.add_argument("--p0-checklist", type=Path)
    parser.add_argument("--sentinel-file", type=Path)
    parser.add_argument("--candidate", help="Commit for --write-failure-codes.")
    parser.add_argument("--expected-candidate", help="Commit for --verify.")
    parser.add_argument("--expected-release", help="Version for --verify.")
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build or verify a service release packet.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        For a build: ``0`` for a signed ``READY`` packet, ``1`` for a signed
        ``NOT_READY`` packet, ``2`` when no safe decision could be produced.
        For ``--verify``: ``0`` when valid, ``1`` with reason codes, ``2``
        when the packet or key is unusable. ``--write-failure-codes`` returns
        ``0`` once the inventory is written.
    """

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.manifest is not None and args.output is None:
        parser.error("--manifest requires --output")
    if args.write_failure_codes is not None and args.candidate is None:
        parser.error("--write-failure-codes requires --candidate")
    try:
        if args.write_failure_codes is not None:
            inventory = build_failure_code_inventory(args.candidate)
            _write_json(inventory, args.write_failure_codes, overwrite=args.overwrite)
            print(f"service-release: wrote {len(inventory['codes'])} failure codes")
            return 0
        key = _load_signing_key(args.signing_key_file)
        checklist = load_service_p0_checklist(args.p0_checklist)
        if args.verify is not None:
            return _run_verify(args, key, checklist)
        manifest = load_service_release_manifest(args.manifest)
        packet = evaluate_service_release(
            manifest,
            repo_root=args.repo_root,
            signing_key=key,
            key_id=args.key_id,
            p0_checklist=checklist,
            sentinels=_load_sentinels(args.sentinel_file),
        )
        write_service_release_packet(packet, args.output, overwrite=args.overwrite)
    except ServiceReleaseError as exc:
        print(f"service-release: {'; '.join(exc.codes)}", file=sys.stderr)
        return 2
    except (OSError, subprocess.SubprocessError):
        print("service-release: release_evidence_unavailable", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(packet.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"service-release: {packet.release['version']} -> {packet.decision}")
        for row in packet.evidence_kinds:
            status = "PASS" if row["state"] == KIND_PRESENT else "FAIL"
            print(f"  [{status}] {row['kind']}: {row['state']} ({row['entry_count']})")
        for item in packet.p0_items:
            status = "PASS" if item["state"] == P0_COMPLETE else "FAIL"
            print(f"  [{status}] P0 {item['issue']}: {item['status']}")
        for code in packet.blocking_codes:
            print(f"  [FAIL] {code}")
    return 0 if packet.decision == SERVICE_RELEASE_READY else 1


def _run_verify(
    args: argparse.Namespace, key: bytes, checklist: Mapping[str, Any]
) -> int:
    expected = args.expected_candidate
    if expected is not None and not _COMMIT_RE.fullmatch(expected):
        raise ServiceReleaseError("expected_candidate_invalid")
    payload = load_service_release_packet(args.verify)
    result = verify_service_release_packet(
        payload,
        key,
        expected_candidate=expected,
        expected_release=args.expected_release,
        expected_p0_checklist_digest=stable_hash(checklist),
        require_ready=args.require_ready,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        state = "VALID" if result.valid else "INVALID"
        print(f"service-release verify: {state}; decision={result.decision}")
        for code in result.reason_codes:
            print(f"  [FAIL] {code}")
    return 0 if result.valid else 1


# --- manifest helpers --------------------------------------------------------


def _validate_release(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != _RELEASE_FIELDS:
        raise ServiceReleaseError("release_fields_invalid")
    version = raw["version"]
    if not isinstance(version, str) or not _SEMVER_RE.fullmatch(version):
        raise ServiceReleaseError("release_version_invalid")
    target = raw["target"]
    major, minor = version.split(".")[:2]
    if not isinstance(target, str) or target != f"v{major}.{minor}":
        raise ServiceReleaseError("release_target_invalid")
    if target != SERVICE_RELEASE_TARGET:
        raise ServiceReleaseError("release_target_unsupported")
    commit = raw["candidate_commit"]
    if not isinstance(commit, str) or not _COMMIT_RE.fullmatch(commit):
        raise ServiceReleaseError("candidate_commit_invalid")
    evaluated_at = _canonical_timestamp(raw["evaluated_at"], "evaluated_at_invalid")
    epoch = raw["source_date_epoch"]
    if type(epoch) is not int or epoch < 0:
        raise ServiceReleaseError("source_date_epoch_invalid")
    if int(_timestamp(evaluated_at, "evaluated_at_invalid").timestamp()) != epoch:
        raise ServiceReleaseError("source_date_epoch_invalid")
    return {
        "candidate_commit": commit,
        "evaluated_at": evaluated_at,
        "source_date_epoch": epoch,
        "target": target,
        "version": version,
    }


def _validate_policy(raw: Any) -> dict[str, int]:
    if not isinstance(raw, Mapping) or set(raw) != _POLICY_FIELDS:
        raise ServiceReleaseError("policy_invalid")
    value = raw["max_evidence_age_seconds"]
    if type(value) is not int or not 1 <= value <= _MAX_AGE_SECONDS:
        raise ServiceReleaseError("policy_invalid")
    return {"max_evidence_age_seconds": value}


def _validate_evidence(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list) or len(raw) > MAX_EVIDENCE_ENTRIES:
        raise ServiceReleaseError("evidence_invalid")
    entries: list[dict[str, str]] = []
    ids: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise ServiceReleaseError("evidence_entry_invalid")
        kind = item.get("kind")
        if not isinstance(kind, str) or kind not in EVIDENCE_KINDS:
            raise ServiceReleaseError("evidence_kind_unknown")
        if set(item) != _ENTRY_FIELDS:
            raise ServiceReleaseError("evidence_entry_invalid")
        entry_id = item["id"]
        if not isinstance(entry_id, str) or not _ID_RE.fullmatch(entry_id):
            raise ServiceReleaseError("evidence_id_invalid")
        if entry_id in ids:
            raise ServiceReleaseError("evidence_id_duplicate")
        ids.add(entry_id)
        path = item["path"]
        if not _portable_path(path):
            raise ServiceReleaseError("evidence_path_invalid")
        if kind == FIXTURE and not path.startswith(_SYNTHETIC_FIXTURE_ROOTS):
            raise ServiceReleaseError("fixture_path_not_synthetic")
        digest = item["sha256"]
        if not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest):
            raise ServiceReleaseError("evidence_digest_invalid")
        commit = item["candidate_commit"]
        if not isinstance(commit, str) or not _COMMIT_RE.fullmatch(commit):
            raise ServiceReleaseError("evidence_candidate_invalid")
        entries.append(
            {
                "candidate_commit": commit,
                "generated_at": _canonical_timestamp(
                    item["generated_at"], "evidence_generated_at_invalid"
                ),
                "id": entry_id,
                "kind": kind,
                "path": path,
                "sha256": digest,
            }
        )
    return sorted(entries, key=_entry_order)


def _validate_moved_items(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or len(raw) > MAX_MOVED_ITEMS:
        raise ServiceReleaseError("moved_items_invalid")
    items: list[dict[str, Any]] = []
    issues: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping) or set(item) != _MOVED_FIELDS:
            raise ServiceReleaseError("moved_item_invalid")
        issue = item["issue"]
        if not _is_issue(issue):
            raise ServiceReleaseError("moved_item_issue_invalid")
        if issue in issues:
            raise ServiceReleaseError("moved_item_issue_duplicate")
        issues.add(issue)
        if item["priority"] not in MOVED_ITEM_PRIORITIES:
            raise ServiceReleaseError("moved_item_priority_invalid")
        destination = item["destination"]
        if destination is not None and not _is_milestone(destination):
            raise ServiceReleaseError("moved_item_destination_invalid")
        for name in ("rationale", "residual_risk"):
            value = item[name]
            if value is not None and not _is_plain_code(value):
                raise ServiceReleaseError(f"moved_item_{name}_invalid")
        claims = item["narrowed_claims"]
        if (
            not isinstance(claims, list)
            or not all(_is_plain_code(claim) for claim in claims)
            or len(set(claims)) != len(claims)
        ):
            raise ServiceReleaseError("moved_item_claims_invalid")
        items.append(
            {
                "destination": item["destination"],
                "issue": issue,
                "narrowed_claims": sorted(claims),
                "priority": item["priority"],
                "rationale": item["rationale"],
                "residual_risk": item["residual_risk"],
            }
        )
    return sorted(items, key=lambda row: _issue_number(row["issue"]))


# --- evaluation helpers ------------------------------------------------------


def _loader_registry(
    loaders: Mapping[str, ServiceEvidenceLoader] | None,
) -> dict[str, ServiceEvidenceLoader]:
    registry = dict(DEFAULT_EVIDENCE_LOADERS)
    for kind, loader in (loaders or {}).items():
        if kind not in EVIDENCE_KINDS or not callable(loader):
            raise ServiceReleaseError("loader_kind_unknown")
        registry[kind] = loader
    return registry


def _validated_sentinels(values: Sequence[str]) -> list[str]:
    if isinstance(values, str):
        raise ServiceReleaseError("sentinel_invalid")
    result = []
    for value in values:
        if not isinstance(value, str) or not 8 <= len(value) <= 256:
            raise ServiceReleaseError("sentinel_invalid")
        result.append(value)
    return result


def _repository_binding(
    root: Path,
    candidate: str,
    resolve_head: Callable[[Path], str] | None,
) -> dict[str, Any]:
    try:
        head = (resolve_head or _git_head)(root)
    except (
        JourneyReleaseError,
        ServiceReleaseError,
        OSError,
        subprocess.SubprocessError,
    ):
        head = None
    if not isinstance(head, str) or not _COMMIT_RE.fullmatch(head):
        return {
            "blocking_codes": ["checkout_unresolved"],
            "head_commit": _ZERO_COMMIT,
            "verified": False,
        }
    codes = [] if hmac.compare_digest(head, candidate) else ["checkout_commit_mismatch"]
    return {"blocking_codes": codes, "head_commit": head, "verified": not codes}


def _git_head(root: Path) -> str:
    return _git_output(root, "rev-parse", "HEAD")


def _evaluate_entry(
    root: Path,
    entry: Mapping[str, str],
    release: Mapping[str, str],
    *,
    max_age: int,
    evaluated_at: datetime,
    registry: Mapping[str, ServiceEvidenceLoader],
    sentinels: Sequence[str],
) -> tuple[dict[str, Any], ServiceEvidenceResult | None]:
    codes: list[str] = []
    if entry["candidate_commit"] != release["candidate_commit"]:
        codes.append("wrong_candidate")
    generated_at = _timestamp(entry["generated_at"], "evidence_generated_at_invalid")
    age = int((evaluated_at - generated_at).total_seconds())
    if age < 0:
        codes.append("evidence_from_future")
        age = 0
    elif age > max_age:
        codes.append("evidence_expired")

    result: ServiceEvidenceResult | None = None
    digest, size, data = _ZERO_DIGEST, 0, None
    try:
        digest, size, data = _read_evidence(root, entry["path"])
    except (OSError, JourneyReleaseError, ServiceReleaseError):
        codes.append("evidence_missing")
    else:
        if not hmac.compare_digest(digest, entry["sha256"]):
            codes.append("stale_digest")
        else:
            if data is not None:
                codes.extend(
                    f"sentinel_detected:{name}"
                    for name in scan_for_sentinels(data, sentinels=sentinels)
                )
            source = ServiceEvidenceSource(
                entry_id=entry["id"],
                kind=entry["kind"],
                digest=digest,
                byte_size=size,
                release=dict(release),
                data=data,
            )
            result = _run_loader(registry[entry["kind"]], source)
            codes.extend(result.blocking_codes)
            if (
                entry["kind"] not in DIGEST_ONLY_KINDS
                and result.loader != LOADER_TYPED
                and "typed_loader_unavailable" not in codes
            ):
                codes.append("typed_loader_unavailable")

    row = {
        "age_seconds": age,
        "blocking_codes": sorted(set(codes)),
        "byte_size": size,
        "candidate_commit": entry["candidate_commit"],
        "digest": digest,
        "id": entry["id"],
        "kind": entry["kind"],
        "loader": result.loader if result is not None else LOADER_UNAVAILABLE,
        "scanned": data is not None and result is not None,
        "state": EVIDENCE_BLOCKED if codes else EVIDENCE_VERIFIED,
        "summary": dict(sorted(result.summary.items())) if result else {},
    }
    return row, result


def _read_evidence(root: Path, relative: str) -> tuple[str, int, bytes | None]:
    path = _resolve_beneath(root, relative)
    if path.is_symlink() or not path.is_file():
        raise ServiceReleaseError("evidence_missing")
    size = path.stat().st_size
    if size <= MAX_TYPED_EVIDENCE_BYTES:
        data = path.read_bytes()
        return "sha256:" + hashlib.sha256(data).hexdigest(), len(data), data
    hasher = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
            total += len(chunk)
    return "sha256:" + hasher.hexdigest(), total, None


def _run_loader(
    loader: ServiceEvidenceLoader, source: ServiceEvidenceSource
) -> ServiceEvidenceResult:
    try:
        result = loader(source)
    except ServiceReleaseError as exc:
        return ServiceEvidenceResult(_sanitized_codes(exc.codes))
    except Exception:
        # An injected loader fails closed; its message may carry evidence text.
        return ServiceEvidenceResult(("evidence_loader_failed",))
    if not isinstance(result, ServiceEvidenceResult):
        return ServiceEvidenceResult(("evidence_loader_failed",))
    codes = list(_sanitized_codes(result.blocking_codes))
    summary: dict[str, str | int | bool] = {}
    for key, value in dict(result.summary).items():
        if _summary_value_ok(key, value):
            summary[key] = value
        else:
            codes.append("loader_summary_invalid")
    loader_kind = (
        result.loader
        if result.loader in (LOADER_TYPED, LOADER_DIGEST_ONLY, LOADER_UNAVAILABLE)
        else LOADER_UNAVAILABLE
    )
    return ServiceEvidenceResult(
        tuple(sorted(set(codes))), summary, result.document, loader_kind
    )


def _sanitized_codes(codes: Any) -> tuple[str, ...]:
    if isinstance(codes, str) or not isinstance(codes, (list, tuple)):
        codes = (codes,)
    return tuple(
        code
        if isinstance(code, str) and _CODE_RE.fullmatch(code)
        else "loader_code_invalid"
        for code in codes
    )


def _summary_value_ok(key: Any, value: Any) -> bool:
    if not isinstance(key, str) or not _SUMMARY_KEY_RE.fullmatch(key):
        return False
    if type(value) is bool:
        return True
    if type(value) is int:
        return 0 <= value <= _MAX_SUMMARY_INT
    if not isinstance(value, str):
        return False
    return bool(_DIGEST_RE.fullmatch(value) or _SUMMARY_TEXT_RE.fullmatch(value))


def _kind_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for kind in EVIDENCE_KINDS:
        entries = [row for row in rows if row["kind"] == kind]
        codes: list[str] = []
        if not entries:
            codes.append(f"evidence_not_present:{kind}")
            state = KIND_NOT_PRESENT
        else:
            if kind in SINGLE_ENTRY_KINDS and len(entries) > 1:
                codes.append(f"evidence_kind_repeated:{kind}")
            blocked = codes or any(row["state"] != EVIDENCE_VERIFIED for row in entries)
            state = KIND_BLOCKED if blocked else KIND_PRESENT
        table.append(
            {
                "blocking_codes": codes,
                "entry_count": len(entries),
                "kind": kind,
                "producer": EVIDENCE_PRODUCERS[kind],
                "required": kind in REQUIRED_EVIDENCE_KINDS,
                "state": state,
            }
        )
    return table


def _single_document(
    rows: Sequence[Mapping[str, Any]],
    results: Mapping[str, ServiceEvidenceResult],
    kind: str,
) -> Mapping[str, Any] | None:
    entries = [row for row in rows if row["kind"] == kind]
    if len(entries) != 1 or entries[0]["state"] != EVIDENCE_VERIFIED:
        return None
    result = results.get(entries[0]["id"])
    document = result.document if result is not None else None
    return document if isinstance(document, Mapping) else None


def _p0_rows(
    checklist: Mapping[str, Any],
    kinds: Sequence[Mapping[str, Any]],
    status: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    states = {row["kind"]: row["state"] for row in kinds}
    issue_states: dict[str, str] = {}
    if status is not None:
        issue_states = {row["issue"]: row["state"] for row in status["issues"]}
    table: list[dict[str, Any]] = []
    for item in checklist["items"]:
        issue = item["issue"]
        missing = [
            kind for kind in item["required_evidence"] if states[kind] != KIND_PRESENT
        ]
        codes = [f"p0_evidence_missing:{issue}:{kind}" for kind in missing]
        issue_state = issue_states.get(issue, ISSUE_UNKNOWN)
        if status is None:
            codes.append(f"p0_status_unverifiable:{issue}")
        elif issue_state == ISSUE_UNKNOWN:
            codes.append(f"p0_status_missing:{issue}")
        elif issue_state != ISSUE_CLOSED:
            codes.append(f"p0_open:{issue}")
        table.append(
            {
                "blocking_codes": sorted(codes),
                "control": item["control"],
                "issue": issue,
                "missing_evidence": missing,
                "required_evidence": list(item["required_evidence"]),
                "state": P0_INCOMPLETE if codes else P0_COMPLETE,
                "status": issue_state,
            }
        )
    return table


def _moved_rows(
    moved: Sequence[Mapping[str, Any]],
    checklist: Mapping[str, Any],
    target: str,
) -> list[dict[str, Any]]:
    p0_issues = {item["issue"] for item in checklist["items"]}
    invariants = set(checklist["protected_invariants"])
    table: list[dict[str, Any]] = []
    for item in moved:
        issue = item["issue"]
        codes: list[str] = []
        if item["priority"] == "P0" or issue in p0_issues:
            codes.append(f"p0_moved:{issue}")
        if item["rationale"] not in MOVED_ITEM_RATIONALES:
            codes.append(f"moved_item_rationale_missing:{issue}")
        if not _later_milestone(item["destination"], target):
            codes.append(f"moved_item_destination_invalid:{issue}")
        if item["residual_risk"] is None:
            codes.append(f"moved_item_residual_risk_missing:{issue}")
        if invariants.intersection(item["narrowed_claims"]):
            codes.append(f"moved_item_weakens_invariant:{issue}")
        table.append(
            {
                "blocking_codes": sorted(codes),
                "destination": item["destination"],
                "issue": issue,
                "narrowed_claims": list(item["narrowed_claims"]),
                "priority": item["priority"],
                "rationale": item["rationale"],
                "residual_risk": item["residual_risk"],
            }
        )
    return table


def _later_milestone(value: Any, target: str) -> bool:
    if not _is_milestone(value):
        return False
    return _milestone_key(value) > _milestone_key(target)


def _milestone_key(value: str) -> tuple[int, int, int]:
    match = _MILESTONE_RE.fullmatch(value)
    assert match is not None
    return (int(match[1]), int(match[2]), int(match[3] or 0))


# --- verification helpers ----------------------------------------------------

_VERIFIER_CODES: Final = (
    "candidate_mismatch",
    "decision_inconsistent",
    "decision_not_ready",
    "evidence_count_mismatch",
    "evidence_kind_missing",
    "evidence_order_invalid",
    "p0_checklist_mismatch",
    "packet_digest_mismatch",
    "packet_kind_unknown",
    "packet_malformed",
    "release_mismatch",
    "release_target_unsupported",
    "signature_algorithm_unsupported",
    "signature_invalid",
    "signature_missing",
    "unknown_schema",
    "wrong_candidate",
)


def _structural_codes(packet: ServiceReleasePacket) -> set[str]:
    codes: set[str] = set()
    order = {kind: index for index, kind in enumerate(EVIDENCE_KINDS)}
    keys = []
    counts: dict[str, int] = {}
    candidate = packet.release.get("candidate_commit")
    if packet.release.get("target") != SERVICE_RELEASE_TARGET:
        codes.add("release_target_unsupported")
    for row in packet.evidence:
        kind, entry_id = row["kind"], row["id"]
        if kind not in order or not isinstance(entry_id, str):
            raise ServiceReleaseError("packet_malformed")
        keys.append((order[kind], entry_id))
        counts[kind] = counts.get(kind, 0) + 1
        # A row bound to another commit must carry the gate's own code.
        if row["candidate_commit"] != candidate and (
            "wrong_candidate" not in row["blocking_codes"]
        ):
            codes.add(f"wrong_candidate:{entry_id}")
    if any(left >= right for left, right in zip(keys, keys[1:])):
        codes.add("evidence_order_invalid")
    listed = [row["kind"] for row in packet.evidence_kinds]
    for kind in REQUIRED_EVIDENCE_KINDS:
        if kind not in listed:
            codes.add(f"evidence_kind_missing:{kind}")
    known = [kind for kind in listed if kind in order]
    if known != sorted(known, key=order.__getitem__) or len(set(listed)) != len(listed):
        codes.add("evidence_order_invalid")
    for row in packet.evidence_kinds:
        kind = row["kind"]
        if kind not in order:
            codes.add("packet_kind_unknown")
            continue
        actual = counts.get(kind, 0)
        if row["entry_count"] != actual or (row["state"] == KIND_NOT_PRESENT) != (
            actual == 0
        ):
            codes.add(f"evidence_count_mismatch:{kind}")
    issues = [row["issue"] for row in packet.p0_items]
    if issues != sorted(issues, key=_issue_number) or len(set(issues)) != len(issues):
        codes.add("evidence_order_invalid")
    if packet.p0_checklist.get("item_count") != len(issues):
        codes.add("evidence_count_mismatch:p0_items")
    ready = packet.decision == SERVICE_RELEASE_READY
    clean = (
        not packet.blocking_codes
        and packet.repository_binding.get("verified") is True
        and all(row["state"] == EVIDENCE_VERIFIED for row in packet.evidence)
        and all(row["state"] == KIND_PRESENT for row in packet.evidence_kinds)
        and all(row["state"] == P0_COMPLETE for row in packet.p0_items)
        and all(not row["blocking_codes"] for row in packet.moved_items)
        and packet.sentinel_scan.get("finding_count") == 0
        and not any(code.startswith("wrong_candidate:") for code in codes)
    )
    if packet.decision not in (SERVICE_RELEASE_READY, SERVICE_RELEASE_NOT_READY) or (
        ready != clean
    ):
        codes.add("decision_inconsistent")
    return codes


def _authentic(packet: ServiceReleasePacket, key: bytes, codes: set[str]) -> bool:
    before = len(codes)
    if packet.recompute_packet_digest() != packet.packet_digest:
        codes.add("packet_digest_mismatch")
    signature = packet.signature
    if signature is None:
        codes.add("signature_missing")
    elif signature.algorithm != SERVICE_RELEASE_SIGNATURE_ALGORITHM:
        codes.add("signature_algorithm_unsupported")
    else:
        signed = {**packet._unsigned_payload(), "packet_digest": packet.packet_digest}
        if not hmac.compare_digest(_release_hmac(key, signed), signature.value):
            codes.add("signature_invalid")
    return len(codes) == before


def _verification(
    codes: Sequence[str], *, authentic: bool
) -> ServiceReleaseVerification:
    return ServiceReleaseVerification(
        valid=False,
        authentic=authentic,
        decision=SERVICE_RELEASE_NOT_READY,
        reason_codes=tuple(sorted(codes)),
    )


# --- small helpers -----------------------------------------------------------


def _key_or_raise(key: bytes | str) -> bytes:
    try:
        return _key_bytes(key)
    except (JourneyReleaseError, AttributeError, TypeError):
        raise ServiceReleaseError("signing_key_invalid") from None


def _load_signing_key(path: Path | None) -> bytes:
    if path is not None:
        try:
            if path.is_symlink() or not path.is_file():
                raise ServiceReleaseError("signing_key_unreadable")
            value = path.read_bytes().strip()
        except OSError:
            raise ServiceReleaseError("signing_key_unreadable") from None
    else:
        value = os.environ.get(SERVICE_RELEASE_SIGNING_ENV, "").encode("utf-8")
    if len(value) < 32:
        raise ServiceReleaseError("signing_key_missing")
    return value


def _load_sentinels(path: Path | None) -> list[str]:
    if path is None:
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        raise ServiceReleaseError("sentinel_invalid") from None
    return [line for line in (item.strip() for item in lines) if line]


def _write_json(
    payload: Mapping[str, Any], path: str | Path, *, overwrite: bool
) -> Path:
    output = Path(path)
    try:
        return _write_release_json(payload, output, overwrite=overwrite)
    except JourneyReleaseError:
        code = "output_exists" if output.exists() else "output_unwritable"
        raise ServiceReleaseError(code) from None
    except OSError:
        raise ServiceReleaseError("output_unwritable") from None


def _decode_json(data: bytes, code: str) -> Any:
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite,
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        JourneyReleaseError,
        RecursionError,
    ):
        raise ServiceReleaseError(code) from None


def _canonical_timestamp(value: Any, code: str) -> str:
    return _timestamp(value, code).isoformat().replace("+00:00", "Z")


def _timestamp(value: Any, code: str) -> datetime:
    try:
        return _parse_timestamp(value, code)
    except JourneyReleaseError:
        raise ServiceReleaseError(code) from None


def _portable_path(value: Any) -> bool:
    if not isinstance(value, str) or not _PORTABLE_PATH_RE.fullmatch(value):
        return False
    parts = value.split("/")
    return all(part not in {"", ".", ".."} for part in parts)


def _entry_order(entry: Mapping[str, str]) -> tuple[int, str]:
    return (EVIDENCE_KINDS.index(entry["kind"]), entry["id"])


def _is_issue(value: Any) -> bool:
    return isinstance(value, str) and _ISSUE_RE.fullmatch(value) is not None


def _issue_number(value: str) -> int:
    return int(value[1:]) if _is_issue(value) else 0


def _is_milestone(value: Any) -> bool:
    return isinstance(value, str) and _MILESTONE_RE.fullmatch(value) is not None


def _is_plain_code(value: Any) -> bool:
    return isinstance(value, str) and _PLAIN_CODE_RE.fullmatch(value) is not None


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _packet_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(k, str) for k in value):
        raise ServiceReleaseError("packet_malformed")
    return dict(value)


def _packet_rows(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        raise ServiceReleaseError("packet_malformed")
    return tuple(_packet_mapping(item) for item in value)


def _packet_codes(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ServiceReleaseError("packet_malformed")
    return tuple(value)


def _packet_str(value: Any) -> str:
    if not isinstance(value, str):
        raise ServiceReleaseError("packet_malformed")
    return value


def _load_bundled_schema(name: str) -> dict[str, Any]:
    resource = resources.files(_SCHEMA_PACKAGE).joinpath(name)
    with resource.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ServiceReleaseError("bundled_schema_invalid")
    return payload


__all__ = [
    "DEFAULT_EVIDENCE_LOADERS",
    "DIGEST_ONLY_KINDS",
    "EVIDENCE_BLOCKED",
    "EVIDENCE_KINDS",
    "EVIDENCE_PRODUCERS",
    "EVIDENCE_VERIFIED",
    "FAILURE_CODE_INVENTORY",
    "FAILURE_CODE_INVENTORY_SCHEMA",
    "FAILURE_CODE_SURFACES",
    "FIXTURE",
    "KIND_BLOCKED",
    "KIND_NOT_PRESENT",
    "KIND_PRESENT",
    "LOADER_DIGEST_ONLY",
    "LOADER_TYPED",
    "LOADER_UNAVAILABLE",
    "MIGRATION_REHEARSAL_MATRIX",
    "MOVED_ITEM_RATIONALES",
    "NO_EGRESS_ATTESTATION",
    "P0_CHECKLIST_SCHEMA",
    "P0_COMPLETE",
    "P0_INCOMPLETE",
    "P0_STATUS",
    "P0_STATUS_SCHEMA",
    "PROTOCOL_MAP",
    "RECOVERY_DRILL_REPORT",
    "RECOVERY_DRILL_SCHEMA",
    "REQUIRED_EVIDENCE_KINDS",
    "SECURITY_PROFILE_REPORT",
    "SENTINEL_PATTERNS",
    "SERVICE_RELEASE_COMPATIBILITY_POLICY",
    "SERVICE_RELEASE_LIMITATIONS",
    "SERVICE_RELEASE_NOT_READY",
    "SERVICE_RELEASE_READY",
    "SERVICE_RELEASE_REASON_CODES",
    "SERVICE_RELEASE_SCHEMA_VERSION",
    "SERVICE_RELEASE_SIGNATURE_ALGORITHM",
    "SERVICE_RELEASE_SIGNING_ENV",
    "SERVICE_RELEASE_TARGET",
    "SINGLE_ENTRY_KINDS",
    "THREAT_MODEL",
    "ServiceEvidenceLoader",
    "ServiceEvidenceResult",
    "ServiceEvidenceSource",
    "ServiceReleaseError",
    "ServiceReleasePacket",
    "ServiceReleaseVerification",
    "build_arg_parser",
    "build_failure_code_inventory",
    "build_p0_status",
    "evaluate_service_release",
    "load_digest_only_evidence",
    "load_failure_code_inventory_evidence",
    "load_p0_status_evidence",
    "load_recovery_drill_evidence",
    "load_service_p0_checklist",
    "load_service_recovery_drill_schema",
    "load_service_release_manifest",
    "load_service_release_manifest_schema",
    "load_service_release_packet",
    "load_service_release_packet_schema",
    "load_unavailable_evidence",
    "main",
    "required_failure_codes",
    "scan_for_sentinels",
    "service_p0_checklist_digest",
    "validate_service_p0_checklist",
    "validate_service_release_manifest",
    "verify_service_release_packet",
    "write_service_release_packet",
]


if __name__ == "__main__":
    raise SystemExit(main())
