"""Fail-closed v4.0 portable-kit release gate and signed evidence packet.

The portable-kit milestone first planned as v3.5 was folded into v4.0, so the
gate, schemas and documentation carry the v4.0 name. The gate binds one exact
candidate commit to every piece of kit release evidence:

* kit artifacts (Swift package source, AAR, npm, Flutter, Tauri, Electron and
  Python distributions where present), by SHA-256;
* the frozen support matrix from :mod:`openmed.eval.kit_support_matrix`;
* the device qualification report from :mod:`openmed.eval.kit_qualification`;
* model bundle manifests, parity and golden fixtures, the license inventory
  and the kit threat model, by SHA-256;
* test-lane results; and
* P1 and P2 work moved out of the release, with a coded rationale, a
  destination and the claims it narrows.

Evidence is read through typed loaders keyed by evidence kind. Callers can
inject a loader for a kind whose producer lands later. A required kind with
no entry is reported as ``not_present`` and blocks the release.

The packet reuses the v3 Journey gate's canonical JSON, caller-keyed
HMAC-SHA256 signature and atomic writer from
:mod:`openmed.eval.journey_release`. It holds controlled codes, counts,
matrix identifiers and digests only: never paths, file contents, device
identifiers, free text or key material. Building or verifying a packet makes
no network call. The gate does not run devices, tag, publish or authorize a
release; a ``READY`` packet is input to a separate promotion step.
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
    JOURNEY_RELEASE_STATES,
    JourneyReleaseError,
    _git_output,
    _key_bytes,
    _parse_timestamp,
    _reject_duplicate_keys,
    _reject_non_finite,
    _release_hmac,
    _resolve_beneath,
    _write_release_json,
)
from openmed.eval.kit_qualification import (
    CELL_STATUSES,
    DEVICE_EVIDENCE,
    FAKE_RUNTIME_EVIDENCE,
    FAKE_RUNTIME_ONLY,
    QUALIFIED,
    REPORT_SCHEMA,
    SIMULATOR_EVIDENCE,
    SIMULATOR_ONLY,
    VIRTUAL_DEVICE_CLASSES,
    KitQualificationError,
    _verify_digest,
    matrix_digest,
)
from openmed.eval.kit_support_matrix import (
    EPIC_SURFACES,
    MATRIX_NAME,
    SUPPORTED,
    identifier_shaped,
    validate_kit_support_matrix,
)
from openmed.eval.kit_support_matrix import (
    SCHEMA_VERSION as MATRIX_SCHEMA_VERSION,
)

KIT_RELEASE_SCHEMA_VERSION: Final = "1.0.0"
_SUPPORTED_SCHEMA_VERSIONS: Final = frozenset({KIT_RELEASE_SCHEMA_VERSION})
KIT_RELEASE_COMPATIBILITY_POLICY: Final = JOURNEY_RELEASE_COMPATIBILITY_POLICY
KIT_RELEASE_SIGNATURE_ALGORITHM: Final = JOURNEY_RELEASE_SIGNATURE_ALGORITHM
KIT_RELEASE_READY: Final = "READY"
KIT_RELEASE_NOT_READY: Final = "NOT_READY"
KIT_RELEASE_TEST_LANE_SCHEMA: Final = "openmed.kit_release.test_lane.v1"
KIT_RELEASE_SIGNING_ENV: Final = "OPENMED_KIT_RELEASE_KEY"

KIT_ARTIFACT: Final = "kit_artifact"
SUPPORT_MATRIX: Final = "support_matrix"
QUALIFICATION_REPORT: Final = "qualification_report"
BUNDLE_MANIFEST: Final = "bundle_manifest"
PARITY_FIXTURE: Final = "parity_fixture"
LICENSE_INVENTORY: Final = "license_inventory"
THREAT_MODEL: Final = "threat_model"
TEST_LANE: Final = "test_lane"
# Canonical order: packet evidence is sorted by this order, then by id.
EVIDENCE_KINDS: Final = (
    KIT_ARTIFACT,
    SUPPORT_MATRIX,
    QUALIFICATION_REPORT,
    BUNDLE_MANIFEST,
    PARITY_FIXTURE,
    LICENSE_INVENTORY,
    THREAT_MODEL,
    TEST_LANE,
)
# Every kind is release-blocking: a kind without an entry is ``not_present``.
REQUIRED_EVIDENCE_KINDS: Final = EVIDENCE_KINDS
SINGLE_ENTRY_KINDS: Final = frozenset(
    {SUPPORT_MATRIX, QUALIFICATION_REPORT, LICENSE_INVENTORY, THREAT_MODEL}
)

# Kit distribution formats and the matrix surfaces each one can ship.
KIT_ARTIFACT_FORMATS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "aar": frozenset({"android", "document-intake", "key-store"}),
        "electron-package": frozenset({"electron"}),
        "flutter-package": frozenset({"flutter"}),
        "npm-tarball": frozenset({"electron", "react-native", "web"}),
        "python-sdist": frozenset({"python"}),
        "python-wheel": frozenset({"python"}),
        "swiftpm-source": frozenset({"document-intake", "key-store", "swift"}),
        "tauri-crate": frozenset({"tauri"}),
    }
)

EVIDENCE_VERIFIED: Final = "verified"
EVIDENCE_BLOCKED: Final = "blocked"
KIND_PRESENT: Final = "present"
KIND_NOT_PRESENT: Final = "not_present"
KIND_BLOCKED: Final = "blocked"
LOADER_TYPED: Final = "typed"
LOADER_DIGEST_ONLY: Final = "digest_only"

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
KIT_RELEASE_LIMITATIONS: Final = (
    "no_autonomous_clinical_action",
    "no_publication_authorization",
    "not_a_medical_device",
    "not_clinically_validated",
    "requires_local_validation",
)

MAX_EVIDENCE_ENTRIES: Final = 512
MAX_MOVED_ITEMS: Final = 512
MAX_TYPED_EVIDENCE_BYTES: Final = 16 * 1024 * 1024
MAX_MANIFEST_BYTES: Final = 4 * 1024 * 1024
MAX_PACKET_BYTES: Final = 16 * 1024 * 1024
_MAX_AGE_SECONDS: Final = 31_536_000
_MAX_COUNT: Final = 10_000_000

_REPRODUCTION_COMMAND: Final = (
    "python scripts/release/kit_release_gate.py --manifest "
    "kit-release-manifest.json --output kit-release-packet.json"
)
_SCHEMA_PACKAGE: Final = "openmed.core.schemas.json"
_ZERO_COMMIT: Final = "0" * 40
_ZERO_DIGEST: Final = "sha256:" + "0" * 64
_COMMIT_RE: Final = re.compile(r"[0-9a-f]{40}\Z")
_DIGEST_RE: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ID_RE: Final = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\Z")
_CODE_RE: Final = re.compile(r"[a-z0-9][a-z0-9_.:#-]{0,159}\Z")
_ISSUE_RE: Final = re.compile(r"#[1-9][0-9]{0,6}\Z")
_MILESTONE_RE: Final = re.compile(r"v[0-9]{1,3}\.[0-9]{1,3}(?:\.[0-9]{1,3})?\Z")
_SEMVER_RE: Final = re.compile(
    r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z.-]{1,32})?\Z"
)
_MATRIX_RELEASE_RE: Final = re.compile(r"v[0-9]{1,3}\.[0-9]{1,3}\Z")
_PORTABLE_PATH_RE: Final = re.compile(r"[A-Za-z0-9_.][A-Za-z0-9_./+@-]{0,511}\Z")
_HEX64_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_SUMMARY_KEY_RE: Final = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_SUMMARY_TEXT_RE: Final = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}\Z")
_MAX_SUMMARY_INT: Final = 2**53

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
    {
        "candidate_commit",
        "evaluated_at",
        "matrix_release",
        "source_date_epoch",
        "version",
    }
)
_POLICY_FIELDS: Final = frozenset({"max_evidence_age_seconds"})
_ENTRY_FIELDS: Final = frozenset(
    {"candidate_commit", "generated_at", "id", "kind", "path", "sha256"}
)
_ARTIFACT_FIELDS: Final = frozenset({"format", "surface"})
_MOVED_FIELDS: Final = frozenset(
    {"destination", "issue", "narrowed_claims", "priority", "rationale"}
)
_LANE_FIELDS: Final = frozenset(
    {
        "candidate_commit",
        "failed_count",
        "lane",
        "schema",
        "skipped_count",
        "state",
        "test_count",
    }
)
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
        "packet_digest",
        "release",
        "repository_binding",
        "reproduction",
        "schema_version",
        "signature",
        "supported_cells",
        "surfaces",
    }
)


class KitReleaseError(ValueError):
    """Raised when kit release input cannot produce a safe decision.

    Attributes:
        codes: Value-free reason codes. They name fields, evidence kinds and
            controlled identifiers only, never submitted values or paths.
    """

    def __init__(self, *codes: str) -> None:
        self.codes = tuple(codes) or ("kit_release_invalid",)
        super().__init__("; ".join(self.codes))


@dataclass(frozen=True)
class KitEvidenceSource:
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
        release: The manifest release: ``version``, ``matrix_release`` and
            ``candidate_commit``.
        attributes: Kind-specific manifest attributes, such as an artifact's
            ``surface`` and ``format``.
        data: The digested bytes, or ``None`` for oversized files.
    """

    entry_id: str
    kind: str
    digest: str
    byte_size: int
    release: Mapping[str, str]
    attributes: Mapping[str, str] = field(default_factory=dict)
    data: bytes | None = field(default=None, repr=False)

    def read_json(self) -> Any:
        """Decode the evidence as strict JSON.

        Returns:
            The decoded JSON value.

        Raises:
            KitReleaseError: ``evidence_too_large`` when the bytes were not
                retained, or ``evidence_unreadable`` when they are not strict
                UTF-8 JSON without duplicate keys or non-finite numbers.
        """

        if self.data is None:
            raise KitReleaseError("evidence_too_large")
        return _decode_json(self.data, "evidence_unreadable")


@dataclass(frozen=True)
class KitEvidenceResult:
    """Value-free outcome of one evidence loader.

    Attributes:
        blocking_codes: Bare reason codes; the gate suffixes the evidence id.
        summary: Code-keyed counts, booleans, codes or digests for the packet.
        document: Parsed document kept in memory for cross-checks only. It
            never enters the packet.
        loader: ``typed`` for parsed evidence, ``digest_only`` when only the
            digest is bound.
    """

    blocking_codes: tuple[str, ...] = ()
    summary: Mapping[str, str | int | bool] = field(default_factory=dict)
    document: Any = field(default=None, repr=False, compare=False)
    loader: str = LOADER_TYPED


KitEvidenceLoader = Callable[[KitEvidenceSource], KitEvidenceResult]


@dataclass(frozen=True)
class KitReleaseVerification:
    """Offline verification outcome for a signed kit release packet.

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
class KitReleasePacket:
    """Signed, reproducible and value-free v4.0 kit release decision."""

    release: Mapping[str, Any]
    decision: str
    repository_binding: Mapping[str, Any]
    evidence: tuple[Mapping[str, Any], ...]
    evidence_kinds: tuple[Mapping[str, Any], ...]
    supported_cells: tuple[Mapping[str, Any], ...]
    surfaces: tuple[Mapping[str, Any], ...]
    moved_items: tuple[Mapping[str, Any], ...]
    blocking_codes: tuple[str, ...]
    manifest_digest: str
    reproduction: Mapping[str, Any]
    limitations: tuple[str, ...] = KIT_RELEASE_LIMITATIONS
    schema_version: str = KIT_RELEASE_SCHEMA_VERSION
    compatibility_policy: str = KIT_RELEASE_COMPATIBILITY_POLICY
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
            "release": _plain(self.release),
            "repository_binding": _plain(self.repository_binding),
            "reproduction": _plain(self.reproduction),
            "schema_version": self.schema_version,
            "supported_cells": [_plain(item) for item in self.supported_cells],
            "surfaces": [_plain(item) for item in self.surfaces],
        }

    def recompute_packet_digest(self) -> str:
        """Recompute the canonical digest without trusting the stored value."""

        return stable_hash(self._unsigned_payload())

    def sign(self, key: bytes | str, *, key_id: str) -> KitReleasePacket:
        """Return a copy signed over every packet field and its digest.

        Args:
            key: Caller-supplied HMAC key of at least 32 bytes.
            key_id: Bounded key identifier recorded instead of the key.

        Returns:
            The signed packet.

        Raises:
            KitReleaseError: If the key is too short or the key id is not a
                bounded code.
        """

        if not isinstance(key_id, str) or not _CODE_RE.fullmatch(key_id):
            raise KitReleaseError("key_id_invalid")
        packet_digest = self.recompute_packet_digest()
        signed = {**self._unsigned_payload(), "packet_digest": packet_digest}
        try:
            value = _release_hmac(key, signed)
        except JourneyReleaseError:
            raise KitReleaseError("signing_key_invalid") from None
        return replace(
            self,
            packet_digest=packet_digest,
            signature=AuditSignature(
                key_id=key_id,
                algorithm=KIT_RELEASE_SIGNATURE_ALGORITHM,
                value=value,
            ),
        )

    def verify(self, key: bytes | str) -> bool:
        """Return whether the packet is authentic and internally consistent.

        A signed ``NOT_READY`` packet verifies; use
        :func:`verify_kit_release_packet` with ``require_ready=True`` to also
        require a ``READY`` decision.
        """

        return verify_kit_release_packet(self.to_dict(), key).valid

    def to_dict(self) -> dict[str, Any]:
        """Return the complete deterministic JSON representation."""

        return {
            **self._unsigned_payload(),
            "packet_digest": self.packet_digest,
            "signature": self.signature.to_dict() if self.signature else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> KitReleasePacket:
        """Restore a packet for independent offline verification.

        Raises:
            KitReleaseError: ``packet_malformed`` when a field is missing or
                has the wrong JSON type.
        """

        if not isinstance(payload, Mapping) or set(payload) != _PACKET_FIELDS:
            raise KitReleaseError("packet_malformed")
        signature = payload["signature"]
        if signature is not None and not isinstance(signature, Mapping):
            raise KitReleaseError("packet_malformed")
        return cls(
            release=_packet_mapping(payload["release"]),
            decision=_packet_str(payload["decision"]),
            repository_binding=_packet_mapping(payload["repository_binding"]),
            evidence=_packet_rows(payload["evidence"]),
            evidence_kinds=_packet_rows(payload["evidence_kinds"]),
            supported_cells=_packet_rows(payload["supported_cells"]),
            surfaces=_packet_rows(payload["surfaces"]),
            moved_items=_packet_rows(payload["moved_items"]),
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


# --- evidence loaders --------------------------------------------------------


def load_support_matrix_evidence(source: KitEvidenceSource) -> KitEvidenceResult:
    """Load the frozen support matrix and check it names the release.

    Args:
        source: The digest-checked matrix file.

    Returns:
        A result whose document is the decoded matrix when it validates.
        Codes: ``evidence_unreadable``, ``unknown_schema``, ``matrix_invalid``,
        ``matrix_not_frozen`` and ``matrix_release_mismatch``.
    """

    try:
        document = source.read_json()
    except KitReleaseError as exc:
        return KitEvidenceResult(exc.codes)
    if not isinstance(document, Mapping):
        return KitEvidenceResult(("matrix_invalid",))
    if (
        document.get("schema_version") != MATRIX_SCHEMA_VERSION
        or document.get("matrix") != MATRIX_NAME
    ):
        return KitEvidenceResult(("unknown_schema",))
    problems = validate_kit_support_matrix(document)
    if problems:
        return KitEvidenceResult(
            ("matrix_invalid",), summary={"problem_count": len(problems)}
        )
    codes: list[str] = []
    if document["state"] != "frozen":
        codes.append("matrix_not_frozen")
    if document["release"] != source.release["matrix_release"]:
        codes.append("matrix_release_mismatch")
    statuses = [cell["status"] for cell in document["cells"]]
    summary: dict[str, str | int | bool] = {
        "canonical_digest": "sha256:" + matrix_digest(document),
        "cells_experimental": statuses.count("experimental"),
        "cells_supported": statuses.count(SUPPORTED),
        "release": document["release"],
        "state": document["state"],
        "unsupported_entries": len(document["unsupported"]),
    }
    return KitEvidenceResult(tuple(codes), summary, document=dict(document))


def load_qualification_report_evidence(
    source: KitEvidenceSource,
) -> KitEvidenceResult:
    """Load a device qualification report and verify its self-digest.

    Args:
        source: The digest-checked report from
            :func:`openmed.eval.kit_qualification.collect_qualification`.

    Returns:
        A result whose document is the decoded report. Codes:
        ``evidence_unreadable``, ``unknown_schema``, ``report_digest_mismatch``,
        ``report_invalid``, ``matrix_release_mismatch`` and
        ``qualification_records_rejected``.
    """

    try:
        document = source.read_json()
    except KitReleaseError as exc:
        return KitEvidenceResult(exc.codes)
    if not isinstance(document, Mapping) or document.get("schema") != REPORT_SCHEMA:
        return KitEvidenceResult(("unknown_schema",))
    try:
        _verify_digest(document, "report_digest", "report_digest_mismatch")
    except KitQualificationError:
        return KitEvidenceResult(("report_digest_mismatch",))
    if not _report_shape_valid(document):
        return KitEvidenceResult(("report_invalid",))
    codes: list[str] = []
    if document["matrix"]["release"] != source.release["matrix_release"]:
        codes.append("matrix_release_mismatch")
    summary_block = document["summary"]
    by_class = summary_block["records_by_class"]
    if summary_block["records_rejected"]:
        codes.append("qualification_records_rejected")
    summary: dict[str, str | int | bool] = {
        "cells_planned": summary_block["cells_planned"],
        "cells_qualified": summary_block["cells_qualified"],
        "device_records": by_class[DEVICE_EVIDENCE],
        "fake_runtime_records": by_class[FAKE_RUNTIME_EVIDENCE],
        "matrix_digest": "sha256:" + document["matrix"]["digest"],
        "records_accepted": summary_block["records_accepted"],
        "records_rejected": summary_block["records_rejected"],
        "report_digest": "sha256:" + document["report_digest"],
        "simulator_records": by_class[SIMULATOR_EVIDENCE],
    }
    return KitEvidenceResult(tuple(codes), summary, document=dict(document))


def load_test_lane_evidence(source: KitEvidenceSource) -> KitEvidenceResult:
    """Load one test-lane result bound to the candidate commit.

    A lane record (``openmed.kit_release.test_lane.v1``) is a closed object
    with ``lane``, ``candidate_commit``, ``state`` (a Journey result state)
    and ``test_count``, ``failed_count`` and ``skipped_count``.

    Args:
        source: The digest-checked lane record.

    Returns:
        A result whose document is the decoded record. Codes:
        ``evidence_unreadable``, ``unknown_schema``, ``lane_invalid``,
        ``wrong_candidate``, ``lane_not_successful``, ``lane_failed`` and
        ``lane_empty``.
    """

    try:
        document = source.read_json()
    except KitReleaseError as exc:
        return KitEvidenceResult(exc.codes)
    if (
        not isinstance(document, Mapping)
        or document.get("schema") != KIT_RELEASE_TEST_LANE_SCHEMA
    ):
        return KitEvidenceResult(("unknown_schema",))
    if (
        set(document) != _LANE_FIELDS
        or not _is_id(document["lane"])
        or not isinstance(document["candidate_commit"], str)
        or not _COMMIT_RE.fullmatch(document["candidate_commit"])
        or document["state"] not in JOURNEY_RELEASE_STATES
        or not all(
            _is_count(document[name])
            for name in ("test_count", "failed_count", "skipped_count")
        )
    ):
        return KitEvidenceResult(("lane_invalid",))
    codes: list[str] = []
    if document["candidate_commit"] != source.release["candidate_commit"]:
        codes.append("wrong_candidate")
    if document["state"] != "success":
        codes.append("lane_not_successful")
    if document["failed_count"]:
        codes.append("lane_failed")
    if document["test_count"] == 0:
        codes.append("lane_empty")
    summary: dict[str, str | int | bool] = {
        "failed_count": document["failed_count"],
        "lane": document["lane"],
        "skipped_count": document["skipped_count"],
        "state": document["state"],
        "test_count": document["test_count"],
    }
    return KitEvidenceResult(tuple(codes), summary, document=dict(document))


def load_kit_artifact_evidence(source: KitEvidenceSource) -> KitEvidenceResult:
    """Bind a kit artifact by digest; the packet records its surface and format.

    Args:
        source: The digest-checked artifact file.

    Returns:
        A digest-only result without codes.
    """

    del source
    return KitEvidenceResult(loader=LOADER_DIGEST_ONLY)


def load_digest_only_evidence(source: KitEvidenceSource) -> KitEvidenceResult:
    """Bind evidence by digest alone, without parsing it.

    This is the default for bundle manifests, parity and golden fixtures,
    license inventories and the threat model, whose producers are separate
    work. Inject a typed loader for a kind to also evaluate its content.

    Args:
        source: The digest-checked evidence file.

    Returns:
        A digest-only result without codes.
    """

    del source
    return KitEvidenceResult(loader=LOADER_DIGEST_ONLY)


DEFAULT_EVIDENCE_LOADERS: Final[Mapping[str, KitEvidenceLoader]] = MappingProxyType(
    {
        KIT_ARTIFACT: load_kit_artifact_evidence,
        SUPPORT_MATRIX: load_support_matrix_evidence,
        QUALIFICATION_REPORT: load_qualification_report_evidence,
        BUNDLE_MANIFEST: load_digest_only_evidence,
        PARITY_FIXTURE: load_digest_only_evidence,
        LICENSE_INVENTORY: load_digest_only_evidence,
        THREAT_MODEL: load_digest_only_evidence,
        TEST_LANE: load_test_lane_evidence,
    }
)


# --- manifest ----------------------------------------------------------------


def validate_kit_release_manifest(manifest: Any) -> dict[str, Any]:
    """Validate and normalize a kit release manifest.

    Structural problems raise; release-readiness problems (missing evidence,
    stale digests, wrong candidates, moved items without a rationale) are
    left for :func:`evaluate_kit_release` to report as blocking codes.

    Args:
        manifest: A decoded manifest.

    Returns:
        The normalized manifest with evidence sorted canonically.

    Raises:
        KitReleaseError: With a value-free code such as ``unknown_schema``.
    """

    if not isinstance(manifest, Mapping):
        raise KitReleaseError("manifest_not_object")
    if not _known_schema(manifest.get("schema_version")):
        raise KitReleaseError("unknown_schema")
    if set(manifest) != _MANIFEST_FIELDS:
        raise KitReleaseError("manifest_fields_invalid")
    if manifest["compatibility_policy"] != KIT_RELEASE_COMPATIBILITY_POLICY:
        raise KitReleaseError("compatibility_policy_unsupported")
    release = _validate_release(manifest["release"])
    policy = _validate_policy(manifest["policy"])
    evidence = _validate_evidence(manifest["evidence"])
    moved = _validate_moved_items(manifest["moved_items"])
    return {
        "compatibility_policy": KIT_RELEASE_COMPATIBILITY_POLICY,
        "evidence": evidence,
        "moved_items": moved,
        "policy": policy,
        "release": release,
        "schema_version": KIT_RELEASE_SCHEMA_VERSION,
    }


def load_kit_release_manifest(path: str | Path) -> dict[str, Any]:
    """Read and validate a manifest, rejecting duplicate keys.

    Args:
        path: Manifest file.

    Returns:
        The normalized manifest.

    Raises:
        KitReleaseError: ``manifest_unreadable`` or a validation code.
    """

    source = Path(path)
    try:
        if source.stat().st_size > MAX_MANIFEST_BYTES:
            raise KitReleaseError("manifest_too_large")
        data = source.read_bytes()
    except OSError:
        raise KitReleaseError("manifest_unreadable") from None
    return validate_kit_release_manifest(_decode_json(data, "manifest_unreadable"))


def load_kit_release_manifest_schema() -> dict[str, Any]:
    """Load the bundled strict manifest JSON Schema."""

    return _load_bundled_schema("kit_release_manifest.schema.json")


def load_kit_release_packet_schema() -> dict[str, Any]:
    """Load the bundled signed-packet JSON Schema."""

    return _load_bundled_schema("kit_release_packet.schema.json")


def load_kit_release_packet(path: str | Path) -> dict[str, Any]:
    """Read a packet file as strict JSON for offline verification.

    Args:
        path: Packet file.

    Returns:
        The decoded packet payload.

    Raises:
        KitReleaseError: ``packet_unreadable`` or ``packet_too_large``.
    """

    source = Path(path)
    try:
        if source.stat().st_size > MAX_PACKET_BYTES:
            raise KitReleaseError("packet_too_large")
        data = source.read_bytes()
    except OSError:
        raise KitReleaseError("packet_unreadable") from None
    payload = _decode_json(data, "packet_unreadable")
    if not isinstance(payload, dict):
        raise KitReleaseError("packet_malformed")
    return payload


# --- evaluation --------------------------------------------------------------


def evaluate_kit_release(
    manifest: Mapping[str, Any],
    *,
    repo_root: str | Path,
    signing_key: bytes | str,
    key_id: str = "kit-release",
    loaders: Mapping[str, KitEvidenceLoader] | None = None,
    resolve_head: Callable[[Path], str] | None = None,
) -> KitReleasePacket:
    """Evaluate and sign one v4.0 kit release manifest.

    Every evidence path is read beneath ``repo_root`` without following
    symlinks, digested and handed to the loader for its kind. The checkout
    must be at the manifest's candidate commit.

    Args:
        manifest: A decoded manifest (see the bundled manifest schema).
        repo_root: The candidate checkout.
        signing_key: Caller-supplied HMAC key of at least 32 bytes.
        key_id: Bounded key identifier recorded in the signature.
        loaders: Optional loaders by evidence kind; they replace the defaults
            in :data:`DEFAULT_EVIDENCE_LOADERS` for those kinds.
        resolve_head: Optional resolver for the checkout's ``HEAD`` commit.
            Defaults to ``git rev-parse HEAD``.

    Returns:
        The signed packet. Its decision is ``READY`` only when every required
        kind is present and every check passes.

    Raises:
        KitReleaseError: If the manifest, loaders or key are unusable.
    """

    normalized = validate_kit_release_manifest(manifest)
    _key_or_raise(signing_key)
    registry = _loader_registry(loaders)
    root = Path(repo_root).resolve()
    release = normalized["release"]
    evaluated_at = _timestamp(release["evaluated_at"], "evaluated_at_invalid")
    binding = _repository_binding(root, release["candidate_commit"], resolve_head)
    loader_release = {
        key: release[key] for key in ("candidate_commit", "matrix_release", "version")
    }

    rows: list[dict[str, Any]] = []
    results: dict[str, KitEvidenceResult] = {}
    for entry in normalized["evidence"]:
        row, result = _evaluate_entry(
            root,
            entry,
            loader_release,
            max_age=normalized["policy"]["max_evidence_age_seconds"],
            evaluated_at=evaluated_at,
            registry=registry,
        )
        rows.append(row)
        if result is not None:
            results[entry["id"]] = result

    # Documents from injected loaders are re-checked before any cross-check.
    matrix_id, matrix = _single_entry_document(rows, results, SUPPORT_MATRIX)
    if matrix is not None and validate_kit_support_matrix(matrix):
        _block(rows, str(matrix_id), "matrix_invalid")
        matrix = None
    report_id, report = _single_entry_document(rows, results, QUALIFICATION_REPORT)
    if report is not None and (
        report.get("schema") != REPORT_SCHEMA or not _report_shape_valid(report)
    ):
        _block(rows, str(report_id), "report_invalid")
        report = None
    if matrix is not None and report is not None and report_id is not None:
        if report["matrix"]["digest"] != matrix_digest(matrix):
            _block(rows, report_id, "stale_matrix_digest")
    kinds = _kind_rows(rows)
    cells = _supported_cells(matrix, report, rows, results)
    surfaces = _surface_rows(matrix, rows)
    moved = _moved_rows(normalized["moved_items"], matrix, release["matrix_release"])

    codes: set[str] = set(binding["blocking_codes"])
    for row in rows:
        codes.update(f"{code}:{row['id']}" for code in row["blocking_codes"])
    for table in (kinds, cells, surfaces, moved):
        for item in table:
            codes.update(item["blocking_codes"])
    if matrix is None:
        codes.add("supported_cells_unverifiable")
    blocking = tuple(sorted(codes))
    packet = KitReleasePacket(
        release=dict(release),
        decision=KIT_RELEASE_NOT_READY if blocking else KIT_RELEASE_READY,
        repository_binding=binding,
        evidence=tuple(rows),
        evidence_kinds=tuple(kinds),
        supported_cells=tuple(cells),
        surfaces=tuple(surfaces),
        moved_items=tuple(moved),
        blocking_codes=blocking,
        manifest_digest=stable_hash(normalized),
        reproduction={
            "command": _REPRODUCTION_COMMAND,
            "evidence_count": len(rows),
            "moved_item_count": len(moved),
        },
    )
    return packet.sign(signing_key, key_id=key_id)


def verify_kit_release_packet(
    payload: Mapping[str, Any],
    key: bytes | str,
    *,
    expected_candidate: str | None = None,
    require_ready: bool = False,
) -> KitReleaseVerification:
    """Verify a packet offline and report stable reason codes.

    The verifier recomputes the canonical digest and HMAC, then checks the
    packet's own structure: canonical evidence order, every required kind,
    per-kind entry counts, candidate binding and decision consistency.

    Args:
        payload: A decoded packet.
        key: The HMAC key the packet was signed with.
        expected_candidate: Optional commit the packet must be bound to.
        require_ready: Also require a ``READY`` decision.

    Returns:
        The verification outcome. Codes include ``unknown_schema``,
        ``packet_malformed``, ``signature_missing``, ``signature_invalid``,
        ``packet_digest_mismatch``, ``evidence_order_invalid``,
        ``evidence_kind_missing:<kind>``, ``evidence_count_mismatch:<kind>``,
        ``wrong_candidate:<id>``, ``candidate_mismatch``,
        ``decision_inconsistent`` and ``decision_not_ready``.

    Raises:
        KitReleaseError: ``signing_key_invalid`` when the key is too short.
    """

    key_bytes = _key_or_raise(key)
    if not isinstance(payload, Mapping):
        return _verification(("packet_malformed",), authentic=False)
    if (
        not _known_schema(payload.get("schema_version"))
        or payload.get("compatibility_policy") != KIT_RELEASE_COMPATIBILITY_POLICY
    ):
        return _verification(("unknown_schema",), authentic=False)
    try:
        packet = KitReleasePacket.from_dict(payload)
        codes = _structural_codes(packet)
        authentic = _authentic(packet, key_bytes, codes)
    except (KitReleaseError, KeyError, TypeError, ValueError, AttributeError):
        return _verification(("packet_malformed",), authentic=False)
    if (
        expected_candidate is not None
        and packet.release.get("candidate_commit") != expected_candidate
    ):
        codes.add("candidate_mismatch")
    if require_ready and packet.decision != KIT_RELEASE_READY:
        codes.add("decision_not_ready")
    decision = (
        packet.decision
        if packet.decision in (KIT_RELEASE_READY, KIT_RELEASE_NOT_READY)
        else KIT_RELEASE_NOT_READY
    )
    return KitReleaseVerification(
        valid=not codes,
        authentic=authentic,
        decision=decision,
        reason_codes=tuple(sorted(codes)),
    )


def write_kit_release_packet(
    packet: KitReleasePacket,
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
        KitReleaseError: ``output_exists`` or ``output_unwritable``.
    """

    output = Path(path)
    try:
        return _write_release_json(packet.to_dict(), output, overwrite=overwrite)
    except JourneyReleaseError:
        code = "output_exists" if output.exists() else "output_unwritable"
        raise KitReleaseError(code) from None
    except OSError:
        raise KitReleaseError("output_unwritable") from None


# --- command line ------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the kit release gate command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Build or verify a signed, fail-closed v4.0 OpenMedKit release packet."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--manifest", type=Path, help="Build from this manifest.")
    mode.add_argument("--verify", type=Path, help="Verify this packet offline.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, help="Packet output for --manifest.")
    parser.add_argument("--signing-key-file", type=Path)
    parser.add_argument("--key-id", default="kit-release")
    parser.add_argument("--expected-candidate", help="Commit for --verify.")
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build or verify a kit release packet.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        For a build: ``0`` for a signed ``READY`` packet, ``1`` for a signed
        ``NOT_READY`` packet, ``2`` when no safe decision could be produced.
        For ``--verify``: ``0`` when valid, ``1`` with reason codes, ``2``
        when the packet or key is unusable.
    """

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.manifest is not None and args.output is None:
        parser.error("--manifest requires --output")
    try:
        key = _load_signing_key(args.signing_key_file)
        if args.verify is not None:
            return _run_verify(args, key)
        manifest = load_kit_release_manifest(args.manifest)
        packet = evaluate_kit_release(
            manifest,
            repo_root=args.repo_root,
            signing_key=key,
            key_id=args.key_id,
        )
        write_kit_release_packet(packet, args.output, overwrite=args.overwrite)
    except KitReleaseError as exc:
        print(f"kit-release: {'; '.join(exc.codes)}", file=sys.stderr)
        return 2
    except (OSError, subprocess.SubprocessError):
        print("kit-release: release_evidence_unavailable", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(packet.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"kit-release: {packet.release['version']} -> {packet.decision}")
        for row in packet.evidence_kinds:
            status = "PASS" if row["state"] == KIND_PRESENT else "FAIL"
            print(f"  [{status}] {row['kind']}: {row['state']} ({row['entry_count']})")
        for code in packet.blocking_codes:
            print(f"  [FAIL] {code}")
    return 0 if packet.decision == KIT_RELEASE_READY else 1


def _run_verify(args: argparse.Namespace, key: bytes) -> int:
    expected = args.expected_candidate
    if expected is not None and not _COMMIT_RE.fullmatch(expected):
        raise KitReleaseError("expected_candidate_invalid")
    payload = load_kit_release_packet(args.verify)
    result = verify_kit_release_packet(
        payload,
        key,
        expected_candidate=expected,
        require_ready=args.require_ready,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        state = "VALID" if result.valid else "INVALID"
        print(f"kit-release verify: {state}; decision={result.decision}")
        for code in result.reason_codes:
            print(f"  [FAIL] {code}")
    return 0 if result.valid else 1


# --- manifest helpers --------------------------------------------------------


def _validate_release(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != _RELEASE_FIELDS:
        raise KitReleaseError("release_fields_invalid")
    version = raw["version"]
    if not isinstance(version, str) or not _SEMVER_RE.fullmatch(version):
        raise KitReleaseError("release_version_invalid")
    matrix_release = raw["matrix_release"]
    major, minor = version.split(".")[:2]
    if (
        not isinstance(matrix_release, str)
        or not _MATRIX_RELEASE_RE.fullmatch(matrix_release)
        or matrix_release != f"v{major}.{minor}"
    ):
        raise KitReleaseError("release_matrix_release_invalid")
    commit = raw["candidate_commit"]
    if not isinstance(commit, str) or not _COMMIT_RE.fullmatch(commit):
        raise KitReleaseError("candidate_commit_invalid")
    evaluated_at = _canonical_timestamp(raw["evaluated_at"], "evaluated_at_invalid")
    epoch = raw["source_date_epoch"]
    if type(epoch) is not int or epoch < 0:
        raise KitReleaseError("source_date_epoch_invalid")
    if int(_timestamp(evaluated_at, "evaluated_at_invalid").timestamp()) != epoch:
        raise KitReleaseError("source_date_epoch_invalid")
    return {
        "candidate_commit": commit,
        "evaluated_at": evaluated_at,
        "matrix_release": matrix_release,
        "source_date_epoch": epoch,
        "version": version,
    }


def _validate_policy(raw: Any) -> dict[str, int]:
    if not isinstance(raw, Mapping) or set(raw) != _POLICY_FIELDS:
        raise KitReleaseError("policy_invalid")
    value = raw["max_evidence_age_seconds"]
    if type(value) is not int or not 1 <= value <= _MAX_AGE_SECONDS:
        raise KitReleaseError("policy_invalid")
    return {"max_evidence_age_seconds": value}


def _validate_evidence(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list) or len(raw) > MAX_EVIDENCE_ENTRIES:
        raise KitReleaseError("evidence_invalid")
    entries: list[dict[str, str]] = []
    ids: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise KitReleaseError("evidence_entry_invalid")
        kind = item.get("kind")
        if not isinstance(kind, str) or kind not in EVIDENCE_KINDS:
            raise KitReleaseError("evidence_kind_unknown")
        expected = _ENTRY_FIELDS | (_ARTIFACT_FIELDS if kind == KIT_ARTIFACT else set())
        if set(item) != expected:
            raise KitReleaseError("evidence_entry_invalid")
        entry_id = item["id"]
        if not _is_id(entry_id):
            raise KitReleaseError("evidence_id_invalid")
        if entry_id in ids:
            raise KitReleaseError("evidence_id_duplicate")
        ids.add(entry_id)
        path = item["path"]
        if not _portable_path(path):
            raise KitReleaseError("evidence_path_invalid")
        digest = item["sha256"]
        if not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest):
            raise KitReleaseError("evidence_digest_invalid")
        commit = item["candidate_commit"]
        if not isinstance(commit, str) or not _COMMIT_RE.fullmatch(commit):
            raise KitReleaseError("evidence_candidate_invalid")
        entry = {
            "candidate_commit": commit,
            "generated_at": _canonical_timestamp(
                item["generated_at"], "evidence_generated_at_invalid"
            ),
            "id": entry_id,
            "kind": kind,
            "path": path,
            "sha256": digest,
        }
        if kind == KIT_ARTIFACT:
            artifact_format, surface = item["format"], item["surface"]
            if (
                not isinstance(artifact_format, str)
                or artifact_format not in KIT_ARTIFACT_FORMATS
            ):
                raise KitReleaseError("kit_artifact_format_invalid")
            if (
                not isinstance(surface, str)
                or surface not in KIT_ARTIFACT_FORMATS[artifact_format]
            ):
                raise KitReleaseError("kit_artifact_surface_invalid")
            entry["format"] = artifact_format
            entry["surface"] = surface
        entries.append(entry)
    return sorted(entries, key=_entry_order)


def _validate_moved_items(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or len(raw) > MAX_MOVED_ITEMS:
        raise KitReleaseError("moved_items_invalid")
    items: list[dict[str, Any]] = []
    issues: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping) or set(item) != _MOVED_FIELDS:
            raise KitReleaseError("moved_item_invalid")
        issue = item["issue"]
        if not isinstance(issue, str) or not _ISSUE_RE.fullmatch(issue):
            raise KitReleaseError("moved_item_issue_invalid")
        if issue in issues:
            raise KitReleaseError("moved_item_issue_duplicate")
        issues.add(issue)
        if item["priority"] not in MOVED_ITEM_PRIORITIES:
            raise KitReleaseError("moved_item_priority_invalid")
        for name in ("destination", "rationale"):
            value = item[name]
            if value is not None and not isinstance(value, str):
                raise KitReleaseError(f"moved_item_{name}_invalid")
        claims = item["narrowed_claims"]
        if (
            not isinstance(claims, list)
            or not all(_is_id(claim) for claim in claims)
            or len(set(claims)) != len(claims)
        ):
            raise KitReleaseError("moved_item_claims_invalid")
        items.append(
            {
                "destination": item["destination"],
                "issue": issue,
                "narrowed_claims": sorted(claims),
                "priority": item["priority"],
                "rationale": item["rationale"],
            }
        )
    return sorted(items, key=lambda row: int(row["issue"][1:]))


# --- evaluation helpers ------------------------------------------------------


def _loader_registry(
    loaders: Mapping[str, KitEvidenceLoader] | None,
) -> dict[str, KitEvidenceLoader]:
    registry = dict(DEFAULT_EVIDENCE_LOADERS)
    for kind, loader in (loaders or {}).items():
        if kind not in EVIDENCE_KINDS or not callable(loader):
            raise KitReleaseError("loader_kind_unknown")
        registry[kind] = loader
    return registry


def _repository_binding(
    root: Path,
    candidate: str,
    resolve_head: Callable[[Path], str] | None,
) -> dict[str, Any]:
    try:
        head = (resolve_head or _git_head)(root)
    except (JourneyReleaseError, KitReleaseError, OSError, subprocess.SubprocessError):
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
    registry: Mapping[str, KitEvidenceLoader],
) -> tuple[dict[str, Any], KitEvidenceResult | None]:
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

    result: KitEvidenceResult | None = None
    attributes = {key: entry[key] for key in sorted(_ARTIFACT_FIELDS) if key in entry}
    digest, size, data = _ZERO_DIGEST, 0, None
    try:
        digest, size, data = _read_evidence(root, entry["path"])
    except (OSError, JourneyReleaseError, KitReleaseError):
        codes.append("evidence_missing")
    else:
        if not hmac.compare_digest(digest, entry["sha256"]):
            codes.append("stale_digest")
        else:
            source = KitEvidenceSource(
                entry_id=entry["id"],
                kind=entry["kind"],
                digest=digest,
                byte_size=size,
                release=dict(release),
                attributes=attributes,
                data=data,
            )
            result = _run_loader(registry[entry["kind"]], source)
            codes.extend(result.blocking_codes)

    row = {
        "age_seconds": age,
        "attributes": attributes,
        "blocking_codes": sorted(set(codes)),
        "byte_size": size,
        "candidate_commit": entry["candidate_commit"],
        "digest": digest,
        "id": entry["id"],
        "kind": entry["kind"],
        "loader": result.loader if result is not None else LOADER_DIGEST_ONLY,
        "state": EVIDENCE_BLOCKED if codes else EVIDENCE_VERIFIED,
        "summary": dict(sorted(result.summary.items())) if result else {},
    }
    return row, result


def _read_evidence(root: Path, relative: str) -> tuple[str, int, bytes | None]:
    path = _resolve_beneath(root, relative)
    if path.is_symlink() or not path.is_file():
        raise KitReleaseError("evidence_missing")
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
    loader: KitEvidenceLoader, source: KitEvidenceSource
) -> KitEvidenceResult:
    try:
        result = loader(source)
    except KitReleaseError as exc:
        return KitEvidenceResult(_sanitized_codes(exc.codes))
    except Exception:
        # An injected loader fails closed; its message may carry evidence text.
        return KitEvidenceResult(("evidence_loader_failed",))
    if not isinstance(result, KitEvidenceResult):
        return KitEvidenceResult(("evidence_loader_failed",))
    codes = list(_sanitized_codes(result.blocking_codes))
    summary: dict[str, str | int | bool] = {}
    for key, value in dict(result.summary).items():
        if _summary_value_ok(key, value):
            summary[key] = value
        else:
            codes.append("loader_summary_invalid")
    loader_kind = (
        result.loader
        if result.loader in (LOADER_TYPED, LOADER_DIGEST_ONLY)
        else LOADER_TYPED
    )
    return KitEvidenceResult(
        tuple(sorted(set(codes))), summary, result.document, loader_kind
    )


def _sanitized_codes(codes: Any) -> tuple[str, ...]:
    if isinstance(codes, str) or not isinstance(codes, (list, tuple)):
        codes = (codes,)
    return tuple(code if _is_code(code) else "loader_code_invalid" for code in codes)


def _is_code(value: Any) -> bool:
    return (
        isinstance(value, str)
        and _CODE_RE.fullmatch(value) is not None
        and not identifier_shaped(value)
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
    if _DIGEST_RE.fullmatch(value):
        return True
    return _SUMMARY_TEXT_RE.fullmatch(value) is not None and not identifier_shaped(
        value
    )


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
                "required": kind in REQUIRED_EVIDENCE_KINDS,
                "state": state,
            }
        )
    return table


def _single_entry_document(
    rows: Sequence[Mapping[str, Any]],
    results: Mapping[str, KitEvidenceResult],
    kind: str,
) -> tuple[str | None, Mapping[str, Any] | None]:
    entries = [row for row in rows if row["kind"] == kind]
    if len(entries) != 1:
        return None, None
    result = results.get(entries[0]["id"])
    document = result.document if result is not None else None
    if not isinstance(document, Mapping):
        return entries[0]["id"], None
    return entries[0]["id"], document


def _block(rows: list[dict[str, Any]], entry_id: str, code: str) -> None:
    for row in rows:
        if row["id"] == entry_id:
            row["blocking_codes"] = sorted({*row["blocking_codes"], code})
            row["state"] = EVIDENCE_BLOCKED


def _supported_cells(
    matrix: Mapping[str, Any] | None,
    report: Mapping[str, Any] | None,
    rows: Sequence[Mapping[str, Any]],
    results: Mapping[str, KitEvidenceResult],
) -> list[dict[str, Any]]:
    if matrix is None:
        return []
    pins = {pin["id"]: pin for pin in matrix["runtime_pins"]}
    evidence_kinds = {row["id"]: row["kind"] for row in matrix["evidence"]}
    report_cells: dict[str, Mapping[str, Any]] = {}
    report_targets: dict[str, list[Mapping[str, Any]]] = {}
    if report is not None:
        report_cells = {row["cell"]: row for row in report["cells"]}
        for target in report["targets"]:
            report_targets.setdefault(target["cell"], []).append(target)
    lanes = _lane_results(rows, results)
    table: list[dict[str, Any]] = []
    for cell in matrix["cells"]:
        if cell["status"] != SUPPORTED:
            continue
        cell_id = cell["id"]
        runtime_kind = pins[cell["runtime"]]["kind"]
        codes: list[str] = []
        if runtime_kind == "inference":
            qualification, cell_codes = _cell_qualification(
                cell,
                report_cells.get(cell_id),
                report_targets.get(cell_id, []),
            )
            codes.extend(cell_codes)
        else:
            qualification = "out_of_scope"
        lane_refs = [
            ref for ref in cell["evidence"] if evidence_kinds[ref] == "ci-lane"
        ]
        for ref in lane_refs:
            outcome = lanes.get(ref)
            if outcome is None:
                codes.append(f"lane_result_missing:{ref}")
            elif not outcome:
                codes.append(f"lane_not_passing:{ref}")
        table.append(
            {
                "blocking_codes": sorted(f"{code}:{cell_id}" for code in set(codes)),
                "cell": cell_id,
                "lane_count": len(lane_refs),
                "qualification": qualification,
                "runtime_kind": runtime_kind,
                "surface": cell["surface"],
            }
        )
    return table


def _lane_results(
    rows: Sequence[Mapping[str, Any]],
    results: Mapping[str, KitEvidenceResult],
) -> dict[str, bool]:
    lanes: dict[str, bool] = {}
    for row in rows:
        if row["kind"] != TEST_LANE:
            continue
        result = results.get(row["id"])
        document = result.document if result is not None else None
        if not isinstance(document, Mapping) or not _is_id(document.get("lane")):
            continue
        lane = str(document["lane"])
        passing = row["state"] == EVIDENCE_VERIFIED
        lanes[lane] = lanes.get(lane, True) and passing
    return lanes


def _cell_qualification(
    cell: Mapping[str, Any],
    row: Mapping[str, Any] | None,
    targets: Sequence[Mapping[str, Any]],
) -> tuple[str, list[str]]:
    if cell["device_class"] in VIRTUAL_DEVICE_CLASSES:
        return SIMULATOR_ONLY, ["simulator_only_evidence"]
    if row is not None and row.get("planned") is False:
        return "unplanned", ["supported_cell_without_evidence"]
    if row is None or not targets:
        return "missing", ["supported_cell_without_evidence"]
    # Recompute from the targets instead of trusting the cell flag: only
    # passing physical-device records may satisfy a real-device requirement.
    device_passed = all(
        target["status"] == QUALIFIED
        and target["evidence"][DEVICE_EVIDENCE]["passed"] >= 1
        and target["evidence"][DEVICE_EVIDENCE]["failed"] == 0
        for target in targets
    )
    device_seen = any(
        target["evidence"][DEVICE_EVIDENCE]["passed"]
        + target["evidence"][DEVICE_EVIDENCE]["failed"]
        for target in targets
    )
    simulator_passed = any(
        target["evidence"][SIMULATOR_EVIDENCE]["passed"] >= 1 for target in targets
    )
    fake_passed = any(
        target["evidence"][FAKE_RUNTIME_EVIDENCE]["passed"] >= 1 for target in targets
    )
    status = row["status"] if row["status"] in CELL_STATUSES else "invalid"
    if row["qualified"] is True and device_passed:
        consistent = (
            status == QUALIFIED
            and row["plan_fresh"] is True
            and not row["coverage_gaps"]
            and row["targets"] == len(targets)
        )
        if consistent:
            return QUALIFIED, []
        return "inconsistent", ["qualification_report_inconsistent"]
    if status == SIMULATOR_ONLY or (not device_seen and simulator_passed):
        return SIMULATOR_ONLY, ["simulator_only_evidence"]
    if status == FAKE_RUNTIME_ONLY or (not device_seen and fake_passed):
        return FAKE_RUNTIME_ONLY, ["fake_runtime_only_evidence"]
    if row["qualified"] is True:
        return "inconsistent", ["qualification_report_inconsistent"]
    if status in ("missing", "unplanned"):
        return status, ["supported_cell_without_evidence"]
    return status, ["supported_cell_not_qualified"]


def _surface_rows(
    matrix: Mapping[str, Any] | None,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    artifacts: dict[str, int] = {}
    for row in rows:
        if row["kind"] == KIT_ARTIFACT:
            surface = row["attributes"]["surface"]
            artifacts[surface] = artifacts.get(surface, 0) + 1
    supported: set[str] = set()
    claims: dict[str, str] = {}
    if matrix is not None:
        supported = {
            cell["surface"] for cell in matrix["cells"] if cell["status"] == SUPPORTED
        }
        claims = {row["id"]: row["support_claim"] for row in matrix["surfaces"]}
    table: list[dict[str, Any]] = []
    for surface in EPIC_SURFACES:
        required = surface in supported
        count = artifacts.get(surface, 0)
        codes = [f"kit_artifact_missing:{surface}"] if required and not count else []
        table.append(
            {
                "artifact_count": count,
                "artifact_required": required,
                "blocking_codes": codes,
                "support_claim": claims.get(surface, "unverified"),
                "surface": surface,
            }
        )
    return table


def _moved_rows(
    items: Sequence[Mapping[str, Any]],
    matrix: Mapping[str, Any] | None,
    matrix_release: str,
) -> list[dict[str, Any]]:
    cells: dict[str, str] = {}
    surfaces: dict[str, str] = {}
    if matrix is not None:
        cells = {cell["id"]: cell["status"] for cell in matrix["cells"]}
        cells.update({row["id"]: "unsupported" for row in matrix["unsupported"]})
        surfaces = {row["id"]: row["support_claim"] for row in matrix["surfaces"]}
    table: list[dict[str, Any]] = []
    for item in items:
        issue = item["issue"]
        codes: list[str] = []
        if item["priority"] == "P0":
            codes.append("moved_item_p0")
        rationale = item["rationale"]
        if rationale is None or not rationale.strip():
            codes.append("moved_item_rationale_missing")
            rationale = None
        elif rationale not in MOVED_ITEM_RATIONALES:
            codes.append("moved_item_rationale_unknown")
            rationale = None
        destination = item["destination"]
        if destination is None or not destination.strip():
            codes.append("moved_item_destination_missing")
            destination = None
        elif destination == matrix_release or not (
            _ISSUE_RE.fullmatch(destination) or _MILESTONE_RE.fullmatch(destination)
        ):
            codes.append("moved_item_destination_invalid")
            destination = None
        claims: list[str] = []
        for claim in item["narrowed_claims"]:
            if matrix is None:
                codes.append("narrowed_claim_unverifiable")
            elif claim in surfaces:
                claims.append(claim)
                if surfaces[claim] == "supported":
                    codes.append(f"narrowed_claim_still_supported:{claim}")
            elif claim in cells:
                claims.append(claim)
                if cells[claim] == SUPPORTED:
                    codes.append(f"narrowed_claim_still_supported:{claim}")
            else:
                codes.append("narrowed_claim_unknown")
        table.append(
            {
                "blocking_codes": sorted({f"{code}:{issue}" for code in codes}),
                "destination": destination,
                "issue": issue,
                "narrowed_claims": claims,
                "priority": item["priority"],
                "rationale": rationale,
            }
        )
    return table


# --- verification helpers ----------------------------------------------------


def _structural_codes(packet: KitReleasePacket) -> set[str]:
    codes: set[str] = set()
    order = {kind: index for index, kind in enumerate(EVIDENCE_KINDS)}
    keys = []
    counts: dict[str, int] = {}
    candidate = packet.release.get("candidate_commit")
    for row in packet.evidence:
        kind, entry_id = row["kind"], row["id"]
        if kind not in order or not _is_id(entry_id):
            raise KitReleaseError("packet_malformed")
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
    ready = packet.decision == KIT_RELEASE_READY
    clean = (
        not packet.blocking_codes
        and packet.repository_binding.get("verified") is True
        and all(row["state"] == EVIDENCE_VERIFIED for row in packet.evidence)
        and all(row["state"] == KIND_PRESENT for row in packet.evidence_kinds)
        and not any(code.startswith("wrong_candidate:") for code in codes)
    )
    if packet.decision not in (KIT_RELEASE_READY, KIT_RELEASE_NOT_READY) or (
        ready != clean
    ):
        codes.add("decision_inconsistent")
    return codes


def _authentic(packet: KitReleasePacket, key: bytes, codes: set[str]) -> bool:
    before = len(codes)
    if packet.recompute_packet_digest() != packet.packet_digest:
        codes.add("packet_digest_mismatch")
    signature = packet.signature
    if signature is None:
        codes.add("signature_missing")
    elif signature.algorithm != KIT_RELEASE_SIGNATURE_ALGORITHM:
        codes.add("signature_algorithm_unsupported")
    else:
        signed = {**packet._unsigned_payload(), "packet_digest": packet.packet_digest}
        if not hmac.compare_digest(_release_hmac(key, signed), signature.value):
            codes.add("signature_invalid")
    return len(codes) == before


def _verification(codes: Sequence[str], *, authentic: bool) -> KitReleaseVerification:
    return KitReleaseVerification(
        valid=False,
        authentic=authentic,
        decision=KIT_RELEASE_NOT_READY,
        reason_codes=tuple(sorted(codes)),
    )


# --- small helpers -----------------------------------------------------------


def _report_shape_valid(document: Mapping[str, Any]) -> bool:
    try:
        matrix = document["matrix"]
        summary = document["summary"]
        by_class = summary["records_by_class"]
        counts_ok = all(
            _is_count(summary[name])
            for name in (
                "cells_planned",
                "cells_qualified",
                "records_accepted",
                "records_rejected",
            )
        ) and all(
            _is_count(by_class[name])
            for name in (DEVICE_EVIDENCE, SIMULATOR_EVIDENCE, FAKE_RUNTIME_EVIDENCE)
        )
        cells_ok = all(
            _is_id(row["cell"])
            and type(row["planned"]) is bool
            and type(row["qualified"]) is bool
            and isinstance(row["status"], str)
            and _is_count(row["targets"])
            and isinstance(row["coverage_gaps"], list)
            for row in document["cells"]
        )
        targets_ok = all(
            _is_id(target["cell"])
            and isinstance(target["status"], str)
            and all(
                _is_count(target["evidence"][name]["passed"])
                and _is_count(target["evidence"][name]["failed"])
                for name in (
                    DEVICE_EVIDENCE,
                    SIMULATOR_EVIDENCE,
                    FAKE_RUNTIME_EVIDENCE,
                )
            )
            for target in document["targets"]
        )
        return (
            isinstance(matrix["digest"], str)
            and _HEX64_RE.fullmatch(matrix["digest"]) is not None
            and isinstance(matrix["release"], str)
            and isinstance(document["report_digest"], str)
            and _HEX64_RE.fullmatch(document["report_digest"]) is not None
            and counts_ok
            and cells_ok
            and targets_ok
        )
    except (KeyError, TypeError, AttributeError):
        return False


def _known_schema(value: Any) -> bool:
    return isinstance(value, str) and value in _SUPPORTED_SCHEMA_VERSIONS


def _entry_order(entry: Mapping[str, Any]) -> tuple[int, str]:
    return (EVIDENCE_KINDS.index(entry["kind"]), entry["id"])


def _is_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and _ID_RE.fullmatch(value) is not None
        and not identifier_shaped(value)
        and not identifier_shaped(value.upper())
    )


def _is_count(value: Any) -> bool:
    return type(value) is int and 0 <= value <= _MAX_COUNT


def _portable_path(value: Any) -> bool:
    if not isinstance(value, str) or not _PORTABLE_PATH_RE.fullmatch(value):
        return False
    parts = value.split("/")
    return all(part not in {"", ".", ".."} for part in parts)


def _timestamp(value: Any, code: str) -> datetime:
    try:
        return _parse_timestamp(value, code)
    except JourneyReleaseError:
        raise KitReleaseError(code) from None


def _canonical_timestamp(value: Any, code: str) -> str:
    return _timestamp(value, code).isoformat().replace("+00:00", "Z")


def _key_or_raise(key: bytes | str) -> bytes:
    if not isinstance(key, (bytes, str)):
        raise KitReleaseError("signing_key_invalid")
    try:
        return _key_bytes(key)
    except JourneyReleaseError:
        raise KitReleaseError("signing_key_invalid") from None


def _load_signing_key(path: Path | None) -> bytes:
    if path is not None:
        try:
            if path.is_symlink() or not path.is_file():
                raise KitReleaseError("signing_key_unreadable")
            value = path.read_bytes().strip()
        except OSError:
            raise KitReleaseError("signing_key_unreadable") from None
    else:
        value = os.environ.get(KIT_RELEASE_SIGNING_ENV, "").encode("utf-8")
    if len(value) < 32:
        raise KitReleaseError("signing_key_missing")
    return value


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
        raise KitReleaseError(code) from None


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _packet_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(k, str) for k in value):
        raise KitReleaseError("packet_malformed")
    return dict(value)


def _packet_rows(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        raise KitReleaseError("packet_malformed")
    return tuple(_packet_mapping(item) for item in value)


def _packet_codes(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise KitReleaseError("packet_malformed")
    return tuple(value)


def _packet_str(value: Any) -> str:
    if not isinstance(value, str):
        raise KitReleaseError("packet_malformed")
    return value


def _load_bundled_schema(name: str) -> dict[str, Any]:
    resource = resources.files(_SCHEMA_PACKAGE).joinpath(name)
    with resource.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise KitReleaseError("bundled_schema_invalid")
    return payload


__all__ = [
    "BUNDLE_MANIFEST",
    "DEFAULT_EVIDENCE_LOADERS",
    "EVIDENCE_BLOCKED",
    "EVIDENCE_KINDS",
    "EVIDENCE_VERIFIED",
    "KIND_BLOCKED",
    "KIND_NOT_PRESENT",
    "KIND_PRESENT",
    "KIT_ARTIFACT",
    "KIT_ARTIFACT_FORMATS",
    "KIT_RELEASE_COMPATIBILITY_POLICY",
    "KIT_RELEASE_LIMITATIONS",
    "KIT_RELEASE_NOT_READY",
    "KIT_RELEASE_READY",
    "KIT_RELEASE_SCHEMA_VERSION",
    "KIT_RELEASE_SIGNATURE_ALGORITHM",
    "KIT_RELEASE_SIGNING_ENV",
    "KIT_RELEASE_TEST_LANE_SCHEMA",
    "LICENSE_INVENTORY",
    "LOADER_DIGEST_ONLY",
    "LOADER_TYPED",
    "MAX_EVIDENCE_ENTRIES",
    "MAX_MOVED_ITEMS",
    "MAX_TYPED_EVIDENCE_BYTES",
    "MOVED_ITEM_PRIORITIES",
    "MOVED_ITEM_RATIONALES",
    "PARITY_FIXTURE",
    "QUALIFICATION_REPORT",
    "REQUIRED_EVIDENCE_KINDS",
    "SINGLE_ENTRY_KINDS",
    "SUPPORT_MATRIX",
    "TEST_LANE",
    "THREAT_MODEL",
    "KitEvidenceLoader",
    "KitEvidenceResult",
    "KitEvidenceSource",
    "KitReleaseError",
    "KitReleasePacket",
    "KitReleaseVerification",
    "build_arg_parser",
    "evaluate_kit_release",
    "load_digest_only_evidence",
    "load_kit_artifact_evidence",
    "load_kit_release_manifest",
    "load_kit_release_manifest_schema",
    "load_kit_release_packet",
    "load_kit_release_packet_schema",
    "load_qualification_report_evidence",
    "load_support_matrix_evidence",
    "load_test_lane_evidence",
    "main",
    "validate_kit_release_manifest",
    "verify_kit_release_packet",
    "write_kit_release_packet",
]


if __name__ == "__main__":
    raise SystemExit(main())
