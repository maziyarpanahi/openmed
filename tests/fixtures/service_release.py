"""Synthetic builders for the v4.0 secure-service release-gate tests.

Every record, digest and commit here is synthetic. The signing key below is
committed on purpose and is TEST-ONLY: it signs fixture packets so offline
verification can be exercised, and it must never sign real release evidence.

The typed loaders and drill steps in this module are TEST-ONLY stand-ins for
producers that are still open work (#3551, #3552, #3574, #3577, #3578, #3548
and #3599). They let the tests reach a ``READY`` packet; they are not
evidence that those controls exist.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from openmed.eval import service_recovery_drill as drill
from openmed.eval.service_release import (
    FIXTURE,
    LOADER_TYPED,
    MIGRATION_REHEARSAL_MATRIX,
    NO_EGRESS_ATTESTATION,
    PROTOCOL_MAP,
    SECURITY_PROFILE_REPORT,
    ServiceEvidenceLoader,
    ServiceEvidenceResult,
    ServiceEvidenceSource,
    build_failure_code_inventory,
    build_p0_status,
    load_service_p0_checklist,
)

# TEST-ONLY committed keys. Never use them for a real release packet.
TEST_ONLY_SIGNING_KEY = b"test-only-openmed-service-release-key-never-for-release"
TEST_ONLY_KEY_ID = "test-only-service-release"
OTHER_TEST_ONLY_KEY = b"test-only-other-service-release-key-for-negative-tests"

CANDIDATE = "3603" * 10
OTHER_COMMIT = "4253" * 10
VERSION = "4.0.0"
EVALUATED_AT = "2026-11-08T00:00:00Z"
EVALUATED_EPOCH = 1_794_096_000
GENERATED_AT = "2026-11-07T12:00:00Z"
PLANTED_SENTINEL = "SYNTH-SENTINEL-3603-DO-NOT-EMIT"

STAND_IN_KINDS = (
    SECURITY_PROFILE_REPORT,
    NO_EGRESS_ATTESTATION,
    MIGRATION_REHEARSAL_MATRIX,
    PROTOCOL_MAP,
)

# Evidence id -> (kind, relative path).
EVIDENCE_FILES = {
    "security-profile": (SECURITY_PROFILE_REPORT, "evidence/security-profile.json"),
    "no-egress": (NO_EGRESS_ATTESTATION, "evidence/no-egress.json"),
    "recovery-drill": ("recovery_drill_report", "evidence/recovery-drill.json"),
    "migration-rehearsal": (
        MIGRATION_REHEARSAL_MATRIX,
        "evidence/migration-rehearsal.json",
    ),
    "protocol-map": (PROTOCOL_MAP, "evidence/protocol-map.json"),
    "synthetic-request": (FIXTURE, "tests/fixtures/service/synthetic-request.json"),
    "failure-codes": ("failure_code_inventory", "evidence/failure-codes.json"),
    "p0-status": ("p0_status", "evidence/p0-status.json"),
    "threat-model": ("threat_model", "docs/security/service-threat-model.md"),
}


def head_resolver(commit: str = CANDIDATE) -> Callable[[Path], str]:
    """Return an injected ``HEAD`` resolver pinned to *commit*."""

    return lambda _root: commit


def stand_in_document(
    kind: str, *, state: str = "success", candidate: str = CANDIDATE
) -> dict[str, Any]:
    """Return a TEST-ONLY synthetic document for a kind without a producer."""

    return {
        "candidate_commit": candidate,
        "schema": f"test-only.{kind}.v1",
        "state": state,
    }


def stand_in_loader(kind: str) -> ServiceEvidenceLoader:
    """Return a TEST-ONLY typed loader for a stand-in document."""

    def loader(source: ServiceEvidenceSource) -> ServiceEvidenceResult:
        document = source.read_json()
        if document.get("schema") != f"test-only.{kind}.v1":
            return ServiceEvidenceResult(("unknown_schema",))
        codes = []
        if document["candidate_commit"] != source.release["candidate_commit"]:
            codes.append("wrong_candidate")
        if document["state"] != "success":
            codes.append("stand_in_not_successful")
        return ServiceEvidenceResult(
            tuple(codes), {"state": document["state"]}, document, LOADER_TYPED
        )

    return loader


def stand_in_loaders() -> dict[str, ServiceEvidenceLoader]:
    """TEST-ONLY typed loaders for every kind whose producer is open work."""

    return {kind: stand_in_loader(kind) for kind in STAND_IN_KINDS}


def _passing(check: str) -> drill.DrillStep:
    def step(context: drill.DrillContext) -> drill.DrillStepOutcome:
        del context
        return drill.DrillStepOutcome(
            drill.STEP_PASSED, check, {"unverified_restored_records": 0}
        )

    return step


def stand_in_drill_steps() -> dict[tuple[str, str], drill.DrillStep]:
    """TEST-ONLY passing steps for every phase that is ``not_available``."""

    return {
        key: _passing(check)
        for key, (_owner, check) in drill.UNAVAILABLE_DRILL_STEPS.items()
    }


def complete_drill_report(candidate: str = CANDIDATE) -> dict[str, Any]:
    """Run the drill with TEST-ONLY stand-ins so every phase passes."""

    return drill.run_recovery_drill(
        candidate_commit=candidate,
        generated_at=GENERATED_AT,
        steps=stand_in_drill_steps(),
    )


def all_closed_status(candidate: str = CANDIDATE) -> dict[str, Any]:
    """P0 status with every committed checklist issue closed."""

    checklist = load_service_p0_checklist()
    return build_p0_status(
        {item["issue"]: "closed" for item in checklist["items"]},
        candidate_commit=candidate,
    )


def default_documents(candidate: str = CANDIDATE) -> dict[str, Any]:
    """Return the synthetic evidence contents keyed by evidence id."""

    def stand_in(kind: str) -> dict[str, Any]:
        return stand_in_document(kind, candidate=candidate)

    return {
        "security-profile": stand_in(SECURITY_PROFILE_REPORT),
        "no-egress": stand_in(NO_EGRESS_ATTESTATION),
        "recovery-drill": complete_drill_report(candidate),
        "migration-rehearsal": stand_in(MIGRATION_REHEARSAL_MATRIX),
        "protocol-map": stand_in(PROTOCOL_MAP),
        "synthetic-request": {
            "document_id": "doc-synthetic-1",
            "text_sha256": "0" * 64,
        },
        "failure-codes": build_failure_code_inventory(candidate),
        "p0-status": all_closed_status(candidate),
        "threat-model": "# Synthetic service threat model\n\nSV-01 control.\n",
    }


def render(content: Any) -> bytes:
    """Render JSON or text evidence deterministically."""

    if isinstance(content, (bytes, bytearray)):
        return bytes(content)
    if isinstance(content, str):
        return content.encode("utf-8")
    return (json.dumps(content, indent=2, sort_keys=True) + "\n").encode("utf-8")


def digest(data: bytes) -> str:
    """Return a prefixed SHA-256 digest."""

    return "sha256:" + hashlib.sha256(data).hexdigest()


def make_service_release(
    tmp_path: Path,
    *,
    documents: Mapping[str, Any] | None = None,
    omit: tuple[str, ...] = (),
    candidate: str = CANDIDATE,
    root: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Write synthetic evidence beneath *tmp_path* and return a manifest.

    Args:
        tmp_path: Directory used as the candidate checkout.
        documents: Overrides for :func:`default_documents` by evidence id.
        omit: Evidence ids to leave out of the manifest and tree.
        candidate: Commit every document and entry names.
        root: Existing checkout to write into instead of ``tmp_path``.

    Returns:
        The checkout root and a manifest that references every written file.
    """

    root = root or tmp_path / "candidate"
    root.mkdir(parents=True, exist_ok=True)
    contents = default_documents(candidate)
    contents.update(documents or {})
    evidence = []
    for entry_id, (kind, relative) in EVIDENCE_FILES.items():
        if entry_id in omit:
            continue
        data = render(contents[entry_id])
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        evidence.append(
            {
                "candidate_commit": candidate,
                "generated_at": GENERATED_AT,
                "id": entry_id,
                "kind": kind,
                "path": relative,
                "sha256": digest(data),
            }
        )
    manifest = {
        "compatibility_policy": "same_major",
        "evidence": evidence,
        "moved_items": [],
        "policy": {"max_evidence_age_seconds": 7 * 24 * 3600},
        "release": {
            "candidate_commit": candidate,
            "evaluated_at": EVALUATED_AT,
            "source_date_epoch": EVALUATED_EPOCH,
            "target": "v4.0",
            "version": VERSION,
        },
        "schema_version": "1.0.0",
    }
    return root, manifest


def entry(manifest: Mapping[str, Any], entry_id: str) -> dict[str, Any]:
    """Return the mutable manifest entry for *entry_id*."""

    return next(item for item in manifest["evidence"] if item["id"] == entry_id)


def rewrite(root: Path, manifest: dict[str, Any], entry_id: str, content: Any) -> None:
    """Replace one evidence file and update its manifest digest."""

    item = entry(manifest, entry_id)
    data = render(content)
    (root / item["path"]).write_bytes(data)
    item["sha256"] = digest(data)


def clone(value: Any) -> Any:
    """Deep-copy JSON-like test data."""

    return copy.deepcopy(value)
