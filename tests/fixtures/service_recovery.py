"""Synthetic service state for encrypted backup and restore tests.

Every key, record, and sentinel here is synthetic.  The builder produces a
state root in the canonical recovery layout: an encrypted local Journey store
with artifacts, evidence, and facts, an async job metadata store, a
model-registry state document, and an audit chain.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openmed.clinical.journey_contracts import (
    ClinicalArtifact,
    ClinicalFact,
    EvidenceLocator,
    canonical_digest,
    sha256_digest,
)
from openmed.core.audit import AuditReport, AuditSpan, DetectorInfo, hash_text
from openmed.core.audit_chain import AuditChain
from openmed.core.key_provider import InMemoryKeyProvider, KeyState
from openmed.core.registry_slots import registry_slot_key
from openmed.service.recovery import (
    AUDIT_CHAIN_DIRECTORY,
    BACKUP_KEY_PURPOSE,
    JOB_STORE_PATH,
    REGISTRY_STATE_FILE,
    ServiceStateSources,
)
from openmed.structured.store import (
    JOURNEY_KEY_PURPOSE,
    JourneyStoreEncryption,
    LocalJourneyStore,
)

SENTINEL = "ZQRECOVERYSENTINEL5821"
SENTINEL_BYTES = SENTINEL.encode("ascii")
JOURNEY_KEY = bytes(range(32))
BACKUP_KEY = bytes(range(64, 96))
OTHER_BACKUP_KEY = bytes(range(96, 128))
ROTATED_BACKUP_KEY = bytes(range(128, 160))
SUBJECT = "subject_aaaaaaaaaaaaaaaa"
COMMITTED_AT = "2026-01-02T03:04:05Z"
SLOT = registry_slot_key("pii", "small", "pytorch")


def journey_provider(
    material: bytes = JOURNEY_KEY, key_id: str = "journey-key-1"
) -> InMemoryKeyProvider:
    """Return a provider holding one active synthetic Journey key."""

    provider = InMemoryKeyProvider()
    provider.add_key(
        key_id, material, purpose=JOURNEY_KEY_PURPOSE, state=KeyState.ACTIVE
    )
    return provider


def backup_provider(
    material: bytes = BACKUP_KEY, key_id: str = "backup-key-1"
) -> InMemoryKeyProvider:
    """Return a provider holding one active synthetic backup key."""

    provider = InMemoryKeyProvider()
    provider.add_key(
        key_id, material, purpose=BACKUP_KEY_PURPOSE, state=KeyState.ACTIVE
    )
    return provider


def journey_encryption(
    provider: InMemoryKeyProvider | None = None,
) -> JourneyStoreEncryption:
    """Return an encrypted Journey configuration."""

    return JourneyStoreEncryption.encrypted(provider or journey_provider())


def _audit_report(index: int) -> AuditReport:
    original = f"Patient {SENTINEL}{index} called 555-01{index:02d}."
    start = original.index(SENTINEL)
    end = start + len(SENTINEL) + len(str(index))
    return AuditReport(
        policy="hipaa_safe_harbor",
        resolved_profile={"method": "mask", "language": "en"},
        detectors=[
            DetectorInfo(source="ml", model_id="synthetic-model", model_format="x")
        ],
        safety_sweep={"spans_added": 0},
        spans=[
            AuditSpan(
                start=start,
                end=end,
                label="NAME",
                canonical_label="PERSON",
                sources=["ml"],
                confidence=0.99,
                threshold=0.5,
                action="mask",
                surrogate="[PERSON]",
                text_hash=hash_text(original[start:end]),
            )
        ],
        thresholds={"PERSON": 0.5},
        residual_risk={},
        openmed_version="0.0.0",
        manifest_hash=hash_text("synthetic manifest"),
        document_length=len(original),
        input_hash=hash_text(original),
        deidentified_text_hash=hash_text("Patient [PERSON] called 555."),
    )


@dataclass
class SyntheticServiceState:
    """A synthetic state root plus the open Journey store that writes it."""

    root: Path
    encryption: JourneyStoreEncryption = field(repr=False)
    store: LocalJourneyStore = field(repr=False)
    notes: int = 0
    audit_entries: int = 0
    jobs: dict[str, dict[str, Any]] = field(default_factory=dict)

    def sources(self) -> ServiceStateSources:
        """Return sources for the canonical layout."""

        return ServiceStateSources.from_state_root(self.root)

    def add_note(self, committed_at: str = COMMITTED_AT) -> ClinicalArtifact:
        """Ingest one artifact with evidence and a fact carrying the sentinel."""

        index = self.notes
        suffix = f"{index:016d}"
        content = f"synthetic note {index} for {SENTINEL}".encode("ascii")
        artifact = ClinicalArtifact(
            artifact_id=f"artifact_{suffix}",
            artifact_type="clinical_note",
            media_type="text/plain",
            content_hash=sha256_digest(content),
            byte_size=len(content),
            source_id="source_aaaaaaaaaaaaaaaa",
            recorded_at=committed_at,
            subject_id=SUBJECT,
        )
        evidence = EvidenceLocator(
            locator_id=f"evidence_{suffix}",
            artifact_id=artifact.artifact_id,
            location_type="text_span",
            location={"start": 0, "end": 9},
        )
        fact = ClinicalFact(
            fact_id=f"fact_{suffix}",
            subject_id=SUBJECT,
            fact_type="condition",
            value={"code": "synthetic-condition", "note": SENTINEL},
            status="active",
            evidence_ids=(evidence.locator_id,),
            derivation_hash=canonical_digest({"fact": index}),
        )
        result = self.store.ingest_graph(
            artifact,
            content,
            evidence=(evidence,),
            facts=(fact,),
            committed_at=committed_at,
        )
        assert result.ok, result.code
        self.notes += 1
        return artifact

    def add_job(self, status: str = "done") -> str:
        """Persist one job metadata record in the local job-store format."""

        job_id = f"job-{len(self.jobs):04d}"
        self.jobs[job_id] = {
            "created_at": COMMITTED_AT,
            "document_count": 1,
            "expires_at": "2099-01-01T00:00:00Z",
            "id": job_id,
            "label": SENTINEL,
            "status": status,
            "updated_at": COMMITTED_AT,
        }
        self.write_jobs()
        return job_id

    def write_jobs(self) -> None:
        """Rewrite the job store atomically."""

        path = self.root / JOB_STORE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(
            json.dumps({"jobs": self.jobs}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def add_audit_entry(self) -> None:
        """Append one synthetic report to the audit chain file."""

        path = self.root / AUDIT_CHAIN_DIRECTORY / "chain-0000.json"
        chain = AuditChain.load(path) if path.exists() else AuditChain()
        chain.append(_audit_report(self.audit_entries))
        chain.write(path)
        self.audit_entries += 1

    def write_registry(self, latest: str | None = "OpenMed/synthetic-pii") -> None:
        """Write a schema-valid registry state with one slot."""

        state: Mapping[str, Any] = {
            "schema_version": 2,
            "slots": {
                SLOT: {
                    "checkpoints": {} if latest is None else {latest: "1.0.0"},
                    "lineage": [],
                    "pointers": {
                        "canary": None,
                        "last_green": latest,
                        "latest": latest,
                    },
                }
            },
        }
        path = self.root / REGISTRY_STATE_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")

    def close(self) -> None:
        """Close the Journey store."""

        self.store.close()


def build_service_state(
    root: Path,
    *,
    encryption: JourneyStoreEncryption | None = None,
    notes: int = 2,
    jobs: int = 2,
    audit_entries: int = 2,
) -> SyntheticServiceState:
    """Create a synthetic service state root in the canonical layout."""

    root.mkdir(parents=True, exist_ok=True)
    selected = encryption or journey_encryption()
    state = SyntheticServiceState(
        root=root,
        encryption=selected,
        store=LocalJourneyStore(root / "journey", encryption=selected),
    )
    for _ in range(notes):
        state.add_note()
    for _ in range(jobs):
        state.add_job()
    for _ in range(audit_entries):
        state.add_audit_entry()
    state.write_registry()
    return state


def logical_snapshot(root: Path, encryption: JourneyStoreEncryption) -> dict[str, Any]:
    """Return a comparable logical view of a state root's durable records."""

    store = LocalJourneyStore(root / "journey", encryption=encryption)
    try:
        facts = store.metadata.list_facts(SUBJECT)
        assert facts.ok, facts.code
        integrity = store.metadata.integrity_check()
        assert integrity.ok, integrity.code
        artifacts = {}
        for fact in facts.value or ():
            artifact_id = "artifact_" + fact.fact_id.removeprefix("fact_")
            artifact = store.metadata.get_artifact(artifact_id)
            assert artifact.ok, artifact.code
            blob = store.artifacts.get_bytes(artifact.value.content_hash)
            assert blob.ok, blob.code
            artifacts[artifact_id] = blob.value
        journey = {
            "artifacts": artifacts,
            "facts": sorted(fact.fact_id for fact in facts.value or ()),
            "integrity": dict(integrity.value or {}),
            "revision": store.metadata.latest_revision,
        }
    finally:
        store.close()
    audit = AuditChain.load(root / AUDIT_CHAIN_DIRECTORY / "chain-0000.json")
    return {
        "audit": (len(audit.entries), audit.head_hash, audit.verify().valid),
        "jobs": json.loads((root / JOB_STORE_PATH).read_text(encoding="utf-8")),
        "journey": journey,
        "registry": json.loads((root / REGISTRY_STATE_FILE).read_text("utf-8")),
    }


def tree_bytes(root: Path) -> bytes:
    """Return the concatenated bytes of every regular file under ``root``."""

    chunks = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            chunks.append(path.read_bytes())
    return b"\n".join(chunks)
