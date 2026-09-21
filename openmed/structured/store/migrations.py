"""Deterministic SQLite migrations for the local Journey metadata store."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StoreMigration:
    """One immutable ordered SQLite migration."""

    version: int
    name: str
    statements: tuple[str, ...]

    @property
    def checksum(self) -> str:
        """Return a deterministic checksum for migration drift detection."""

        payload = "\n-- statement --\n".join(self.statements).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


MIGRATIONS = (
    StoreMigration(
        version=1,
        name="initial_append_only_store",
        statements=(
            """
            CREATE TABLE store_revisions (
                revision INTEGER PRIMARY KEY AUTOINCREMENT,
                committed_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE artifacts (
                artifact_id TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL UNIQUE,
                payload_hash TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL,
                created_revision INTEGER NOT NULL
                    REFERENCES store_revisions(revision)
            )
            """,
            """
            CREATE TABLE evidence_locators (
                locator_id TEXT PRIMARY KEY,
                artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
                payload_hash TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL,
                created_revision INTEGER NOT NULL
                    REFERENCES store_revisions(revision)
            )
            """,
            """
            CREATE TABLE clinical_facts (
                fact_id TEXT PRIMARY KEY,
                subject_id TEXT NOT NULL,
                fact_type TEXT NOT NULL,
                payload_hash TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL,
                created_revision INTEGER NOT NULL
                    REFERENCES store_revisions(revision)
            )
            """,
            """
            CREATE TABLE fact_evidence (
                fact_id TEXT NOT NULL REFERENCES clinical_facts(fact_id),
                locator_id TEXT NOT NULL REFERENCES evidence_locators(locator_id),
                PRIMARY KEY (fact_id, locator_id)
            )
            """,
            """
            CREATE TABLE fact_parents (
                fact_id TEXT NOT NULL REFERENCES clinical_facts(fact_id),
                parent_fact_id TEXT NOT NULL REFERENCES clinical_facts(fact_id),
                PRIMARY KEY (fact_id, parent_fact_id)
            )
            """,
            """
            CREATE TABLE conflict_sets (
                conflict_id TEXT PRIMARY KEY,
                subject_id TEXT NOT NULL,
                payload_hash TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL,
                created_revision INTEGER NOT NULL
                    REFERENCES store_revisions(revision)
            )
            """,
            """
            CREATE TABLE conflict_facts (
                conflict_id TEXT NOT NULL REFERENCES conflict_sets(conflict_id),
                fact_id TEXT NOT NULL REFERENCES clinical_facts(fact_id),
                PRIMARY KEY (conflict_id, fact_id)
            )
            """,
            """
            CREATE TABLE resolution_events (
                resolution_id TEXT PRIMARY KEY,
                conflict_id TEXT NOT NULL REFERENCES conflict_sets(conflict_id),
                supersedes_resolution_id TEXT
                    REFERENCES resolution_events(resolution_id),
                payload_hash TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL,
                created_revision INTEGER NOT NULL
                    REFERENCES store_revisions(revision)
            )
            """,
            """
            CREATE TABLE dataset_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                payload_hash TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL,
                created_revision INTEGER NOT NULL
                    REFERENCES store_revisions(revision)
            )
            """,
            """
            CREATE TABLE canonical_record_versions (
                canonical_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                subject_id TEXT NOT NULL,
                fact_id TEXT NOT NULL REFERENCES clinical_facts(fact_id),
                payload_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_revision INTEGER NOT NULL
                    REFERENCES store_revisions(revision),
                PRIMARY KEY (canonical_id, version),
                UNIQUE (canonical_id, payload_hash)
            )
            """,
            """
            CREATE TABLE job_metadata_versions (
                job_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                state TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_revision INTEGER NOT NULL
                    REFERENCES store_revisions(revision),
                PRIMARY KEY (job_id, version),
                UNIQUE (job_id, payload_hash)
            )
            """,
            """
            CREATE INDEX clinical_facts_subject_revision_idx
            ON clinical_facts(subject_id, created_revision, fact_id)
            """,
            """
            CREATE INDEX canonical_records_revision_idx
            ON canonical_record_versions(canonical_id, created_revision, version)
            """,
            """
            CREATE INDEX job_metadata_revision_idx
            ON job_metadata_versions(job_id, created_revision, version)
            """,
        ),
    ),
    StoreMigration(
        version=2,
        name="ingestion_control_plane",
        statements=(
            """
            CREATE TABLE ingestion_manifests (
                manifest_digest TEXT PRIMARY KEY,
                manifest_id TEXT NOT NULL UNIQUE,
                payload_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE ingestion_jobs (
                job_id TEXT PRIMARY KEY,
                manifest_digest TEXT NOT NULL UNIQUE
                    REFERENCES ingestion_manifests(manifest_digest)
            )
            """,
            """
            CREATE TABLE ingestion_job_versions (
                job_id TEXT NOT NULL REFERENCES ingestion_jobs(job_id),
                version INTEGER NOT NULL,
                state TEXT NOT NULL,
                checkpoint_sequence INTEGER NOT NULL,
                recorded_at TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (job_id, version),
                UNIQUE (job_id, payload_hash)
            )
            """,
            """
            CREATE TABLE ingestion_replay_audits (
                replay_id TEXT PRIMARY KEY,
                manifest_digest TEXT NOT NULL
                    REFERENCES ingestion_manifests(manifest_digest),
                job_id TEXT NOT NULL REFERENCES ingestion_jobs(job_id),
                action TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                payload_hash TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE ingestion_leases (
                lease_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL REFERENCES ingestion_jobs(job_id),
                worker_id TEXT NOT NULL,
                epoch INTEGER NOT NULL,
                acquired_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                payload_hash TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL,
                UNIQUE (job_id, epoch)
            )
            """,
            """
            CREATE TABLE ingestion_lease_releases (
                lease_id TEXT PRIMARY KEY REFERENCES ingestion_leases(lease_id),
                released_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE ingestion_checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL REFERENCES ingestion_jobs(job_id),
                sequence INTEGER NOT NULL,
                step TEXT NOT NULL,
                input_digest TEXT NOT NULL,
                output_digest TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                payload_hash TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL,
                UNIQUE (job_id, sequence),
                UNIQUE (job_id, step, input_digest)
            )
            """,
            """
            CREATE TABLE ingestion_retries (
                retry_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL REFERENCES ingestion_jobs(job_id),
                classification TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                recorded_at TEXT NOT NULL,
                payload_hash TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE ingestion_cancellations (
                cancellation_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL UNIQUE REFERENCES ingestion_jobs(job_id),
                requested_at TEXT NOT NULL,
                payload_hash TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE ingestion_quarantine_results (
                quarantine_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL REFERENCES ingestion_jobs(job_id),
                classification TEXT NOT NULL,
                created_at TEXT NOT NULL,
                payload_hash TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE ingestion_quarantine_promotions (
                promotion_id TEXT PRIMARY KEY,
                quarantine_id TEXT NOT NULL UNIQUE
                    REFERENCES ingestion_quarantine_results(quarantine_id),
                promoted_at TEXT NOT NULL,
                payload_hash TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL
            )
            """,
            """
            CREATE INDEX ingestion_job_state_idx
            ON ingestion_job_versions(job_id, version, state)
            """,
            """
            CREATE INDEX ingestion_lease_job_epoch_idx
            ON ingestion_leases(job_id, epoch)
            """,
            """
            CREATE INDEX ingestion_checkpoint_job_sequence_idx
            ON ingestion_checkpoints(job_id, sequence)
            """,
        ),
    ),
)

LATEST_MIGRATION_VERSION = MIGRATIONS[-1].version
