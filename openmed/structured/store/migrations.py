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
)

LATEST_MIGRATION_VERSION = MIGRATIONS[-1].version
