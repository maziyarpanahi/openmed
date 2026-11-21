"""Local append-only, hash-linked history of tabular release aggregates."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

from openmed.core.audit import stable_hash
from openmed.core.audit_chain import AuditChain, AuditChainEntry
from openmed.risk.release import AnonymizationResult


class ReleaseHistory:
    """Keep a verified history for one opaque dataset lineage in a SQLite file.

    Args:
        path: Local SQLite file, shared by any number of lineages.
        lineage_id: Stable dataset lineage identifier. Only its digest is stored.
    """

    def __init__(self, path: str | Path, lineage_id: str):
        if not isinstance(lineage_id, str) or not lineage_id.strip():
            raise ValueError("a lineage identifier is required")
        self.path = Path(path)
        self._lineage = stable_hash(["openmed-release-lineage", lineage_id])
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS release_entries (
                lineage TEXT NOT NULL, sequence INTEGER NOT NULL, record TEXT NOT NULL,
                chain_entry TEXT NOT NULL, PRIMARY KEY(lineage, sequence))""")
            db.execute("""CREATE TABLE IF NOT EXISTS release_heads (
                lineage TEXT PRIMARY KEY, entry_count INTEGER NOT NULL, head_hash TEXT NOT NULL)""")

    def append(self, release: AnonymizationResult, *, released_on: date) -> str:
        """Append the safe fields of a release and return its new chain head."""
        if (
            not isinstance(release, AnonymizationResult)
            or type(released_on) is not date
        ):
            raise ValueError("a validated release and release date are required")
        record = {
            "dataset_digest": release.released_dataset_digest,
            "schema_digest": release.released_schema_digest,
            "quasi_identifiers": sorted(release.policy.quasi_identifiers),
            "generalization_levels": {
                field: level for field, level, _, _ in release.generalization.levels
            },
            "suppressed_rows": release.generalization.suppressed_rows,
            "achieved_k": release.after.achieved_k,
            "released_on": released_on.isoformat(),
        }
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            _, chain = self._read(db)
            entry = AuditChainEntry(
                sequence=chain.next_sequence,
                previous_hash=chain.head_hash,
                report_hash=stable_hash([self._lineage, record]),
                input_hash=release.released_schema_digest,
                deidentified_text_hash=release.released_dataset_digest,
                spans=(),
                entry_hash=AuditChain.GENESIS_HASH,
            )
            entry = replace(entry, entry_hash=entry.compute_hash())
            db.execute(
                "INSERT INTO release_entries VALUES (?, ?, ?, ?)",
                (
                    self._lineage,
                    entry.sequence,
                    json.dumps(record, sort_keys=True),
                    json.dumps(entry.to_dict(), sort_keys=True),
                ),
            )
            db.execute(
                "INSERT OR REPLACE INTO release_heads VALUES (?, ?, ?)",
                (self._lineage, entry.sequence + 1, entry.entry_hash),
            )
        return entry.entry_hash

    def entries(self) -> tuple[dict[str, Any], ...]:
        """Load aggregate records in order, raising ValueError if verification fails."""
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("BEGIN")
            return self._read(db)[0]

    def verify(self) -> bool:
        """Check the retained head, chain links and each record's digest."""
        try:
            self.entries()
        except (ValueError, TypeError, KeyError):
            return False
        return True

    def _read(self, db: sqlite3.Connection) -> tuple[tuple[dict, ...], AuditChain]:
        head = db.execute(
            "SELECT entry_count,head_hash FROM release_heads WHERE lineage=?",
            (self._lineage,),
        ).fetchone()
        count, digest = head or (0, AuditChain.GENESIS_HASH)
        records = []
        entries = []
        for sequence, payload, encoded in db.execute(
            "SELECT sequence,record,chain_entry FROM release_entries WHERE lineage=? ORDER BY sequence",
            (self._lineage,),
        ):
            record = json.loads(payload)
            entry = AuditChainEntry.from_dict(json.loads(encoded))
            if sequence != entry.sequence or entry.report_hash != stable_hash(
                [self._lineage, record]
            ):
                raise ValueError("release history record mismatch")
            records.append(record)
            entries.append(entry)
        chain = AuditChain(
            entries, declared_entry_count=count, declared_head_hash=digest
        )
        if not chain.verify().valid:
            raise ValueError("invalid release history chain")
        return tuple(records), chain
