"""Backend-neutral composition of Journey artifact and metadata stores."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from openmed.clinical.journey_contracts import (
    ClinicalArtifact,
    ClinicalFact,
    ConflictSet,
    EvidenceLocator,
)

from .protocols import (
    CommitStatusUnknown,
    CompensatingArtifactStore,
    StoreResult,
    StoreState,
    TransactionalJourneyStore,
)


@dataclass(frozen=True, slots=True)
class StoredJourneyGraph:
    """Value-safe counts for one committed backend-neutral graph."""

    artifact: ClinicalArtifact = field(repr=False)
    evidence_count: int
    fact_count: int
    conflict_count: int


class ComposedJourneyStore:
    """Join one content store to one transactional metadata implementation.

    This facade provides the same ingestion behavior for local/local,
    object/local, local/PostgreSQL, and object/PostgreSQL combinations.  Both
    stores must be dedicated to the caller while an ingestion is running.
    """

    def __init__(
        self,
        artifacts: CompensatingArtifactStore,
        metadata: TransactionalJourneyStore,
    ) -> None:
        if not isinstance(artifacts, CompensatingArtifactStore):
            raise TypeError("artifact backend does not support compensation")
        if not isinstance(metadata, TransactionalJourneyStore):
            raise TypeError("metadata backend does not satisfy store contracts")
        self.artifacts = artifacts
        self.metadata = metadata

    def close(self) -> None:
        """Close the metadata backend when it exposes a close method."""

        closer = getattr(self.metadata, "close", None)
        if callable(closer):
            closer()

    def __enter__(self) -> "ComposedJourneyStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def ingest_graph(
        self,
        artifact: ClinicalArtifact,
        content: bytes,
        *,
        evidence: Iterable[EvidenceLocator] = (),
        facts: Iterable[ClinicalFact] = (),
        conflicts: Iterable[ConflictSet] = (),
        committed_at: str,
    ) -> StoreResult[StoredJourneyGraph]:
        """Persist verified bytes and one atomic metadata graph."""

        locators = tuple(evidence)
        fact_records = tuple(facts)
        conflict_records = tuple(conflicts)
        blob = self.artifacts.put_bytes(artifact, content)
        if not blob.ok:
            return StoreResult.outcome(
                blob.state,
                blob.code or "artifact_write_failed",
            )

        failure: StoreResult[Any] | None = None
        created_metadata = False
        revision: int | None = None
        try:
            with self.metadata.transaction(committed_at=committed_at) as transaction:
                revision = transaction.revision
                artifact_result = transaction.put_artifact(artifact)
                if not artifact_result.ok:
                    failure = artifact_result
                else:
                    created_metadata = created_metadata or artifact_result.created
                for locator in locators:
                    if failure is not None:
                        break
                    result = transaction.put_evidence(locator)
                    if not result.ok:
                        failure = result
                    else:
                        created_metadata = created_metadata or result.created
                for fact in fact_records:
                    if failure is not None:
                        break
                    result = transaction.put_fact(fact)
                    if not result.ok:
                        failure = result
                    else:
                        created_metadata = created_metadata or result.created
                for conflict in conflict_records:
                    if failure is not None:
                        break
                    result = transaction.put_conflict(conflict)
                    if not result.ok:
                        failure = result
                    else:
                        created_metadata = created_metadata or result.created
        except CommitStatusUnknown:
            return StoreResult.outcome(StoreState.UNKNOWN, "commit_status_unknown")
        except Exception:
            failure = StoreResult.outcome(StoreState.FAILURE, "transaction_failed")

        if failure is not None:
            if blob.created:
                self.artifacts.discard_if_created(artifact.content_hash)
            return StoreResult.outcome(
                failure.state,
                failure.code or "transaction_failed",
            )
        graph = StoredJourneyGraph(
            artifact=artifact,
            evidence_count=len(locators),
            fact_count=len(fact_records),
            conflict_count=len(conflict_records),
        )
        return StoreResult.success(
            graph,
            created=blob.created or created_metadata,
            revision=revision if created_metadata else self.metadata.latest_revision,
        )


__all__ = ["ComposedJourneyStore", "StoredJourneyGraph"]
