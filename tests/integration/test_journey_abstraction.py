"""Offline SQLite snapshot-to-abstraction integration with synthetic records."""

import json
from pathlib import Path

import pytest

from openmed.agent.workflows import (
    AbstractionEvidenceError,
    AbstractionFieldBinding,
    AbstractionReviewReceipt,
    ReviewerState,
    SourceKind,
    TransformationKind,
    build_journey_abstraction_evidence,
)
from openmed.clinical.journey import JourneyQuery, query_journey
from openmed.clinical.journey_contracts import (
    ClinicalArtifact,
    ClinicalFact,
    EvidenceLocator,
)
from openmed.structured.store import SQLiteJourneyStore

pytestmark = pytest.mark.integration


def test_sqlite_journey_snapshot_requires_source_coverage_and_explicit_review(tmp_path):
    fixture = json.loads(
        (
            Path(__file__).parents[1] / "fixtures/structured/journey_contracts.json"
        ).read_text()
    )
    artifact = ClinicalArtifact.from_dict(fixture["clinical_artifact"])
    fact = ClinicalFact.from_dict(fixture["clinical_fact"])
    locator = EvidenceLocator.from_dict(fixture["evidence_locator"])
    store = SQLiteJourneyStore(tmp_path / "synthetic.sqlite3")
    try:
        with store.transaction(committed_at="2026-01-02T04:00:00Z") as transaction:
            assert transaction.put_artifact(artifact).ok
            assert transaction.put_evidence(locator).ok
            assert transaction.put_fact(fact).ok
        result = query_journey(store, JourneyQuery(subject_id=fact.subject_id))
        assert result.ok and result.value is not None
        page = result.value
        field_id = "registry.primary_diagnosis"
        args = {
            "snapshot": page.snapshot,
            "fields": (
                AbstractionFieldBinding(
                    field_id, (fact.fact_id,), TransformationKind.RULE
                ),
            ),
            "facts": tuple(event.fact for event in page.events),
            "locators": tuple(
                path.locator for event in page.events for path in event.evidence_paths
            ),
            "artifacts": tuple(
                path.artifact for event in page.events for path in event.evidence_paths
            ),
            "source_kinds": {artifact.artifact_id: SourceKind.CLINICAL_RECORD},
        }
        pending = build_journey_abstraction_evidence(**args)
        with pytest.raises(AbstractionEvidenceError):
            pending.finalize((field_id,))
        receipt = AbstractionReviewReceipt(
            field_id, pending.chains[0].chain_digest, ReviewerState.APPROVED
        )
        approved = build_journey_abstraction_evidence(
            **args, review_receipts=(receipt,)
        )
        assert approved.finalize((field_id,)).chain_count == 1
        incomplete = build_journey_abstraction_evidence(
            **{**args, "locators": ()}, review_receipts=(receipt,)
        )
        with pytest.raises(AbstractionEvidenceError) as caught:
            incomplete.finalize((field_id,))
        assert "missing_locator_evidence" in {
            issue.code for issue in caught.value.report.issues
        }
        assert "synthetic-condition" not in approved.evaluate((field_id,)).to_json()
    finally:
        store.close()
