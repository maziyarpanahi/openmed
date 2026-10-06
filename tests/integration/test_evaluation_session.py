"""Deterministic offline session integrating every shipped governance primitive."""

import hashlib

import pytest

from openmed.eval.governance import (
    ComparisonCase,
    ConflictOfInterestMetadata,
    FeedbackBudgetLedger,
    FeedbackBudgetPolicy,
    RubricCriterion,
    SourceEvidence,
    commit_holdout_manifests,
)
from openmed.eval.workflows.sealed_manifest import (
    SEALED_MANIFEST_COMPONENTS,
    seal_workflow_manifest,
)
from openmed.eval.workflows.session import (
    EvaluationSession,
    SQLiteSessionStore,
    evaluation_case_digest,
)


@pytest.mark.integration
def test_end_to_end_is_deterministic_with_an_injected_clock(tmp_path):
    def digest(value):
        return "sha256:" + hashlib.sha256(value.encode()).hexdigest()

    components = {name: digest(name) for name in SEALED_MANIFEST_COMPONENTS}
    manifest = seal_workflow_manifest(components)
    cases = tuple(
        ComparisonCase(
            f"case-{i}",
            (SourceEvidence(f"source-{i}", f"Synthetic source {i}"),),
            {"submission": "Placeholder", "baseline": f"Synthetic baseline {i}"},
        )
        for i in range(3)
    )
    manifests = {
        "case": tuple(evaluation_case_digest(c) for c in cases),
        **{k: (digest(k),) for k in ("label", "template", "randomization")},
    }
    holdout = commit_holdout_manifests("synthetic-e2e", manifests)
    records, feedbacks = [], []
    for i in range(2):
        store = SQLiteSessionStore(tmp_path / f"run-{i}.sqlite")
        ticks = iter(range(6))
        reviews = []
        try:
            session = EvaluationSession(
                session_key=digest("e2e-session"),
                manifest=manifest,
                component_snapshot=lambda: components,
                holdout=holdout,
                holdout_manifests=manifests,
                cases=cases,
                candidate_identity="submission",
                candidate_manifests={
                    "submission": manifest.manifest_digest,
                    "baseline": digest("baseline"),
                },
                runner=lambda evidence: f"Synthetic output {evidence[0].evidence_ref}",
                scorer=lambda case, output: 0.9,
                review=reviews.append,
                provider_digests={k: digest(k) for k in ("runner", "scorer", "review")},
                rubric=(RubricCriterion("grounding", "Synthetic evidence?", 1, 5),),
                conflict_of_interest=ConflictOfInterestMetadata(
                    "reviewer-1", digest("coi")
                ),
                randomization_key=b"synthetic-key-for-packet-order-0001",
                public_items=("Synthetic public corpus",),
                shadow_items=("Synthetic shadow corpus",),
                canaries=("SYNTHETIC-CANARY",),
                ledger=FeedbackBudgetLedger(
                    [FeedbackBudgetPolicy(holdout.commitment_digest, 1, 0, (0.5, 0.8))]
                ),
                store=store,
                clock=lambda: next(ticks),
            )
            feedbacks.append(session.run())
            records.append(session.record.to_json())
            assert len(reviews) == 1 and len(reviews[0]) == 3
            assert [s[0] for s in session.record.steps] == [
                "verified",
                "cases",
                "forensics",
                "adjudication",
                "release_claimed",
                "released",
            ]
        finally:
            store.close()
    assert records[0] == records[1]
    assert feedbacks[0] == feedbacks[1]
    assert feedbacks[0].per_case_scores == (0.9, 0.9, 0.9)
