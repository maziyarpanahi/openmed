"""Offline synthetic packet-to-summary-gate integration; no clinician evidence."""

import json

import pytest

from openmed.eval.summary_gate import evaluate_summary_gate
from openmed.eval.summary_review import import_summary_review
from tests.unit.eval.test_summary_gate import sample
from tests.unit.eval.test_summary_review import vector

pytestmark = pytest.mark.integration


def gate_inputs(args):
    """Adapt synthetic evaluated artifacts to the existing summary gate."""
    values = sample()
    values["deidentified"].deidentified_text = args["source"]
    values.update(
        summary=args["summary"],
        adjudications=None,
        claims=[
            {**claim, "claim_class": "finding", "evidence_ids": claim["citations"]}
            for claim in args["claims"]
        ],
        source_evidence=args["evidence"],
        support_evidence=[
            dict(
                evidence_id=row["evidence_id"],
                claim_id=claim["claim_id"],
                relation="supported",
                approved=True,
            )
            for row, claim in zip(args["evidence"], args["claims"])
        ],
    )
    return values


def test_synthetic_review_metrics_round_trip_but_adjudication_gate_fails():
    _, _, bundle, args = vector()
    review = import_summary_review(bundle, **args)
    values = gate_inputs(args)
    checks = {
        check.gate: check
        for check in evaluate_summary_gate(**values, review_import=review)
    }
    assert checks["summary_machine_spans"].passed
    assert not checks["summary_review"].passed
    assert checks["summary_review"].reason == "synthetic_review"
    assert not checks["summary_citation_support"].passed
    serialized = json.dumps([check.to_dict() for check in checks.values()])
    assert args["source"] not in serialized
    assert "private-claim" not in serialized
    assert "hidden-model" not in serialized


def test_caller_declared_reviewer_receipt_is_bound_and_complete():
    # Still synthetic test data: exercise reviewer provenance without claiming
    # actual clinician recruitment, credentials or clinical validation.
    _, _, bundle, args = vector(evidence_kind="reviewer", count=1)
    review = import_summary_review(bundle, **args)
    values = gate_inputs(args)
    checks = evaluate_summary_gate(**values, review_import=review)
    assert all(check.passed for check in checks), [check.to_dict() for check in checks]
    values["summary"] += " Changed."
    assert not evaluate_summary_gate(**values, review_import=review)[0].passed


@pytest.mark.parametrize("state", ["missing", "incomplete", "disputed"])
def test_partial_reviews_fail_even_with_zero_support_threshold(state):
    _, _, bundle, args = vector(evidence_kind="reviewer")
    if state == "missing":
        bundle["decisions"] = []
    elif state == "incomplete":
        bundle["decisions"] = bundle["decisions"][:1]
    else:
        for index, row in enumerate(bundle["decisions"]):
            row["reason"] = "evidence_quality"
            row["label"] = "supports" if index % 2 == 0 else "contradicts"
    review = import_summary_review(bundle, **args)
    values = gate_inputs(args)
    values["thresholds"]["citation_support_min"] = 0
    checks = {
        check.gate: check
        for check in evaluate_summary_gate(**values, review_import=review)
    }
    assert not checks["summary_review"].passed
    assert not checks["summary_citation_support"].passed


def test_import_cannot_mix_legacy_labels_or_stale_citation_spans():
    _, _, bundle, args = vector(evidence_kind="reviewer")
    review = import_summary_review(bundle, **args)
    values = gate_inputs(args)
    values["adjudications"] = sample()["adjudications"]
    assert not evaluate_summary_gate(**values, review_import=review)[0].passed
    values["adjudications"] = None
    values["claims"][0]["end"] -= 1
    assert not evaluate_summary_gate(**values, review_import=review)[0].passed
