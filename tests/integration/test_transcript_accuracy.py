"""Offline fixed-pair accuracy evidence pipeline with synthetic data only."""

import json

import pytest

from openmed.eval.transcript_accuracy import (
    ClinicalTermClass,
    ReferenceTermSpan,
    score_transcript,
    transcript_accuracy_report,
)


@pytest.mark.integration
def test_fixed_reference_to_suppressed_aggregate_evidence_without_providers():
    reference = "SyntheticAda takes 5 mg"
    spans = [
        ReferenceTermSpan(0, 12, ClinicalTermClass.IDENTIFIER),
        ReferenceTermSpan(19, 20, ClinicalTermClass.DOSE),
        ReferenceTermSpan(21, 23, ClinicalTermClass.UNIT),
    ]
    scores = [
        score_transcript(
            reference, hypothesis, spans=spans, partials=["SyntheticAda takes"]
        )
        for hypothesis in [reference] * 5 + [reference.replace("5", "50")] * 5
    ]
    report = transcript_accuracy_report(scores, bootstrap_resamples=100)
    assert report["wer"]["rate"] == 5 / 40
    assert report["clinical_terms"]["dose"]["rate"] == 0.5
    assert report["clinical_terms"]["identifier"]["recall"] == 1
    assert report["clinical_terms"]["medication"]["suppressed"]
    assert report["final_revision_churn"]["rate"] == 0
    payload = json.dumps(report, sort_keys=True)
    for value in ("SyntheticAda", "takes", "5 mg", "50 mg", "reference", "hypothesis"):
        assert value not in payload
    assert json.loads(payload) == report
