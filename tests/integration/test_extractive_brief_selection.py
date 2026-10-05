"""Offline reviewed synthetic evidence through the complete brief pipeline."""

import json

import pytest

from openmed.clinical.brief import STAGES, build_clinical_brief
from tests.unit.clinical.test_brief import SENTENCES, fixture_context

pytestmark = pytest.mark.integration


def test_late_mandatory_medication_preserves_original_citations_and_all_gates():
    sentences = SENTENCES + ("Synthetic medication continued.",)
    value, context = fixture_context(sentences)
    brief = build_clinical_brief(value, model="extractive", context=context)
    assert brief.refusal_reason is None
    assert brief.summary == value.deidentified_text
    assert brief.metrics["coverage"]["recall"] == 1
    assert len(brief.citations) == 4
    assert brief.to_dict()["stages"] == list(STAGES)
    for citation in brief.citations:
        assert (
            value.deidentified_text[citation["source_start"] : citation["source_end"]]
            == brief.summary[citation["output_start"] : citation["output_end"]]
        )
    assert sentences[-1] not in json.dumps(brief.to_dict())
    assert brief.metrics["leakage"]["passed"]
    assert brief.to_dict()["status"] == "needs_review"
    baseline = build_clinical_brief(value, model="extractive-baseline", context=context)
    assert sentences[-1] not in baseline.summary
    assert baseline.metrics["coverage"]["recall"] == 0.75


def test_impossible_reviewed_extract_reports_budget_failure_without_partial_output():
    sentences = SENTENCES + (
        "A long synthetic medication instruction requires review before continuation.",
    )
    value, context = fixture_context(sentences)
    brief = build_clinical_brief(value, model="extractive", context=context)
    assert brief.refusal_reason is not None
    assert brief.summary == ""
    audit = brief.to_dict()
    assert audit["citations"] == []
    assert brief.metrics["extractive_selection"]["status"] == "insufficient_budget"
    assert not brief.metrics["extractive_selection"]["omission_budget"]["passed"]
    assert sentences[-1] not in json.dumps(audit)
