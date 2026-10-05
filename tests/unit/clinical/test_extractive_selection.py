"""Synthetic sentence-selection and non-compensable budget regressions."""

import hashlib
import itertools
import json
from dataclasses import replace
from datetime import datetime

import pytest

from openmed.clinical.extractive_selection import (
    ExtractiveFact,
    ExtractiveSelectionError,
    select_extractive_sentences,
)
from openmed.clinical.summarize import summarize_deidentified
from openmed.clinical.summarize_backends import ExtractiveSummarizerBackend
from openmed.clinical.summary_length_budget import build_summary_length_budget
from openmed.clinical.summary_omission_budget import ImportanceClassPolicy
from openmed.core.pii import DeidentificationResult


def opaque(value):
    return "sha256:" + hashlib.sha256(str(value).encode()).hexdigest()


MANDATORY = ImportanceClassPolicy(opaque("required"), 10, mandatory=True)
OPTIONAL = ImportanceClassPolicy(opaque("optional"), 1, omission_limit=8)


def fact(text, sentence, index=0, policy=MANDATORY, length_class="key_findings"):
    start = text.index(sentence)
    return ExtractiveFact(
        opaque(index), policy.class_id, start, start + len(sentence), length_class
    )


def select(text, evidence, cap=160, policies=(MANDATORY, OPTIONAL), demands=None):
    return select_extractive_sentences(
        text,
        evidence=tuple(evidence),
        importance_classes=policies,
        length_budget=build_summary_length_budget(
            cap, demands or {"key_findings": cap}
        ),
    )


@pytest.mark.parametrize(
    "late", ["Start synthetic medication.", "A new finding is present."]
)
def test_required_fact_after_third_sentence_preserves_offsets_and_order(late):
    text = "Admission occurred. No fever. Improved. " + late
    evidence = (fact(text, "No fever.", 1), fact(text, late, 2))
    selected = select(text, evidence)
    assert selected.status == "selected"
    assert selected.summary == "No fever. " + late
    audit = selected.to_dict()
    assert audit["coverage"]["recall"] == 1
    assert audit["omission_budget"]["passed"]
    for citation in audit["citations"]:
        assert (
            text[citation["source_start"] : citation["source_end"]]
            == selected.summary[citation["output_start"] : citation["output_end"]]
        )
    assert late not in json.dumps(audit)
    assert late not in repr(selected)
    assert ExtractiveSummarizerBackend().summarize(text).endswith("Improved.")


def test_duplicates_and_alternate_ids_cannot_inflate_coverage():
    text = "A finding. Another finding."
    first, second = fact(text, "A finding.", 1), fact(text, "Another finding.", 2)
    expected = select(text, (first, second))
    repeated = select(
        text, (first, second, first, replace(first, evidence_id=opaque(3)))
    )
    assert select(text, (first, second, first)).to_dict() == expected.to_dict()
    assert repeated.summary == expected.summary
    assert repeated.to_dict()["coverage"] == expected.to_dict()["coverage"]
    assert (
        repeated.to_dict()["omission_budget"] == expected.to_dict()["omission_budget"]
    )
    assert repeated.to_dict()["coverage"]["source_fact_count"] == 2


def test_conflicting_duplicate_and_cross_sentence_evidence_fail_closed():
    text = "First finding. Next finding."
    first = fact(text, "First finding.")
    for other in (
        replace(first, start=15, end=len(text)),
        replace(first, importance_class_id=OPTIONAL.class_id),
        replace(first, end=len(text)),
    ):
        assert select(text, (first, other)).status == "invalid_evidence"
    assert select(text, (replace(first, end=len(text)),)).status == "invalid_evidence"


def test_ties_empty_evidence_and_impossible_budgets_are_stable():
    text = "AA. BB. CC."
    evidence = tuple(
        fact(text, s, i, OPTIONAL) for i, s in enumerate(("AA.", "BB.", "CC."))
    )
    a = select(text, evidence, cap=3)
    b = select(text, tuple(reversed(evidence)), cap=3)
    assert a.summary == b.summary == "AA."
    assert a.to_dict() == b.to_dict()
    empty = select(text, ())
    assert empty.status == "empty_evidence"
    assert empty.to_dict()["coverage"]["fail_closed"]
    failure = select(
        text, (replace(evidence[-1], importance_class_id=MANDATORY.class_id),), cap=2
    )
    assert failure.status == "insufficient_budget"
    assert failure.summary == "" and failure.to_dict()["citations"] == []
    assert not failure.to_dict()["omission_budget"]["passed"]


def test_mandatory_class_cannot_be_compensated_by_optional_coverage():
    text = "A. B. C. Required medication."
    evidence = tuple(
        fact(text, s, i, OPTIONAL) for i, s in enumerate(("A.", "B.", "C."))
    ) + (fact(text, "Required medication.", 4),)
    assert select(text, evidence, cap=10).status == "insufficient_budget"
    assert select(text, evidence, cap=20).summary == "Required medication."


def test_independent_length_caps_and_utf8_separator_charges():
    text = "é. Medication."
    evidence = (
        fact(text, "é.", 1),
        fact(text, "Medication.", 2, length_class="medications"),
    )
    assert (
        select(
            text, evidence, cap=15, demands={"key_findings": 3, "medications": 12}
        ).status
        == "selected"
    )
    assert (
        select(
            text, evidence, cap=15, demands={"key_findings": 2, "medications": 13}
        ).status
        == "insufficient_budget"
    )
    assert (
        select(
            text, evidence, cap=14, demands={"key_findings": 3, "medications": 12}
        ).status
        == "insufficient_budget"
    )


def test_exact_search_agrees_with_exhaustive_small_oracle():
    sentences = ("Long optional finding.", "Short.", "Required.", "Other.")
    text = " ".join(sentences)
    evidence = tuple(
        fact(text, s, i, MANDATORY if i == 2 else OPTIONAL)
        for i, s in enumerate(sentences)
    )
    for cap in range(1, len(text.encode()) + 1):
        possibilities = []
        for count in range(1, 5):
            for indices in itertools.combinations(range(4), count):
                summary = " ".join(sentences[i] for i in indices)
                if 2 in indices and len(summary.encode()) <= cap:
                    possibilities.append((indices, summary))
        actual = select(text, evidence, cap)
        if not possibilities:
            assert actual.status == "insufficient_budget"
        else:
            expected = min(
                possibilities, key=lambda p: (-len(p[0]), len(p[1].encode()), p[0])
            )
            assert actual.summary == expected[1]


def test_search_resource_limit_is_distinct_from_infeasibility(monkeypatch):
    import openmed.clinical.extractive_selection as module

    monkeypatch.setattr(module, "MAX_SELECTION_STATES", 1)
    text = "One. Two."
    actual = select(
        text, (fact(text, "One.", 1, OPTIONAL), fact(text, "Two.", 2, OPTIONAL))
    )
    assert actual.status == "selection_limit_exceeded"
    assert actual.summary == ""


def test_backend_refusal_survives_guarded_pipeline_without_private_context():
    text = "Required medication."
    backend = ExtractiveSummarizerBackend(
        evidence=(fact(text, text),),
        importance_classes=(MANDATORY,),
        length_budget=build_summary_length_budget(2, {"key_findings": 2}),
    )
    value = DeidentificationResult(text, text, [], "mask", datetime(2026, 1, 1))
    with pytest.raises(ExtractiveSelectionError) as caught:
        summarize_deidentified(value, model=backend)
    assert caught.value.result.status == "insufficient_budget"
    assert text not in str(caught.value)
    assert caught.value.__context__ is None
