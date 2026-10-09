"""Synthetic ambient evaluation and blinded-review safety controls."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from openmed.eval.ambient_drafts import (
    REVIEW_SCHEMA_VERSION,
    AmbientDraft,
    AmbientFact,
    evaluate_ambient_drafts,
    import_ambient_reviews,
)

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures/eval/ambient_drafts.json"


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def load_case(name):
    rows = json.loads(FIXTURE.read_text())["encounters"]
    row = next(r for r in rows if r["case"] == name)

    def facts(key):
        return tuple(
            AmbientFact(**{**f, "citations": tuple(f["citations"])}) for f in row[key]
        )

    assert digest(row["draft_text"]) == row["text_digest"]
    return AmbientDraft(
        row["blinded_case_id"], row["text_digest"], facts("truth"), facts("statements")
    )


def repeated(draft, n=5):
    # Replicated rows verify arithmetic/suppression, never clinical sample size.
    return [replace(draft, blinded_case_id=digest(str(i))) for i in range(n)]


def reviews(drafts, decisions=("accept", "accept"), adjudicated=False):
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "rows": [
            {
                "blinded_case_id": d.blinded_case_id,
                "draft_revision": d.revision,
                "reviewer_id": digest(f"reviewer:{i}"),
                "decision": decision,
                "reason": "clinical_interpretation"
                if len(set(decisions)) > 1
                else None,
                "adjudicated": adjudicated,
            }
            for d in drafts
            for i, decision in enumerate(decisions)
        ],
    }


@pytest.mark.parametrize(
    ("case", "metric", "numerator", "denominator"),
    [
        ("omitted_medication", "omission", 5, 10),
        ("invented_statement", "unsupported", 5, 15),
        ("inverted_experiencer", "misattribution", 5, 10),
        ("wrong_speaker", "misattribution", 5, 10),
        ("negation_flip", "negation", 5, 10),
        ("negation_flip", "contradiction", 5, 10),
    ],
)
def test_required_errors(case, metric, numerator, denominator):
    report = evaluate_ambient_drafts(repeated(load_case(case)))
    cell = report["slices"]["overall"][metric]
    assert cell["numerator"] == numerator
    assert cell["denominator"] == denominator
    assert cell["rate"] == numerator / denominator
    assert cell["ci95"][0] <= cell["rate"] <= cell["ci95"][1]
    assert report["reviewer_confirmation_required"] is True


def test_clean_negative_control():
    report = evaluate_ambient_drafts(repeated(load_case("clean")))
    assert all(r["numerator"] == 0 for r in report["slices"]["overall"].values())
    assert "score" not in report


def test_attribution_cannot_be_offset_by_citation_coverage():
    draft = load_case("inverted_experiencer")
    assert {c for s in draft.statements for c in s.citations} == {
        f.fact_id for f in draft.truth
    }
    cells = evaluate_ambient_drafts(repeated(draft))["slices"]["overall"]
    assert cells["misattribution"]["rate"] == 0.5
    assert cells["omission"]["rate"] == 0.5  # Wrong experiencer is not covered.
    assert cells["unsupported"]["rate"] == 0.5


def test_omitted_medication_slice():
    slices = evaluate_ambient_drafts(repeated(load_case("omitted_medication")))[
        "slices"
    ]
    assert slices["class:medication"]["omission"]["rate"] == 1
    assert slices["class:symptom"]["omission"]["rate"] == 0


@pytest.mark.parametrize("case", ["clean", "invented_statement", "negation_flip"])
def test_small_slices_suppress_counts_rates_and_intervals(case):
    report = evaluate_ambient_drafts([load_case(case)])
    for metrics in report["slices"].values():
        for cell in metrics.values():
            assert cell == {
                "numerator": None,
                "denominator": None,
                "rate": None,
                "ci95": None,
                "suppressed": True,
            }


def test_complementary_and_cross_slice_suppression():
    drafts = repeated(load_case("clean"), 10)
    one = load_case("inverted_experiencer")
    report = evaluate_ambient_drafts([*drafts, one])
    assert all(s["misattribution"]["suppressed"] for s in report["slices"].values())
    assert report["slices"]["overall"]["negation"]["suppressed"]


def test_unknown_and_incorrect_proposition_citations_fail_closed():
    draft = load_case("clean")
    for changes in (
        {"citations": (digest("unknown"),)},
        {"proposition": digest("unsupported different proposition")},
    ):
        changed = replace(draft, statements=(replace(draft.statements[0], **changes),))
        cells = evaluate_ambient_drafts(repeated(changed))["slices"]["overall"]
        assert cells["unsupported"]["rate"] == 1
        assert cells["omission"]["rate"] == 1


def test_wrong_section_does_not_cover_required_fact():
    draft = load_case("clean")
    changed = replace(
        draft, statements=tuple(replace(s, section="plan") for s in draft.statements)
    )
    cells = evaluate_ambient_drafts(repeated(changed))["slices"]["overall"]
    assert cells["omission"]["rate"] == 1
    assert cells["unsupported"]["rate"] == 0


def test_optional_truth_not_counted_as_omission():
    draft = load_case("omitted_medication")
    changed = replace(
        draft, truth=(replace(draft.truth[0], required=False), draft.truth[1])
    )
    assert (
        evaluate_ambient_drafts(repeated(changed))["slices"]["overall"]["omission"][
            "rate"
        ]
        == 0
    )


def test_no_statements_or_no_required_truth():
    draft = load_case("clean")
    empty = replace(draft, statements=())
    cells = evaluate_ambient_drafts(repeated(empty))["slices"]["overall"]
    assert cells["omission"]["rate"] == 1
    assert cells["unsupported"]["suppressed"]
    optional = replace(
        draft, truth=tuple(replace(f, required=False) for f in draft.truth)
    )
    assert evaluate_ambient_drafts(repeated(optional))["slices"]["overall"]["omission"][
        "suppressed"
    ]


@pytest.mark.parametrize(
    "changes",
    [
        {"fact_id": "Patient Jane Private"},
        {"proposition": "private diagnosis"},
        {"speaker": "private name"},
        {"experiencer": "unknown"},
        {"negated": 1},
        {"section": "private note"},
        {"fact_class": "PrivateClinicalClass"},
        {"required": 0},
        {"citations": [digest("a")]},
        {"citations": (digest("a"), digest("a"))},
    ],
)
def test_invalid_fact_diagnostics_do_not_echo_values(changes):
    with pytest.raises(ValueError) as caught:
        replace(load_case("clean").truth[0], **changes)
    assert not any(str(v) in str(caught.value) for v in changes.values())


def test_duplicate_and_missing_truth_rejected():
    draft = load_case("clean")
    for changes in (
        {"truth": ()},
        {"truth": (draft.truth[0], draft.truth[0])},
        {"statements": (draft.statements[0], draft.statements[0])},
        {"truth": draft.statements},
    ):
        with pytest.raises(ValueError):
            replace(draft, **changes)
    with pytest.raises(ValueError, match="duplicate_encounter"):
        evaluate_ambient_drafts([draft, draft])
    with pytest.raises(ValueError, match="empty_encounter"):
        evaluate_ambient_drafts([])


@pytest.mark.parametrize("minimum", [True, 0, 1, 2.5, "5"])
def test_invalid_minimum(minimum):
    with pytest.raises(ValueError):
        evaluate_ambient_drafts([load_case("clean")], minimum_cell_size=minimum)


def test_reports_and_reprs_exclude_source_payloads_and_identifiers():
    draft = load_case("clean")
    payload = json.dumps(evaluate_ambient_drafts(repeated(draft)))
    for value in (
        draft.blinded_case_id,
        draft.text_digest,
        draft.revision,
        draft.truth[0].proposition,
        draft.truth[0].fact_id,
        "synthetic-med",
    ):
        assert value not in payload
        assert value not in repr(draft)
        assert value not in repr(draft.truth[0])


def test_review_agreement_and_disagreement_remain_separate():
    drafts = repeated(load_case("clean"))
    report = import_ambient_reviews(reviews(drafts), drafts)
    assert report["rates"]["agreement"]["rate"] == 1
    disputed = import_ambient_reviews(reviews(drafts, ("accept", "revise")), drafts)
    assert disputed["rates"]["agreement"]["rate"] == 0
    assert disputed["rates"]["adjudication"]["rate"] == 0
    adjudicated = import_ambient_reviews(
        reviews(drafts, ("accept", "revise"), True), drafts
    )
    assert adjudicated["rates"]["agreement"]["rate"] == 0
    assert adjudicated["rates"]["adjudication"]["rate"] == 1
    assert adjudicated["reviewer_confirmation_required"]


def test_unclear_review_not_hidden_by_agreement():
    drafts = repeated(load_case("clean"))
    report = import_ambient_reviews(reviews(drafts, ("unclear", "unclear")), drafts)
    assert report["rates"]["agreement"]["rate"] == 1
    assert report["rates"]["unclear"]["rate"] == 1
    assert report["reviewer_confirmation_required"]


@pytest.mark.parametrize(
    "field",
    [
        "candidate_name",
        "patient_name",
        "transcript",
        "model",
        "reviewer_name",
        "comment",
    ],
)
def test_review_rejects_unblinded_fields(field):
    draft = load_case("clean")
    payload = reviews([draft])
    payload["rows"][0][field] = "private payload"
    with pytest.raises(ValueError, match="unblinded_or_invalid_review") as caught:
        import_ambient_reviews(payload, [draft])
    assert "private payload" not in str(caught.value)


def test_review_rejects_stale_text_truth_or_statement_revision():
    draft = load_case("clean")
    for changed in (
        replace(draft, text_digest=digest("changed text")),
        replace(
            draft, truth=tuple(replace(f, negated=not f.negated) for f in draft.truth)
        ),
        replace(draft, statements=(draft.statements[0],)),
    ):
        with pytest.raises(ValueError, match="stale_or_unknown_review"):
            import_ambient_reviews(reviews([draft]), [changed])


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown",
        "duplicate",
        "missing",
        "single",
        "reason",
        "status",
        "schema",
        "identity",
    ],
)
def test_review_negative_controls(mutation):
    draft = load_case("clean")
    payload = copy.deepcopy(reviews([draft], ("accept", "revise")))
    row = payload["rows"][0]
    if mutation == "unknown":
        row["blinded_case_id"] = digest("unknown case")
    if mutation == "duplicate":
        payload["rows"].append(copy.deepcopy(row))
    if mutation == "missing":
        payload["rows"] = []
    if mutation == "single":
        payload["rows"].pop()
    if mutation == "reason":
        row["reason"] = None
    if mutation == "status":
        row["adjudicated"] = "true"
    if mutation == "schema":
        payload["schema_version"] = "stale"
    if mutation == "identity":
        row["reviewer_id"] = "Doctor Private"
    with pytest.raises(ValueError):
        import_ambient_reviews(payload, [draft])


def test_small_review_cells_and_identifier_privacy():
    draft = load_case("clean")
    report = import_ambient_reviews(reviews([draft]), [draft])
    assert all(r["suppressed"] for r in report["rates"].values())
    assert draft.blinded_case_id not in json.dumps(report)
    assert draft.revision not in json.dumps(report)
