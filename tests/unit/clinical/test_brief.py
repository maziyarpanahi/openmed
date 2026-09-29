"""Synthetic composer contract tests; fake scores are never release evidence."""

import json
from dataclasses import replace
from datetime import datetime

import pytest

from openmed.clinical.brief import (
    STAGES,
    BriefContext,
    BriefFact,
    BriefRefusal,
    _compose,
    _digest,
    brief_policy_fingerprint,
    build_clinical_brief,
)
from openmed.clinical.evidence_packet import (
    build_evidence_packet,
    fingerprint_evidence_review,
)
from openmed.clinical.nli_gate import NLIThresholds
from openmed.clinical.review_state_machine import (
    ReviewState,
    ReviewStateMachine,
    make_opaque_event_id,
)
from openmed.core.pii import DeidentificationResult

SENTENCES = (
    "The admission problem was dehydration.",
    "The discharge diagnosis was dehydration.",
    "Symptoms improved after fluids.",
)


def fixture_context():
    text = " ".join(SENTENCES)
    result = DeidentificationResult(text, text, [], "mask", datetime(2026, 1, 1))
    facts = [
        BriefFact(
            "synthetic:ref-" + str(i), field, "affirmed", "certain", "recent", "patient"
        )
        for i, field in enumerate(
            ("admission_reason", "discharge_diagnoses", "hospital_course")
        )
    ]
    policy = brief_policy_fingerprint(text, tuple(facts))
    rows = []
    for i, (sentence, field) in enumerate(
        zip(SENTENCES, ("admission_reason", "discharge_diagnoses", "hospital_course"))
    ):
        ref = "synthetic:ref-" + str(i)
        start = text.index(sentence)
        row = dict(
            reference_id=ref,
            source_id="synthetic:document",
            start=start,
            end=start + len(sentence),
            policy_fingerprint=policy,
            review_state="approved",
            synthetic=True,
            verified=True,
        )
        fingerprint = fingerprint_evidence_review(
            **{
                k: row[k]
                for k in (
                    "reference_id",
                    "source_id",
                    "start",
                    "end",
                    "policy_fingerprint",
                )
            }
        )
        machine = ReviewStateMachine()
        for state in (ReviewState.IN_REVIEW, ReviewState.APPROVED):
            machine.transition(
                state, make_opaque_event_id((ref, state.value)), fingerprint
            )
        row["review_transitions"] = machine.transitions
        rows.append(row)
    thresholds = NLIThresholds(
        calibration_id="synthetic-test-only", calibration_method="synthetic-fixture"
    )
    context = BriefContext(
        build_evidence_packet(rows, policy_fingerprint=policy),
        _digest(text),
        tuple(facts),
        lambda p, h: {
            "entailment": 1.0,
            "contradiction": 0.0,
            "neutral": 0.0,
            "calibration_id": "synthetic-test-only",
        },
        thresholds,
        lambda _: [],
    )
    return result, context


def test_composes_every_stage_and_preserves_safe_serialization():
    result, context = fixture_context()
    # Call the internal composer once so failures expose a stack during development.
    brief = _compose(result, "extractive", "bhc", context, [])
    assert brief.refusal_reason is None
    assert brief.summary == result.deidentified_text
    assert brief.to_dict()["stages"] == list(STAGES)
    assert len(brief.citations) == len(brief.verdicts) == 3
    assert brief.envelope["requires_human_review"]
    assert not brief.envelope["is_diagnostic"]
    assert brief.to_dict()["metrics"]["leakage"]["passed"]
    assert brief.metrics["coverage"]["recall"] == 1.0
    assert SENTENCES[0] not in json.dumps(brief.to_dict())
    assert SENTENCES[0] not in repr(brief)
    assert brief.to_response()["summary"] == brief.summary
    assert (
        build_clinical_brief(result, model="extractive", context=context).digest
        == brief.digest
    )


def test_review_is_never_invented():
    result, _ = fixture_context()
    brief = build_clinical_brief(result, model="extractive")
    assert brief.refusal_reason is BriefRefusal.REVIEW_REQUIRED
    assert brief.summary == ""


@pytest.mark.parametrize("label", ["contradiction", "neutral"])
def test_nli_rejects_contradiction_and_abstention(label):
    result, context = fixture_context()
    context = replace(
        context,
        nli_predict=lambda p, h: {
            "label": label,
            "score": 1.0,
            "calibration_id": "synthetic-test-only",
        },
    )
    brief = build_clinical_brief(result, model="extractive", context=context)
    assert brief.refusal_reason is BriefRefusal.NLI_REJECTED
    assert brief.summary == ""


def test_empty_evidence_refuses_before_generator():
    result, context = fixture_context()
    context = replace(
        context,
        packet=build_evidence_packet(
            [],
            policy_fingerprint=brief_policy_fingerprint(result.deidentified_text, ()),
        ),
        facts=(),
    )
    brief = build_clinical_brief(
        result, context=context, model=lambda _: pytest.fail("generation")
    )
    assert brief.refusal_reason is BriefRefusal.EMPTY_EVIDENCE


def test_fabricated_claim_refuses():
    result, context = fixture_context()
    brief = build_clinical_brief(
        result, context=context, model=lambda _: "The patient has pneumonia."
    )
    assert brief.refusal_reason is BriefRefusal.UNSUPPORTED_CLAIM


def test_input_digest_is_required():
    result, context = fixture_context()
    brief = build_clinical_brief(
        result,
        model="extractive",
        context=replace(context, content_digest=_digest("changed")),
    )
    assert brief.refusal_reason is BriefRefusal.INVALID_EVIDENCE


def test_private_backend_error_is_not_serialized():
    result, context = fixture_context()

    def broken(_):
        raise RuntimeError("PRIVATE_SENTINEL")

    brief = build_clinical_brief(result, context=context, model=broken)
    assert brief.refusal_reason is BriefRefusal.STAGE_FAILED
    assert "PRIVATE_SENTINEL" not in repr(brief.to_dict())


def test_stages_cannot_be_configured():
    result, context = fixture_context()
    with pytest.raises(TypeError):
        build_clinical_brief(result, context=context, stages=())


def test_remote_model_never_reaches_generation():
    result, context = fixture_context()
    assert (
        build_clinical_brief(
            result, context=context, model="https://example.com"
        ).refusal_reason
        is not None
    )


def test_annotation_change_requires_new_review():
    result, context = fixture_context()
    context = replace(
        context,
        facts=(replace(context.facts[0], negation="negated"), *context.facts[1:]),
    )
    assert (
        build_clinical_brief(result, context=context, model="extractive").refusal_reason
        is BriefRefusal.INVALID_EVIDENCE
    )


def test_final_packet_metadata_is_privacy_scanned():
    result, context = fixture_context()
    seen = []

    def detector(text):
        seen.append(text)
        if '"schema_version"' in text:
            return [{"label": "NAME", "start": 0, "end": 1, "critical": True}]
        return []

    brief = build_clinical_brief(
        result, context=replace(context, privacy_detector=detector), model="extractive"
    )
    assert brief.refusal_reason is BriefRefusal.PRIVACY
    assert brief.summary == ""
    assert len(seen) == 2


def test_missing_privacy_stage_fails_closed():
    result, context = fixture_context()
    brief = build_clinical_brief(
        result, context=replace(context, privacy_detector=None), model="extractive"
    )
    assert brief.refusal_reason is BriefRefusal.STAGE_FAILED


def test_missing_calibration_binding_fails_closed():
    result, context = fixture_context()
    brief = build_clinical_brief(
        result,
        context=replace(
            context, nli_predict=lambda p, h: {"label": "entailment", "score": 1.0}
        ),
        model="extractive",
    )
    assert brief.refusal_reason is BriefRefusal.NLI_UNAVAILABLE


def test_reemitted_identifier_never_reaches_packet():
    result, context = fixture_context()
    result.mapping = {"[PERSON]": "PRIVATE_SENTINEL"}
    brief = build_clinical_brief(
        result, context=context, model=lambda _: "PRIVATE_SENTINEL was discharged."
    )
    assert brief.refusal_reason is BriefRefusal.PRIVACY
    assert "PRIVATE_SENTINEL" not in json.dumps(brief.to_dict())


def test_result_views_are_defensive_copies():
    result, context = fixture_context()
    brief = build_clinical_brief(result, context=context, model="extractive")
    digest = brief.digest
    brief.to_dict()["stages"].clear()
    brief.citations[0]["source_start"] = 9000
    assert brief.digest == digest
    assert brief.citations[0]["source_start"] == 0


@pytest.mark.parametrize(
    "module,name",
    [
        ("summary_section_plan", "require_summary_section_plan"),
        ("summary_length_budget", "build_summary_length_budget"),
        ("summary_profiles", "get_summary_profile"),
        ("summary_citations", "compute_summary_citation_metrics"),
        ("citation_boundaries", "validate_citation_boundaries"),
        ("citation_minimality", "check_citation_minimality"),
        ("summary_temporal_order", "validate_summary_temporal_order"),
        ("guarded_provenance", "build_guarded_provenance_record"),
    ],
)
def test_required_stage_failure_is_not_skipped(monkeypatch, module, name):
    import importlib

    result, context = fixture_context()

    def fail(*args, **kwargs):
        raise RuntimeError("PRIVATE_STAGE_ERROR")

    monkeypatch.setattr(
        importlib.import_module("openmed.clinical." + module), name, fail
    )
    brief = build_clinical_brief(result, context=context, model="extractive")
    assert brief.refusal_reason is BriefRefusal.STAGE_FAILED
    assert brief.summary == ""
    assert "PRIVATE_STAGE_ERROR" not in json.dumps(brief.to_dict())
