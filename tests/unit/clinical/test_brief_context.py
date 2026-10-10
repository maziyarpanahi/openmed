"""Synthetic offline adapter vectors; no model or clinical qualification claims."""

import json
from dataclasses import asdict, replace
from datetime import datetime
from types import SimpleNamespace

import pytest

from openmed.clinical.brief import (
    BriefContext,
    BriefFact,
    _digest,
    brief_policy_fingerprint,
    build_clinical_brief,
)
from openmed.clinical.brief_context import (
    BriefContextCode,
    BriefContextError,
    BriefExtraction,
    BriefFactMapping,
    LocalBriefContextProvider,
    plan_brief_context,
)
from openmed.clinical.evidence_packet import (
    build_evidence_packet,
    fingerprint_evidence_review,
)
from openmed.clinical.journey_contracts import (
    ClinicalFact,
    ConflictSet,
    EvidenceLocator,
)
from openmed.clinical.nli_gate import NLIThresholds
from openmed.clinical.review_state_machine import (
    ReviewState,
    ReviewStateMachine,
    make_opaque_event_id,
)
from openmed.core.pii import DeidentificationResult, PIIEntity

REVIEW_ID = "a" * 64
SENTENCES = (
    "The admission problem was dehydration.",
    "The discharge diagnosis was dehydration.",
    "Symptoms improved after fluids.",
)
FIELDS = ("admission_reason", "discharge_diagnoses", "hospital_course")


def adapter_fixture():
    """Construct reviewed evidence independently of the adapter's mapping."""
    text = " ".join(SENTENCES)
    artifact = DeidentificationResult(text, text, [], "mask", datetime(2026, 1, 1))
    facts, locators, mappings, manual_facts, rows = [], [], [], [], []
    source_id = "synthetic:" + _digest("source-1")[7:]
    for i, (sentence, profile_field) in enumerate(zip(SENTENCES, FIELDS)):
        locator_id = f"locator_{i:016d}"
        start = text.index(sentence)
        locators.append(
            EvidenceLocator(
                locator_id,
                "artifact_0000000000000001",
                "text_span",
                {"start": start, "end": start + len(sentence)},
            )
        )
        facts.append(
            ClinicalFact(
                f"fact_{i:016d}",
                "subject_0000000000000001",
                "condition",
                "dehydration",
                "active",
                (locator_id,),
                "sha256:" + "b" * 64,
                attributes={
                    "assertion": "affirmed",
                    "certainty": "certain",
                    "experiencer": "patient",
                    "field_states": {"value": "known"},
                },
            )
        )
        mappings.append(BriefFactMapping(f"fact_{i:016d}", profile_field, "recent"))
        ref = "synthetic:" + _digest(locator_id)[7:]
        manual_facts.append(
            BriefFact(ref, profile_field, "affirmed", "certain", "recent", "patient")
        )
        rows.append(
            dict(
                reference_id=ref,
                source_id=source_id,
                start=start,
                end=start + len(sentence),
            )
        )
    policy = brief_policy_fingerprint(text, tuple(manual_facts))
    # Explicit synthetic reviewer decisions exist before the provider runs.
    for row in rows:
        fingerprint = fingerprint_evidence_review(**row, policy_fingerprint=policy)
        machine = ReviewStateMachine()
        for state in (ReviewState.IN_REVIEW, ReviewState.APPROVED):
            machine.transition(
                state,
                make_opaque_event_id((row["reference_id"], state.value)),
                fingerprint,
            )
        row.update(
            policy_fingerprint=policy,
            review_state="approved",
            synthetic=True,
            verified=True,
            review_transitions=machine.transitions,
        )
    thresholds = NLIThresholds(
        calibration_id="synthetic-test-only", calibration_method="synthetic-fixture"
    )
    nli_predict = lambda p, h: {
        "entailment": 1.0,
        "contradiction": 0.0,
        "neutral": 0.0,
        "calibration_id": "synthetic-test-only",
    }
    manual = BriefContext(
        build_evidence_packet(rows, policy_fingerprint=policy),
        _digest(text),
        tuple(manual_facts),
        nli_predict,
        thresholds,
        lambda _: [],
    )
    extraction = BriefExtraction(
        artifact,
        "artifact_0000000000000001",
        "source-1",
        tuple(facts),
        tuple(locators),
        tuple(mappings),
        synthetic=True,
    )
    reviewed_plan = plan_brief_context(extraction, profile="bhc", thresholds=thresholds)
    resolver = SimpleNamespace(resolve=lambda text, review_id: extraction)
    verifier = SimpleNamespace(
        verify=lambda review_id, plan: (
            (reviewed_plan.binding.digest, manual.packet)
            if review_id == REVIEW_ID
            else None
        )
    )
    provider = LocalBriefContextProvider(
        extraction_provider=resolver,
        review_verifier=verifier,
        thresholds=thresholds,
        nli_predict=nli_predict,
        privacy_detector=manual.privacy_detector,
    )
    return extraction, manual, provider, resolver, verifier


def test_matches_hand_built_context_and_brief():
    extraction, manual, provider, _, _ = adapter_fixture()
    outcome = provider.build(extraction.artifact.original_text, REVIEW_ID)
    assert outcome.code is BriefContextCode.READY
    assert outcome.context == manual
    expected = build_clinical_brief(
        extraction.artifact, model="extractive", context=manual
    )
    actual = build_clinical_brief(
        outcome.artifact, model="extractive", context=outcome.context
    )
    assert actual.refusal_reason is None
    assert actual.to_response() == expected.to_response()
    assert provider(extraction.artifact.original_text, REVIEW_ID) == (
        outcome.artifact,
        manual,
    )
    assert "dehydration" not in json.dumps(outcome.to_dict())
    assert "dehydration" not in repr(outcome)


@pytest.mark.parametrize(
    "change,code",
    [
        ("empty", "missing_facts"),
        ("mapping", "missing_mapping"),
        ("extra_mapping", "missing_mapping"),
        ("duplicate_fact", "conflicted_facts"),
        ("duplicate_locator", "ambiguous_source"),
        ("conflict", "conflicted_facts"),
        ("unknown", "unknown_fact"),
        ("missing_axis", "unknown_fact"),
        ("unknown_status", "unknown_fact"),
        ("unknown_state", "unknown_fact"),
        ("unsupported_axis", "unsupported_mapping"),
        ("field", "unsupported_mapping"),
        ("temporal", "unknown_fact"),
        ("missing_locator", "missing_source"),
        ("artifact", "missing_source"),
        ("multiple", "ambiguous_source"),
        ("nontext", "unsupported_mapping"),
        ("transform", "unsupported_mapping"),
        ("overflow", "invalid_offsets"),
        ("overlap", "ambiguous_source"),
        ("mixed_subject", "ambiguous_source"),
        ("demographic", "unsupported_mapping"),
    ],
)
def test_negative_mapping_controls_never_reach_verifier(change, code):
    extraction, _, provider, resolver, verifier = adapter_fixture()
    first = extraction.facts[0]
    attrs = dict(first.attributes)
    if change == "empty":
        extraction = replace(extraction, facts=())
    elif change == "mapping":
        extraction = replace(extraction, mappings=extraction.mappings[1:])
    elif change == "extra_mapping":
        extraction = replace(
            extraction,
            mappings=(
                *extraction.mappings,
                BriefFactMapping("extra", FIELDS[0], "recent"),
            ),
        )
    elif change == "duplicate_fact":
        extraction = replace(extraction, facts=(*extraction.facts, first))
    elif change == "duplicate_locator":
        extraction = replace(
            extraction, locators=(*extraction.locators, extraction.locators[0])
        )
    elif change == "conflict":
        extraction = replace(
            extraction,
            conflicts=(
                ConflictSet(
                    "conflict_0000000000000001",
                    "subject_0000000000000001",
                    "contradiction",
                    ("fact_0000000000000000", "fact_0000000000000001"),
                    "open",
                    "synthetic",
                    "sha256:" + "c" * 64,
                ),
            ),
        )
    elif change in {
        "unknown",
        "missing_axis",
        "unsupported_axis",
        "unknown_state",
        "demographic",
    }:
        if change == "missing_axis":
            attrs.pop("assertion")
        elif change == "unknown_state":
            attrs["field_states"] = {"value": "unknown"}
        elif change == "demographic":
            attrs["demographic_class"] = "name"
        else:
            attrs["assertion"] = "unknown" if change == "unknown" else "conditional"
        extraction = replace(
            extraction, facts=(replace(first, attributes=attrs), *extraction.facts[1:])
        )
    elif change == "unknown_status":
        extraction = replace(
            extraction, facts=(replace(first, status="unknown"), *extraction.facts[1:])
        )
    elif change in {"field", "temporal"}:
        m = extraction.mappings[0]
        m = (
            replace(m, profile_field="invented")
            if change == "field"
            else replace(m, temporality="unknown")
        )
        extraction = replace(extraction, mappings=(m, *extraction.mappings[1:]))
    elif change in {"missing_locator", "multiple", "mixed_subject"}:
        changes = {
            "missing_locator": {"evidence_ids": ("locator_9999999999999999",)},
            "multiple": {
                "evidence_ids": ("locator_0000000000000000", "locator_0000000000000001")
            },
            "mixed_subject": {"subject_id": "subject_0000000000000002"},
        }[change]
        extraction = replace(
            extraction, facts=(replace(first, **changes), *extraction.facts[1:])
        )
    else:
        loc = extraction.locators[0]
        changes = {
            "artifact": {"artifact_id": "artifact_0000000000000002"},
            "nontext": {
                "location_type": "json_pointer",
                "location": {"pointer": "/field"},
            },
            "transform": {"transform": {"kind": "ocr"}},
            "overflow": {"location": {"start": 0, "end": 20000}},
            "overlap": {
                "location": {
                    "start": 0,
                    "end": len(extraction.artifact.deidentified_text),
                }
            },
        }[change]
        extraction = replace(
            extraction, locators=(replace(loc, **changes), *extraction.locators[1:])
        )
    resolver.resolve = lambda *args: extraction
    verifier.verify = lambda *args: pytest.fail("invalid extraction reached reviewer")
    outcome = provider.build(extraction.artifact.original_text, REVIEW_ID)
    assert outcome.code.value == code
    assert outcome.context is None


@pytest.mark.parametrize(
    "change",
    [
        "text",
        "fact_value",
        "fact_axis",
        "locator",
        "profile_field",
        "source_id",
        "artifact_id",
        "calibration",
        "profile",
        "original",
    ],
)
def test_changed_evidence_or_policy_requires_new_review(change):
    extraction, manual, provider, resolver, _ = adapter_fixture()
    if change == "text":
        extraction.artifact.deidentified_text += " Changed."
    elif change in {"fact_value", "fact_axis"}:
        first = extraction.facts[0]
        changes = (
            {"value": "PRIVATE_CANARY"}
            if change == "fact_value"
            else {"attributes": {**first.attributes, "assertion": "negated"}}
        )
        extraction = replace(
            extraction, facts=(replace(first, **changes), *extraction.facts[1:])
        )
    elif change == "locator":
        loc = extraction.locators[0]
        extraction = replace(
            extraction,
            locators=(
                replace(loc, location={"start": 1, "end": loc.location["end"]}),
                *extraction.locators[1:],
            ),
        )
    elif change == "profile_field":
        extraction = replace(
            extraction,
            mappings=(
                replace(extraction.mappings[0], profile_field=FIELDS[2]),
                *extraction.mappings[1:],
            ),
        )
    elif change == "source_id":
        extraction = replace(extraction, source_id="artifact_0000000000000002")
    elif change == "artifact_id":
        extraction = replace(
            extraction,
            artifact_id="artifact_0000000000000002",
            locators=tuple(
                replace(l, artifact_id="artifact_0000000000000002")
                for l in extraction.locators
            ),
        )
    elif change == "calibration":
        provider._thresholds = replace(manual.thresholds, margin=0.07)
    elif change == "profile":
        provider._profile = "discharge_summary"
    else:
        extraction.artifact.original_text += " PRIVATE_CANARY"
    resolver.resolve = lambda *args: extraction
    outcome = provider.build(extraction.artifact.original_text, REVIEW_ID)
    assert outcome.code is BriefContextCode.EVIDENCE_CHANGED
    assert "PRIVATE_CANARY" not in json.dumps(outcome.to_dict())


def test_unreviewed_revoked_wrong_packet_and_failed_providers():
    extraction, manual, provider, resolver, verifier = adapter_fixture()
    text = extraction.artifact.original_text
    verifier.verify = lambda *args: None
    assert provider.build(text, REVIEW_ID).code is BriefContextCode.REVIEW_REQUIRED
    with pytest.raises(BriefContextError, match="^review_required$") as error:
        provider(text, REVIEW_ID)
    assert error.value.__context__ is None
    plan = plan_brief_context(extraction, profile="bhc", thresholds=manual.thresholds)
    verifier.verify = lambda *args: (
        plan.binding.digest,
        replace(
            manual.packet,
            references=manual.packet.references[1:],
            rejection_report=None,
        ),
    )
    assert provider.build(text, REVIEW_ID).code is BriefContextCode.INVALID_EVIDENCE
    resolver.resolve = lambda *args: None
    assert provider.build(text, REVIEW_ID).code is BriefContextCode.MISSING_SOURCE

    def broken(*args):
        raise RuntimeError("PRIVATE_CANARY")

    resolver.resolve = broken
    assert provider.build(text, REVIEW_ID).code is BriefContextCode.PROVIDER_UNAVAILABLE
    with pytest.raises(BriefContextError) as error:
        provider(text, REVIEW_ID)
    assert error.value.__context__ is None
    assert "PRIVATE_CANARY" not in repr(error.value)


def test_order_is_canonical_and_profiles_have_no_silent_fallback():
    extraction, manual, _, _, _ = adapter_fixture()
    baseline = plan_brief_context(
        extraction, profile="bhc", thresholds=manual.thresholds
    )
    reordered = replace(
        extraction,
        facts=extraction.facts[::-1],
        mappings=extraction.mappings[::-1],
        locators=extraction.locators[::-1],
    )
    assert (
        plan_brief_context(reordered, profile="bhc", thresholds=manual.thresholds)
        == baseline
    )
    with pytest.raises(BriefContextError, match="^unsupported_mapping$") as error:
        plan_brief_context(
            extraction, profile="PRIVATE_CANARY", thresholds=manual.thresholds
        )
    assert error.value.__context__ is None
    assert "PRIVATE_CANARY" not in repr(error.value)
    assert "dehydration" not in json.dumps(asdict(baseline.binding))


def test_non_synthetic_input_is_not_relabelled_or_approved():
    extraction, _, provider, resolver, verifier = adapter_fixture()
    resolver.resolve = lambda *args: replace(extraction, synthetic=False)
    verifier.verify = lambda *args: pytest.fail("non-synthetic input reached verifier")
    assert (
        provider.build(extraction.artifact.original_text, REVIEW_ID).code
        is BriefContextCode.UNSUPPORTED_MAPPING
    )


@pytest.mark.parametrize(
    "private",
    ["PRIVATE_CANARY", "患者識別子１２３", "/private/patient/chart", "1980-01-02"],
)
def test_private_source_mapping_values_never_enter_plan_or_diagnostics(private):
    extraction, manual, provider, _, _ = adapter_fixture()
    extraction.artifact.mapping = {"[PERSON]": private}
    extraction.artifact.pii_entities = [
        PIIEntity(text=private, label="NAME", start=0, end=len(private), confidence=1.0)
    ]
    plan = plan_brief_context(extraction, profile="bhc", thresholds=manual.thresholds)
    safe = json.dumps(asdict(plan.binding)) + repr(plan.spans) + repr(extraction)
    assert private not in safe
    outcome = provider.build(extraction.artifact.original_text, REVIEW_ID)
    assert outcome.code is BriefContextCode.EVIDENCE_CHANGED
    assert private not in json.dumps(outcome.to_dict())


def test_changed_evidence_during_verification_refuses():
    extraction, _, provider, _, verifier = adapter_fixture()
    original = verifier.verify

    def change(review_id, plan):
        result = original(review_id, plan)
        extraction.artifact.deidentified_text += " Changed."
        return result

    verifier.verify = change
    assert (
        provider.build(extraction.artifact.original_text, REVIEW_ID).code
        is BriefContextCode.EVIDENCE_CHANGED
    )


def test_unknown_field_state_has_no_negative_fact_fallback():
    extraction, _, provider, resolver, _ = adapter_fixture()
    first = extraction.facts[0]
    attrs = {**first.attributes, "field_states": {"value": "novel_state"}}
    resolver.resolve = lambda *args: replace(
        extraction, facts=(replace(first, attributes=attrs), *extraction.facts[1:])
    )
    assert (
        provider.build(extraction.artifact.original_text, REVIEW_ID).code
        is BriefContextCode.UNKNOWN_FACT
    )


def test_local_protocol_socket_attempt_is_refused_without_content():
    import socket

    extraction, _, provider, _, verifier = adapter_fixture()

    def network(*args):
        socket.create_connection(("127.0.0.1", 9), timeout=0.01)
        pytest.fail("outbound socket was permitted")

    verifier.verify = network
    assert (
        provider.build(extraction.artifact.original_text, REVIEW_ID).code
        is BriefContextCode.PROVIDER_UNAVAILABLE
    )


@pytest.mark.parametrize("invalid", [None, 0, "", "PRIVATE_CANARY", "A" * 64])
def test_invalid_review_reference_never_calls_resolver(invalid):
    extraction, _, provider, resolver, _ = adapter_fixture()
    resolver.resolve = lambda *args: pytest.fail(
        "invalid review reference reached resolver"
    )
    assert (
        provider.build(extraction.artifact.original_text, invalid).code
        is BriefContextCode.INVALID_EVIDENCE
    )


@pytest.mark.parametrize("change", ["null_value", "missing_source_identity"])
def test_missing_values_and_identities_refuse(change):
    extraction, _, provider, resolver, _ = adapter_fixture()
    if change == "null_value":
        extraction = replace(
            extraction,
            facts=(replace(extraction.facts[0], value=None), *extraction.facts[1:]),
        )
        expected = BriefContextCode.UNKNOWN_FACT
    else:
        extraction = replace(extraction, source_id="")
        expected = BriefContextCode.MISSING_SOURCE
    resolver.resolve = lambda *args: extraction
    assert provider.build(extraction.artifact.original_text, REVIEW_ID).code is expected


def test_unicode_character_offsets_and_section_crossing():
    extraction, manual, _, _, _ = adapter_fixture()
    prefix = "History:\n合成🩺\n"
    text = (
        prefix
        + extraction.artifact.deidentified_text
        + "\nAssessment:\nSynthetic finding."
    )
    artifact = replace(extraction.artifact, original_text=text, deidentified_text=text)
    locators = tuple(
        replace(
            loc,
            location={
                "start": loc.location["start"] + len(prefix),
                "end": loc.location["end"] + len(prefix),
            },
        )
        for loc in extraction.locators
    )
    extraction = replace(extraction, artifact=artifact, locators=locators)
    plan = plan_brief_context(extraction, profile="bhc", thresholds=manual.thresholds)
    for span, sentence in zip(plan.spans, SENTENCES):
        assert text[span.start : span.end] == sentence
    crossing = replace(
        locators[0], location={"start": locators[0].location["start"], "end": len(text)}
    )
    with pytest.raises(BriefContextError, match="^ambiguous_source$"):
        plan_brief_context(
            replace(extraction, locators=(crossing, *locators[1:])),
            profile="bhc",
            thresholds=manual.thresholds,
        )
