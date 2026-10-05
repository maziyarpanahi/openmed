"""Entirely synthetic tests of the reviewed-local (non-synthetic) provenance."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from openmed.clinical.brief import (
    BriefRefusal,
    ReviewedLocalBriefContext,
    brief_policy_fingerprint,
    build_clinical_brief,
)
from openmed.clinical.reviewed_local_evidence import (
    LocalReviewReceipt,
    ReviewAdmissionError,
    ReviewAdmissionRefusal,
    ReviewAuthorityStatus,
    ReviewedLocalEvidence,
    ReviewedLocalReference,
    admit_reviewed_local_evidence,
    reviewed_source_digest,
)
from tests.unit.clinical.test_brief import fixture_context

NOW = 1000


class Source:
    def __init__(self, digest):
        self.digest = digest
        self.calls = 0

    def current_digest(self, source_id):
        assert source_id == "source:" + "a" * 64
        self.calls += 1
        return self.digest


class Authority:
    """Test-only independent registry, never trusts a caller's approval marker."""

    def __init__(self, receipt):
        self.receipt = receipt
        self.revoked = False
        self.calls = 0

    def verify(self, receipt, *, evidence_digest, now):
        self.calls += 1
        if self.revoked:
            return ReviewAuthorityStatus.REVOKED
        if receipt != self.receipt or evidence_digest != self.receipt.evidence_digest:
            return ReviewAuthorityStatus.MISMATCHED
        return ReviewAuthorityStatus.CURRENT


def reviewed_fixture():
    result, synthetic = fixture_context()
    facts = tuple(
        replace(fact, reference_id="ref:" + f"{i + 1:064x}")
        for i, fact in enumerate(synthetic.facts)
    )
    packet = ReviewedLocalEvidence(
        source_id="source:" + "a" * 64,
        source_digest=reviewed_source_digest(result.deidentified_text),
        source_length=len(result.deidentified_text),
        policy_digest=brief_policy_fingerprint(result.deidentified_text, facts),
        references=tuple(
            ReviewedLocalReference(f.reference_id, ref.start, ref.end)
            for f, ref in zip(facts, synthetic.packet.references)
        ),
    )
    receipt = LocalReviewReceipt(
        "receipt:" + "b" * 64,
        "authority:" + "c" * 64,
        packet.evidence_digest,
        900,
        1100,
    )
    packet = replace(packet, review_receipt=receipt)
    context = ReviewedLocalBriefContext(
        packet,
        synthetic.content_digest,
        facts,
        synthetic.nli_predict,
        synthetic.thresholds,
        synthetic.privacy_detector,
        Source(packet.source_digest),
        Authority(receipt),
        lambda: NOW,
    )
    return result, context


def admit(context):
    return admit_reviewed_local_evidence(
        context.packet,
        source_digest=context.packet.source_digest,
        policy_digest=context.packet.policy_digest,
        source=context.source,
        authority=context.authority,
        clock=context.clock,
    )


def test_round_trip_admission_and_full_guarded_generation():
    result, context = reviewed_fixture()
    assert context.packet.provenance_class == "reviewed_local"
    assert ReviewedLocalEvidence.from_json(context.packet.to_json()) == context.packet
    assert admit(context) == context.packet
    brief = build_clinical_brief(result, context=context, model="extractive")
    assert brief.refusal_reason is None
    assert brief.summary == result.deidentified_text
    assert context.authority.calls == context.source.calls == 3
    assert brief.metrics["reviewed_evidence"] == context.packet.to_dict()
    assert result.deidentified_text not in json.dumps(brief.to_dict())
    assert result.deidentified_text not in repr(context)


@pytest.mark.parametrize(
    "failure,reason",
    [
        ("missing", BriefRefusal.REVIEW_RECEIPT_MISSING),
        ("expired", BriefRefusal.REVIEW_RECEIPT_EXPIRED),
        ("future", BriefRefusal.REVIEW_RECEIPT_MISMATCHED),
        ("binding", BriefRefusal.REVIEW_RECEIPT_MISMATCHED),
        ("unknown", BriefRefusal.REVIEW_RECEIPT_MISMATCHED),
        ("forged_expiry", BriefRefusal.REVIEW_RECEIPT_MISMATCHED),
        ("revoked", BriefRefusal.REVIEW_RECEIPT_REVOKED),
        ("source", BriefRefusal.REVIEW_SOURCE_CHANGED),
        ("custody", BriefRefusal.REVIEW_SOURCE_CHANGED),
        ("policy", BriefRefusal.REVIEW_POLICY_CHANGED),
        ("offsets", BriefRefusal.REVIEW_RECEIPT_MISMATCHED),
        ("unavailable_source", BriefRefusal.REVIEW_SOURCE_UNAVAILABLE),
        ("unavailable_authority", BriefRefusal.REVIEW_AUTHORITY_UNAVAILABLE),
        ("boolean_authority", BriefRefusal.REVIEW_AUTHORITY_UNAVAILABLE),
    ],
)
def test_refusals_happen_before_generation(failure, reason):
    result, context = reviewed_fixture()
    packet = context.packet
    receipt = packet.review_receipt
    if failure == "missing":
        packet = replace(packet, review_receipt=None)
    elif failure == "expired":
        context = replace(context, clock=lambda: 1100)
    elif failure == "future":
        context = replace(context, clock=lambda: 899)
    elif failure == "binding":
        packet = replace(
            packet,
            review_receipt=replace(receipt, evidence_digest="sha256:" + "d" * 64),
        )
    elif failure == "unknown":
        packet = replace(
            packet, review_receipt=replace(receipt, receipt_id="receipt:" + "d" * 64)
        )
    elif failure == "forged_expiry":
        packet = replace(packet, review_receipt=replace(receipt, expires_at=9999))
    elif failure == "revoked":
        context.authority.revoked = True
    elif failure == "source":
        packet = replace(packet, source_digest="sha256:" + "d" * 64)
    elif failure == "custody":
        context.source.digest = "sha256:" + "d" * 64
    elif failure == "policy":
        packet = replace(packet, policy_digest="sha256:" + "d" * 64)
    elif failure == "offsets":
        packet = replace(
            packet,
            references=(replace(packet.references[0], end=2), *packet.references[1:]),
        )
    elif failure == "unavailable_source":
        context.source.digest = None
    else:

        def verify(*a, **kw):
            if failure == "boolean_authority":
                return True
            raise RuntimeError("SYNTHETIC_PRIVATE_PATH")

        context.authority.verify = verify
    brief = build_clinical_brief(
        result,
        context=replace(context, packet=packet),
        model=lambda _: pytest.fail("generation reached"),
    )
    assert brief.refusal_reason is reason
    assert brief.summary == ""
    assert brief.to_dict()["stages"][-1] == "evidence"
    assert "SYNTHETIC_PRIVATE_PATH" not in json.dumps(brief.to_response())


@pytest.mark.parametrize("drift", ["source", "revocation", "expiry"])
def test_custody_and_authority_are_rechecked_at_generation(drift):
    result, context = reviewed_fixture()
    calls = []
    original = context.authority.verify

    def verify(*a, **kw):
        status = original(*a, **kw)
        if context.authority.calls == 1:
            if drift == "source":
                context.source.digest = "sha256:" + "d" * 64
            elif drift == "revocation":
                context.authority.revoked = True
        return status

    context.authority.verify = verify
    if drift == "expiry":

        def clock():
            calls.append(1)
            return 1000 if len(calls) == 1 else 1100

        context = replace(context, clock=clock)
    brief = build_clinical_brief(
        result, context=context, model=lambda _: pytest.fail("generation reached")
    )
    assert brief.refusal_reason in {
        BriefRefusal.REVIEW_SOURCE_CHANGED,
        BriefRefusal.REVIEW_RECEIPT_REVOKED,
        BriefRefusal.REVIEW_RECEIPT_EXPIRED,
    }
    assert brief.to_dict()["stages"][-1] == "generation"


def test_synthetic_context_cannot_be_relabelled_or_receive_local_packet():
    result, synthetic = fixture_context()
    _, reviewed = reviewed_fixture()
    brief = build_clinical_brief(
        result,
        context=replace(synthetic, packet=reviewed.packet),
        model=lambda _: pytest.fail("generation"),
    )
    assert brief.refusal_reason is BriefRefusal.INVALID_EVIDENCE
    with pytest.raises(ReviewAdmissionError):
        ReviewedLocalEvidence.from_dict(
            synthetic.packet.to_dict() | {"provenance_class": "reviewed_local"}
        )
    assert (
        build_clinical_brief(
            result, context=synthetic, model="extractive"
        ).refusal_reason
        is None
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_id", "/private/patient.txt"),
        ("source_id", "PatientName"),
        ("offset_convention", "utf8_bytes"),
        ("provenance_class", "synthetic"),
        ("schema_version", True),
        ("source_length", True),
        ("source_digest", "private content"),
        ("text", "SYNTHETIC_PRIVATE_TEXT"),
    ],
)
def test_wire_rejects_private_values_unknown_fields_and_wrong_coordinates(field, value):
    _, context = reviewed_fixture()
    with pytest.raises(ReviewAdmissionError) as error:
        ReviewedLocalEvidence.from_dict(context.packet.to_dict() | {field: value})
    assert error.value.reason is ReviewAdmissionRefusal.INVALID
    assert str(value) not in str(error.value)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["references"][0].update(start=True),
        lambda p: p["references"][0].update(end=p["source_length"] + 1),
        lambda p: p["references"][0].update(text="SYNTHETIC_PRIVATE_TEXT"),
        lambda p: p["references"].append(p["references"][0]),
        lambda p: p["review_receipt"].update(token="SYNTHETIC_SECRET"),
    ],
)
def test_nested_span_integrity_and_value_free_shape(mutate):
    _, context = reviewed_fixture()
    packet = context.packet.to_dict()
    mutate(packet)
    with pytest.raises(ReviewAdmissionError):
        ReviewedLocalEvidence.from_dict(packet)


def test_tampered_frozen_record_is_revalidated():
    _, context = reviewed_fixture()
    object.__setattr__(
        context.packet.references[0], "end", context.packet.source_length + 1
    )
    with pytest.raises(ReviewAdmissionError):
        admit(context)


@pytest.mark.parametrize(
    "clock", [lambda: float("inf"), lambda: float("nan"), lambda: True, lambda: -1]
)
def test_invalid_clock_fails_closed(clock):
    _, context = reviewed_fixture()
    with pytest.raises(ReviewAdmissionError) as error:
        admit(replace(context, clock=clock))
    assert error.value.reason is ReviewAdmissionRefusal.AUTHORITY_UNAVAILABLE


def test_malformed_json_errors_are_value_free_without_chains():
    with pytest.raises(ReviewAdmissionError) as error:
        ReviewedLocalEvidence.from_json('{"SYNTHETIC_PRIVATE":')
    assert error.value.__context__ is None
    assert "SYNTHETIC_PRIVATE" not in str(error.value)
    with pytest.raises(ReviewAdmissionError):
        ReviewedLocalEvidence.from_json('{"kind":1,"kind":2}')


def test_shared_swift_wire_fixture_and_unicode_coordinates():
    _, context = reviewed_fixture()
    fixture = (
        Path(__file__).parents[2] / "fixtures/clinical/brief_parity/reviewed_local.json"
    )
    assert ReviewedLocalEvidence.from_json(fixture.read_text()) == context.packet
    assert (
        reviewed_source_digest("A😀क。")
        == "sha256:9da25197118f0b51f8a6d65c45ccffdefee37003d3d8cb37ef6693bed7a91050"
    )


def test_changed_reviewed_axes_require_new_policy_review():
    result, context = reviewed_fixture()
    context = replace(
        context,
        facts=(replace(context.facts[0], negation="negated"), *context.facts[1:]),
    )
    brief = build_clinical_brief(
        result, context=context, model=lambda _: pytest.fail("generation")
    )
    assert brief.refusal_reason is BriefRefusal.REVIEW_POLICY_CHANGED


def test_private_source_store_and_clock_failures_do_not_escape():
    result, context = reviewed_fixture()

    def broken(*a, **kw):
        raise RuntimeError("SYNTHETIC_PRIVATE_PATH")

    for target, reason in [
        ("source", BriefRefusal.REVIEW_SOURCE_UNAVAILABLE),
        ("clock", BriefRefusal.REVIEW_AUTHORITY_UNAVAILABLE),
    ]:
        if target == "source":
            context.source.current_digest = broken
            changed = context
        else:
            changed = replace(context, clock=broken)
        brief = build_clinical_brief(
            result, context=changed, model=lambda _: pytest.fail("generation")
        )
        assert brief.refusal_reason is reason
        assert "SYNTHETIC_PRIVATE_PATH" not in json.dumps(brief.to_response())


def test_safe_serialization_revalidates_altered_frozen_records():
    _, context = reviewed_fixture()
    object.__setattr__(context.packet, "source_id", "SYNTHETIC_PRIVATE_PATH")
    for serialize in (context.packet.to_dict, context.packet.to_json):
        with pytest.raises(ReviewAdmissionError) as error:
            serialize()
        assert error.value.reason is ReviewAdmissionRefusal.INVALID
        assert "SYNTHETIC_PRIVATE_PATH" not in str(error.value)
