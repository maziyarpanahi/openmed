"""Synthetic deterministic critical-token, privacy and review-gate controls."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from openmed.multimodal.transcript_holds import (
    CONFIDENCE_THRESHOLDS,
    POLICY_VERSION,
    CriticalTokenClass,
    DoseCheckStatus,
    DraftTokenCitation,
    HoldConfirmation,
    TokenDoseEvidence,
    TokenIdentity,
    TranscriptDose,
    TranscriptHoldError,
    TranscriptHoldGate,
    TranscriptToken,
    check_transcript_doses,
    classify_critical_token,
)

FIXTURE = Path(__file__).parents[2] / "fixtures/multimodal/transcript_hold_cases.json"
CASES = json.loads(FIXTURE.read_text())["cases"]
REF = TokenIdentity(1, 0)


def gate(token, *, dose_evidence=(), revision=1):
    return TranscriptHoldGate(
        [token],
        [
            DraftTokenCitation(10, (token.identity,)),
            DraftTokenCitation(11, (token.identity,)),
        ],
        revision=revision,
        medication_names=("syntheticmed",),
        dose_evidence=dose_evidence,
    )


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("delta", [-0.000001, 0, 0.000001])
def test_versioned_threshold_table_and_acoustic_disagreement(case, delta):
    confidence = case["threshold"] + delta
    result = gate(
        TranscriptToken(REF, case["text"], confidence, tuple(case["alternatives"]))
    )
    disagreement = any(value != case["text"] for value in case["alternatives"])
    assert bool(result.holds) == (delta < 0 or disagreement)
    assert bool(result.statement_holds(10)) == bool(result.holds)
    assert bool(result.statement_holds(11)) == bool(result.holds)
    if result.holds:
        record = result.holds[0].to_dict()
        assert record["classes"] == case["classes"]
        assert record["threshold"] == case["threshold"]
        assert record["policy_version"] == POLICY_VERSION == 1


@pytest.mark.parametrize("kind", list(CriticalTokenClass))
def test_every_class_has_a_frozen_threshold(kind):
    assert CONFIDENCE_THRESHOLDS[kind] == (
        0.90 if kind == CriticalTokenClass.UNCERTAINTY else 0.95
    )
    with pytest.raises(TypeError):
        CONFIDENCE_THRESHOLDS[kind] = 0


def test_negative_controls_consistent_spelling_and_unknown_words():
    assert not gate(TranscriptToken(REF, "Fifteen.", 1, ("fifteen",))).holds
    assert not gate(TranscriptToken(REF, "ordinary", 0.1)).holds
    assert gate(TranscriptToken(REF, "ordinary", 1, ("no",))).holds
    assert not classify_critical_token("ordinary")


def test_missing_confidence_is_not_a_pass():
    assert gate(TranscriptToken(REF, "no", None)).holds


@pytest.mark.parametrize(
    "score", [-1, 1.1, float("nan"), float("inf"), True, "PRIVATE"]
)
def test_invalid_confidence_is_content_free(score):
    with pytest.raises(TranscriptHoldError, match="^invalid_confidence$"):
        TranscriptToken(REF, "PRIVATE", score)


def test_adjacent_number_unit_and_dose_scores():
    other = TokenIdentity(1, 1)
    result = TranscriptHoldGate(
        [TranscriptToken(REF, "fifteen", 0.94), TranscriptToken(other, "mg", 1)],
        [DraftTokenCitation(1, (other,)), DraftTokenCitation(2, (REF, other))],
        revision=1,
    )
    assert CriticalTokenClass.DOSE in result.holds[0].classes
    assert not result.statement_holds(1)
    assert result.statement_holds(2)
    # An explicit unchecked dose cannot silently become an in-range dose.
    for status, expected in [
        (DoseCheckStatus.IN_RANGE, 0),
        (DoseCheckStatus.NOT_CHECKED, 0.5),
        (DoseCheckStatus.FLAGGED, 1),
    ]:
        checked = gate(
            TranscriptToken(REF, "15", 1),
            dose_evidence=[TokenDoseEvidence(REF, status)],
        )
        assert bool(checked.holds) == bool(expected)
        if expected:
            assert checked.holds[0].dose_flag_score == expected


def test_unchanged_dosing_checker_and_no_private_payload_in_diagnostics():
    dose = TranscriptDose((REF,), "syntheticmed", "oral", 50, "mg")
    ranges = {"syntheticmed": {"oral": {"low": 1, "high": 20, "unit": "mg"}}}
    assert check_transcript_doses([dose], ranges)[0].status == DoseCheckStatus.FLAGGED
    assert (
        check_transcript_doses([replace(dose, amount=15)], ranges)[0].status
        == DoseCheckStatus.IN_RANGE
    )
    assert (
        check_transcript_doses([replace(dose, unit="ml")], ranges)[0].status
        == DoseCheckStatus.NOT_CHECKED
    )
    assert check_transcript_doses([dose], None)[0].status == DoseCheckStatus.NOT_CHECKED
    assert "syntheticmed" not in repr(dose)
    with pytest.raises(TranscriptHoldError, match="^dose_check_failed$"):
        check_transcript_doses([dose], {"PRIVATE": {"oral": {"high": "PRIVATE"}}})


def test_hold_resolution_is_bound_authorized_explicit_and_retains_original_records():
    result = gate(TranscriptToken(REF, "fifteen", 0.2, ("fifty",)))
    receipt = HoldConfirmation(REF, result.evidence_digest, 7, True)
    for statement in (10, 11):
        with pytest.raises(TranscriptHoldError, match="unresolved_token_hold"):
            result.export_reviewed(statement, reviewer_confirmed=True)
    with pytest.raises(TranscriptHoldError, match="reviewer_denied"):
        result.resolve(receipt, authorize=lambda _: False)
    with pytest.raises(TranscriptHoldError, match="invalid_confirmation"):
        result.resolve(replace(receipt, confirmed=False), authorize=lambda _: True)
    for changed in (
        gate(TranscriptToken(REF, "fifty", 0.2)),
        gate(TranscriptToken(REF, "fifteen", 0.2, ("fifty",)), revision=2),
    ):
        with pytest.raises(TranscriptHoldError, match="stale_confirmation"):
            changed.resolve(receipt, authorize=lambda _: True)
    result.resolve(receipt, authorize=lambda reviewer: reviewer == 7)
    assert (
        result.holds
        and not result.statement_holds(10)
        and not result.statement_holds(11)
    )
    with pytest.raises(TranscriptHoldError, match="review_required"):
        result.export_reviewed(10, reviewer_confirmed=False)
    assert result.export_reviewed(10, reviewer_confirmed=True)["notice"].startswith(
        "Non-diagnostic"
    )


def test_authority_failure_does_not_echo_error_or_clear_hold():
    result = gate(TranscriptToken(REF, "no", 0.1))

    def denied(_):
        raise RuntimeError("PRIVATE")

    with pytest.raises(TranscriptHoldError, match="^reviewer_denied$"):
        result.resolve(
            HoldConfirmation(REF, result.evidence_digest, 1, True), authorize=denied
        )
    assert result.statement_holds(10)


@pytest.mark.parametrize(
    "payload",
    [
        "Invented Person",
        "patient@example.invalid",
        "رقم-خيالي",
        "/private/synthetic",
        "secret-token",
    ],
)
def test_multilingual_private_values_never_enter_hold_records_or_reprs(payload):
    token = TranscriptToken(REF, payload, 0.1, ("no",))
    result = gate(token)
    serialized = json.dumps(result.holds[0].to_dict())
    assert payload not in serialized + repr(result.holds) + repr(token) + repr(result)
    assert set(result.holds[0].to_dict()) == {
        "segment_id",
        "token_id",
        "classes",
        "confidence",
        "threshold",
        "disagreement_score",
        "dose_flag_score",
        "policy_version",
    }


def test_unknown_duplicate_citations_and_identity_collision_fail_closed():
    token = TranscriptToken(REF, "no", 0.1)
    with pytest.raises(TranscriptHoldError, match="duplicate_identity"):
        TranscriptHoldGate([token, token], [], revision=1)
    with pytest.raises(TranscriptHoldError, match="unknown_citation"):
        TranscriptHoldGate(
            [token], [DraftTokenCitation(1, (TokenIdentity(2, 0),))], revision=1
        )
    with pytest.raises(TranscriptHoldError, match="duplicate_citation"):
        DraftTokenCitation(1, (REF, REF))
    with pytest.raises(TranscriptHoldError, match="missing_citations"):
        DraftTokenCitation(1, ())
    with pytest.raises(TranscriptHoldError, match="invalid_identity"):
        TokenIdentity("PRIVATE", 0)


def test_deterministic_sweep_and_segment_separation():
    tokens = [
        TranscriptToken(TokenIdentity(segment, 0), "no", 0.1) for segment in range(8)
    ]
    statements = [
        DraftTokenCitation(segment, (token.identity,))
        for segment, token in enumerate(tokens)
    ]
    first = TranscriptHoldGate(tokens, statements, revision=1)
    second = TranscriptHoldGate(reversed(tokens), reversed(statements), revision=1)
    assert (
        first.holds == second.holds and first.evidence_digest == second.evidence_digest
    )
    for segment in range(8):
        assert [
            record.identity.segment_id for record in first.statement_holds(segment)
        ] == [segment]


def test_confirmation_and_records_reject_uncontrolled_fields():
    from openmed.multimodal.transcript_holds import TokenHoldRecord

    with pytest.raises(TranscriptHoldError, match="^invalid_confirmation$"):
        HoldConfirmation(REF, "PRIVATE", 1, True)
    with pytest.raises(TranscriptHoldError, match="^invalid_hold_record$"):
        TokenHoldRecord(REF, ("PRIVATE",), 1, 0.95, 0, 0)


def test_changed_policy_inputs_citations_and_doses_invalidate_receipts():
    token = TranscriptToken(REF, "no", 0.1)
    result = gate(token)
    receipt = HoldConfirmation(REF, result.evidence_digest, 1, True)
    changed = [
        TranscriptHoldGate([token], [DraftTokenCitation(12, (REF,))], revision=1),
        gate(token, dose_evidence=[TokenDoseEvidence(REF, DoseCheckStatus.FLAGGED)]),
        TranscriptHoldGate(
            [token],
            [DraftTokenCitation(10, (REF,)), DraftTokenCitation(11, (REF,))],
            revision=1,
            medication_names=("othermed",),
        ),
    ]
    for other in changed:
        with pytest.raises(TranscriptHoldError, match="stale_confirmation"):
            other.resolve(receipt, authorize=lambda _: True)


def test_complete_token_dependencies_do_not_require_source_offsets_or_text():
    token = TranscriptToken(REF, "right", 0.1)
    result = gate(token)
    assert [record.identity for record in result.statement_holds(10)] == [REF]
    with pytest.raises(TranscriptHoldError, match="unknown_dose_token"):
        TranscriptHoldGate(
            [token],
            [],
            revision=1,
            dose_evidence=[
                TokenDoseEvidence(TokenIdentity(99, 0), DoseCheckStatus.FLAGGED)
            ],
        )
    with pytest.raises(TranscriptHoldError, match="duplicate_identity"):
        TranscriptHoldGate(
            [token],
            [DraftTokenCitation(1, (REF,)), DraftTokenCitation(1, (REF,))],
            revision=1,
        )
    with pytest.raises(TranscriptHoldError, match="unknown_statement"):
        result.export_reviewed(99, reviewer_confirmed=True)
