"""Offline fixed transcript-to-dose-check-to-reviewed-permit safety path."""

import pytest

from openmed.multimodal.transcript_holds import (
    DraftTokenCitation,
    HoldConfirmation,
    TokenIdentity,
    TranscriptDose,
    TranscriptHoldError,
    TranscriptHoldGate,
    TranscriptToken,
    check_transcript_doses,
)


def test_every_dependent_statement_stays_held_until_each_correction_is_confirmed():
    refs = tuple(TokenIdentity(0, index) for index in range(4))
    tokens = [
        TranscriptToken(refs[0], "syntheticmed", 1),
        TranscriptToken(refs[1], "fifteen", 0.7, ("fifty",)),
        TranscriptToken(refs[2], "mg", 1),
        TranscriptToken(refs[3], "", 1, ("no",)),
    ]
    dose = TranscriptDose(refs[:3], "syntheticmed", "oral", 50, "mg")
    findings = check_transcript_doses(
        [dose], {"syntheticmed": {"oral": {"high": 20, "unit": "mg"}}}
    )
    gate = TranscriptHoldGate(
        tokens,
        [
            DraftTokenCitation(0, refs[:3]),
            DraftTokenCitation(1, refs[1:]),
            DraftTokenCitation(2, (refs[3],)),
        ],
        revision=1,
        medication_names=("syntheticmed",),
        dose_evidence=findings,
    )
    for ref in refs:
        for statement in range(3):
            if gate.statement_holds(statement):
                with pytest.raises(TranscriptHoldError, match="unresolved_token_hold"):
                    gate.export_reviewed(statement, reviewer_confirmed=True)
        gate.resolve(
            HoldConfirmation(ref, gate.evidence_digest, 5, True),
            authorize=lambda reviewer: reviewer == 5,
        )
    assert len(gate.holds) == 4
    assert all(
        gate.export_reviewed(statement, reviewer_confirmed=True)["evidence_digest"]
        == gate.evidence_digest
        for statement in range(3)
    )
    # Corrections rebuild fixed evidence; old receipts do not clear new findings.
    next_gate = TranscriptHoldGate(
        tokens, [DraftTokenCitation(0, refs)], revision=2, dose_evidence=findings
    )
    with pytest.raises(TranscriptHoldError, match="stale_confirmation"):
        next_gate.resolve(
            HoldConfirmation(refs[0], gate.evidence_digest, 5, True),
            authorize=lambda _: True,
        )
