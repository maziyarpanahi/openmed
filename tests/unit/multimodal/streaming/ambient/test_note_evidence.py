"""Synthetic source screening, lineage, contradiction and clinician-custody gates."""

import json
import socket
from dataclasses import replace

import pytest

from openmed.clinical.nli import HeuristicNLIBackend
from openmed.clinical.reviewed_local_evidence import (
    LocalReviewReceipt,
    ReviewAuthorityStatus,
)
from openmed.multimodal.streaming.ambient.note_evidence import (
    AmbientNoteSession,
    ExtractionOperation,
    NliNoteVerifier,
    NoteAssertion,
    NoteCitation,
    NoteEvidenceError,
    NoteExtractionStep,
    NotePolicy,
    NoteProposal,
    NoteProviderDeclaration,
    NoteSection,
    NoteSectionKind,
    NoteStatement,
    NoteUncertainty,
    NoteVerificationResult,
    StatementSupport,
    SupportVerdict,
)
from openmed.multimodal.streaming.ambient.speaker_roles import (
    RoleAttributionReview,
    RoleHypothesis,
    RoleObservation,
    TranscriptSource,
    assess_speaker_roles,
)
from openmed.multimodal.streaming.privacy import (
    PhiRedactionSpan,
    PhiResolution,
    PhiScreenDeclaration,
    PhiScreenResult,
    SegmentPhiDecision,
)
from openmed.multimodal.streaming.speech import ClinicalRole, SpeakerAliases
from openmed.multimodal.streaming.transcript import RevisionLedger, TranscriptToken

PRIVATE = "SYNTHETIC-PRIVATE-NAME"
CLAIM = "The patient reports cough."
EXTRACTOR = NoteProviderDeclaration(artifact_digest="a" * 64, policy_digest="b" * 64)
VERIFIER = NoteProviderDeclaration(artifact_digest="c" * 64, policy_digest="d" * 64)


def ref(index):
    return f"{index:032x}"


class Screener:
    declaration = PhiScreenDeclaration(artifact_digest="f" * 64)

    def __init__(self):
        self.calls = []
        self.extra = None
        self.resolution = PhiResolution.RESOLVED
        self.callback = None

    def screen(self, request):
        self.calls.append(request)
        if self.callback:
            self.callback()
        decisions = []
        for segment in request.segments:
            spans = []
            for value in (PRIVATE, self.extra):
                if value and value in segment.text:
                    start = segment.text.index(value)
                    spans.append(PhiRedactionSpan(start, start + len(value)))
            decisions.append(
                SegmentPhiDecision(
                    segment_ref=segment.segment_ref,
                    revision=segment.revision,
                    resolution=self.resolution,
                    redactions=tuple(spans),
                )
            )
        return PhiScreenResult(
            request_digest=request.request_digest, decisions=tuple(decisions)
        )


class Extractor:
    declaration = EXTRACTOR

    def __init__(self):
        self.requests = []
        self.transform = None
        self.callback = None

    def extract(self, request):
        self.requests.append(request)
        if self.callback:
            self.callback()
        statements, steps = [], []
        for index, evidence in enumerate(request.evidence):
            start = evidence.text.find(CLAIM)
            if start < 0:
                start = 0
            text = CLAIM if CLAIM in evidence.text else evidence.text
            statement = NoteStatement(
                statement_ref=ref(1000 + index),
                text=text,
                citations=(
                    NoteCitation(evidence.evidence_ref, start, start + len(text)),
                ),
                step_ref=ref(1100 + index),
                assertion=NoteAssertion.AFFIRMED,
                uncertainty=NoteUncertainty.NONE_DECLARED,
                concept_ref=ref(1200),
                subject_ref=ref(1300),
                temporal_ref=ref(1400),
            )
            statements.append(statement)
            steps.append(
                NoteExtractionStep(
                    statement.step_ref,
                    ExtractionOperation.PARAPHRASE,
                    (evidence.evidence_ref,),
                    self.declaration.artifact_digest,
                )
            )
        proposal = NoteProposal(
            request.request_digest,
            (NoteSection(NoteSectionKind.HISTORY, tuple(statements)),),
            tuple(steps),
        )
        return self.transform(proposal) if self.transform else proposal


class Verifier:
    declaration = VERIFIER

    def __init__(self):
        self.requests = []
        self.verdict = SupportVerdict.ENTAILED
        self.confidence = 999_000
        self.transform = None
        self.callback = None

    def verify(self, request):
        self.requests.append(request)
        if self.callback:
            self.callback()
        result = NoteVerificationResult(
            request.request_digest,
            tuple(
                StatementSupport(item.statement_ref, self.verdict, self.confidence)
                for item in request.statements
            ),
        )
        return self.transform(result) if self.transform else result


class Authority:
    def __init__(self, receipt):
        self.receipt = receipt
        self.status = ReviewAuthorityStatus.CURRENT
        self.calls = []
        self.callback = None

    def verify(self, receipt, *, evidence_digest, now):
        self.calls.append((receipt, evidence_digest, now))
        if self.callback:
            self.callback()
        if receipt != self.receipt or evidence_digest != self.receipt.evidence_digest:
            return ReviewAuthorityStatus.MISMATCHED
        return self.status


class CurrentSource:
    def __init__(self, packet):
        self.packet = packet
        self.callback = None
        self.calls = 0
        self.changed = False

    def current_digest(self, source_id):
        self.calls += 1
        if self.callback:
            self.callback(self.calls)
        if self.changed or source_id != self.packet.source_id:
            return None
        return self.packet.source_digest


def fixture(
    *, count=1, text=None, finalized=True, policy=None, verifier=None, offsets=None
):
    aliases = SpeakerAliases(ref(1))
    aliases.apply_revision(0, (ref(2),))
    sources, observations = [], []
    for index in range(count):
        content = text or f"{PRIVATE}. {CLAIM}"
        ledger = RevisionLedger(
            ref(1),
            ref(10 + index),
            len(content),
            (
                TranscriptToken(
                    token_ref=ref(60 + index),
                    start=0,
                    end=len(content),
                    start_ms=0,
                    end_ms=1000,
                ),
            ),
        )
        start, end = offsets or (0, len(content))
        citation = ledger.capture_span(ref(20 + index), start, end)
        if finalized:
            ledger.finalize(expected_revision=0)
        sources.append(TranscriptSource(ledger=ledger, text=content))
        observations.append(
            RoleObservation(
                observation_id=ref(30 + index),
                cluster_ref=ref(2),
                diarization_revision=0,
                citation=citation,
                hypotheses=(
                    RoleHypothesis(role=ClinicalRole.PATIENT, evidence=(citation,)),
                ),
            )
        )
    sources, observations = tuple(sources), tuple(observations)
    context = assess_speaker_roles(aliases, sources, observations)
    reviews = tuple(
        RoleAttributionReview(
            observation_id=item.observation_id,
            context_digest=context.context_digest,
            role=ClinicalRole.PATIENT,
            review_id=ref(40 + index),
            reviewer_id=ref(50),
            decision_digest="e" * 64,
            reviewer_confirmed=True,
        )
        for index, item in enumerate(observations)
    )
    screen, extract, verify = Screener(), Extractor(), verifier or Verifier()
    session = AmbientNoteSession(
        aliases,
        sources,
        observations,
        reviews,
        screener=screen,
        extractor=extract,
        verifier=verify,
        policy=policy or NotePolicy(),
        clock=lambda: 0.0,
    )
    return session, screen, extract, verify


def receipt_for(draft):
    return LocalReviewReceipt(
        "receipt:" + "1" * 64,
        "authority:" + "2" * 64,
        draft.review_packet.evidence_digest,
        100,
        200,
    )


def finalize(
    session, draft, *, authority=None, source=None, receipt=None, clock=lambda: 150
):
    receipt = receipt or receipt_for(draft)
    return session.finalize(
        receipt,
        current_source=source or CurrentSource(draft.review_packet),
        authority=authority or Authority(receipt),
        clock=clock,
    )


def mutate_statement(proposal, **values):
    section = proposal.sections[0]
    first = replace(section.statements[0], **values)
    return replace(
        proposal,
        sections=(replace(section, statements=(first,) + section.statements[1:]),),
    )


def reasons(draft, index=0):
    return json.loads(draft.report_json)["evaluation"]["statements"][index]["blockers"]


def test_end_to_end_screening_original_offsets_and_real_clinician_admission():
    session, screen, extractor, verifier = fixture()
    with session:
        draft = session.compose(ref(80))
        assert draft.text == "history\n" + CLAIM
        assert not reasons(draft)
        evidence = extractor.requests[0].evidence[0]
        assert PRIVATE not in evidence.text and "[REDACTED]" in evidence.text
        assert (
            evidence.alias == "speaker_0001" and evidence.role is ClinicalRole.PATIENT
        )
        report = json.loads(draft.to_json())
        cite = report["evaluation"]["statements"][0]["citations"][0]
        assert cite["start"] == len(PRIVATE) + 2
        assert cite["end"] == len(PRIVATE) + 2 + len(CLAIM)
        assert cite["segment_ref"] == ref(10) and cite["registered_span_ref"] == ref(20)
        assert cite["revision"] == 1
        assert all(
            PRIVATE not in item.premise for item in verifier.requests[0].statements
        )
        receipt = receipt_for(draft)
        authority = Authority(receipt)
        result = finalize(session, draft, authority=authority)
        assert result.text == draft.text and result.data_json == draft.data_json
        assert authority.calls and len(extractor.requests) == 1
        assert len(verifier.requests) == 2
        approved = json.loads(result.to_json())
        assert approved["state"] == "clinician_reviewed"
        assert approved["clinician_review_required"] is False
        for public in (
            draft.to_json(),
            result.to_json(),
            repr(draft),
            repr(result),
            repr(session),
        ):
            assert PRIVATE not in public and CLAIM not in public
        assert len(screen.calls) == 4  # source and generated text at both gates


def test_default_review_packet_is_not_an_approval_and_authority_can_revoke():
    session, _, _, _ = fixture()
    draft = session.compose(ref(80))
    assert draft.review_packet.review_receipt is None
    receipt = receipt_for(draft)
    authority = Authority(receipt)
    authority.status = ReviewAuthorityStatus.REVOKED
    with pytest.raises(NoteEvidenceError, match="finalization_blocked"):
        finalize(session, draft, authority=authority)
    authority.status = ReviewAuthorityStatus.CURRENT
    assert finalize(session, draft, authority=authority).text == draft.text
    session.close()


@pytest.mark.parametrize("change", ["revoked", "mismatched", "unavailable"])
def test_late_source_lookup_cannot_reuse_earlier_clinician_authorization(change):
    session, _, _, _ = fixture()
    draft = session.compose(ref(80))
    receipt = receipt_for(draft)
    authority, source = Authority(receipt), CurrentSource(draft.review_packet)

    def update_review(_):
        if not authority.calls:
            return
        if change == "unavailable":
            authority.callback = lambda: (_ for _ in ()).throw(RuntimeError(PRIVATE))
        else:
            authority.status = ReviewAuthorityStatus(change)

    source.callback = update_review
    with pytest.raises(NoteEvidenceError, match="finalization_blocked") as caught:
        finalize(session, draft, authority=authority, source=source)
    assert caught.value.__context__ is None and PRIVATE not in str(caught.value)
    session.close()


@pytest.mark.parametrize(
    "fault", ["missing", "mismatched", "expired", "unavailable", "source_changed"]
)
def test_current_custody_and_exact_review_binding_are_mandatory(fault):
    session, _, _, _ = fixture()
    draft = session.compose(ref(80))
    receipt = receipt_for(draft)
    authority, source = Authority(receipt), CurrentSource(draft.review_packet)
    if fault == "missing":
        receipt = None
    elif fault == "mismatched":
        receipt = replace(receipt, evidence_digest="sha256:" + "0" * 64)
    elif fault == "expired":
        receipt = replace(receipt, expires_at=149)
    elif fault == "unavailable":
        authority.callback = lambda: (_ for _ in ()).throw(RuntimeError(PRIVATE))
    else:
        source.changed = True
    with pytest.raises(NoteEvidenceError) as caught:
        session.finalize(
            receipt, current_source=source, authority=authority, clock=lambda: 150
        )
    assert caught.value.__context__ is None and PRIVATE not in str(caught.value)
    session.close()


def test_uncited_statement_stays_visible_but_cannot_finalize():
    session, _, extractor, verifier = fixture()

    def uncited(proposal):
        proposal = mutate_statement(proposal, citations=())
        return replace(proposal, steps=(replace(proposal.steps[0], input_refs=()),))

    extractor.transform = uncited
    draft = session.compose(ref(80))
    assert "uncited_statement" in reasons(draft)
    assert CLAIM in draft.text and verifier.requests[0].statements == ()
    with pytest.raises(NoteEvidenceError):
        finalize(session, draft)
    session.close()


@pytest.mark.parametrize(
    "verdict,expected",
    [
        (SupportVerdict.CONTRADICTED, "contradicted_statement"),
        (SupportVerdict.NEUTRAL, "unsupported_statement"),
        (SupportVerdict.ABSTAINED, "unsupported_statement"),
    ],
)
def test_evidence_verification_cannot_be_overridden_by_human_receipt(verdict, expected):
    session, _, _, verifier = fixture()
    verifier.verdict = verdict
    draft = session.compose(ref(80))
    assert expected in reasons(draft)
    with pytest.raises(NoteEvidenceError):
        finalize(session, draft)
    session.close()


def test_low_support_and_unresolved_uncertainty_block_finalization():
    session, _, extractor, verifier = fixture()
    verifier.confidence = 300_000
    extractor.transform = lambda proposal: mutate_statement(
        proposal, uncertainty=NoteUncertainty.UNRESOLVED
    )
    draft = session.compose(ref(80))
    assert set(reasons(draft)) == {"insufficient_support", "unresolved_uncertainty"}
    with pytest.raises(NoteEvidenceError):
        finalize(session, draft)
    session.close()


@pytest.mark.parametrize("fault", ["scope", "uncertainty"])
def test_missing_claim_scope_and_unpreserved_uncertainty_are_explicit(fault):
    session, _, extractor, _ = fixture()
    values = (
        {"concept_ref": None}
        if fault == "scope"
        else {"assertion": NoteAssertion.UNCERTAIN}
    )
    extractor.transform = lambda proposal: mutate_statement(proposal, **values)
    draft = session.compose(ref(80))
    assert (
        "unresolved_claim_scope" if fault == "scope" else "uncertainty_not_preserved"
    ) in reasons(draft)
    session.close()


def test_qualified_uncertainty_is_retained_in_a_reviewed_note():
    session, _, extractor, _ = fixture(text="The patient may have cough.")
    extractor.transform = lambda proposal: mutate_statement(
        proposal,
        assertion=NoteAssertion.UNCERTAIN,
        uncertainty=NoteUncertainty.QUALIFIED,
    )
    draft = session.compose(ref(80))
    assert not reasons(draft)
    result = finalize(session, draft)
    assert "may" in result.text
    assert (
        json.loads(result.to_json())["evaluation"]["statements"][0]["uncertainty"]
        == "qualified"
    )
    session.close()


@pytest.mark.parametrize("conflict", ["assertion", "value"])
def test_builtin_cross_statement_conflicts_survive_favorable_verifier(conflict):
    session, _, extractor, _ = fixture(count=2)

    def transform(proposal):
        if conflict == "assertion":
            return mutate_statement(proposal, assertion=NoteAssertion.NEGATED)
        section = proposal.sections[0]
        statements = tuple(
            replace(item, value_digest=str(index + 1) * 64)
            for index, item in enumerate(section.statements)
        )
        return replace(proposal, sections=(replace(section, statements=statements),))

    extractor.transform = transform
    draft = session.compose(ref(80))
    assert all("conflicting_statements" in reasons(draft, i) for i in (0, 1))
    with pytest.raises(NoteEvidenceError):
        finalize(session, draft)
    session.close()


def test_different_experiencer_or_time_is_not_silently_collapsed_into_same_fact():
    session, _, extractor, _ = fixture(count=2)
    extractor.transform = lambda proposal: mutate_statement(
        proposal, assertion=NoteAssertion.NEGATED, subject_ref=ref(999)
    )
    draft = session.compose(ref(80))
    assert not reasons(draft, 0) and not reasons(draft, 1)
    assert finalize(session, draft).text == draft.text
    session.close()


@pytest.mark.parametrize("fault", ["redacted", "outside"])
def test_citations_cannot_treat_redaction_placeholders_or_invalid_offsets_as_evidence(
    fault,
):
    session, _, extractor, verifier = fixture()
    end = 10 if fault == "redacted" else 65_536
    extractor.transform = lambda proposal: mutate_statement(
        proposal, citations=(NoteCitation(ref(30), 0, end),)
    )
    draft = session.compose(ref(80))
    assert (
        "redacted_citation" if fault == "redacted" else "invalid_citation"
    ) in reasons(draft)
    assert verifier.requests[0].statements == ()
    with pytest.raises(NoteEvidenceError):
        finalize(session, draft)
    session.close()


def test_observation_clipped_inside_identifier_cannot_reveal_a_partial_name():
    session, _, extractor, _ = fixture(offsets=(4, len(PRIVATE) + 2 + len(CLAIM)))
    draft = session.compose(ref(80))
    text = extractor.requests[0].evidence[0].text
    assert text.startswith("[REDACTED]") and PRIVATE[4:] not in text
    cite = json.loads(draft.to_json())["evaluation"]["statements"][0]["citations"][0]
    assert cite["start"] == len(PRIVATE) + 2
    session.close()


def test_generated_identifiers_are_screened_before_the_draft_is_returned():
    session, _, extractor, verifier = fixture()
    extractor.transform = lambda proposal: mutate_statement(
        proposal, text=PRIVATE + ". " + CLAIM
    )
    draft = session.compose(ref(80))
    assert PRIVATE not in draft.text + draft.data_json + draft.to_json()
    assert "[REDACTED]" in draft.text and "generated_phi_redacted" in reasons(draft)
    assert PRIVATE not in verifier.requests[0].statements[0].hypothesis
    with pytest.raises(NoteEvidenceError):
        finalize(session, draft)
    session.close()


@pytest.mark.parametrize("fault", ["provisional", "roles", "phi_unresolved"])
def test_ineligible_source_never_reaches_the_note_extractor(fault):
    session, screen, extractor, _ = fixture(finalized=fault != "provisional")
    if fault == "roles":
        session._reviews = (replace(session._reviews[0], reviewer_confirmed=False),)
    elif fault == "phi_unresolved":
        screen.resolution = PhiResolution.UNRESOLVED
    with pytest.raises(NoteEvidenceError):
        session.compose(ref(80))
    assert extractor.requests == [] and session._closed


@pytest.mark.parametrize("stage", ["screen", "extract", "verify"])
def test_context_mutation_during_callbacks_discards_the_whole_draft(stage):
    session, screen, extractor, verifier = fixture()
    callback = lambda: session._aliases.close()
    {"screen": screen, "extract": extractor, "verify": verifier}[
        stage
    ].callback = callback
    with pytest.raises(NoteEvidenceError):
        session.compose(ref(80))
    assert session._closed


@pytest.mark.parametrize(
    "mutation",
    ["source_text", "aliases", "roles", "policy", "rescreen", "verification"],
)
def test_finalization_recomputes_current_context_and_verification(mutation):
    session, screen, extractor, verifier = fixture()
    draft = session.compose(ref(80))
    if mutation == "source_text":
        original = session._sources[0]
        session._sources = (
            replace(original, text=original.text.replace("cough", "fever")),
        )
    elif mutation == "aliases":
        session._aliases.apply_revision(1, (ref(2),))
    elif mutation == "roles":
        session._reviews = (replace(session._reviews[0], role=ClinicalRole.CLINICIAN),)
    elif mutation == "policy":
        extractor.declaration = replace(EXTRACTOR, policy_digest="0" * 64)
    elif mutation == "rescreen":
        screen.extra = "cough"
    else:
        verifier.verdict = SupportVerdict.CONTRADICTED
    with pytest.raises(NoteEvidenceError):
        finalize(session, draft)
    session.close()


@pytest.mark.parametrize("stage", ["authority", "last_source_lookup"])
def test_late_custody_callbacks_cannot_change_roles_and_release_old_note(stage):
    session, _, _, _ = fixture()
    draft = session.compose(ref(80))
    receipt = receipt_for(draft)
    authority, source = Authority(receipt), CurrentSource(draft.review_packet)
    if stage == "authority":
        authority.callback = session._aliases.close
    else:
        source.callback = lambda count: session._aliases.close() if count == 2 else None
    with pytest.raises(NoteEvidenceError):
        finalize(session, draft, authority=authority, source=source)
    session.close()


def test_expiry_during_late_source_lookup_blocks_export():
    session, _, _, _ = fixture()
    draft = session.compose(ref(80))
    now = [150]
    source = CurrentSource(draft.review_packet)
    source.callback = lambda count: now.__setitem__(0, 201) if count == 2 else None
    with pytest.raises(NoteEvidenceError):
        finalize(session, draft, source=source, clock=lambda: now[0])
    session.close()


@pytest.mark.parametrize(
    "fault",
    [
        "stale_extract",
        "wrong_step",
        "missing_verification",
        "stale_verification",
        "duplicate_statement",
    ],
)
def test_invalid_provider_contracts_fail_without_releasing_partial_drafts(fault):
    session, _, extractor, verifier = fixture(count=2)
    if fault == "stale_extract":
        extractor.transform = lambda proposal: replace(
            proposal, request_digest="0" * 64
        )
    elif fault == "wrong_step":
        extractor.transform = lambda proposal: replace(
            proposal,
            steps=(
                replace(proposal.steps[0], artifact_digest="0" * 64),
                proposal.steps[1],
            ),
        )
    elif fault == "missing_verification":
        verifier.transform = lambda result: replace(
            result, decisions=result.decisions[:1]
        )
    elif fault == "stale_verification":
        verifier.transform = lambda result: replace(result, request_digest="0" * 64)
    else:
        extractor.transform = lambda proposal: mutate_statement(
            proposal, statement_ref=proposal.sections[0].statements[1].statement_ref
        )
    with pytest.raises(NoteEvidenceError) as caught:
        session.compose(ref(80))
    assert caught.value.__context__ is None
    assert session._draft is None and session._closed


@pytest.mark.parametrize("stage", ["screen", "extract", "verify"])
def test_provider_exception_text_is_not_exposed_in_error_context(stage):
    session, screen, extractor, verifier = fixture()

    def fail():
        raise RuntimeError(PRIVATE)

    {"screen": screen, "extract": extractor, "verify": verifier}[stage].callback = fail
    with pytest.raises(NoteEvidenceError) as caught:
        session.compose(ref(80))
    assert caught.value.__context__ is None and caught.value.__cause__ is None
    assert PRIVATE not in str(caught.value)


def test_resource_bounds_and_closed_session_cannot_be_bypassed():
    session, _, _, _ = fixture(policy=NotePolicy(max_note_chars=10))
    with pytest.raises(NoteEvidenceError):
        session.compose(ref(80))
    with pytest.raises(NoteEvidenceError):
        session.compose(ref(81))
    with pytest.raises(NoteEvidenceError):
        fixture(count=2, policy=NotePolicy(max_sources=1))
    with pytest.raises(NoteEvidenceError):
        NoteProviderDeclaration(
            artifact_digest="a" * 64, policy_digest="b" * 64, requires_network=True
        )
    with pytest.raises(NoteEvidenceError):
        NotePolicy(min_support_ppm=True)


def test_actual_local_nli_adapter_detects_negation_and_checks_identical_evidence():
    verifier = NliNoteVerifier(HeuristicNLIBackend(), VERIFIER)
    session, _, extractor, _ = fixture(text=CLAIM, verifier=verifier)
    draft = session.compose(ref(80))
    assert not reasons(draft)
    assert finalize(session, draft).text == draft.text
    session.close()
    session, _, extractor, _ = fixture(
        text="The patient denies cough.", verifier=verifier
    )
    extractor.transform = lambda proposal: mutate_statement(proposal, text=CLAIM)
    draft = session.compose(ref(80))
    assert "contradicted_statement" in reasons(draft)
    with pytest.raises(NoteEvidenceError):
        finalize(session, draft)
    session.close()


def test_offline_deterministic_reports_and_no_automatic_record_writes(monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("network call")

    monkeypatch.setattr(socket, "socket", no_network)
    outputs = []
    for _ in range(2):
        session, _, _, _ = fixture()
        with session:
            draft = session.compose(ref(80))
            outputs.append(finalize(session, draft).to_json())
            assert PRIVATE not in draft.data_json
    assert outputs[0] == outputs[1]


def test_role_change_during_screening_blocks_before_note_model_runs():
    session, screen, extractor, verifier = fixture()
    screen.callback = session._aliases.close
    with pytest.raises(NoteEvidenceError):
        session.compose(ref(80))
    assert extractor.requests == [] and verifier.requests == []


def test_all_context_sources_must_be_finalized_even_if_not_directly_cited():
    session, _, extractor, _ = fixture()
    text = "Synthetic extra context"
    ledger = RevisionLedger(ref(1), ref(99), len(text))
    session._sources += (TranscriptSource(ledger=ledger, text=text),)
    context = assess_speaker_roles(
        session._aliases, session._sources, session._observations
    )
    session._reviews = tuple(
        replace(item, context_digest=context.context_digest)
        for item in session._reviews
    )
    with pytest.raises(NoteEvidenceError):
        session.compose(ref(80))
    assert extractor.requests == []


def test_unicode_source_offsets_remain_original_code_points_after_redaction():
    text = f"préfixe😀 {PRIVATE}。 {CLAIM}"
    session, _, _, _ = fixture(text=text)
    draft = session.compose(ref(80))
    cite = json.loads(draft.to_json())["evaluation"]["statements"][0]["citations"][0]
    assert cite["start"] == text.index(CLAIM)
    assert text[cite["start"] : cite["end"]] == CLAIM
    assert finalize(session, draft).text == draft.text
    session.close()


def test_overlapping_citations_are_rejected_before_verification():
    session, _, extractor, verifier = fixture(text=CLAIM)
    extractor.transform = lambda proposal: mutate_statement(
        proposal,
        citations=(
            NoteCitation(ref(30), 0, len(CLAIM)),
            NoteCitation(ref(30), 1, len(CLAIM)),
        ),
    )
    with pytest.raises(NoteEvidenceError):
        session.compose(ref(80))
    assert verifier.requests == []


def test_verifier_receives_reviewed_caregiver_attribution_with_actual_cited_text():
    session, _, _, verifier = fixture()
    session._reviews = (replace(session._reviews[0], role=ClinicalRole.CAREGIVER),)
    draft = session.compose(ref(80))
    statement = verifier.requests[0].statements[0]
    assert statement.premise == "speaker_0001 [caregiver]: " + CLAIM
    assert PRIVATE not in statement.premise
    cite = json.loads(draft.to_json())["evaluation"]["statements"][0]["citations"][0]
    assert cite["role"] == "caregiver"
    session.close()
