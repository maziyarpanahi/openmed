# Cited ambient note drafts

`openmed.multimodal.streaming.ambient.note_evidence` assembles protected draft sections from finalized transcripts, actual local PHI-screening results and reviewed speaker roles. Each statement carries citations, an extraction step, explicit uncertainty and a current evidence-verification result. Uncited, contradicted, unsupported or conflicting statements remain visible in the draft but block finalization. Finalization additionally requires current clinician review resolved through trusted local custody.

The module returns a local Python result. It does not write a medical record, select an EHR destination, make a mandatory network call or grant authority to act on a clinical statement. The application supplies and qualifies local screening, extraction and verification backends, authenticates clinicians and controls their review records. No model, clinical validation, Swift capture UI or compliance certification is bundled.

## The generation boundary

Create an `AmbientNoteSession` with the original `TranscriptSource` objects, `SpeakerAliases`, complete `RoleObservation` inventory and current `RoleAttributionReview` decisions from the [speaker-role gate](ambient-speaker-roles.md). Every supplied source must already be finalized, including context that is not directly cited. The original source ledgers remain authoritative; the note builder never marks an unfinished source as finalized on the caller's behalf.

The session invokes the actual [PHI withholding buffer](phi-withholding-buffer.md) over complete current source text. Its temporary screening ledger is separate from original revision history. Only actual released text is projected into `ScreenedRoleEvidence`, and only this screened evidence reaches `LocalNoteExtractor.extract`. Arbitrary caller-constructed release objects cannot bypass screening. Speaker roles are freshly checked after screening before invoking the note model; the outer role gate also checks the complete source and role context after provider calls.

Projection preserves the mapping from screened observation offsets to original current transcript offsets. If a role observation starts inside an identifier, it receives the complete redaction placeholder rather than a surviving fragment. `NoteCitation` offsets are Unicode code points within one screened observation. A citation overlapping a placeholder cannot serve as factual evidence. Valid citations map back to the original segment, registered span, revision and offsets, retaining the reviewed alias and role. Overlapping or duplicate citations within one statement are refused.

The extractor's returned statement text is screened again before being returned in a draft or passed to the evidence verifier. Detected generated identifiers are replaced by the actual screen output, and the affected statement gets a `generated_phi_redacted` hold. An unresolved screen refuses the whole composition. Screening is an operational gate, not proof that a detector has perfect recall; production backends require their own privacy qualification.

## Sections, statements and extraction lineage

`NoteProposal` binds its complete response to the exact screened request digest. It contains ordered `NoteSection` objects with controlled headings: history, examination, medications, assessment, plan or other. A `NoteStatement` contains protected text, citations, an extraction-step reference, assertion and uncertainty axes, and opaque semantic scope references.

Each `NoteExtractionStep` identifies extraction, paraphrasing or synthesis, the pinned extractor artifact and the complete set of cited input observations. Unknown input identities, stale request digests, duplicate statement identities, missing steps, unused steps, changed model declarations or mismatched step inputs refuse the response. The implementation preserves section and statement order; it does not silently discard an unsupported statement to make a draft pass.

The default limits are 16 source segments, 65,536 total source characters, 64 role observations, 64 statements, six sections, eight non-overlapping citations per statement, 4,096 characters per statement and 32,768 rendered note characters. Source count can increase to 32 and note length to 65,536; the underlying role gate also bounds the complete serialized proposal to 65,536 characters, including citation and step metadata. Screening operations have a configurable deadline up to 60 seconds. Limits are checked before provider use or before accepting their output.

`NoteProviderDeclaration` pins a model or algorithm digest and policy/decoding digest. It requires local execution and deterministic behavior. These are application/provider contracts; a declaration does not sandbox arbitrary Python code or establish model quality. The module selects no default generation model and has no cloud fallback.

## Evidence support and contradictions

The independent `LocalNoteVerifier` receives the actual screened premise assembled from each statement's citations and the screened statement text. Each cited excerpt is prefixed with its current reviewed alias and role, so verification sees speaker attribution as well as the quoted words. Its response must match the exact request and every eligible statement, in order. Contradiction, neutral, abstention and entailment below the configured support threshold all block finalization. The default threshold is 900,000 parts per million. Scores are backend outputs, not automatically calibrated clinical confidence.

`NliNoteVerifier` adapts an explicitly supplied local OpenMed clinical NLI backend. It does not select a registry alias or download a model. The existing `HeuristicNLIBackend` can demonstrate deterministic exact-support and negation behavior with synthetic text; it is not a clinically qualified verifier. Production applications must independently qualify their screening, extraction and verification backends for the languages and tasks they expose.

The builder also reuses OpenMed's deterministic claim-conflict review across statements sharing the same opaque subject, concept and temporal scope. Incompatible assertion states or different declared normalized-value digests in an affirmative group block every affected statement even if per-statement NLI reports support. Different experiencers or times are not collapsed into one fact. Missing scope, unknown assertion, unresolved uncertainty, or uncertain/hypothetical assertions that omit their uncertainty declaration also block finalization.

Semantic grouping, normalized values and clinical interpretation depend on the qualified extractor and verifier. Opaque references are not identity resolution, and these checks do not detect every clinical contradiction. Clinician review remains mandatory. A reviewer cannot override an uncited or contradictory statement with a confirmation flag; correct it and compose a fresh draft with fresh evidence and review.

## Clinician review and local finalization

`AmbientNoteDraft.text` and `data_json` are protected output, including screened note text. `to_json()` returns only value-free audit metadata: controlled section/claim states, opaque references, hashes, offsets, provenance, support decisions and blockers. The generated data remains protected even after screening; do not put it in logs or public artifacts.

The draft supplies a `ReviewedLocalEvidence` packet with no approval. Its source digest binds the exact rendered note, its references identify the statement ranges, and its policy digest binds the complete proposal, source/role context, screening and model declarations, verification results, citations, uncertainty and policy. The application displays the protected draft and its evidence to an authenticated clinician, stores the reviewed version under the packet's source ID, and issues a `LocalReviewReceipt` in its trusted local review store.

`finalize` uses the existing `admit_reviewed_local_evidence` boundary. The `ReviewAuthorityVerifier` must resolve the exact receipt in its own custody, enforce clinician permission and compare the complete evidence binding and current revocation state. `CurrentLocalSource` must return the current reviewed draft's digest. A serialized packet, a caller boolean or an extractor verdict cannot independently approve the note. The synthetic-only legacy evidence packet is not used to mislabel real de-identified sources as synthetic.

Before finalization, the session screens sources and generated statements again, reruns evidence verification and deterministic conflicts, and requires the same evaluation and role/source binding as the draft. Changed screen results, evidence, declarations, roles, confidence or support verdicts require a new draft and review. Source custody and role context are checked after external review callbacks, and expiry is checked again before returning output. Missing, stale, revoked, unavailable or expired review blocks finalization.

A successful `ReviewedAmbientNote` is an assistive, clinician-reviewed local artifact. It remains subject to the application's clinical workflow and authorization; the library never writes it to an EHR. This session owns one immutable draft. Correction ledgers, fixed-draft evaluation, FHIR export and CLI adapters remain separate capabilities. Close the session or use its context manager to release owned references; Python strings and caller/provider copies are not claimed to be securely erased.

## Synthetic runnable example

This example demonstrates the actual screening and local NLI paths. Its synthetic screen recognizes only the fixture token; it is not a production PHI detector. The example intentionally stops at a draft so no clinician authority is fabricated.

```python
from openmed.clinical.nli import HeuristicNLIBackend
from openmed.multimodal.streaming.ambient.note_evidence import (
    AmbientNoteSession, ExtractionOperation, NliNoteVerifier, NoteAssertion,
    NoteCitation, NoteExtractionStep, NoteProposal, NoteProviderDeclaration,
    NoteSection, NoteSectionKind, NoteStatement, NoteUncertainty,
)
from openmed.multimodal.streaming.ambient.speaker_roles import (
    RoleAttributionReview, RoleHypothesis, RoleObservation, TranscriptSource,
    assess_speaker_roles,
)
from openmed.multimodal.streaming.privacy import (
    PhiRedactionSpan, PhiResolution, PhiScreenDeclaration, PhiScreenResult,
    SegmentPhiDecision,
)
from openmed.multimodal.streaming.speech import ClinicalRole, SpeakerAliases
from openmed.multimodal.streaming.transcript import RevisionLedger, TranscriptToken


def ref(number):
    return f"{number:032x}"


class SyntheticScreen:
    declaration = PhiScreenDeclaration(artifact_digest="a" * 64)

    def screen(self, request):
        decisions = []
        for segment in request.segments:
            start = segment.text.find("SYNTHETIC-NAME")
            redactions = () if start < 0 else (
                PhiRedactionSpan(start, start + len("SYNTHETIC-NAME")),
            )
            decisions.append(SegmentPhiDecision(
                segment_ref=segment.segment_ref, revision=segment.revision,
                resolution=PhiResolution.RESOLVED, redactions=redactions,
            ))
        return PhiScreenResult(
            request_digest=request.request_digest, decisions=tuple(decisions),
        )


class SyntheticExtractor:
    declaration = NoteProviderDeclaration(
        artifact_digest="b" * 64, policy_digest="c" * 64,
    )

    def extract(self, request):
        evidence = request.evidence[0]
        text = "The patient reports cough."
        start = evidence.text.index(text)
        statement = NoteStatement(
            statement_ref=ref(20), text=text,
            citations=(NoteCitation(evidence.evidence_ref, start, start + len(text)),),
            step_ref=ref(21), assertion=NoteAssertion.AFFIRMED,
            uncertainty=NoteUncertainty.NONE_DECLARED,
            concept_ref=ref(22), subject_ref=ref(23), temporal_ref=ref(24),
        )
        return NoteProposal(
            request.request_digest,
            (NoteSection(NoteSectionKind.HISTORY, (statement,)),),
            (NoteExtractionStep(
                ref(21), ExtractionOperation.EXTRACT, (evidence.evidence_ref,),
                self.declaration.artifact_digest,
            ),),
        )


text = "SYNTHETIC-NAME. The patient reports cough."
aliases = SpeakerAliases(ref(1))
aliases.apply_revision(0, (ref(2),))
ledger = RevisionLedger(ref(1), ref(3), len(text), (
    TranscriptToken(token_ref=ref(4), start=0, end=len(text), start_ms=0, end_ms=1000),
))
citation = ledger.capture_span(ref(5), 0, len(text))
ledger.finalize(expected_revision=0)
sources = (TranscriptSource(ledger=ledger, text=text),)
observations = (RoleObservation(
    observation_id=ref(6), cluster_ref=ref(2), diarization_revision=0,
    citation=citation,
    hypotheses=(RoleHypothesis(role=ClinicalRole.PATIENT, evidence=(citation,)),),
),)
assessment = assess_speaker_roles(aliases, sources, observations)
role_reviews = (RoleAttributionReview(
    observation_id=ref(6), context_digest=assessment.context_digest,
    role=ClinicalRole.PATIENT, review_id=ref(7), reviewer_id=ref(8),
    decision_digest="d" * 64, reviewer_confirmed=True,
),)
verifier = NliNoteVerifier(HeuristicNLIBackend(), NoteProviderDeclaration(
    artifact_digest="e" * 64, policy_digest="f" * 64,
))
with AmbientNoteSession(
    aliases, sources, observations, role_reviews,
    screener=SyntheticScreen(), extractor=SyntheticExtractor(), verifier=verifier,
) as session:
    draft = session.compose(ref(9))
    assert draft.text == "history\nThe patient reports cough."
    assert "SYNTHETIC-NAME" not in draft.data_json
    assert draft.review_packet.review_receipt is None
    # An authenticated clinician and trusted review store are required before
    # session.finalize(receipt, current_source=store, authority=authority).
```

## Validation

Run `.venv/bin/python -m pytest tests/unit/multimodal/streaming/ambient/test_note_evidence.py -q`. Synthetic tests exercise real withholding and NLI integration, original Unicode offsets after redaction, clipped identifiers, output screening, incomplete and stale model responses, uncertainty and cross-statement conflicts, current source and clinician custody, callback mutation, expiry, resource bounds, private exception sanitization and deterministic offline reports. Model qualification and real clinician authentication remain application responsibilities.
