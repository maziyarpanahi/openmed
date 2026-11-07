# Reviewed speaker roles for ambient extraction

`openmed.multimodal.streaming.ambient.speaker_roles` keeps clinical role hypotheses separate from acoustic speaker aliases and checks attribution before role-sensitive extraction and note finalization. Patient, clinician, caregiver and interpreter guesses remain unverified regardless of confidence. Every required speech span needs a current, explicit reviewer decision bound to its source text, transcript lineage and diarization context before it reaches the extractor.

The module is a deterministic local Python gate. It does not infer roles from voices, alias numbers, names, cluster order or confidence; it does not load an ASR model, diarizer or note model. It performs no file writes or mandatory network calls. The application supplies a trusted local extractor, authenticates reviewers and retains approval evidence. No Swift review UI or clinical qualification is introduced.

## Separate aliases, hypotheses and reviewed roles

The existing [speaker alias registry](speaker-aliases.md) manages acoustic-cluster lineage. An alias such as `speaker_0001` is not a patient identifier or clinical role. This module uses its current session, active clusters and diarization revision without changing the registry.

`TranscriptSource` pairs the application's protected current text with its original [revision ledger](transcript-revisions.md). The ledger tracks registered character/token/time citations but does not store text. The gate checks current text length and includes the exact text digest in the review binding, so replacing text with a different string of the same length still invalidates old review.

`RoleObservation` identifies one primary registered speech span, its acoustic cluster, the exact diarization revision and zero to five distinct `RoleHypothesis` objects. Each hypothesis contains a `ClinicalRole`, one to eight registered supporting transcript citations and an optional confidence in parts per million. Supporting evidence can come from another supplied segment in the same session. Confidence is uncalibrated metadata, not proof of role correctness.

| Hypotheses | Report state | Admission |
| --- | --- | --- |
| None | `unknown` | Requires explicit reviewer correction to a known role. |
| One | `single_hypothesis` | Still requires current explicit review. |
| Two or more | `multiple_roles` | Requires reviewer selection or correction; no best-score winner. |

Review applies per speech span. A speaker can have different reviewed roles in different contexts, such as caregiver and interpreter, without assigning a permanent role to every statement from that alias. Primary role-sensitive spans must not overlap within a segment; duplicate coverage cannot silently produce two conflicting attributions.

## Current evidence and exact-content review

`assess_speaker_roles` resolves each original registered citation through its ledger and reports current offsets, token references and stream-relative timing. It separately reports hard blockers for a closed session, inactive cluster, changed diarization revision or invalidated citation. Unfinalized transcript evidence is identified explicitly. The report contains no transcript text and is never an approval.

`RoleAttributionReview` binds an observation ID, the complete current context digest, the selected or reviewer-corrected role, a review-event ID, reviewer reference and digest of externally retained approval evidence. `reviewer_confirmed` defaults to false. A reviewer can choose a lower-confidence hypothesis or explicitly correct an unknown/unlisted role. The original hypotheses remain intact in the audit report.

The context digest binds the alias snapshot and lineage, all supplied source text and ledger metadata, and the complete ordered observation set. Changing a hypothesis, confidence, source string, citation, timestamp, finality, cluster lineage or diarization revision invalidates prior review. Source-list ordering does not alter the binding. Existing alias-level `RoleReview` records are not evidence that a reviewer examined these current span-level hypotheses; they cannot bypass this gate.

Applications must authenticate real reviewers and provide truthful cluster-to-span assignments. Digests detect stale or mismatched inputs; they are not signatures or an authentication boundary against a caller that fabricates confirmed decisions. The gate cannot establish from text metadata which voice actually spoke an utterance.

## Guarded extraction and finalization

`extract_with_reviewed_roles` freshly assesses the complete input and requires exactly one confirmed current decision for every observation. Missing, extra, duplicate, stale or unconfirmed decisions block the whole call before the extractor runs. Current finalized evidence is required by default, including supporting citations. Review cannot override invalidated evidence or inactive speaker lineage.

The extractor receives immutable `ReviewedRoleSpan` objects containing the exact protected source slice, reviewed role, session-local alias, original/current citation and decision provenance. A role decision does not rewrite or normalize the transcript. Confidence never selects the role passed to the extractor.

For streaming previews, explicitly set `allow_provisional=True` to permit extraction from current unfinalized transcripts after all roles have been reviewed. The returned `RoleBoundDraft.provisional` flag and report's `provisional_sources` field remain true. This option permits a provisional draft only; it cannot relax note finalization. Once ASR finalizes or revises the source, obtain review against that new context and rerun extraction before finalization. An old provisional draft cannot simply be relabeled final.

The gate checks the complete context again after the callback. A speaker revision or other mutable attribution change during extraction prevents returning its output. The application must serialize access to the alias registry and ledgers. The callback is trusted local code: the gate cannot undo its side effects or sandbox network access. Applications must keep protected workflows local. Callback failures are replaced with a fixed exception without the original sensitive exception context.

`finalize_role_bound_note` freshly validates all source evidence and decisions again, requires finalized transcripts, and checks the draft's original text digest, context digest and review binding. Changed note text, newly selected roles, different source inventories or stale diarization cannot finalize an old extraction. Even fresh role decisions for a revised context require a new extraction. The function rebuilds audit metadata from current inputs instead of trusting the draft's stored report JSON.

`RoleCheckedNote.data_json` is protected output containing the note and complete role/source decision lineage. Its state is `role_attribution_checked`, and the audit still says `clinical_review_required: true`. This gate does not establish note factuality, clinical correctness, patient identity, consent, de-identification or authorization to act. It must not trigger clinical decisions automatically. Source-inventory completeness and the extractor's interpretation remain application responsibilities; unresolved required observations must not be omitted to make the gate pass.

Closing the alias registry destroys its attribution state and blocks subsequent finalization. Coordinate that lifecycle with capture, retention and review policies; this module does not restore destroyed evidence or authorize retaining audio.

## Synthetic example

All text, identifiers and confidence values below are synthetic API controls. A real application obtains the explicit role decision from its authenticated reviewer.

```python
import json

from openmed.multimodal.streaming.ambient.speaker_roles import (
    RoleAttributionReview, RoleHypothesis, RoleObservation, TranscriptSource,
    assess_speaker_roles, extract_with_reviewed_roles, finalize_role_bound_note,
)
from openmed.multimodal.streaming.speech.speaker_aliases import (
    ClinicalRole, SpeakerAliases,
)
from openmed.multimodal.streaming.transcript.revision_ledger import (
    RevisionLedger, TranscriptToken,
)

text = "Synthetic source statement"
aliases = SpeakerAliases("1" * 32)
aliases.apply_revision(0, ("2" * 32,))
ledger = RevisionLedger("1" * 32, "3" * 32, len(text), (
    TranscriptToken(token_ref="4" * 32, start=0, end=len(text), start_ms=0, end_ms=800),
))
citation = ledger.capture_span("5" * 32, 0, len(text))
ledger.finalize(expected_revision=0)
sources = (TranscriptSource(ledger=ledger, text=text),)
observations = (RoleObservation(
    observation_id="6" * 32, cluster_ref="2" * 32, diarization_revision=0,
    citation=citation, hypotheses=(
        RoleHypothesis(role=ClinicalRole.PATIENT, evidence=(citation,), confidence_ppm=600_000),
        RoleHypothesis(role=ClinicalRole.CLINICIAN, evidence=(citation,), confidence_ppm=990_000),
    ),
),)
report = assess_speaker_roles(aliases, sources, observations)
reviews = (RoleAttributionReview(
    observation_id="6" * 32, context_digest=report.context_digest,
    role=ClinicalRole.PATIENT, review_id="7" * 32, reviewer_id="8" * 32,
    decision_digest="a" * 64, reviewer_confirmed=True,
),)

def synthetic_extractor(spans):
    assert spans[0].role is ClinicalRole.PATIENT  # Explicit lower-score choice.
    assert spans[0].text == text
    return "Synthetic note from reviewed source attribution"

draft = extract_with_reviewed_roles(aliases, sources, observations, reviews, synthetic_extractor)
note = finalize_role_bound_note(draft, aliases, sources, observations, reviews)
assert not draft.provisional
assert json.loads(note.data_json)["state"] == "role_attribution_checked"
assert "Synthetic source statement" not in note.to_json()
assert "Synthetic note" not in note.to_json()
```

## Privacy, bounds and verification

Protected transcript slices and notes are excluded from representations and report methods. `RoleReport.to_json()`, `RoleBoundDraft.to_json()` and `RoleCheckedNote.to_json()` contain only attribution metadata, confidence, offsets, timing, opaque references and digests. Fixed `SpeakerRoleError` messages do not include patient values or reviewer names. Generic dataclass serialization is not redaction. These metadata remain linkable and require application access and retention controls; hashes do not de-identify records.

A call accepts one to 32 transcript sources, each with one to 65,536 valid UTF-8 code points and at most 262,144 total code points. There are one to 128 observations and at most 1024 primary/supporting citation links. Each observation permits at most five distinct role hypotheses and each hypothesis one to eight evidence references. Extractor output is bounded to 65,536 nonblank valid UTF-8 code points. Existing alias-registry and transcript-ledger limits also apply. Bounds fail closed without partial note data.

```bash
.venv/bin/python -m pytest tests/unit/multimodal/streaming/ambient/test_speaker_roles.py -q
```

Synthetic controls cover unknown/multiple/high-confidence hypotheses, explicit lower-score and corrected roles, per-span attribution, complete review admission, alias merge/split/relabel changes, transcript/timing/finality changes, provisional drafts, stale finalization, callback failures and context changes, privacy and bounded work. They establish software behavior only, not role-inference accuracy or clinical qualification.
