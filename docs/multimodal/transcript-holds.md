# Critical transcript token holds

`openmed.multimodal.transcript_holds` and OpenMedKit's `TranscriptHoldGate`
classify fixed, finalized token evidence and place confirmation holds on every
statement citing a risky token. Processing is deterministic and offline. A
reviewed-export permit always carries a non-diagnostic notice and requires
explicit clinician confirmation; this aid never changes a dose or triggers an
order.

## Policy v1

| Class | Minimum confidence |
| --- | --- |
| Number, unit, dose, negation, laterality, medication, hypo/hyper modifier | 0.95 |
| Uncertainty cue | 0.90 |

The boundary is inclusive: confidence equal to the threshold passes. Missing
confidence holds a critical token. Any alternative disagreeing after case and
sentence-ending punctuation normalization holds it regardless of confidence.
Classification considers both the selected text and **all supplied alternatives**,
so an empty selected token with a `no` alternative represents a possible dropped
negation. An empty alternative to `no` also creates a hold. Leading decimal
points are preserved: `.5` and `5` disagree.

Local rules recognize English number words, numeric amounts, common units,
compact dose-shaped tokens such as `15mg`, negation/uncertainty/laterality cues,
and selected hypo/hyper modifiers. Adjacent number/unit tokens with consecutive
IDs in the same segment receive the dose class. Medication tokens use an exact,
caller-owned local lexicon; no formulary or dosing guidance is bundled.
Unknown words create no hold solely due to low confidence. This bounded English
rule set is not a complete clinical lexicon, multilingual safety classifier,
ASR qualification, calibration result or clinical validation.

A dropped cue absent from **all** supplied evidence cannot be inferred here.
Callers must preserve omission alternatives (including empty token markers)
and supply complete token dependencies, including negation tokens, to every
statement. IDs are opaque non-negative integers; they are never patient IDs.

## Python example

```python
from openmed.multimodal.transcript_holds import (
    DraftTokenCitation,
    HoldConfirmation,
    TokenIdentity,
    TranscriptHoldGate,
    TranscriptToken,
)

ref = TokenIdentity(segment_id=0, token_id=0)
gate = TranscriptHoldGate(
    [TranscriptToken(ref, "fifteen", 0.7, ("fifty",))],
    [DraftTokenCitation(0, (ref,)), DraftTokenCitation(1, (ref,))],
    revision=1,
)
assert gate.statement_holds(0) and gate.statement_holds(1)
# export_reviewed(..., reviewer_confirmed=True) still refuses unresolved holds.
receipt = HoldConfirmation(ref, gate.evidence_digest, reviewer_id=7, confirmed=True)
gate.resolve(receipt, authorize=lambda reviewer: reviewer == 7)
permit = gate.export_reviewed(0, reviewer_confirmed=True)
assert permit["revision"] == 1
```

`resolve` is the hold-resolution entry point for a separate correction workflow.
It accepts only an explicit clinician confirmation, a current snapshot digest
and an authorized numeric reviewer identity. Authorization is injected and
local; callback exceptions fail closed without copying their messages. Original
hold records remain immutable after confirmation. Confirming a token clears
its hold on every dependent statement; other token holds still block export.
This allows clinician acceptance of the current evidence after independently
reviewing its basis. Replacement text requires a new evidence revision and a
new gate, with fresh confirmation of any remaining holds.

The digest binds the revision, private token evidence, classes, medication
lexicon, dose scores and citation graph. Old receipts fail on changed evidence.
Digests are platform-local snapshot identities, not a cross-platform correction
ledger wire contract. The future correction workflow (#3682) owns history,
review invalidation and propagation across revisions. This module owns neither
ASR (#2812/#2814) nor note assembly (#2816). Export permits contain no note text;
callers must enforce the permit against the **current** snapshot on their actual
export path. Retaining an older gate does not make its permit current.

## Dosing integration and Swift parity

Python `check_transcript_doses` accepts private caller-extracted `TranscriptDose`
inputs and caller-owned local reference ranges. It invokes
`openmed.clinical.dosing_check.check_dose_ranges` unchanged and emits only
`TokenDoseEvidence` identities and controlled statuses. Supply **all** tokens
supporting each dose, including its medication, amount and unit. In-range gives
score 0, not-checked (missing/incompatible references) gives 0.5, and an
out-of-range flag gives 1. Both nonzero scores hold all those tokens. Scores are
controlled status encodings, not probabilities or measured clinical risk.
Checker failures yield only `dose_check_failed`; no reference values are logged.

OpenMedKit accepts the same value-free dose statuses from a caller's offline
dosing adapter. It does not duplicate or replace the Python dosing checker.
`FixedTranscriptToken`, `DraftTokenCitation`, `TokenDoseEvidence`,
`TranscriptHoldConfirmation`, `resolve` and `exportReviewed` provide the matching
classification, propagation and review gate. A shared synthetic fixture checks
classes, thresholds, omissions and alternative disagreement on both platforms.
No provider, model assets, network transport or cloud fallback is involved.

## Privacy boundary

Hold serialization contains exactly segment/token identities, controlled class
names, confidence/threshold, disagreement/dose scores and numeric policy version.
It contains no selected token, alternative, dose, unit value, medication name,
note text, source payload or private path. Private input representations hide
text and alternatives; diagnostics use controlled codes. These are confirmation
requests, never diagnoses, recommended doses, or evidence of clinical validation.
