# Handwriting candidates and transcription review

`openmed.multimodal.vision.handwriting.uncertainty` keeps handwritten recognition hypotheses separate from reviewed transcriptions. It represents alternatives, uncalibrated confidence and original visual evidence, then builds a structured export only after checking an explicit reviewer decision for every current candidate. It performs no OCR, handwriting recognition, image decoding, clinical interpretation, file write or network transfer.

The caller identifies a region as handwritten and supplies its hypotheses. This module does not infer handwriting style or claim recognition accuracy. Printed OCR should remain in its own evidence path. A high recognition score never promotes a handwritten candidate to reviewed or clinically verified text.

## Candidate and review states

`HandwritingCandidate` contains an opaque candidate ID, an original `PIXELS` [evidence node](../fused-evidence-graphs.md), zero to sixteen ordered `TranscriptionAlternative` objects and optional recognition-provenance digest. Each alternative holds protected text, an opaque ID and the existing provider `Confidence` type: a score in `[0, 1]`, or `None` for unknown. Scores are not calibrated probabilities. Alternatives keep their original spelling, punctuation and order.

Every candidate's state is always `unverified`, including a candidate with confidence `1.0`. There is no writable verified flag or implicit best-score selection. Empty alternatives represent an unreadable region, not a fabricated blank value. The candidate cites the original visual region rather than treating hypotheses of different lengths as original printed-OCR spans.

`HandwritingReview` records a reviewer reference, review-event ID, external approval-evidence digest, candidate ID and complete candidate digest. Its `reviewer_confirmed` flag defaults to false. `review_handwriting` requires explicit confirmation and resolves one of four actions:

| Action | Required choice | Result |
| --- | --- | --- |
| `SELECT` | Exact existing alternative ID | `reviewed_transcription`, with the selected provider score retained unchanged. |
| `CORRECT` | Explicit nonempty corrected transcription | `reviewed_transcription`, with unknown confidence. Human review does not manufacture a calibrated score. |
| `REJECT` | No selected or corrected value | `rejected`, without a value; blocked from export. |
| `ABSTAIN` | No selected or corrected value | `abstained`, without a value; blocked from export. |

A reviewer can transcribe an unreadable region through `CORRECT` after inspecting its original evidence. Selecting an unknown alternative, combining selection with correction, or supplying a value for rejection/abstention is rejected. Review produces a separate immutable result; it does not mutate or silently relabel the original candidate.

## Exact-content binding

The candidate digest binds all hypotheses, confidence values, their order, candidate ID, original source digest, visual node ID, page geometry, bounds and recognition provenance. Changing any of them invalidates an earlier review, even if the selected hypothesis's text is unchanged. The review digest additionally binds the chosen alternative or corrected text, action, explicit confirmation and reviewer/approval references.

Applications must authenticate the reviewer and retain the real approval evidence. Opaque references and digests detect accidental stale or mismatched decisions; they do not authenticate an application that fabricates a new confirmed review. Reviewer confirmation concerns transcription only. It does not establish clinical validity, correct field association, medication safety or permission to trigger a clinical decision.

## Guarded structured export

`export_handwriting(candidates, reviews)` accepts original candidates and decisions, then freshly resolves every review. It does not trust previously created `ReviewedHandwriting` objects as authorization. The batch requires exactly one review per candidate, unique candidate and review IDs, current candidate bindings, explicit confirmation and an accepted transcription for every entry. Missing, stale, conflicting, rejected or abstained entries prevent the entire export. No partial JSON is returned.

The protected `HandwritingExport.data_json` contains reviewed values together with `source_kind: "handwriting"`, `reviewed_transcription` state, original visual coordinates, source/candidate digests and reviewer/decision lineage. Candidate order determines export order; review-list ordering does not. JSON encoding keeps values as strings, including quotes and punctuation. No date, number, dose, unit or locale conversion is attempted. Field grounding, schema mapping, FHIR/OMOP conversion and clinical review require their own downstream controls.

`to_report_json()` excludes all recognized and corrected text. It retains the evidence coordinates, selected alternative ID, confidence, state and review references. `HandwritingCandidate.to_json()` and `ReviewedHandwriting.to_json()` are also content-free report methods. `data_json` and the protected in-memory text fields are intended for authorized clinical-data paths, never logs or public reports. Generic dataclass serialization is not a redaction mechanism.

## Synthetic example

The values and digests below are synthetic API controls, not a recommendation or a patient record.

```python
import json

from openmed.multimodal.evidence import EvidenceNode, NodeKind, PageGeometry
from openmed.multimodal.vision.handwriting.uncertainty import (
    HandwritingCandidate, HandwritingReview, HandwritingState, ReviewAction,
    TranscriptionAlternative, export_handwriting,
)
from openmed.multimodal.vision.providers.base import Confidence

candidate = HandwritingCandidate(
    candidate_id="1" * 32,
    evidence=EvidenceNode(
        node_id="2" * 32, kind=NodeKind.PIXELS, source_digest="a" * 64,
        geometry=PageGeometry(page=0, width=1000, height=800),
        region=(10, 20, 100, 60),
    ),
    alternatives=(
        TranscriptionAlternative(
            alternative_id="3" * 32, text="0.5 mg", confidence=Confidence(score=.7),
        ),
        TranscriptionAlternative(
            alternative_id="4" * 32, text="5 mg", confidence=Confidence(score=.95),
        ),
    ),
    recognizer_digest="b" * 64,
)
assert candidate.state is HandwritingState.UNVERIFIED

# Explicit synthetic reviewer choice: the higher-score alternative is not chosen.
decision = HandwritingReview(
    review_id="5" * 32, reviewer_id="6" * 32, decision_digest="c" * 64,
    candidate_id=candidate.candidate_id, candidate_digest=candidate.digest,
    action=ReviewAction.SELECT, alternative_id="3" * 32,
    reviewer_confirmed=True,
)
exported = export_handwriting((candidate,), (decision,))
record = json.loads(exported.data_json)["items"][0]
assert record["value"] == "0.5 mg"
assert record["evidence"]["region"] == [10, 20, 100, 60]
assert record["state"] == "reviewed_transcription"
assert "0.5 mg" not in exported.to_report_json()
assert candidate.state is HandwritingState.UNVERIFIED
```

For a correction, use `ReviewAction.CORRECT`, omit `alternative_id`, and supply the reviewed `corrected_text`. Retain a new decision event for that exact correction. Do not reuse approval evidence for a different value or source revision.

## Privacy, limits and verification

Recognized and corrected text is excluded from object representations and report methods. Malformed declarations and blocked exports raise fixed `HandwritingReviewError` messages without patient values, source paths or reviewer names. Reports use opaque IDs, digests, confidence and source coordinates. These remain linkable metadata and require application access and retention controls; hashes are not de-identification. The module does not log, persist or transmit anything.

Each hypothesis or correction is valid UTF-8 text containing one to 4096 Unicode code points and at least one non-whitespace character. Whitespace and punctuation are preserved. Candidates contain at most sixteen alternatives with unique IDs. Export batches contain one to 128 candidates and at most 262,144 total code points across all hypotheses and corrections. Bounds are deterministic and fail closed. Source geometry uses the existing fused-evidence validation.

```bash
.venv/bin/python -m pytest tests/unit/multimodal/vision/handwriting/test_uncertainty.py -q
```

Synthetic controls verify that high scores do not authorize export, reviewers can select a lower-score hypothesis, corrections retain exact spelling, source coordinates survive export, stale evidence invalidates approval, mixed batches cannot partially export, unknown confidence stays unknown, and diagnostic surfaces omit protected values. This Python representation/export slice does not introduce a Swift review UI, a handwriting recognition model or clinical qualification.
