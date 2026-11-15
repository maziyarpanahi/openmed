# Fixed ambient draft evaluation

`openmed.eval.ambient_drafts` evaluates already annotated drafts against fixed
encounter truth, locally and without a model, transport, cloud fallback or new
dependency. This Python evaluation-harness slice does not assemble notes, generate
dialogues, recruit clinicians or change OpenMedKit runtime behavior.

**Evaluation only; non-diagnostic. Explicit clinician review is required.** Every
report carries this notice and `reviewer_confirmation_required=true`. Reports
never authorize export, clinical action, provider qualification or a release.
Downstream consequential workflows must enforce their own explicit reviewer
confirmation against the current evidence. Synthetic fixtures establish software
behavior, not clinical validation.

## Annotated input contract

Import `AmbientFact`, `AmbientDraft`, `evaluate_ambient_drafts` and
`import_ambient_reviews` from `openmed.eval.ambient_drafts`.

Each `AmbientDraft` contains a random 64-hex `blinded_case_id`, the SHA-256
`text_digest` of the exact reviewed draft, nonempty `truth` and atomic
`statements` tuples. The caller annotates **every** statement, including uncited
and invented ones. Raw text, transcripts, model identities and patient identities
are not retained by these objects. Their representations hide field values.

Each `AmbientFact` has these fields:

| Field | Contract |
| --- | --- |
| `fact_id` | Random 64-hex reference, unique within truth or statements |
| `proposition` | SHA-256 of a caller-defined canonical proposition, excluding speaker, experiencer and negation |
| `section` | `subjective`, `objective`, `assessment`, `plan`, `other` |
| `fact_class` | `allergy`, `diagnosis`, `finding`, `history`, `lab`, `medication`, `other`, `procedure`, `symptom`, `treatment`, `vital` |
| `speaker`, `experiencer` | Independently `patient`, `clinician`, `caregiver` |
| `negated` | Explicit boolean polarity |
| `required` | True for required truth; optional truth may use False; statements use True |
| `citations` | Tuple of truth references on statements; empty on truth |

The speaker identifies who supplied information; the experiencer identifies whose
clinical state it describes. A caregiver speaking about the patient is different
from a patient speaking about a caregiver. The truth section is the required
placement for a fact. Optional facts remain available as supporting evidence.

Matching is exact over caller annotations. A digest is not proof of semantic
support, a secret or a de-identification mechanism. Protect local process memory
and never derive public case/reviewer references from patient or candidate names.
Use random opaque aliases; protect their private mapping separately. Input checks
cannot verify reviewer credentials, actual blinding, annotation completeness or
whether a supplied text digest was calculated from the displayed draft.

## Separate error rates and slices

The evaluator reuses `summary_unsupported_claims` for four-state claim support and
`summary_coverage` for required-fact coverage. Citations alone do not imply support.

| Metric | Numerator | Denominator |
| --- | --- | --- |
| `omission` | Required truth not correctly supported in its required section | Required truth facts |
| `unsupported` | Uncited, unresolved or contradicted statements | All draft statements |
| `contradiction` | Statements assigned the contradicted state by the existing claim metric | All draft statements |
| `misattribution` | Statements with a wrong speaker or experiencer relative to known cited truth | Statements citing at least one known truth fact |
| `negation` | Statements with polarity different from known cited truth | Statements citing at least one known truth fact |

Wrong experiencers and speakers count as contradictions and separately as
misattribution. A polarity flip counts as contradiction and negation. Conflicting
support/contradiction citations are unresolved under the existing four-state
contract; their attribution and negation errors remain separately visible.
An unrelated proposition, unknown citation or class mismatch is unresolved.
A correctly supported statement placed in the wrong section does not cover the
required fact in its intended section. A corrupted statement can therefore count
as both unsupported and failure to express required truth.

There is no combined score: high coverage cannot cancel misattribution. Known
citation denominators also mean that uncited statements cannot obtain an
attribution pass; their unsupported rate remains visible. Empty denominators are
suppressed rather than represented as perfect results.

The fixed report keys are `overall`, `section:<code>`, `class:<code>`,
`speaker:<code>` and `experiencer:<code>`. Omissions are sliced by truth; other
metrics are sliced by draft statements. Each metric cell has `numerator`,
`denominator`, `rate`, `ci95` and `suppressed`. Intervals are 95% Wilson intervals
for atomic binary observations. They are descriptive; correlated statements
within encounters violate an independent-sampling interpretation. They are not
encounter-bootstrap intervals or evidence of clinical generalization.

## Suppression and privacy

The default `minimum_cell_size` is five and cannot be less than two. Suppress a
cell if its denominator, nonzero numerator or nonzero complement is below the
minimum. Suppressed counts, rates and intervals are all null. If any nonempty
slice is unsafe, suppress that entire metric family, including the overall cell,
to prevent reconstruction by subtraction across partitions. Empty slices are
always suppressed. No encounter counts or identifiers are added outside these
cells. This conservative policy can hide an otherwise large overall sample.

Reports contain controlled schema/slice/notice codes, counts, rates and intervals,
never source text, draft values, reviewer identities, case references, proposition
digests or private paths. Diagnostics use controlled messages. Keep source
payloads out of application logs and retain any private review mapping locally.
Repeated releases over overlapping cohorts need caller-managed privacy controls;
this suppression policy is not differential privacy.

## Blinded clinician review return format

The importer accepts a mapping with **exactly** `schema_version` and `rows`.
The schema version is `openmed.ambient_draft_reviews.v1`. Each row has exactly:

```json
{
  "blinded_case_id": "<random 64-hex case alias>",
  "draft_revision": "<AmbientDraft.revision, 64 hex characters>",
  "reviewer_id": "<random 64-hex reviewer alias>",
  "decision": "revise",
  "reason": "clinical_interpretation",
  "adjudicated": false
}
```

The placeholders above document the format; they are deliberately not valid
identifiers. Decisions are `accept`, `revise` or `unclear`. Reasons are null or a
`DisagreementReason` value: `clinical_interpretation`, `evidence_quality`,
`guideline_ambiguity`, `label_definition`, `reviewer_error`, `other`.

Distribute blinded review materials separately, compute the exact text digest,
and capture `draft.revision` before review. The revision binds the evaluator
version, alias, text digest, truth, statement annotations and citations. Pass
**current** snapshots to `import_ambient_reviews`; changed text, truth or
annotations invalidate old returns. Retain the random alias-to-source mapping in
a separate protected local store; never include it in the return payload.

Every current draft needs at least two distinct reviewers. Reject missing cases,
duplicate reviewers, unknown aliases, wrong schemas, stale revisions, unknown
fields, plaintext identifiers, candidate/model identities and free-text reasons.
Disagreement handling reuses `reviewer_disagreement`: disagreeing rows need one
consistent typed reason and adjudication status. Agreed cases use a null reason
and cannot be adjudicated. `adjudicated` is a caller assertion of a separate
review process; the importer neither resolves differences nor verifies evidence
of adjudication. Disagreement remains counted after adjudication.

Reports separately publish case-level unanimous agreement, adjudication among
disagreed cases and unclear decisions among reviewer rows, with the same minimum
cell and Wilson-interval rules. Reason partitions are omitted. Agreement on
`unclear` remains visible as an unclear-review rate; it never implies acceptance.
Review imports do not replace machine scores or create a release-gate decision.
The pending summary-review workflow (#3655) remains a separate integration.

## Synthetic fixture example

`tests/fixtures/eval/ambient_drafts.json` contains six hand-authored encounters,
including transcripts, explicit roles, truth and draft annotations: clean,
omitted medication, invented statement, inverted experiencer, flipped negation
and wrong speaker. No generator or external media/model assets are required.

A patient saying "I take synthetic-med A. I do not have a cough" supplies two
required facts. The draft "Patient denies cough" omits the medication. The draft
"Patient takes synthetic-med A. Caregiver denies cough" has complete citations
but wrong experiencer. Tests replicate synthetic rows solely to verify count
arithmetic and publication thresholds; replication supplies no independent
clinical evidence. Evaluating a single fixture suppresses all cells.

Focused offline validation:

```sh
.venv/bin/python -m pytest tests/unit/eval/test_ambient_drafts.py tests/integration/test_ambient_drafts.py -q
```
