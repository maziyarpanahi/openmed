# Privacy-safe reviewer disagreement

`openmed.eval.reviewer_disagreement` measures whether pseudonymous clinical
reviewers agree and whether disagreements receive adjudication. It runs locally,
uses no network service, and emits only aggregate counts and rates. To collect
decisions from native-language experts without exposing reviewer identity, see
[Native-language review packets](#native-language-review-packets).

## Input contract

Each case needs at least two `ReviewerDecision` records with distinct
pseudonymous reviewer identifiers. Decision codes are intentionally opaque.
Disagreement cases must use one consistent `DisagreementReason` and one
consistent adjudication status across their records. Free-text reasons are
rejected so clinical text cannot accidentally enter an aggregate report.

```python
from openmed.eval.reviewer_disagreement import (
    DisagreementReason,
    ReviewerDecision,
    reviewer_disagreement_report,
)

decisions = [
    ReviewerDecision("case-01", "reviewer-a", "accept"),
    ReviewerDecision("case-01", "reviewer-b", "accept"),
    ReviewerDecision(
        "case-02",
        "reviewer-a",
        "accept",
        reason=DisagreementReason.EVIDENCE_QUALITY,
        adjudicated=True,
    ),
    ReviewerDecision(
        "case-02",
        "reviewer-b",
        "revise",
        reason=DisagreementReason.EVIDENCE_QUALITY,
        adjudicated=True,
    ),
]

report = reviewer_disagreement_report(decisions, minimum_cell_size=2)
payload = report.to_dict()
```

The agreement denominator is all reviewed cases; its numerator is the number
with unanimous decisions. The adjudication denominator is disagreement cases;
its numerator is the number marked adjudicated. Per-reason cells report the
same adjudication rate within a typed reason.

## Minimum-cell protection

The default minimum cell size is five. A rate is suppressed when its denominator
is below the threshold or when either nonzero side of its binary split is below
the threshold. Suppressing the complementary cell matters: publishing four
agreements out of five cases would reveal the single disagreement even if its
count were hidden separately.

Suppressed rates contain `null` for numerator, denominator, and rate. Reason
categories whose cells are unsafe are omitted, and the report records only how
many reason cells were suppressed. A zero-sized side is safe to publish because
it describes no reviewer or case. An empty adjudication denominator has a
`null` rate and unsuppressed zero counts.

Reports never include case identifiers, reviewer identifiers, decision codes,
or free-text payloads. Exceptions and `ReviewerDecision` representations are
also value-free. Identifiers are still held in process while grouping, so the
caller remains responsible for pseudonymizing them before constructing records
and for protecting process memory.

This metric supports review-quality monitoring. It is not a compliance
certification and does not make or trigger clinical decisions.

## Native-language review packets

Language validation needs native experts, but a review package can leak PHI or
show more institutional text than the review needs.
`openmed.eval.native_review_packets` builds minimal packets for native-language
reviewers and imports their decisions back through opaque item identifiers.
Building, verifying and importing run locally and make no network calls.

### Build a packet

Every `NativeReviewItem` must declare `privacy_status="synthetic"` or
`privacy_status="deidentified"` together with a SHA-256 digest of its generator
or de-identification provenance record. Any other status is refused when the
item is constructed. Item text is capped by `max_item_characters` (1,000 code
points by default, 4,000 at most), so reviewers see short segments rather than
whole notes. Proposed labels use half-open Unicode code-point offsets, the
annotation-interchange `unicode_codepoint` convention, and must come from the
task's closed label vocabulary. Offsets that would split a combining mark, an
Indic vowel sign, or a joiner sequence are rejected.

```python
from openmed.eval.native_review_packets import (
    NativeReviewItem,
    NativeReviewSpan,
    NativeReviewTask,
    build_native_review_packet,
)

task = NativeReviewTask(
    task_code="label_check",
    instructions=(
        "Confirm whether each highlighted span carries the stated label in "
        "natural Swahili clinical usage."
    ),
    decision_codes=("accept", "reject", "revise", "unsure"),
    label_codes=("CONDITION", "MEDICATION"),
    reason_codes=("boundary", "label", "morphology"),
    correction_decision_code="revise",
)
items = [
    NativeReviewItem(
        source_ref="projection-batch-7:item-0001",
        task_code="label_check",
        privacy_status="synthetic",
        provenance_digest="sha256:" + "a" * 64,
        text="Mgonjwa ana homa kali tangu jana.",
        spans=(NativeReviewSpan(start=12, end=16, label="CONDITION"),),
    ),
]

packet, manifest = build_native_review_packet(
    items,
    packet_ref="sw-assignment-0001",
    language="sw",
    tasks=(task,),
    seed=20261024,
)
```

`packet.to_json()` is the reviewer artifact. It contains the task instructions
and closed vocabularies, the allowed reviewer-role codes, a short reviewer
notice, the items in a seeded randomized order, and a provenance commitment.
Items are named only by `nri_...` identifiers derived from the packet
reference, seed, position and reviewed content. The packet never contains
source references, provenance digests, the seed or reviewer identity. The same
items, tasks and seed produce byte-identical packets in any input order.

`manifest.to_json()` is the evaluator-private binding from each opaque item to
its `source_ref`, provenance digest, text digest and projection digest. Store
it apart from the packet and never send it to reviewers.
`verify_native_review_manifest(packet, manifest)` checks that the manifest
describes exactly the distributed packet; the packet's provenance commitment
also binds the source mapping, so an edited or swapped manifest entry fails
with a closed reason code.

Issue one packet per reviewer assignment, each with its own `packet_ref` and
`seed`. The assignment reference then separates reviewers without naming them.

### Import reviewer decisions

`packet.submission_template()` returns the document a reviewer completes:

```json
{
  "schema_version": "openmed.eval.native_review.submission.v1",
  "packet_ref": "sw-assignment-0001",
  "packet_digest": "sha256:...",
  "reviewer_role": "native_clinician",
  "decisions": [
    {
      "item_id": "nri_...",
      "decision": "revise",
      "reason_code": "boundary",
      "corrections": [{"start": 12, "end": 21, "label": "CONDITION"}]
    }
  ]
}
```

```python
from openmed.eval.native_review_packets import import_native_review_decisions

imported = import_native_review_decisions(
    submission_json,
    packet=packet,
    manifest=manifest,
)
```

Import first verifies the packet against the manifest, then requires the
submission to name the same packet reference and digest. The reviewer role must
be one of the packet's role codes, and decision, reason and correction label
codes must come from the task's closed vocabularies. Corrections are a complete
replacement span set, required for the task's correction decision and refused
for every other decision. Unknown or duplicate item identifiers, duplicate JSON
keys and non-finite numbers are rejected. Fields that look like reviewer
identity, such as `reviewer_name`, `email` or `annotator_id`, fail with
`reviewer_identity_forbidden`, and comment or note fields fail with
`free_text_forbidden`, so neither can reach an imported record.

Completeness is checked against the manifest. A `null` decision or a missing
row leaves the item undecided, and by default import raises
`NativeReviewIncompleteError`, whose `completeness` attribute lists the missing
opaque item identifiers. Pass `require_complete=False` to keep the decided
subset with the same completeness summary.

Each imported `NativeReviewDecision` carries the reviewer-role code, decision
and reason codes, corrections, and lineage: packet reference and digest,
submission digest, source reference, language, privacy status, and the
provenance, text and projection digests the reviewer saw. It holds no reviewer
identity, free text or source text. `decision_id` is derived from the packet
digest, item identifier and submission digest, so re-importing the same
submission is idempotent, while a changed resubmission yields a new decision
that shares the `(packet_ref, item_id)` assignment key. `to_dict()` and
`from_dict()` round-trip each record and revalidate that lineage. Adjudication
of conflicting decisions is a separate, explicit step; to measure agreement,
map decisions to `ReviewerDecision` records keyed by `source_ref`, with
`packet_ref` as the pseudonymous reviewer identifier.

The privacy status is a declaration bound to a provenance digest; this module
cannot prove that text is synthetic or de-identified. Treat packets as protected
review artifacts, keep their text out of logs and general reports, and remember
that the grapheme check is a conservative guard, not full Unicode segmentation.
Errors carry only field names and closed codes. Native review supports language
validation only and does not make or trigger clinical decisions.
