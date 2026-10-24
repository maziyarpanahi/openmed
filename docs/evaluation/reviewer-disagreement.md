# Privacy-safe reviewer disagreement

`openmed.eval.reviewer_disagreement` measures whether pseudonymous clinical
reviewers agree and whether disagreements receive adjudication. It runs locally,
uses no network service, and emits only aggregate counts and rates.

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

## Adjudicating native-review labels

`openmed.training.native_review_adjudication` merges native-language reviewer
decisions into accepted training and evaluation labels. Nothing is accepted
without an explicit `AdjudicationPolicy`, and every merge returns a new
immutable `AdjudicationRevision` that names the revision it follows. The merge
runs locally, makes no network calls, and raises errors that carry only field
names and closed codes.

### Inputs

- A `LabelProjection` per source item: an opaque `source_ref`, the language and
  task codes, the current reviewed text, and the proposed spans. The text stays
  in memory to compute its digest and to revalidate offsets; it is never
  serialized. Use source references that are not derived from patient
  identifiers.
- Reviewer decisions in the `openmed.eval.native_review.decision.v1` shape, as
  mappings or objects exposing `to_dict()`. Each becomes a `DecisionOverlay`
  carrying the reviewer-role code, packet and submission digests, and the text
  and projection digests the reviewer saw. The `decision_id` must match its
  packet, item and submission lineage. Fields that look like reviewer identity
  fail with `reviewer_identity_forbidden`, and comment, snippet or concordance
  fields fail with `free_text_forbidden`.
- Optional `AdjudicationEvent` records and the parent revision.

```python
from openmed.training.native_review_adjudication import (
    AdjudicationPolicy,
    LabelProjection,
    LabelState,
    ReviewSpan,
    build_adjudication_event,
    merge_native_review_labels,
)

policy = AdjudicationPolicy(
    policy_id="sw_label_review",
    policy_version="1",
    label_codes=("CONDITION", "MEDICATION"),
    adjudicator_roles=("clinical_linguist",),
)
projection = LabelProjection(
    source_ref="projection-batch-7:item-0001",
    language="sw",
    task_code="label_check",
    text="Mgonjwa ana homa kali tangu jana.",
    spans=(ReviewSpan(start=12, end=16, label="CONDITION"),),
)

# decisions: imported native-review decision documents for this item.
revision = merge_native_review_labels([projection], decisions, policy=policy)
item = revision.item("projection-batch-7:item-0001")
if item.state is LabelState.UNRESOLVED:
    event = build_adjudication_event(
        item,
        policy=policy,
        adjudicator_role="clinical_linguist",
        outcome="accept_spans",
        spans=(ReviewSpan(start=12, end=21, label="CONDITION"),),
    )
    revision = merge_native_review_labels(
        [projection], decisions, policy=policy, events=[event], previous=revision
    )
```

### Resolution rules

Accept, reject and revise decision codes are disjoint policy vocabularies; any
other code, such as `unsure`, is an abstention. A revise decision must carry
corrections, which are the reviewer's complete label set. Each item then
resolves in this order:

1. Ambiguous duplicate decisions, language or task mismatches, invalid
   corrections, unknown labels, and offsets that are out of range or split a
   grapheme cluster set the state to `invalidated`. So does an item whose
   decisions all refer to other text or another projection.
2. One valid adjudication event decides the item. Two valid event heads leave
   it `unresolved` with `conflicting_adjudication`.
3. Without an event, disagreeing or abstaining reviewers leave the item
   `unresolved`. Fewer than `minimum_reviewers` counted assignments leave it
   `pending`. Unanimous reviewers resolve it to `accepted` or `rejected` only
   when `resolve_unanimous` is true; otherwise it waits with
   `adjudication_required`.

An adjudication event is valid only when its policy digest matches, its
adjudicator role is in `adjudicator_roles`, and its basis, text and projection
digests match the current item. `build_adjudication_event()` copies those
digests from the revision entry the adjudicator reviewed, so a new decision or
an edited projection makes the event stale and returns the item to review. A
later event replaces an earlier one only by naming it in `supersedes_event_id`.

### Supersession and invalidation

Decisions sharing a `(packet_ref, item_id)` assignment are resubmissions. A
newer decision replaces an older one only through `supersedes_decision_id`, for
example `DecisionOverlay.from_decision(resubmission, supersedes_decision_id=...)`.
Two current decisions for one assignment, a supersession cycle, or one
`decision_id` with different content mark the item `duplicate_decision`.
Decisions about changed text (`text_changed`) or a regenerated projection
(`stale_projection`) are excluded from the evidence. Fresh decisions can then
resolve the item again. When the parent revision accepted labels that no longer
stand for the same text, the item sets `prior_acceptance_revoked`.

Re-importing identical decisions does not change the revision. Input order does
not change the output.

### Exports

- `revision.export_labels()` lists accepted items with their spans,
  `source_ref`, text and projection digests, and decision lineage: decision
  identifiers, reviewer-role codes, decision codes, the applied event and its
  adjudicator role, superseded and excluded identifiers, and the policy and
  revision identifiers. It contains no reviewer identity, assignment reference
  or text.
- `revision.to_audit_dict()` reports state and reason counts and one entry per
  item keyed by a subject digest instead of `source_ref`.
- `export_adjudicated_annotations(revision, projections, hash_secret=...)`
  returns an annotation-interchange envelope of `entity` records with keyed
  surface hashes, lowercased labels, and the revision identifier as `batch_id`.
  It fails if the text changed since the revision.
- `revision.to_reviewer_decisions()` produces `ReviewerDecision` records for
  `reviewer_disagreement_report`, with subject digests as case identifiers and
  decision identifiers as reviewer identifiers.

`revision.to_json()` and `AdjudicationRevision.from_dict()` round-trip a
revision and revalidate its identifiers and digests, so a stored revision can
be passed back as `previous`. Treat stored revisions as protected evaluator
records because they hold source references. Adjudication curates language
labels only. It does not retrain models or make or trigger clinical decisions.

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
