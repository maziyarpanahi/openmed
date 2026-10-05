# Claim-level citation support metrics

`openmed.eval.citation_support_metrics` evaluates whether citations attached to
atomic claims point to usable evidence and, when an approved local review set
is supplied, whether that evidence supports the claim. It is an evaluation aid
only. It is not a compliance certification, a clinical decision, or a
guarantee of clinical truth. Qualified human review remains required before
clinical use.

## Local input contract

The evaluator accepts three independent collections:

- atomic claims, each with a stable identifier, an optional half-open summary
  span, and zero or more cited evidence identifiers;
- evidence spans, each with an identifier, a half-open source span, and an
  optional source identifier and source length;
- optional clinician adjudications, each labeling a cited claim/evidence pair
  as `supports`, `contradicts`, `irrelevant`, or `unclear`.

Identifiers are used only to connect input records. They are not emitted in
reports. Production callers should use opaque references such as
`sha256:<64 lowercase hex characters>` and keep source text outside the
evaluation artifact. The evaluator ignores input `text` fields after deriving
an optional length; it never stores or renders them.

```python
from openmed.eval import compute_citation_support_metrics

claims = [
    {
        "claim_id": "claim-001",
        "start": 0,
        "end": 12,
        "source_id": "sha256:" + "1" * 64,
        "source_length": 40,
        "citations": ["evidence-001"],
    },
    {
        "claim_id": "claim-002",
        "start": 13,
        "end": 25,
        "source_id": "sha256:" + "1" * 64,
        "source_length": 40,
        "citations": [],
    },
]
evidence = [
    {
        "evidence_id": "evidence-001",
        "source_id": "sha256:" + "2" * 64,
        "start": 4,
        "end": 10,
        "source_length": 24,
    },
    {
        "evidence_id": "evidence-unused",
        "source_id": "sha256:" + "2" * 64,
        "start": 12,
        "end": 18,
        "source_length": 24,
    },
]
adjudications = [
    {
        "claim_id": "claim-001",
        "evidence_id": "evidence-001",
        "label": "supports",
    }
]

report = compute_citation_support_metrics(
    claims,
    evidence,
    adjudications=adjudications,
)
assert report.citation_precision == 1.0
assert report.support_recall == 0.5
assert report.orphan_claim_count == 1
assert report.unused_evidence_count == 1
```

The summary and evidence source identifiers in this example are different on
purpose: a generated claim and the source passage supporting it can belong to
different local documents. A source mismatch is checked only when an explicit
citation record supplies a `source_id` that disagrees with its evidence span.

## Metric definitions

The report is atomic-claim based; it does not average document-level scores.

| Metric | Definition |
| --- | --- |
| Citation precision | Supporting adjudicated citation edges divided by reviewed, span-valid citation edges. |
| Support recall | Claims with at least one supporting adjudicated citation divided by all atomic claims. |
| Orphan claims | Atomic claims with no citation edge, with a count and rate. |
| Unused evidence | Supplied evidence spans never cited by a known claim, with a count and rate. |

Citation precision and support recall are `None`/`n/a` when no usable clinician
adjudication is supplied. Unreviewed citations are reported as coverage gaps;
they are not silently treated as supporting. A claim with both supporting and
non-supporting evidence counts as supported for recall, while each reviewed
citation remains visible in the precision denominator.

An adjudication applies to a claim/evidence pair and therefore labels every
distinct valid cited subspan for that pair. Exact duplicate citation edges count
once. Conflicting labels for a pair become `unclear`; contradictory aliases
inside one review record are rejected rather than resolved by field order.

Single citation and adjudication mappings are accepted as individual records.
Typed records are revalidated before evaluation. Collection consumption is
bounded: at most 100,000 claims, 200,000 evidence records, 500,000 citation edges,
and 500,000 adjudications, with one extra item consumed to detect overflow.
These limits also apply to compact mapping inputs and combined embedded inputs.

## Deterministic span checks

`report.deterministic` is independent from semantic adjudication. It checks
that:

- supplied claim and evidence spans use `0 <= start < end`;
- a bounded `source_length`, when supplied, contains the span;
- citation references resolve to known claims and evidence;
- explicit citation spans are valid and lie within their evidence span; and
- an explicit citation source identifier agrees with its evidence source.

The result reports valid and invalid counts plus fixed reason codes such as
`missing_evidence`, `invalid_evidence_span`, `invalid_citation_span`,
`citation_outside_evidence`, `missing_claim`, and `source_mismatch`. It does
not infer support from span overlap. Missing claim spans are reported as
unverified, while citation relationship metrics can still be computed for
structured claims that have no summary offsets.

## Privacy, provenance, and review

`to_json()` and `to_markdown()` expose counts, fixed labels, offsets, and
one-way SHA-256 digests for the input collections. They do not expose claim
IDs, evidence IDs, source identifiers, source text, reviewer comments, or
exception values. Validation errors use fixed categories and do not echo
submitted values. Reports set `human_review_required` to `true` and carry the
evaluation-only disclaimer.

Report constructors validate count/rate consistency, fixed reason labels, and
digest syntax; nested count mappings are immutable. The input digest binds all
citation edges, including references to missing claims. Digests are provenance
pseudonyms, not a guarantee of anonymization. Caller-selected artifact write
failures use value-free errors without retaining the underlying path exception.

The implementation uses only the Python standard library. It performs no
model loading, telemetry, or mandatory network request, so the same local
inputs produce byte-stable JSON and Markdown regardless of input collection
order.

## Importing blinded summary reviews

`openmed.eval.summary_review` adds a local export/return workflow over the existing
[blinded adjudication packets](blinded-adjudication.md). It owns result ingestion,
not packet rendering or clinical judgment. This is the Python evaluation harness;
OpenMedKit's on-device brief validation and human-review requirement are unchanged.
No network, model download, telemetry or new dependency is required.

1. Render packets and retain their existing `SealedIdentityMapping` and evaluator
   key locally. Use approved opaque references and access-controlled artifacts.
2. Call `export_summary_review` with those packets/mapping, `ReviewBinding` rows,
   the exact evaluated `claims`, `evidence`, `summary` and deidentified `source`,
   an integer `rubric_version >= 1`, and `evidence_kind="synthetic"` (default) or
   `"reviewer"`. Choose the kind before export; imports cannot promote synthetic
   requests to reviewer evidence.
3. Send only the returned request and the existing blinded packets to authorized
   reviewers. Retain `SealedSummaryReview.private_json` and its `commitment`
   separately. The HMAC seal authenticates integrity; it is **not encryption**.
   Protect the private JSON with local access controls and keep the key separate.
4. Import the returned decision document using `import_summary_review`, the
   sealed document/key and the current evaluated artifacts. Serialize only
   `ImportedSummaryReview.to_dict()` into public evaluation reports.
5. Pass that result as `review_import=` to `evaluate_summary_gate`, with no legacy
   `adjudications`. The gate checks it against current citation inputs, the exact
   summary, and `deidentified.deidentified_text`; drift fails closed.

Every evaluated claim must cite evidence, and each claim/evidence pair must have
exactly one private binding. Uncited claims cannot be exported as adjudicable
evidence. The exporter
verifies the existing sealed identity mapping, the candidate's exact output digest,
and the cited evidence excerpt against the source offsets. Claim and evidence
spans must fit the actual artifacts. The request binds rubric content/version,
packet content, mapping commitment, source and output content digests, and the
citation metric's claim/evidence digests. Altered same-length text is stale too.
Each public request entry uses a keyed `review_ref`, zero-based packet/evidence
indices, a blinded candidate alias and claim offsets. It contains no private
candidate identity, claim ID, source content or private path.

### Return document v1

The return is a JSON-compatible dictionary with **exactly** these top-level fields:

```python
returned = {
    # Copy these five fields unchanged from the exported request:
    "schema_version": request["schema_version"],
    "request_digest": request["request_digest"],
    "rubric_version": request["rubric_version"],
    "rubric_digest": request["rubric_digest"],
    "evidence_kind": request["evidence_kind"],
    # SHA-256 of an access-controlled local review receipt, not its contents:
    "review_evidence_digest": "sha256:" + "1" * 64,
    "decisions": [
        {
            "review_ref": request["reviews"][0]["review_ref"],
            "reviewer_ref": "sha256:" + "2" * 64,
            "label": "supports",
            "reason": None,
        },
        {
            "review_ref": request["reviews"][0]["review_ref"],
            "reviewer_ref": "sha256:" + "3" * 64,
            "label": "supports",
            "reason": None,
        },
    ],
    "resolutions": [],
}
```

This example is synthetic syntax, not clinician evidence. Digest values and
reviewer references must be generated from the evaluator's approved local process,
not copied as production evidence. The importer verifies binding and completeness,
not reviewer credentials, recruitment, receipt authenticity or clinical truth.
Calling an export `reviewer` is the evaluator's provenance assertion; a digest is
not proof of human participation. Keep genuine receipts under local access control.

Labels are `supports`, `contradicts`, `irrelevant`, or `unclear`. Each edge requires
at least two distinct pseudonymous reviewer digests. A reviewer may decide an edge
only once; exact duplicates are rejected too. Unanimous rows require `reason=None`.
Disagreeing rows require one consistent `DisagreementReason` code. An optional
resolution has exactly `review_ref`, `label`, `reason`, and `adjudicator_ref`; the
adjudicator must be distinct from every initial reviewer and the reason must match.
Resolutions for absent, single-reviewer or unanimous cases are rejected. Unknown
fields, unblinded identities, arbitrary comments, unsupported versions, stale
artifacts, rubric mismatches, broken seals and duplicate resolutions are rejected
with the controlled `invalid_review_import` error. Exceptions do not retain the
underlying input exception or its payload.

### Aggregate behavior and gate limits

| State | Citation metric behavior |
| --- | --- |
| `missing` | No decision; edge remains unreviewed. |
| `incomplete` | Fewer than two reviewers; edge remains unreviewed. |
| `disputed` | Disagreement without a separate final resolution; edge remains unreviewed. |
| `unclear` | Final or unanimous unclear label; edge remains unevaluable. |
| `accepted` | Unanimous or separately resolved usable label joins citation metrics. Non-supporting labels remain non-supporting. |

Reports expose state counts and separate `machine_span_checks`, citation metrics,
reviewer-agreement metrics, and the declared `evidence_kind`. They never expose the
private bindings, model identities, case references, reviewer references, individual
decisions or source text. A stable decisions digest binds the returned rows without
publishing them. Agreement metrics retain the existing minimum-cell/complementary
suppression (default five, configurable minimum two); incomplete cases are excluded
from agreement rates and remain explicitly counted in review states.

The import-aware summary gate adds `summary_machine_spans` and `summary_review`.
Synthetic decisions can populate metrics for offline development, but **cannot pass
adjudication**, even with a zero support threshold. Missing, partial, disputed or
unclear reviews also fail adjudication regardless of the measured support recall.
Only complete caller-declared reviewer evidence can satisfy the review prerequisite;
all existing fact, unsupported-claim, citation-support threshold and leakage gates
still apply. No medical action, release certification or publication is authorized.
The legacy direct `adjudications=` API remains a caller-asserted, unbound input;
use the importer when exact blinded-review provenance is required. Existing
synthetic benchmark artifacts still refuse absent adjudication and are unchanged.

Collections are bounded to 100,000 records per input and 32 million characters of
sealed private JSON. No raw review receipt or private artifact is written by these
APIs. Synthetic unit and integration vectors cover stable round trips, stale spans
and artifacts, rubric drift, duplicate rows, broken seals, incomplete review,
disagreement resolution, diagnostic leakage and gate refusal.
