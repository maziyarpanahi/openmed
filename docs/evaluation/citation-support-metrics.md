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

The implementation uses only the Python standard library. It performs no
model loading, telemetry, or mandatory network request, so the same local
inputs produce byte-stable JSON and Markdown regardless of input collection
order.
