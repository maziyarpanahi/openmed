# Unsupported-claim rates for clinical summaries

`openmed.eval.summary_unsupported_claims` measures how often atomic claims in a
generated clinical summary lack approved supporting evidence. It is an
evaluation aid only: it is not a compliance certification, a clinical
decision, or a guarantee that a summary is faithful.

## Evidence contract

The evaluator consumes two caller-provided, local collections:

- summary claims, each with a stable `claim_id`, a normalized `claim_class`, an
  input-only `claim_key`, and zero or more cited `evidence_ids`;
- evidence rows, each with an `evidence_id`, an optional claim link, an
  `approved` boolean, and a relation of `supports`, `contradicts`, or
  `unresolved`.

`claim_key` can be a structured fact key or an opaque fingerprint. The
evaluator fingerprints it immediately and never emits it. Use opaque claim and
evidence identifiers in production; do not pass patient names, note text,
diagnosis strings, or other low-entropy sensitive values as report metadata.

```python
from openmed.eval.summary_unsupported_claims import score_summary_claims

claims = [
    {
        "claim_id": "claim-001",
        "claim_class": "diagnosis",
        "claim_key": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "evidence_ids": ["evidence-001"],
    },
    {
        "claim_id": "claim-002",
        "claim_class": "medication",
        "claim_key": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "evidence_ids": [],
    },
]
approved_evidence = [
    {
        "evidence_id": "evidence-001",
        "claim_key": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "approved": True,
        "relation": "supports",
    }
]

report = score_summary_claims(
    claims,
    approved_evidence,
    n_resamples=1_000,
    seed=7,
)
print(report.by_claim_class["diagnosis"].unsupported_rate)
print(report.to_json())
```

## Four-state scoring

Each atomic claim receives exactly one state:

- `supported`: every cited evidence row is approved, linked to the claim, and
  marked as support;
- `contradicted`: every cited evidence row is approved, linked, and marks a
  contradiction, with no support relation;
- `unresolved`: a citation is present but is missing, unapproved, mismatched,
  unknown, or conflicting;
- `uncited`: the claim has no evidence citation.

The unsupported count is `contradicted + unresolved + uncited`, and the
unsupported rate divides that count by all claims. The report contains the
four-state counts and unsupported rate overall and for each normalized claim
class. Non-supported claims should be routed for human review before any
downstream clinical use.

## Deterministic confidence intervals

The report attaches a non-parametric, claim-level bootstrap interval to the
overall rate and every claim class. A fixed `seed`, input collection, and
bootstrap configuration produce the same JSON and Markdown. Empty and
single-claim collections produce zero-width intervals marked as degenerate.
The implementation uses only the Python standard library and performs no
model loading, telemetry, or mandatory network call.

Reports expose counts, normalized claim classes, states, rates, and interval
metadata, plus a digest and counts for the evidence set used. They do not
include claim IDs, evidence IDs, claim keys, source text, or evidence values.
The relation labels must come from an already-approved local evidence or
adjudication process; this evaluator does not select or validate a clinical
source.
