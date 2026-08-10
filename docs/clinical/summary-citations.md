# Summary citation consistency

`openmed.clinical.summary_citations` checks whether structured summary claims
are supported by the evidence records supplied to the local review workflow.
It is a deterministic review aid, not a clinical-quality certification,
compliance certification, or clinical decision guarantee.

## Citation contract

Each evidence record must expose an opaque identifier such as `id` or
`evidence_id`, or an exact half-open source span with `start` and `end`
character offsets. A claim can cite an identifier or an exact span directly,
or through a `citations` collection:

```python
from openmed.clinical.summary_citations import (
    build_summary_citation_report,
)

report = build_summary_citation_report(
    claims=[
        {
            "claim": "synthetic claim",
            "citations": [{"evidence_id": "evidence-1"}],
        }
    ],
    evidence=[
        {
            "id": "evidence-1",
            "start": 0,
            "end": 12,
            "value": "synthetic source value",
        }
    ],
)

if not report.passed:
    print(report.metrics.failure_reasons)
```

The checker uses only opaque identifiers and exact offsets for matching. It
never reads claim text or evidence values. A repeated citation within one
claim, an unknown evidence identifier, or a span absent from the evidence set
fails the strict report verdict. A claim without a valid citation is counted
as an abstention.

## Metrics and safety behavior

`coverage` is the fraction of claims with at least one valid citation;
`abstention_rate` is the fraction without valid support. The report also
contains aggregate counts for duplicate citations, missing source records,
unavailable spans, and referenced evidence records. JSON and Markdown output
contains no claim text, evidence values, identifiers, or citation payloads.

Missing, invalid, or duplicate source evidence fails closed with zero coverage.
Use `assert_summary_citation_gate` when an incomplete citation contract must
stop a local review step:

```python
from openmed.clinical.summary_citations import assert_summary_citation_gate

assert_summary_citation_gate(claims, evidence)
```

The module makes no model, filesystem-discovery, telemetry, or network call.
Use synthetic or caller-controlled offline records in tests and examples. The
output is an assistive review signal and must not be used to automate a
diagnosis or treatment decision.
