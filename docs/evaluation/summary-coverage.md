# Clinical summary fact coverage

`openmed.eval.summary_coverage` measures whether structured summary claims
remain connected to the source facts that were supplied as evidence. It is a
local evaluation metric, not a clinical-quality certification or a clinical
decision guarantee.

## Citation contract

The source-fact collection is the allowed evidence set. Each source record
should provide a stable opaque `id` or `fact_id`. A source record may also
provide a span such as `evidence: {start: 120, end: 136}`; spans are useful
when the summary pipeline cites source offsets instead of fact IDs.

Each summary item represents one claim and can cite a source fact with
`source_fact_id` or `fact_id`, or cite a source span with `start` and `end`:

```python
from openmed.eval.summary_coverage import run_summary_coverage

report = run_summary_coverage(
    source_facts=[
        {"fact_id": "fact-1", "evidence": {"start": 120, "end": 136}},
        {"fact_id": "fact-2", "evidence": {"start": 180, "end": 194}},
    ],
    summary_citations=[
        {"source_fact_id": "fact-1"},
        {"citations": [{"start": 180, "end": 194}]},
    ],
)
```

The matcher uses only opaque identifiers and offsets. Fields such as `value`,
`text`, and `claim` are ignored for scoring and are never copied into the
report. Do not place raw clinical text in committed fixtures or report
metadata.

## Metrics and verdict

For a valid source-evidence set:

```text
recall = distinct cited source facts / source facts
omission_count = source facts with no citation
unsupported_fact_count = summary claims with no valid citation
```

Repeated citations do not inflate recall. A citation that names an unknown
source fact increments `invalid_citation_count`; a claim with no valid citation
is counted as unsupported. `passed` is strict: source evidence must be
available, every source fact must be cited, and every summary claim must have
valid support.

An empty or malformed source-evidence set fails closed. The report emits
`recall: 0.0`, `fail_closed: true`, and a reason code instead of treating the
absence of evidence as perfect coverage. Callers can use
`assert_summary_coverage_gate(...)` when a missing or incomplete citation set
must stop an evaluation.

## Privacy and execution boundaries

The metric performs no model inference, filesystem discovery, telemetry, or
network call. JSON and Markdown reports contain aggregate counts, booleans,
schema information, and reason codes only. They do not contain source values,
summary text, fact IDs, citation payloads, or offsets that were not explicitly
used as structured references. The metric is intended for synthetic or
caller-controlled local evaluation data and does not make autonomous clinical
decisions.
