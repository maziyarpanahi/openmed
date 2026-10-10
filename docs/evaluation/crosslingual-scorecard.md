# Cross-lingual extraction scorecard

`openmed.eval.crosslingual_scorecard` combines existing local evaluation
reports into one aggregate-only view of extraction recall, critical leakage,
abstention, and latency.

## Inputs

The builder accepts `BenchmarkReport` objects and serialized report mappings.
Reports can identify one language with `language` or `metadata.language`, or
provide language slices under `per_language`/`by_language`. A model family is
read from `family` or `metadata.family`; `family_by_model` and manifest rows are
available as explicit fallbacks.

```python
from openmed.eval.crosslingual_scorecard import build_crosslingual_scorecard

scorecard = build_crosslingual_scorecard(
    reports,
    expected_languages=("de", "en", "es", "fr"),
)
scorecard.write_json("artifacts/crosslingual-scorecard.json")
scorecard.write_markdown("artifacts/crosslingual-scorecard.md")
```

Rate metrics use numerator/denominator evidence when a report provides it;
otherwise the report's fixture count is the weight. Family aggregation counts
each report once, while language-sliced metrics are combined using their
available count evidence. Latency retains mean, p50, and p95 summaries and
uses the mean (then p50, then p95) as the headline `latency_ms` value when a
single value is needed.

## Missing evidence and privacy boundary

Pass the expected language set to make missing evidence explicit. The JSON
payload exposes `missing_languages`, `missing_language_evidence`, and counts
for reports that did not have a safe language label. Those reports still
contribute to the family view, but their metrics are not attributed to a
language.

Both renderers are deterministic and aggregate-only. They include language and
family labels, report/fixture counts, rates, and latency summaries; they omit
source text, spans, fixture identifiers, unsafe examples, model metadata, and
all other arbitrary report fields. The scorecard performs no network call and
does not load a model or a dataset.

Registered language codes and known model-family labels remain readable.
Unknown single-token labels are replaced with stable SHA-256 keys so untrusted
report metadata cannot be copied into the artifacts.

The scorecard is evaluation evidence, not a compliance certification or a
clinical decision guarantee. Missing metrics are rendered as `n/a` and should
be investigated before comparing language or family rows.

## Evidence limits

Missing fixture counts are reported as zero (unknown), not inferred from
character support. Metrics without count evidence use a unit weight.
Explicit zero-support language buckets do not establish language coverage.
A leakage rate without a valid denominator is not a leakage event count.
Weighted p50/p95 values summarize the supplied report percentiles; they are
not percentiles of pooled raw latency measurements.

Inputs are bounded to 8,192 reports/items per collection and 8,192 structural
nodes per report, with a maximum nesting depth of 32. Invalid input callbacks,
recursive metadata, and arithmetic overflow fail with a fixed error that
does not retain the original exception context. Rows keep only allowed
aggregate fields, including when constructed directly, and nested state is
immutable. JSON indentation is bounded to 0–8.

Hashed unknown labels remain linkable pseudonyms, not anonymization. This
adapter cannot establish whether upstream metrics were evaluated correctly.
