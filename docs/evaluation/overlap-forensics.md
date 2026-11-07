# Benchmark overlap forensics

OpenMed can scan a sealed benchmark submission for exact, normalized, fuzzy,
canary, and public-versus-shadow signals before results are released. The
scanner is deterministic, runs locally, and performs no network requests. It
reports detected signals and the comparisons that were run; it does not certify
that a submission is free from contamination or hardcoding.

## Prepare evaluator-held inputs

Provide four ordered collections:

- submitted text artifacts from the sealed workflow;
- public benchmark material that a submitter could legitimately have seen;
- evaluator-held shadow material;
- evaluator-held canary markers.

The order is used only to create stable references such as
`submission:000001` and `shadow:000002`. Reports never contain source text.
Keep the report beside the private evaluator mapping if authorized reviewers
need to resolve an ordinal reference.

The scan also requires the sealed workflow manifest and the pre-publication
holdout commitment. Their public digests associate the report with the governed
submission and holdout version. Invalid or mutable evidence is rejected with a
closed error code before text is compared.

```python
from openmed.eval.governance.overlap_forensics import scan_overlap_forensics

report = scan_overlap_forensics(
    submission_items=submission_text,
    public_items=public_benchmark_text,
    shadow_items=private_shadow_text,
    canaries=private_canary_markers,
    submission_manifest=sealed_manifest,
    holdout_commitment=holdout_commitment,
)
evidence_json = report.to_json()
```

Do not write the four input collections to logs, exceptions, or report
metadata. The scanner processes them in memory and emits only ordinal
references, aggregate counts, integer scores, configured thresholds, schema
versions, and the two upstream evidence digests.

## Interpret the checks

The versioned v1 policy applies these checks:

1. **Exact overlap** compares the original Python strings code point for code
   point.
2. **Normalized overlap** applies Unicode NFKC normalization, case folding,
   converts non-letter and non-number runs to spaces, and collapses whitespace.
   It reports matches not already classified as exact.
3. **Fuzzy overlap** computes character-trigram Jaccard similarity over the
   normalized strings. It reports non-exact, non-normalized matches at or above
   the configured threshold, which defaults to 8,500 basis points.
4. **Canary signal** checks whether a normalized evaluator-held marker occurs
   within a normalized submission artifact.
5. **Public versus shadow** compares each submission artifact's strongest
   fuzzy score in both corpora. It reports a signal when the shadow score meets
   the fuzzy threshold and exceeds the public score by the configured margin,
   which defaults to 1,000 basis points.

Basis-point scores are integers from 0 to 10,000. Exact and normalized matches
use 10,000. Treat thresholds as a versioned evaluation policy: record deliberate
changes rather than tuning them after inspecting a submission.

## Review coverage and findings

`report.coverage` records the number of submission, public, shadow, and canary
items, the five checks, and pairwise comparison counts. `report.signal_counts`
records findings by check. Individual findings expose only ordinal references,
scope, score, and—for public-versus-shadow findings—the strongest public score
used for comparison.

A detected signal is a review lead, not proof of contamination or hardcoding.
A report with no signals says only that none were detected within the configured
checks and supplied inputs. It does not establish their absence. Corpus
coverage, normalization choices, threshold sensitivity, paraphrases, semantic
equivalence, and canary placement remain limitations reviewers must assess.

The report does not provide a compliance certification or an autonomous
clinical decision guarantee. Use only synthetic committed examples; restricted
or DUA-gated benchmark content remains evaluator-held.
