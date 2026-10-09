# Fixed-pair transcript accuracy

`openmed.eval.transcript_accuracy` scores caller-owned reference/final-hypothesis
pairs locally, without a provider, network service, weights or clinical lexicon.
The issue scopes this evaluator to Python `openmed/eval`; it does not add a
Swift inference or provider qualification surface. All examples are synthetic.

```python
from openmed.eval.transcript_accuracy import (
    ClinicalTermClass,
    ReferenceTermSpan,
    score_transcript,
    transcript_accuracy_report,
)

reference = "SyntheticAda takes 5 mg"
spans = [
    ReferenceTermSpan(0, 12, ClinicalTermClass.IDENTIFIER),
    ReferenceTermSpan(19, 20, ClinicalTermClass.DOSE),
    ReferenceTermSpan(21, 23, ClinicalTermClass.UNIT),
]
scores = [
    score_transcript(reference, hypothesis, spans=spans,
                     partials=["SyntheticAda takes"])
    for hypothesis in [reference] * 5 + ["SyntheticAda takes 50 mg"] * 5
]
report = transcript_accuracy_report(scores, bootstrap_resamples=100, seed=0)
assert report["wer"]["rate"] == 0.125
assert report["clinical_terms"]["dose"]["rate"] == 0.5
assert report["clinical_terms"]["identifier"]["recall"] == 1.0
```

Repeating rows here demonstrates arithmetic only. Real uncertainty estimates
require independent transcript pairs, not copies of a single observation.

## Normalization and alignment

The explicit policies `en`, `es`, `fr`, and `de` currently share
`transcript-nfc32-v1`: NFC with Python's fixed Unicode 3.2 database, a fixed
ASCII/Latin-1 uppercase-to-lowercase map, and collapsed/trimmed whitespace from
the fixed set in the module. Other language codes fail closed. Unicode outside
the case map is preserved, including multilingual identifiers and emoji. This
limited policy deliberately avoids host locale and changing Unicode case tables.

Words are whitespace-delimited tokens, preserving punctuation, apostrophes,
hyphens, decimal points, units, and accents. No transliteration, spelling
correction, `ß` expansion, spoken-number rewriting or clinical equivalence is
inferred. `5`, `five`, `.5`, `5.0`, `mg` and `µg` remain distinct. CER counts
normalized Unicode scalars with inter-token whitespace excluded, not bytes or
grapheme clusters. `É` and `E` plus combining acute are equivalent.

Both metrics use unit-cost Levenshtein alignment. Ties prefer diagonal
(match/substitution), then deletion, then insertion. `EditCounts` preserves
reference/hypothesis lengths and exact substitution/deletion/insertion counts.
Rate is `(S + D + I) / reference_length` and may exceed one. Empty-reference
rates are `None`, even for an empty hypothesis; insertions are still counted.
Pooled reports sum counts, so empty-reference insertions contribute to the
numerator when other pairs supply a denominator. Each word/character alignment
is bounded to 2,000,000 matrix cells; larger inputs fail with a controlled error.
Callers should score fixed utterances rather than unbounded recordings.

## Clinical annotations and churn

Annotations use half-open Unicode-scalar offsets in the **original reference**.
They must cover whole tokens, without leading/trailing whitespace. Normalization
does not change annotation offsets. Same-class overlaps and duplicates are
rejected; overlaps between classes are allowed. Classes are `medication`,
`dose`, `unit`, `negation`, and `identifier`, supplied by the annotator.

Each span contributes one term. Any aligned substitution/deletion in it, or
insertion strictly inside its token interval, makes that term erroneous.
Insertions on span boundaries remain unassigned: alignment alone cannot decide
which adjacent clinical term they modify. WER/CER still count these insertions.
Use annotations spanning a complete multi-token expression when interior
insertions must be counted. Slice rates are erroneous terms / annotated terms,
not a medication or dose correctness decision. Identifier recall is one minus
the identifier term error rate under normalization. It measures ASR preservation
of supplied identifiers, **not PHI detector recall or redaction safety**.

`partials` are chronological snapshots of the same utterance. For each adjacent
snapshot, including the last partial to final, churn is the number of previous
tokens after the longest common prefix. Appending speech incurs zero churn;
replacement/deletion counts the previously emitted suffix. `revision_churn`
pools all transitions; `final_revision_churn` measures only the last transition.
Denominators are previously emitted token opportunities. With no partials or no
previous tokens, churn has no denominator and its aggregate cell is suppressed.

## Reports, uncertainty and privacy

`TranscriptScore` contains only counts and a controlled language policy. It is
an in-memory intermediate, not a publication-safe single-case report. Raw strings
exist transiently during normalization/alignment; callers must protect memory
and must not log transcripts, the normalizer's returned text or tracebacks with
local variables. The evaluator writes no files, caches, logs or telemetry.
Reports contain no transcript, token, annotation text, source path or case ID.

Reports micro-average counts and bootstrap **whole independent pairs**, including
zero-contribution pairs in each slice. Numeric scores are canonically sorted
before SHA-256 counter resampling, making reports independent of input order,
hash seed, RNG implementation and host Unicode tables. `seed` defaults to zero,
resamples to 1,000. Intervals use the floor/ceil 2.5%/97.5% order statistics;
resamples with zero denominator are excluded (conditional on a defined rate).
Identifier recall intervals invert the error-rate bounds. These are descriptive
percentile intervals, not clinical validation or a release claim.

The default minimum cell size is five, configurable to at least two. Each cell
requires both enough denominator units and enough contributing pairs. Nonzero
small erroneous-pair and error-free-pair cells are suppressed too. WER/CER also
check substitution, deletion, insertion and correctly aligned unit counts;
term/churn rates check both error and complement counts. An entire unsafe cell
has `null` size, contributors, errors, rate and confidence bounds (and recall).
Absent clinical classes are suppressed. Zero-sided splits may be published when
the other requirements pass. Aggregate pair count is hidden below the threshold.
This is minimum-cell protection, not differential privacy; overlapping cohorts
and repeated releases still need a caller-owned disclosure policy.

Every report binds a non-diagnostic notice and
`reviewer_confirmation_required: true`. Consumers must obtain explicit reviewer
confirmation before consequential use. Reports carry no approval, provider
qualification, model ranking gate or autonomous clinical action.
