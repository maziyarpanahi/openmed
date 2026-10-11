# Post-de-identification summarization

See [local backends and typed outcomes](local-backends.md) for executable
synthetic provider wiring and the complete exception/refusal tables.

OpenMed's clinical summarization stage is generative-last. The public
`openmed.clinical.summarize()` entry point de-identifies a note first, then
passes only the de-identified text to a summarizer backend. The default `mlx`
backend resolves the reviewed, revision-pinned Maple alias through the registry.
It loads cached artifacts only, with outbound Python sockets blocked. It never
downloads a model or silently falls back to extraction.

```python
from openmed.clinical import summarize

result = summarize(note, model="extractive")  # explicit, offline CPU baseline
assert result.leakage_check.passed
print(result.summary)
```

The leakage guard compares the summary with the source spans identified by the
de-identification result. A backend that re-emits a source identifier is
rejected before a result is returned. The check exposes counts and digests,
not plaintext identifiers.

Comparison removes the detector's supported zero-width controls, decomposes
canonical variants before its width, confusable and combining-mark defenses,
then recomposes the result for script matching. Case matching retains the prior
Unicode case-insensitive equivalents.
Source hashes and counts retain their existing meaning; normalized source text is
never included in a `LeakageCheck` or exception. The complete source surface and
each whitespace-separated part are compared. Latin and other spaced scripts
retain word boundaries. Han, Kana, Thai, Lao, Khmer and Myanmar surfaces use
substring matching; Hangul surfaces also allow a bounded inventory of trailing
particles and common honorifics, with a word boundary after the suffix.

This is a source-surface guard, not a morphological analyzer. Inflected forms
such as Cyrillic case endings and unrecognized Korean suffixes remain a residual
risk. The guard also depends on the original detector finding the source spans;
keep the separate privacy detector and qualified review in the brief workflow.

Pipeline code that already performed de-identification may call
`summarize_deidentified()` with its `DeidentificationResult`. Passing a plain
string to that guarded stage raises an ordering error.

Use `model="mlx"` (or `"maple"`) on Apple Silicon after separately installing
`openmed[mlx]` and downloading the pinned artifact. Raw-note input also requires
the cached `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1-mlx` artifact and its
source model's tokenizer. Download these separately before offline inference;
no conversion, dependency installation or download runs inside `summarize()`.
With the default cache locations, provision once using:

```bash
hf download deepgrove/maple-preview-2bit-mlx --revision 361db5da5e74ff6fcdd852d478e1f266ce11013a
hf download OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1-mlx --cache-dir ~/.cache/openmed
hf download OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1 --cache-dir ~/.cache/openmed --include '*.json' --include '*.txt' --include '*.model'
```

Missing dependencies raise
`MissingOptionalDependencyError`; missing weights, context overflow, insufficient
estimated memory, and malformed responses fail closed with content-free errors.
`LocalSummarizerError.reason` is a controlled code: `unsupported_mode`,
`unregistered_alias`, `artifact_not_cached`, `invalid_configuration`,
`capability_unsupported`, `invalid_memory_budget`, `memory_budget_exceeded`,
`input_limit_exceeded`, `context_exceeded`, `invalid_output`,
`output_limit_exceeded`, `deidentification_unavailable`, `runtime_unavailable`,
`invalid_backend`, or `execution_failed`. Remote aliases raise the compatible
subclass `RemoteSummarizerError` with `reason="remote_backend"`. Reasons and
messages never include the model alias, filesystem path, note, or underlying
exception. Custom backend errors are reconstructed without their exception chain;
existing `except LocalSummarizerError` handlers keep working. Both error types,
`resolve_summarizer_backend`, `ExtractiveSummarizerBackend`, and
`MLXSummarizerBackend` are exported from `openmed.clinical`.
The adapter runs the shared `summarize` Maple task, template digest, offline
capability probe and conservative memory preflight before model construction.
The default memory estimate is capped at 16 GiB; applications can pass an
`MLXSummarizerBackend(memory_budget_bytes=...)` with an explicit device budget.
This estimate is not a measured peak-memory guarantee.
The bounded generation allowance is 2,048 tokens, including internal reasoning;
it is reserved before model loading. Final summary text is limited to 8 KiB.

`result.metadata` contains only a fixed backend ID and template digest. Do not
log `result.summary` or `result.to_dict()`: the explicit output contains clinical
text. Custom callables are trusted local application code, not sandboxed plugins;
they run with OpenMed's Python socket guard but must not spawn remote tools.

Migration: callers relying on the former implicit sentence-picker must now use
`model="extractive"`. Maple is a general model, not a released clinical SLM or
clinically validated summarizer. Summaries require qualified clinical review;
the source-token leakage check cannot prove that every identifier was detected.

The explicit extractive baseline uses OpenMed's local, script-aware
`segment_text()` and returns at most the first three sentences in source order.
CJK terminators and Indic danda are recognized, while ordinary abbreviations
and decimals are preserved. Selected sentences keep their exact source text
(only surrounding whitespace is trimmed); one space separates selections.
The algorithm identity is `extractive-script-aware-first-three-v2`, bound into
`template_digest`. This is sentence selection, not fact-coverage or model
qualification. An overlarge custom output produces `output_limit_exceeded`,
distinct from a non-string `invalid_output` result.

## Evidence-aware explicit extraction

`build_clinical_brief(..., model="extractive", context=reviewed_context)` selects
whole sentences covering every reviewed fact, including required medication or
finding evidence after the third sentence. Source order, exact text and original
source/output citation offsets are retained. This Python CPU selection path is
independent of generative backends and OpenMedKit's on-device model generation.

Applications owning reviewed de-identified facts can configure
`ExtractiveSummarizerBackend` directly:

```python
from openmed.clinical.extractive_selection import ExtractiveFact
from openmed.clinical.summarize import summarize_deidentified
from openmed.clinical.summarize_backends import ExtractiveSummarizerBackend
from openmed.clinical.summary_length_budget import build_summary_length_budget
from openmed.clinical.summary_omission_budget import ImportanceClassPolicy

# Opaque digests and offsets are supplied by local evidence review.
backend = ExtractiveSummarizerBackend(
    evidence=(ExtractiveFact(fact_digest, class_digest, start, end),),
    importance_classes=(ImportanceClassPolicy(class_digest, 10, mandatory=True),),
    length_budget=build_summary_length_budget(160, {"key_findings": 160}),
)
result = summarize_deidentified(deidentified_result, model=backend)
```

Each fact names its existing length class (`key_findings` by default). UTF-8
bytes, including joining spaces, conservatively charge both global and class
allowances. A sentence with facts from several length classes is charged in
full to each class, once per class. This estimate does not claim a measured
tokenizer bound for arbitrary model vocabularies. No class cap or metric
threshold is increased. The brief adapter maps reviewed profile fields to their
existing policy classes and supplies the resulting immutable length budget to
the selector. Per-class charges include the separator before each selected sentence after
the first; the global charge includes every joining separator.

The bounded exact search maximizes unique fact coverage, then severity-weighted
coverage, then prefers shorter extracts and earlier sentence indices. Each
importance class's omission allowance is enforced independently. Duplicate
identities or source spans cannot inflate coverage; conflicting duplicates or
facts crossing sentence boundaries refuse. No clinical fact or approval is
inferred from the note. Coverage metrics and downstream clinical/privacy gates
remain unchanged; satisfying an omission policy does not waive a coverage gate.
Malformed Unicode returns a controlled refusal without retaining source values
in a chained decoder exception.

`backend.select(deidentified_text)` returns an `ExtractiveSelection` with
`status="selected"`, or `empty_evidence`, `invalid_evidence`,
`insufficient_budget`, or `selection_limit_exceeded`. At most 64 unique facts
and 50,000 search states are admitted. Resource exhaustion is distinguished
from proven infeasibility. Failed selection never returns partial text or
citations. `summarize_deidentified` raises `ExtractiveSelectionError` carrying
that safe result; the brief returns an empty `unsupported_claim` refusal with
the explicit selection status in `metrics.extractive_selection`. Diagnostic
serialization contains only controlled codes, counts, offsets and digests;
`selection.summary` is protected clinical output and must never be logged.

The bundled brief audit and response schemas also validate these emitted
failure metrics. The closed diagnostic contract rejects source text, partial
citations or charged output, unknown selection states and malformed policy IDs.

Without explicit evidence, `model="extractive"` retains the historical
first-three-sentence baseline. `model="extractive-baseline"` also selects that
baseline inside a reviewed brief, for comparison. Neither mode invents evidence.

Reproduce the synthetic comparison with:

```bash
.venv/bin/python -m openmed.eval.summary_benchmark --model extractive --output eval/suites/summaries/extractive.json
.venv/bin/python -m openmed.eval.summary_benchmark --model extractive-baseline --output /tmp/extractive-baseline.json
```

The unchanged seeded fixture and scoring protocol records 1.0 fact recall for
evidence-aware selection and 0.75 for the baseline. Both reports remain failed:
clinician adjudication is absent. Synthetic literal alignment is not clinical
validation, model qualification or release authorization.
