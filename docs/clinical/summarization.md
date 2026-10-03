# Post-de-identification summarization

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
