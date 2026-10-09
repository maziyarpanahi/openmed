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

## Guarded local CLI

`openmed summarize` uses the same post-de-identification API and leakage guard.
The CLI accepts only `--model mlx` (default) or explicit `--model extractive`,
and only `--mode bhc`. Provision the required cached PII/runtime artifacts
separately, including for the extractive baseline. Remote names and URLs are
rejected; the command never downloads models or changes its backend on failure.

```bash
openmed summarize synthetic-note.txt --model extractive \
  --summary-output protected-summary.txt --metadata-output summary-metadata.json --json
```

Input must be a nonempty, regular UTF-8 file of at most 16 KiB. Final symlinks
and special files are refused. The two output destinations must be distinct,
new files with existing parent directories. Both are created with permissions
`0600`; existing files and final symlinks are never overwritten. Ordinary
reservation or write failures remove files created by this invocation. This
two-file operation is not crash-atomic: a process kill or power loss can leave
partial files for the operator to remove.

Only the protected summary file contains generated text, bounded to 8 KiB.
The separate metadata file uses the shared CLI JSON envelope with leakage
counts, backend ID, template digest, summary length and a human-review flag.
`--json` prints that same value-free envelope. Human output is a fixed review
advisory. Neither console stream nor metadata includes note or summary text,
input/output paths, detected identifiers, or backend exception messages. The
guard covers detected source identifiers; qualified clinical review remains
required. See the [machine contract](../cli/machine-contract.md#guarded-summary-and-nli-commands)
for limits, exit codes and stable failure codes.
