# Post-de-identification summarization

OpenMed's clinical summarization stage is generative-last. The public
`openmed.clinical.summarize()` entry point de-identifies a note first, then
passes only the de-identified text to a summarizer backend. A deterministic
extractive fallback is used when no backend is supplied.

```python
from openmed.clinical import summarize

result = summarize(note)
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

The default path is local and deterministic. A trained SLM backend is a
separate task and must be supplied as a local or on-device callable; this
stage never selects a cloud model or sends raw PHI to one. Summaries are
assistive outputs and require qualified clinical review.
