# Code-switch span projection

`openmed.clinical.languages.code_switch_projection` extends source-aligned language
routing with overlapping hypotheses. It never normalizes or changes source text.
`code_switch_windows(text, router=..., overlap=16)` expands each routed run on both
sides and clips at original character boundaries. The default router needs no
model or network; routing confidence is not clinical accuracy.

Run a local detector separately on each `text[window.start:window.end]`, then bind
its local character spans to `WindowEntity` with the window ID, canonical label,
score, and evidence digest. `reconcile_code_switch_entities(..., text_length=...)`
projects spans to the original text. Exact span/label matches merge provenance,
retain all language hypotheses, and record the largest declared detector score.
Overlapping different labels or boundaries remain separate and all require
review. Adjacent spans do not conflict. No score chooses a clinical meaning.

Reports contain offsets, controlled labels, language codes, scores and digests,
never source snippets. Candidate status is only a detector hypothesis; this layer
neither authorizes de-identification release nor replaces a clinical review gate.
Invalid source/window/span bindings fail with controlled messages. Source size,
window count, candidate count, overlap and conflict comparisons are bounded;
dense ambiguous input fails rather than silently discarding candidates.
