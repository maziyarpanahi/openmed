# Clinical NLI verification

See [local backends and typed outcomes](local-backends.md) for executable
synthetic provider wiring and the complete exception/refusal tables.

OpenMed exposes a small, backend-neutral natural-language-inference (NLI)
stage for checking whether a generated or grounded claim is supported by a
source span:

```python
from openmed.clinical.nli import nli, verify

pair = nli(
    "Synthetic patient has pneumonia.",
    "The patient has pneumonia.",
    backend="heuristic",  # explicit development-only option
)

checks = verify(
    ["The patient has no pneumonia.", "The patient has pneumonia."],
    "Synthetic patient has pneumonia.",
    backend="heuristic",
)
```

`nli` returns a value-free mapping with `label`, a finite `score` in `[0, 1]`,
and `backend_id`. Labels are `entailment`, `contradiction`, `neutral`, or
`abstention`. The default selects a released local NLI checkpoint. Until a
checkpoint with an immutable revision, class map, and calibrated thresholds is
registered, it fails closed. The heuristic is available only by explicitly
selecting `backend="heuristic"`; it is not a trained clinical model.

`verify` evaluates every claim and returns only its index, label, score,
backend id, and contradiction or review flags. It never returns premise or
hypothesis text. Structured numeric and medication-status prechecks run before
the model when both records provide those fields. The local backend accepts
only cached PyTorch or ONNX sequence-classification artifacts. Model class
meanings and calibrated thresholds come from the pinned release metadata; the
runtime never guesses them or falls back to a remote service.

The encoder evaluates complete tokenized pairs only. It abstains without model
inference if the pair exceeds 512 tokens or a smaller tokenizer/model limit;
it never silently truncates the source or claim before issuing a decision.

## Opt-in summarization and grounding hooks

`summarize()`, `summarize_deidentified()` and `ground()` accept keyword-only
`verify=True`, an explicit local NLI registry alias, or a caller-owned NLI
backend. Verification is off by default: existing output text, concepts,
iteration and `to_dict()` serialization remain unchanged, and no NLI backend
is resolved. `verify=True` selects the configured NLI backend; missing local
release metadata or cached artifacts raise the existing `LocalNLIError`, with
no heuristic or remote fallback. Use `verify="heuristic"` explicitly for
synthetic development checks.

For example, with a caller's existing `DeidentificationResult`:

```python
from openmed.clinical.summarize import summarize

result = summarize(
    safe_note,
    model=local_summarizer,
    verify="heuristic",
)
checks = [check.to_dict() for check in result.verification]
```

Summarization verifies the existing segmenter's claim slices against the
complete de-identified source as an explicit source span. The original raw
note is never sent to NLI. Both leakage guards must pass first. Every claim
is retained: contradiction, neutral and abstention outcomes require review,
and segmentation's review flags remain set even if NLI returns entailment.
The existing two-item summary/leakage-check unpacking contract is unchanged.

For grounding, enable the same hook on a caller-owned local snapshot:

```python
from openmed.clinical.grounding import ground

result = ground(
    source_text,
    systems=["icd10cm"],
    snapshot=local_snapshot,
    verify="heuristic",
)
checks = [check.to_dict() for check in result.verification]
```

Grounding verifies each retained concept's display against the exact original
raw-text slice at its source offsets. It does not send the entire document or
invent context for an entity. A missing display/code, absent source text,
surface mismatch or out-of-bounds offset yields abstention without inference.
Entity-list inputs lack the complete source text and therefore abstain even
when their supplied surface can be grounded. No concept, code or assertion
is dropped or rewritten. A display/negation mismatch is assistive review
evidence, not an automatic coding decision.

The additive `verification` field contains `ClaimVerification` records in
summary-claim order or grounding-concept order. Their dictionaries contain
only indices, labels, finite scores, controlled backend/review metadata,
half-open offsets and domain-separated SHA-256 digests. They contain no source
or claim text. Grounding JSON round trips preserve these records and reject
rebound offsets, surfaces or claim displays. Summary records likewise bind
their claim digest to the output slice. The ordinary result's summary and
concept surfaces remain protected application data; serialize the verification
records separately for value-free audit evidence. NLI network access is blocked
even for caller-owned backends, and local inference failures have value-free
messages. Backends, calibration thresholds and the clinical brief pipeline
are unchanged. These Python hooks add no REST, MCP or CLI operation.

Controlled verification refusals discard private provider/encoding exception
context. When a direct de-identified summary supplies a cancellation context,
the hook checks it before and after each local NLI call and emits no late result.
## Local CLI verification

`openmed nli verify --source synthetic-note.txt --claims synthetic-claims.json
--json` selects the released local checkpoint and fails closed if one is
unavailable. For the explicit development baseline, use:

```bash
openmed nli verify --source synthetic-note.txt --claims synthetic-claims.json \
  --backend heuristic --json
```

The source must be a nonempty regular UTF-8 file of at most 16 KiB. The claims
file must be a regular UTF-8 JSON file of at most 64 KiB containing a nonempty
array of at most 128 nonempty strings, each at most 4 KiB. Final symlinks and
special files are refused. The command emits only claim index, canonical label,
finite score, fixed backend ID and contradiction/review flags. It writes no
output files and never prints source or claim text. Exit `0` requires every
claim to be entailed; other valid labels return exit `1` with an `ok: true`
evidence envelope. This is advisory verification requiring clinical review.

Applications may explicitly provide an installed local factory with
`--backend-factory trusted_module:create`. The factory takes no arguments and
returns a callable or object with `predict(premise, hypothesis)`, using the
existing Python NLI contract. Its console backend ID is always
`caller-supplied-local`. The factory is trusted local code running in this
process, not a plugin sandbox; it must not spawn external processes or remote
tools. Factory import and inference run under the existing Python socket guard,
with Python/native stdout, stderr, logging and warnings suppressed. The CLI
does not isolate arbitrary native networking or writes from trusted code.
Factories cannot implicitly select the built-in heuristic; that baseline
requires `--backend heuristic`. See the [machine contract](../cli/machine-contract.md#guarded-summary-and-nli-commands)
for controlled failure codes.

## MedNLI data policy

MedNLI is DUA-gated and eval-only. The BigBio mirror is represented by a gated
stub; OpenMed does not bundle, download, or use MedNLI at runtime. Authorized
evaluation code must provide its own approved local access through the existing
eval-only dataset boundary. No MedNLI records or model weights belong in the
repository.
