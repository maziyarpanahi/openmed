# Clinical NLI verification

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
