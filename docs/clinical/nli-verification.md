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

## MedNLI data policy

MedNLI is DUA-gated and eval-only. The BigBio mirror is represented by a gated
stub; OpenMed does not bundle, download, or use MedNLI at runtime. Authorized
evaluation code must provide its own approved local access through the existing
eval-only dataset boundary. No MedNLI records or model weights belong in the
repository.
