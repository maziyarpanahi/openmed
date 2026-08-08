# Clinical NLI verification

OpenMed exposes a small, backend-neutral natural-language-inference (NLI)
stage for checking whether a generated or grounded claim is supported by a
source span:

```python
from openmed.clinical.nli import nli, verify

pair = nli(
    "Synthetic patient has pneumonia.",
    "The patient has pneumonia.",
)

checks = verify(
    ["The patient has no pneumonia.", "The patient has pneumonia."],
    "Synthetic patient has pneumonia.",
)
```

`nli` always returns a mapping with `label` (`entailment`, `contradiction`, or
`neutral`) and a finite `score` in `[0, 1]`. The default backend is a
deterministic, dependency-free heuristic for offline development and synthetic
fixtures. It is intentionally conservative and is not a trained clinical
model.

`verify` can be called from a future `verify=True` option on a summarization or
grounding stage. It evaluates every claim and returns its original claim and
source together with the NLI result and an explicit `contradicted` flag. A
contradiction is surfaced for review; it is never silently dropped. A local
trained MLX head can replace the backend through the `backend=` argument or
`set_default_backend()` while preserving the `nli` and `verify` APIs.

## MedNLI data policy

MedNLI is DUA-gated and eval-only. The BigBio mirror is represented by a gated
stub; OpenMed does not bundle, download, or use MedNLI at runtime. Authorized
evaluation code must provide its own approved local access through the existing
eval-only dataset boundary. No MedNLI records or model weights belong in the
repository.
