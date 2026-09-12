# Model-provenance diff reports

`openmed.eval.model_provenance_diff` compares two synthetic evaluation run
manifests without carrying prompts, notes, generated clinical text, or other
free-form run metadata into the report. The comparison is local-only and uses
deterministic canonical JSON hashing; it does not download or resolve any
artifact.

## Manifest shape

Each manifest declares immutable provenance for the model, tokenizer, policy,
and fixture set. Every component has a safe `fingerprint` and `version`.
Evaluation slices can be names, or declarations with their own optional
fingerprint and version:

```python
from openmed.eval.model_provenance_diff import diff_model_provenance

before = {
    "model": {"fingerprint": "sha256:model-a", "version": "v1"},
    "tokenizer": {"fingerprint": "sha256:tokenizer-a", "version": "v3"},
    "policy": {"fingerprint": "sha256:policy-a", "version": "v2"},
    "fixtures": {"fingerprint": "sha256:fixtures-a", "version": "v4"},
    "evaluation_slices": [
        {
            "name": "baseline",
            "fingerprint": "sha256:baseline-a",
            "version": "v1",
        }
    ],
}

after = {**before, "model": {"fingerprint": "sha256:model-b", "version": "v2"}}
report = diff_model_provenance(before, after)
print(report.to_json())
```

The normalized report contains the before/after manifest fingerprints, changed
component names, fingerprint/version change reasons, and added, removed, or
changed slice declarations. It never copies unknown manifest fields. Inputs
with a free-form value in a known provenance field fail with an exception that
identifies only the safe field name and never echoes the rejected value.

`report.changed`, `report.drift_detected`, and `report.has_drift` are equivalent
boolean checks. `report.to_dict()` is JSON-ready and deterministic, while
`write_model_provenance_manifest` and `load_model_provenance_manifest` provide
local JSON persistence when a run manifest needs to be archived.

This is an audit aid for comparing evaluation inputs, not a compliance
certification or a clinical decision guarantee. Keep actual prompts, notes, and
generated clinical text outside manifests and reports.
