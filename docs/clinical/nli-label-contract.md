# Clinical NLI label contract

OpenMed exposes a backend-neutral result for clinical natural-language
inference (NLI). Every result has one of four canonical labels:

| Canonical label | Meaning for review |
| --- | --- |
| `entailment` | The evidence supports the claim. |
| `contradiction` | The evidence conflicts with the claim. |
| `neutral` | The evidence does not support or conflict with the claim. |
| `abstention` | The backend or policy cannot safely classify the pair; defer to review. |

These are assistive evidence labels. They do not diagnose, recommend treatment,
or make an autonomous clinical decision. A qualified clinician must review the
source evidence and claim before downstream use.

## Map a backend label

Backends use different names and class indexes. Declare the meaning of every
label before adapting a backend result; OpenMed never guesses the order of
numeric classes.

```python
from openmed.clinical import BackendLabelMapping, build_nli_result

mapping = BackendLabelMapping.from_mapping(
    {
        "class_0": "contradiction",
        "class_1": "neutral",
        "class_2": "entailment",
        "class_3": "abstention",
    },
    backend="local-nli",
)

result = build_nli_result(
    "class_2",
    mapping=mapping,
    raw_scores={
        "class_0": 0.02,
        "class_1": 0.08,
        "class_2": 0.88,
        "class_3": 0.02,
    },
)

assert result.label.value == "entailment"
assert result.score == 0.88
```

`BackendLabelMapping.resolve()` rejects a label that is absent from the
declared map. This fail-closed behavior prevents a provider change from being
silently interpreted as a different clinical state. Call
`validate_backend_label_mapping(..., require_complete=True)` when a backend is
expected to expose all four states. Three-state models may omit `abstention`
when an external policy supplies it.

## Score and privacy rules

`raw_scores` are optional metadata only. Each value must be a finite number in
the closed interval `[0, 1]`; booleans, NaN, infinities, logits, nested provider
payloads, and unknown score labels are rejected. Scores are normalized to
canonical label keys and frozen in the returned result. `to_dict()` and
`to_json()` contain only the schema version, backend identifiers, labels, and
bounded numeric metadata. They do not accept or emit premise, hypothesis,
source text, identifiers, or model payloads.

The module is dependency-free and local-first. Constructing a mapping or result
performs no model download, network call, logging, persistence, or remote
provider access. Supply model output from a separately controlled local
adapter, and keep any clinical source text in the caller's review boundary.

## Validation

Run the focused offline contract tests from the repository root:

```text
.venv/bin/python -m pytest tests/unit/clinical/test_nli_labels.py -q
```

The tests use synthetic labels and scores only. They cover all four states,
case and whitespace normalization, complete-map validation, deterministic
serialization, unknown-label rejection without echoing the input, and score
bounds.
