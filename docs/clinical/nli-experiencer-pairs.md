# Experiencer-aware clinical NLI pairs

Clinical natural-language inference can produce a false patient claim when a
premise is about a family member or caregiver. The
`openmed.clinical.nli_experiencer_pairs` module carries the subject of each
pair side into a small, backend-neutral record and applies a conservative
entailment gate.

The NLI-specific vocabulary is deliberately explicit:

| Class | Meaning |
| --- | --- |
| `patient` | The statement is about the patient represented by the note. |
| `family` | The statement is about a relative or family history. |
| `caregiver` | The statement is about a caregiver or care provider. |
| `unknown` | The subject is missing, unresolved, or not safely classifiable. |

Known matching classes are compatible. Different known classes are
`incompatible`, and any `unknown` side is `unresolved`. Both states block an
entailment decision and set `label` to `review_required`. Unknown is not a
wildcard: treating it as compatible could turn an unresolved family or
caregiver statement into a patient assertion.

## Build a pair

Pair sides can be strings with explicit metadata or mappings carrying `text`
and an `experiencer` field. The following uses synthetic values:

```python
from openmed.clinical import build_experiencer_nli_pair

pair = build_experiencer_nli_pair(
    {"text": "synthetic family finding", "experiencer": "family"},
    {"text": "synthetic patient claim", "experiencer": "patient"},
    predicted_label="entailment",
)

pair.entailment_allowed  # False
pair.experiencer_compatibility  # "incompatible"
pair.label  # "review_required"
```

Missing metadata is retained as `unknown` rather than silently defaulting to
the patient. Existing `ClinicalAssertion` or experiencer-assignment records
can be supplied as side metadata; the older `other` class is treated as
`unknown` because it is not safe to infer that every non-patient subject is a
caregiver.

## Privacy and review boundary

`to_model_input()` and `to_text_pair()` are the explicit boundary for sending
text to a caller-controlled local model. `to_dict()`, `to_audit_dict()`,
`to_json()`, `pair_id`, and `repr(pair)` contain only text hashes, lengths,
offsets, controlled classes, and decisions. They do not contain source text or
subject cue surfaces. Validation errors likewise avoid echoing caller values.

Construction performs no model loading, filesystem access, network call, or
telemetry. Pair records are deterministic and remain assistive metadata; a
qualified clinician must review NLI outputs before clinical use.
