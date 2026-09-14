# Empty-evidence summary refusal

OpenMed's local summary boundary must not produce plausible clinical prose when
the approved evidence collection is empty or every candidate is malformed or
explicitly excluded. `openmed.clinical.summary_empty_evidence` provides a
deterministic gate for that boundary.

The gate is local-only. It does not load a model, make a network request, or
decide whether a clinical finding is true. It accepts an already-filtered
collection, removes invalid or explicitly unapproved candidates, and invokes a
caller-supplied generator only when at least one candidate remains.

## Guard a local generator

```python
from openmed.clinical import (
    SummaryEmptyEvidenceRefusal,
    guard_summary_generation,
)


def local_generator(evidence: tuple[object, ...]) -> str:
    return f"generated from {len(evidence)} approved record(s)"


result = guard_summary_generation(
    [{"approved": False, "text": "synthetic source value"}],
    local_generator,
)

assert isinstance(result, SummaryEmptyEvidenceRefusal)
assert result.refusal_code == "empty_approved_evidence"
assert result.to_dict()["input_evidence_count"] == 1
```

The generator is never called for `None`, an empty collection, a collection of
malformed records, or a collection whose records are all explicitly rejected.
When some candidates are valid, the generator receives a tuple containing only
the candidates that survived the gate.

## Refusal contract

`SummaryEmptyEvidenceRefusal.to_dict()` and `.to_json()` contain the stable
`empty_approved_evidence` code, aggregate input/approved/excluded counts, and
fixed human-review guardrails. They never copy source text, extracted values,
identifiers, or upstream exception messages. The refusal is not a diagnosis,
autonomous clinical decision, or compliance certification.

For callers that prefer an exception boundary,
`require_summary_evidence()` returns the filtered tuple or raises
`SummaryEmptyEvidenceError`. The exception message contains only the fixed
`summary_refused_empty_approved_evidence` code; the value-free refusal report
is available as `error.refusal`.

Evidence without an explicit approval field is treated as already approved
because the function parameter is an approved-evidence boundary. Callers that
want explicit approval should provide `approved: True` and reject records
before invoking the local generator. An explicit `approved: False`, a rejected
review status, or `valid: False` is always excluded.
