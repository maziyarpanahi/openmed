# Clinical Refusal Reason Taxonomy

Guarded clinical surfaces can use the refusal taxonomy to report why a request
was declined without copying request text, note text, identifiers, or a
free-form safety message into an exception, log, or report. The taxonomy is a
deterministic local helper and performs no network calls.

## Stable categories

| Identifier | Meaning | Remediation hint |
| --- | --- | --- |
| `missing_evidence` | Required evidence was not available. | Provide the minimum required evidence and retry when it is available. |
| `policy_block` | An applicable safety or workflow policy prevents the operation. | Review the applicable policy and use an approved workflow. |
| `ambiguity` | The request cannot be resolved unambiguously. | Clarify the request or supply disambiguating context. |
| `unsupported_request` | The requested operation is outside the guarded surface's capabilities. | Use a supported clinical workflow or consult the capability guidance. |

Category identifiers are the stable contract. Upstream code is responsible for
choosing a category; this module does not inspect or classify arbitrary request
or clinical-note text.

## Building a refusal

```python
from openmed.clinical import RefusalCategory, build_refusal

refusal = build_refusal(RefusalCategory.MISSING_EVIDENCE)
refusal.to_dict()
# {
#     "category": "missing_evidence",
#     "count": 1,
#     "remediation_hint": (
#         "Provide the minimum required evidence and retry when it is available."
#     ),
# }
```

The same function accepts the canonical string identifier. Other values are
rejected with an error that does not echo the rejected input.

## Aggregating and serializing

Use `aggregate_refusals()` for a batch of category identifiers or counted
`RefusalReason` values. The output is ordered by the taxonomy, independent of
input order:

```python
from openmed.clinical import aggregate_refusals, serialize_refusals

report = aggregate_refusals(
    ["policy_block", "ambiguity", "policy_block"]
)
report.to_dict() == serialize_refusals(
    ["policy_block", "ambiguity", "policy_block"]
)
# True
```

The serialized report contains only two fields:

```json
{
  "counts": {
    "policy_block": 2,
    "ambiguity": 1
  },
  "remediation_hints": {
    "policy_block": "Review the applicable policy and use an approved workflow.",
    "ambiguity": "Clarify the request or supply disambiguating context."
  }
}
```

No free-form reason, request, note, identifier, traceback, or caller metadata is
retained by `RefusalReason` or `RefusalReport`. These outputs are review and
routing aids only; they do not make a clinical decision or provide a compliance
certification.
