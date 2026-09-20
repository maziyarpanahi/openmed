# Assertion-aware clinical NLI pairs

Clinical natural-language inference must keep the source assertion state next
to the premise and hypothesis. A negated, uncertain, or conditional finding
is not equivalent to affirmed evidence, even when the surface concept is the
same. OpenMed therefore requires complete assertion metadata on both sides of
every constructed pair.

These pairs are deterministic assistive inputs for qualified human review.
They do not diagnose, recommend treatment, or make an autonomous clinical
decision.

## Construct a pair

Use the existing `ClinicalAssertion` axes or a mapping with equivalent fields.
The required axes are:

| Axis | Allowed values | NLI-facing aliases |
| --- | --- | --- |
| `negation` | `affirmed`, `negated` | `negated` / `is_negated` booleans |
| `certainty` | `certain`, `uncertain` | `uncertainty`, `uncertain` booleans |
| `temporality` | `recent`, `historical`, `hypothetical` | `hypothetical` / `is_hypothetical` booleans |

Both the premise and the hypothesis must provide these axes. Missing metadata
is rejected rather than defaulted. If redundant forms are supplied, they must
agree; for example, `temporality="recent"` with `hypothetical=True` is
rejected.

```python
from openmed.clinical import build_nli_pair

pair = build_nli_pair(
    {
        "text": "synthetic source finding",
        "assertion": {
            "negation": "negated",
            "uncertainty": True,
            "hypothetical": False,
        },
        "start": 12,
        "end": 35,
    },
    {
        "text": "synthetic target claim",
        "assertion": {
            "negation": "affirmed",
            "certainty": "certain",
            "temporality": "recent",
        },
    },
)
```

`pair.premise_assertion` and `pair.hypothesis_assertion` expose the normalized
axes. The pair does not infer an NLI label: a downstream local model or
rule-based verifier must retain these fields when interpreting its result.
Every pair is marked `requires_clinician_review=True`.

## Privacy boundary and offline behavior

The pair retains premise and hypothesis text only for the caller-controlled
model-input boundary:

```python
model_input = pair.to_model_input()
# model_input keeps premise_assertion and hypothesis_assertion beside the text.
```

For a backend that accepts only a text tuple, use `pair.to_text_pair()` and
retain the assertion sidecars from the pair next to the backend result.

`to_dict()`, `to_audit_dict()`, `to_json()`, and `repr(pair)` never include raw
text. They contain only text hashes, lengths, optional source offsets, the
assertion axes, a deterministic pair identifier, and the review advisory.
Validation exceptions identify only the side and field that failed; they do
not echo the supplied text or invalid value. Keep raw text inside the local
review boundary and use the audit representation for logs, reports, and
persisted artifacts.

Construction performs no model loading, filesystem access, network call,
logging, telemetry, or remote provider lookup. Batch construction preserves
input order and produces the same pair identifiers and JSON for the same
inputs.

## Limitations

Assertion metadata is an evidence safeguard, not a clinical truth guarantee.
The ConText layer and any upstream extractor remain responsible for producing
the metadata. A qualified reviewer must inspect source evidence and the model
or rule-based NLI result before downstream use. This feature does not bundle
clinical corpora, restricted terminology, credentials, or proprietary model
services.

## Validation

The focused suite uses synthetic values only and runs offline:

```text
.venv/bin/python -m pytest tests/unit/clinical/test_nli_assertion_pairs.py -q
```

When the optional checkout-local virtual environment is unavailable, the
equivalent command is `python3 -m pytest ...`.
