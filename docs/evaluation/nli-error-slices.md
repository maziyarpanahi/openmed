# Clinical NLI Error Slices

`openmed.eval.nli_error_slices` produces a deterministic, local-first report
for finding failures that an aggregate clinical NLI score can hide. It reports
five fixed clinical phenomena:

| Slice | What it isolates |
|---|---|
| `negation` | Negated versus affirmed clinical claims |
| `temporality` | Historical, current, or future claim context |
| `experiencer` | Patient, family, or other experiencer context |
| `numbers` | Numeric values, quantities, and dose-like distinctions |
| `medication_status` | Medication start, stop, hold, and active-status distinctions |

The report accepts only already-computed labels and opaque fixture identifiers.
It does not accept or render premise text, hypothesis text, examples, model
outputs, confidence traces, or exception messages. Fixture identifiers remain
available for human review, while all metrics are aggregate counts.

```python
from openmed.eval import (
    NLIErrorSliceCase,
    build_nli_error_slice_report,
)

report = build_nli_error_slice_report(
    [
        NLIErrorSliceCase(
            fixture_id="synthetic-negation-001",
            phenomena=("negation",),
            gold_label="entailment",
            predicted_label="contradiction",
        ),
        NLIErrorSliceCase(
            fixture_id="synthetic-medication-001",
            phenomena=("medication_status",),
            gold_label="neutral",
            predicted_label=None,
            abstained=True,
        ),
    ],
    fixture_set_id="synthetic-clinical-nli-v1",
    model_id="local-model-v1",
)

report.write_json("nli-error-slices.json")
report.write_markdown("nli-error-slices.md")
```

Each slice includes a full `entailment`/`contradiction`/`neutral` confusion
matrix with `abstain` as an additional prediction column, the retained fixture
IDs, scored and abstained counts, accuracy over scored cases, abstention rate,
and abstention coverage. Coverage is the fraction of cases with a usable
prediction; the report also exposes the complementary abstention rate.

The input order is canonicalized by fixture identifier and all five slices are
always emitted, including empty slices. Provenance contains a fixture-set
digest and an optional model digest; arbitrary model metadata is hashed before
serialization. The artifact declares that raw text and examples are
suppressed and that no network is required. No model or dataset download is
performed by the report builder.

This is evaluation evidence for human review, not a compliance certification or
an autonomous clinical decision guarantee. Use synthetic or separately
authorized fixtures only; restricted datasets and credentials are not bundled.
