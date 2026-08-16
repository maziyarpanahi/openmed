# Synthetic clinical fixture generator

`openmed.eval.clinical_fixtures` provides small, deterministic documents for
offline extraction-profile evaluation. The generator uses only the Python
standard library; it does not download models, terminology, or datasets and it
does not modify global random state.

These are synthetic evaluation inputs, not clinical ground truth, a medical
device, or a substitute for qualified clinical judgment.

## Generate fixtures

```python
from openmed.eval.clinical_fixtures import generate_fixtures

fixtures = generate_fixtures(
    profiles=("progress_note", "radiology_report"),
    seed=17,
)

fixture = fixtures[0]
document = fixture.text  # Pass to a local extraction runner.
gold_spans = fixture.gold_spans
expected_fields = fixture.expected_structured_fields
```

The canonical profiles are:

| Profile | Coverage |
|---|---|
| `progress_note` | history, negation, historical context, assessment, and plan |
| `radiology_report` | indication, modality, anatomy, findings, and uncertainty |
| `lab_report` | coded test, quantity, unit, and interpretation |
| `discharge_summary` | diagnosis, absent finding, medication, and follow-up |
| `pathology_report` | specimen, microscopy, and diagnostic uncertainty |

`generic`, `clinical_note`, `radiology`, `lab`, `discharge`, `progress`, and
`pathology` are accepted as short aliases. A profile-specific derivation of the
requested seed makes each document stable even when the order of a selected
profile list changes.

## Gold contract

Each `GoldSpan` contains `start` and `end` character offsets, a label, its
section, assertion axes, and an optional `CodedValue`. It intentionally does
not require a copied mention string for scoring. `fixture.span_text(span)` is
available when a local model test needs the in-memory substring.

Codes use the `openmed.synthetic` system and local code tokens. They exercise
code-system and code propagation without bundling a restricted terminology
vocabulary or making a network call.

`ExpectedField` records link structured output expectations to span IDs. This
keeps field assertions traceable without duplicating source text. The fixture
validates that section ranges, span ranges, and field references are
consistent when it is created.

## Privacy-safe artifacts

`fixture.to_dict()` and `fixture.to_json()` omit the document and span text by
default. They retain only offsets, labels, assertion axes, coded values, field
references, synthetic metadata, and a `sha256:` document fingerprint; scalar
field values are also omitted. This is the safe form for reports, logs, and
audit artifacts:

```python
safe_report = fixture.to_dict()
assert "text" not in safe_report
```

`include_text=True` is an explicit local round-trip opt-in that also retains
scalar expected-field values. Do not use that form for reports or logs. The
committed tests and generated metadata are synthetic-only and mark
`synthetic=True` and `phi=False`.

The generator is an evaluation aid only. It does not certify privacy, coding
accuracy, clinical safety, or production model behavior.
