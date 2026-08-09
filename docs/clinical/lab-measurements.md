# Lab Measurement Normalization

`openmed.clinical.normalize_lab_measurement` turns an already extracted
laboratory row into a deterministic typed record. It normalizes common units
through the local UCUM subset, parses numeric reference ranges, and derives a
descriptive interpretation (`low`, `normal`, `high`, `critical`, or `unknown`).
It is review support, not a diagnosis or clinical decision.

## Normalize an extracted row

```python
from openmed.clinical import normalize_lab_measurement

record = normalize_lab_measurement(
    {
        "analyte": "Glucose",
        "value": 120,
        "unit": "mg/dL",
        "reference_range": "70-99 mg/dL",
        "flag": "H",
        "start": 12,
        "end": 34,
    }
)

record["value"]             # 120.0, in the reported unit
record["unit"]              # "mg/dL"
record["canonical_value"]   # 1.2, in the canonical unit
record["canonical_unit"]    # "g/L"
record["interpretation"]    # "high"
record["source_offsets"]    # {"start": 12, "end": 34}
```

Scalar values and embedded unit strings are also accepted:

```python
record = normalize_lab_measurement(
    "4.2 mmol/L",
    reference_range="3.5-5.1",
    source_offsets=(20, 31),
)
```

Reference ranges use inclusive bounds by default. One-sided forms such as
`"<5 mg/dL"`, `"<=5 mg/dL"`, `">10"`, and `">=10"` preserve their boundary
semantics. A range without a unit can be compared in the measurement's
reported unit because it is attached to that measurement; no cross-unit
conversion is inferred from an unrelated range.

## Unknown units fail closed

Missing, ambiguous, and unrecognized units produce `status="unknown_unit"`.
The record retains the typed source value and supplied unit evidence, but
`canonical_value` and `canonical_unit` remain `None`, and no unit is guessed.
An explicit originating-laboratory flag such as `"H"` is retained as
interpretation evidence even when numeric comparison is unavailable.

Invalid numeric values and malformed ranges have separate `invalid_value` and
`invalid_range` statuses. Callers should route those records to review rather
than coercing them.

## Provenance and privacy

`source_offsets` uses half-open character offsets and is the only source
location retained by the normalizer. The result contains normalized numeric
fields, units, range bounds, qualifiers, and safe provenance markers; it does
not copy the source measurement string into provenance, logs, or exceptions.
The implementation is rules-based, deterministic, local-first, and performs no
network call or wall-clock lookup.
