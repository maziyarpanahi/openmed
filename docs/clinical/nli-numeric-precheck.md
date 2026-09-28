# Numeric contradiction precheck

`numeric_contradiction_precheck()` compares structured numeric evidence before
a clinical NLI model runs. It is deterministic, local-only, and limited to
exact value, unit-dimension, and reference-interval incompatibilities.

```python
from openmed.clinical.nli_numeric_precheck import numeric_contradiction_precheck

result = numeric_contradiction_precheck(
    {"measurement_key": "synthetic-measurement", "value": 10, "unit": "mg/dL"},
    {"measurement_key": "synthetic-measurement", "value": 0.2, "unit": "g/L"},
)

assert result.contradiction
assert result.inference_allowed is False
assert result.evidence[0].reason.value == "value_mismatch"
```

## Contract

Inputs are mappings or `NumericClaim` objects. A claim may carry an exact
`value` and `unit`, a `reference_interval` with `low`, `high`, and `unit`, or
one-sided `reference_low` / `reference_high` bounds. Optional
`measurement_key` values prevent accidental comparison of two different
measurements. Optional source offsets are returned as provenance.

The result status is one of:

- `compatible`: comparable structured evidence agrees, so model inference may
  continue.
- `contradiction`: exact values differ after unit normalization, dimensions are
  incompatible, or explicit intervals are disjoint. Model inference is
  withheld.
- `review_required`: a unit is absent, unknown, or ambiguous, or the two claim
  shapes cannot be compared safely. Model inference is withheld.
- `not_applicable`: both claims identify different measurements.

Unit conversion uses OpenMed's deterministic UCUM subset. A tiny fixed numeric
tolerance absorbs only floating-point conversion noise; it is not a clinical
tolerance.

## Privacy and clinical boundary

Reports contain reason codes, dimensions, offsets, and domain-separated
SHA-256 fingerprints. They do not contain source text, measurement identifiers,
raw values, or raw units. Errors are value-free. The precheck performs no
network call and reads no environment state.

The module never interprets a measurement as normal, abnormal, safe, or
actionable. Reference intervals are compared only for exact overlap. Results
are assistive consistency evidence and require the same clinical review as the
surrounding NLI workflow.
