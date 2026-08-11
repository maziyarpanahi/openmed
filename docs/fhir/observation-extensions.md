# Observation Extension Validation

`check_observation_extensions()` performs a small, deterministic validation
pass over the `extension` array of a FHIR `Observation`. It is deliberately
offline: it does not resolve canonical URLs, download profiles, call a
terminology service, or contact a FHIR server.

The checker accepts only the built-in allowlist or a caller-supplied explicit
allowlist. The built-in rules cover the OpenMed unknown-state extension and the
privacy-safe nested evidence shapes emitted by the FHIR helpers. Every rule
specifies its value type or nested shape, cardinality, and supported FHIR mode.

## Explicit unknown states

An unknown observation state must be represented by the dedicated extension and
an explicit `valueCode`:

```python
from openmed.clinical.exporters.fhir import (
    OBSERVATION_UNKNOWN_STATE_EXTENSION_URL,
    check_observation_extensions,
)

observation = {
    "resourceType": "Observation",
    "extension": [
        {
            "url": OBSERVATION_UNKNOWN_STATE_EXTENSION_URL,
            "valueCode": "asked-unknown",
        }
    ],
}

assert check_observation_extensions(observation, fhir_version="R5") == []
```

The built-in states are `unknown`, `not-asked`, `asked-unknown`, and
`temporarily-unknown`. Inferred, predicted, derived, and estimated states are
rejected. Omitting the extension is not treated as an unknown state.

## Caller-supplied allowlists

For a projection-specific extension, provide a URL-to-spec mapping. Value type
names use FHIR `value[x]` field names. `max_occurs="*"` means unbounded.

```python
from openmed.clinical.exporters.fhir import check_observation_extensions

rules = {
    "https://synthetic.example/fhir/StructureDefinition/review-flag": {
        "value_types": ["valueBoolean"],
        "min_occurs": 0,
        "max_occurs": 1,
        "fhir_versions": ["R4", "R5"],
    }
}

findings = check_observation_extensions(
    {
        "resourceType": "Observation",
        "extension": [
            {
                "url": (
                    "https://synthetic.example/fhir/StructureDefinition/"
                    "review-flag"
                ),
                "valueBoolean": True,
            }
        ],
    },
    allowed_extensions=rules,
)
assert findings == []
```

The checker rejects unknown URLs, unsupported value types, multiple values,
mixed value-and-nested shapes, cardinality violations, and fields that suggest
inferred content. It never includes supplied extension values in findings.
Findings use FHIRPath-style element expressions and can be passed directly to
the shared `OperationOutcome` builder. For a FHIR-native result, use
`validate_observation_extensions()`.

Both `R4` and `R5` are supported through the `fhir_version` parameter. Dotted
release aliases such as `4.0.1` and `5.0.0` are accepted for convenience. This
checker is an assistive structural guard only; it is not a complete FHIR
conformance validator or a clinical decision.
