# FHIR profile-declaration checks

`check_profile_declarations()` audits the `meta.profile` arrays of synthetic
or caller-owned FHIR resources against an explicit local catalog. The check is
deterministic and offline: canonical URLs are compared as strings, and no
profile package, FHIR server, or terminology service is contacted.

## Use an explicit catalog

Catalog entries identify the canonical URL, targeted resource type, and FHIR
release. R4 and R5 aliases such as `4.0.1` and `5.0.0` are accepted.

```python
from openmed.clinical.exporters.fhir import check_profile_declarations

catalog = {
    "https://synthetic.example/fhir/StructureDefinition/patient": {
        "resource_type": "Patient",
        "fhir_versions": ["R4", "R5"],
    }
}

findings = check_profile_declarations(
    {
        "resourceType": "Patient",
        "meta": {
            "profile": [
                "https://synthetic.example/fhir/StructureDefinition/patient"
            ]
        },
    },
    catalog,
    fhir_version="R4",
)
assert findings == []
```

The catalog can also be supplied as `profile_catalog`, `profile_registry`, or
`canonical_urls`. A resource-type-to-URL mapping is accepted for small
allowlists. `ProfileDeclarationSpec` is available when a typed immutable
catalog entry is preferable.

## Findings

The checker classifies missing, duplicate, unknown, resource-type-mismatched,
FHIR-release-mismatched, and validation-mode-mismatched declarations. Every
finding contains a safe structural expression and a generic diagnostic; it
does not include the declared canonical URL, resource ID, profile version, or
any other resource value. Missing declarations are required by default. Set
`require_profile=False` when a caller wants to inspect only resources that
already declare profiles, while `expected_profiles` still enforces explicitly
required profiles.

For a FHIR-native report, use `validate_profile_declarations()`:

```python
from openmed.clinical.exporters.fhir import validate_profile_declarations

outcome = validate_profile_declarations(resource, catalog, fhir_version="R5")
blocking = [
    issue for issue in outcome["issue"] if issue["severity"] in {"fatal", "error"}
]
```

This is an export consistency guard, not a complete FHIR conformance
validator, compliance certification, or clinical decision guarantee. Keep
catalogs and checked resources synthetic or apply the caller's data-access and
privacy policy before invoking it.
