# Importless plugin quarantine report

`openmed.plugins.quarantine` evaluates static plugin metadata before a caller
allows any third-party entry point to load. It does not enumerate installed
packages, resolve or import entry points, read credentials, contact package
indexes, or make network calls.

## Build a report

Pass one metadata mapping or an iterable of mappings to
`build_quarantine_report`:

```python
from openmed.plugins.quarantine import build_quarantine_report

report = build_quarantine_report(
    [
        {
            "name": "local-recognizer",
            "api_version": "1.0.0",
            "capabilities": ["recognizer"],
        },
        {
            "name": "optional-exporter",
            "api_version": "1.0.0",
            "capabilities": ["exporter"],
            "disabled": True,
        },
    ]
)

report.available       # accepted static metadata
report.disabled        # explicitly disabled metadata
report.quarantined     # malformed, incompatible, or duplicate metadata
report.to_dict()       # deterministic, JSON-compatible output
```

The evaluator accepts the OpenMed plugin API major version (`1`) and the
capabilities `recognizer`, `anonymizer_provider`, `exporter`,
`interop_adapter`, and `language_pack` by default. These values come directly
from the existing OpenMed plugin SDK contract so the preflight cannot drift
from runtime discovery. `sdk_version` and `kind` are accepted as compatibility
aliases for `api_version` and `capabilities`. Existing component metadata is
identified by its stable `plugin_id:component_id`; a human-readable `name`
never overrides those identifiers.

## Categories and safety

Each record has one stable category:

| Category | Meaning |
| --- | --- |
| `available` | The API version and capability declaration are compatible. |
| `disabled` | The metadata explicitly sets `disabled: true`, `enabled: false`, or `state: "disabled"`. |
| `quarantined` | Metadata is malformed, targets an unsupported API major, declares an unsupported capability, or shares a name with another available candidate. Every colliding candidate is quarantined. |

Quarantine reasons are machine-readable (`invalid_api_version`,
`missing_capabilities`, `unsupported_capability`, `unsupported_api_version`,
`duplicate_name`, and `invalid_metadata`). Messages are fixed registry-authored
text and never interpolate input values. Reports retain only normalized API and
capability values; unknown fields such as descriptions, credentials, and
secrets are ignored. Names that are not compact public identifiers are emitted
as a stable digest instead of raw text. Prerelease and build-version suffixes
affect the metadata digest but are not emitted in the report. Every record also
carries a SHA-256 metadata digest so duplicate classification and output order
do not depend on the input order.

This report is a preflight classification, not a sandbox or a compatibility
guarantee. A caller must still decide whether and how to load an `available`
plugin. The existing lazy registry remains responsible for runtime component
loading and protocol validation.

One report accepts at most 10,000 metadata records and 64 capability
declarations per record. Larger or non-terminating iterables fail closed as
quarantined metadata instead of being materialized without a bound.
