# Unit-display normalization audit

OpenMed provides a deterministic, offline audit for localized unit display
labels. It checks a synthetic or caller-owned locale table against the allowed
canonical unit codes and an explicit alias table. The audit is a consistency
check for exports and review tooling; it is not a compliance certification or a
clinical decision guarantee.

## Run an audit

The input table is keyed by locale and then by canonical unit code. Each value
is the display label that the locale would render. Alias tables map a localized
label back to its canonical code.

```python
from openmed.clinical.units.display_audit import audit_unit_display_labels

report = audit_unit_display_labels(
    {
        "en": {"mg/dL": "mg/dL", "Cel": "Celsius"},
        "es": {"mg/dL": "mg por decilitro", "Cel": "℃"},
    },
    ("Cel", "mg/dL"),
    {
        "en": {"Celsius": "Cel"},
        "es": {
            "mg por decilitro": "mg/dL",
            "℃": "Cel",
        },
    },
)

report.summary
# {
#   "canonical_units": 2,
#   "conflict": 0,
#   "display_labels": 4,
#   "duplicate": 0,
#   "issues": 0,
#   "locales": 2,
#   "missing": 0,
# }
```

When `alias_tables` is omitted, the registered local clinical normalization
lexicons provide the alias tables. Supplying the mapping explicitly is
preferred for release audits because it makes the audit input self-contained.
Locale identifiers are normalized to their primary language subtag, so `fr-FR`
and `fr` cannot be silently audited as separate language packs.

## Findings and privacy

The report marks each finding as `missing`, `duplicate`, or `conflict`:

- `missing` means a canonical code has no non-empty display label in a locale.
- `duplicate` means one normalized display surface is used for more than one
  canonical code in the same locale.
- `conflict` means an alias table is internally inconsistent, a label resolves
  to a different code, a label references an unknown code, or a non-canonical
  label is absent from the explicit alias table.

`to_dict()`, `to_json()`, and `to_markdown()` contain counts, normalized locale
identifiers, issue reasons, and SHA-256 hashes of labels and codes. They never
include submitted display labels or canonical code strings. The reproducibility
hash covers that source-free payload, and repeated runs with the same inputs
produce byte-identical JSON.

The implementation performs no network calls, emits no logs, reads no wall
clock, and does not alter the registered lexicons. Keep source tables synthetic
or apply the caller's data-access and PHI handling policy before invoking the
audit.
