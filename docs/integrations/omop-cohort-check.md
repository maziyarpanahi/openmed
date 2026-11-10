# OMOP Cohort Export Check

OpenMed provides a local-only validator for the relationship, vocabulary, and
provenance invariants used by its OMOP cohort exports. It accepts the
`OmopCdmTables` returned by the loader or a mapping of table names to row
iterables:

```python
from openmed.interop.omop import load_grounded_notes
from openmed.interop.omop_cohort_check import validate_omop_cohort_export

tables = load_grounded_notes(synthetic_notes)
report = validate_omop_cohort_export(tables)

if not report.is_valid:
    print(report.to_dict())
```

The check is deterministic and makes no network call. It verifies primary-key
uniqueness, person/visit/note relationships, concept references, vocabulary
metadata for source-to-target mappings, and the OpenMed NOTE and NOTE_NLP
provenance chain when those fields are present. Identifiers must be lossless,
signed 64-bit integers or integer strings. Null, fractional, out-of-range, and
duplicate primary keys are invalid; duplicate keys are not available as
relationship targets. Provenance hashes must be lowercase, 64-character
SHA-256 values.

Reports contain table and column names, failure reasons, counts, and stable
`sha256:` row fingerprints. They do not contain source identifiers, note text,
or other row values. Invalid iterators, cyclic or excessively nested row
values, and values that fail during fingerprinting produce bounded errors
without copying the source value. Fingerprints can correlate identical rows
and should be handled as diagnostic metadata rather than anonymous data. The
validator is a structural quality check; it is not a compliance certification
or a clinical decision guarantee.

For a fail-closed boundary, use the assertion helper:

```python
from openmed.interop.omop_cohort_check import assert_valid_omop_cohort_export

assert_valid_omop_cohort_export(tables)
```

The resulting `OmopCohortExportValidationError` includes the aggregate report
and an exception message with only the number of failed row-level invariants.
