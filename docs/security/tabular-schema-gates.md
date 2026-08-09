# Tabular schema-drift privacy gates

OpenMed can compare each incoming tabular batch with a versioned schema
contract before a privacy-sensitive release. The comparison is deterministic,
local-only, and does not require a network call or row materialization.

## Define a contract

Use a stable `field_id` when a logical column may be renamed. The identifier
must be supplied by the application and must not be derived from a row value.

```python
from openmed.risk import SchemaContract, SchemaField

contract = SchemaContract(
    version="v1",
    columns=(
        SchemaField(
            "subject_token",
            "string",
            role="direct_identifier",
            field_id="subject",
        ),
        SchemaField(
            "age_band",
            "integer",
            role="quasi_identifier",
            field_id="age",
        ),
        SchemaField(
            "measure",
            "float64",
            nullable=True,
            role="sensitive",
            field_id="measure",
        ),
    ),
)
```

An incoming schema can be a sequence of `SchemaField` objects, a mapping of
column names to field definitions, or a JSON-compatible mapping with
`columns`/`fields` and an optional `version`/`schema_version`.

## Compare and gate

```python
from openmed.risk import compare_schema_drift, enforce_schema_contract

report = compare_schema_drift(contract, incoming_schema)
ci_evidence = report.to_dict()  # aggregate counts only
report.raise_if_blocked()

# Or compare and raise in one operation:
enforce_schema_contract(contract, incoming_schema)
```

The report classifies added, removed, renamed, type-changed, nullability-
changed, and role-changed columns. Its serialized form contains only the
contract version, status booleans, and integer counts. It does not contain
column names, field identifiers, row values, or sample data. A
`SchemaDriftError` also contains counts only, and exposes the report through
its `report` attribute for CI handling.

## What blocks a privacy release

The gate blocks when any of the following is observed:

- an incoming version is explicitly present and does not match the contract;
- a role changes into or out of `direct_identifier`, `quasi_identifier`, or
  `sensitive`;
- a protected, unknown-role, or custom-role column is added or removed; or
- a protected, unknown-role, or custom-role column is renamed, changes type,
  or changes nullability.

Changes limited to `non_sensitive` and `excluded` roles remain visible in the
counts but do not block this privacy gate. Applications with stricter schema
compatibility requirements can block on any non-zero drift count as an
additional CI policy.

Without a stable `field_id`, a name change is conservatively classified as an
added column plus a removed column. The matcher never guesses that two
different names represent the same logical field.

Keep the full contract, including its column names, inside the application's
protected configuration boundary. Publish only the counts-only report as CI
evidence, and never place raw tabular values in logs, exception text, reports,
or committed fixtures.
