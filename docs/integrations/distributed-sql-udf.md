# Distributed SQL De-identification UDF

OpenMed provides a Python entrypoint for distributed SQL UDF bridges that
execute code inside worker processes. The SQL function is logically scalar:

```sql
openmed_deidentify(text VARCHAR, profile VARCHAR) -> VARCHAR
```

The Python bridge should invoke that function in vectorized mode. Each worker
keeps one lazy OpenMed model loader and sends row windows through
`process_batch`, avoiding one model initialization and one inference call per
row. OpenMed does not ship a compiled engine plugin JAR; provisioning and
securing the engine-specific Python bridge remains an operator responsibility.

## Registration descriptor

The engine-neutral descriptor is available as
`OPENMED_DEIDENTIFY_DESCRIPTOR`:

```python
from openmed.integrations.distributed_sql_udf import (
    OPENMED_DEIDENTIFY_DESCRIPTOR,
)

descriptor = OPENMED_DEIDENTIFY_DESCRIPTOR
```

Its registration fields are:

```yaml
name: openmed_deidentify
language: python
entrypoint: openmed.integrations.distributed_sql_udf:deidentify_batch
arguments:
  - name: text
    sql_type: VARCHAR
    python_batch: texts
  - name: profile
    sql_type: VARCHAR
    python_batch: profiles
return_type: VARCHAR
vectorized: true
null_handling: called_on_null_input
default_batch_size: 64
```

Map the descriptor to the equivalent fields in the engine's Python UDF
registration system. In particular, configure the bridge to pass arrays of
`text` and `profile` values to `deidentify_batch`; the returned array aligns
one-to-one with the input rows. SQL `NULL` stays `NULL`, and an empty string
stays empty without loading the model.

## Worker lifecycle

For direct integration or an offline registration test, construct the callable
once during worker setup and reuse it for every vector window:

```python
from openmed.integrations.distributed_sql_udf import (
    DistributedSQLDeidentifyUDF,
    DistributedSQLUDFConfig,
)

openmed_deidentify = DistributedSQLDeidentifyUDF(
    config=DistributedSQLUDFConfig(
        default_profile="hipaa_safe_harbor",
        batch_size=64,
    )
)

redacted = openmed_deidentify(
    [
        "Patient Jane Roe has hypertension.",
        None,
        "",
    ],
    ["hipaa_safe_harbor", None, "hipaa_safe_harbor"],
)
```

The module-level `deidentify_batch` entrypoint uses the same process-local
worker pattern automatically. `deidentify(text, profile)` is also available
for scalar-only bridges, but a bridge that calls Python once per row cannot
benefit from vector batching.

Install the OpenMed model artifact on every worker before accepting queries if
the deployment must remain fully offline. Raw input text is not logged by this
adapter; engine query logs, failure capture, and spill configuration should be
reviewed separately so they do not retain PHI.
