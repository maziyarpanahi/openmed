# Pandas-on-Spark De-identification

OpenMed registers pandas-on-Spark DataFrame and Series accessors so existing
Pandas-style workflows can redact free-text columns without switching to a
row-at-a-time Spark UDF.

Install the optional Spark extra:

```bash
pip install "openmed[spark]"
```

Import the integration once to register the `.deid` accessors:

```python
import pyspark.pandas as ps
import openmed.integrations.pandas_on_spark  # registers the accessors

records = ps.DataFrame(
    {
        "record_id": ["a", "b"],
        "clinical_note": [
            "Patient Jane Roe called 555-0100.",
            "Follow-up contains no identifiers.",
        ],
    }
)

redacted = records.deid.deidentify(
    columns="clinical_note",
    policy="hipaa_safe_harbor",
)
print(redacted.to_pandas())
```

The DataFrame method has the same public signature as the local Pandas
accessor: pass one column name or a sequence through `columns`, select a
`method`, and optionally set a policy profile. Extra de-identification options,
including `model_name`, `confidence_threshold`, and language settings, are
forwarded to OpenMed's batch processor.

Series use the same distributed path:

```python
notes = records["clinical_note"].deid.deidentify(
    policy="strict_no_leak",
    use_safety_sweep=True,
)
```

Under the hood, pandas-on-Spark sends Arrow row groups through a pandas UDF.
OpenMed calls `process_batch` once for each row group and reuses a worker-local
model loader, so the model backbone is not loaded once per row. The integration
does not log source text or emit raw PHI in progress metadata.

This accessor is for the pandas API on Spark. Use the Structured Streaming
helpers for streaming sinks, or the Dask accessor for Dask DataFrames.
