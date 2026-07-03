# Dask DataFrame De-identification

OpenMed can redact Dask DataFrames partition by partition so larger-than-memory
tables use the same local de-identification path as in-memory batches.

Install the optional Dask extra:

```bash
pip install "openmed[dask]"
```

Import the integration once to register the `.deid` accessors:

```python
import dask.dataframe as dd
import openmed.integrations.dask_accessor  # registers DataFrame/Series accessors

pdf = ...
records = dd.from_pandas(pdf, npartitions=8)

redacted = records.deid.deidentify(
    target_columns=["clinical_note"],
    policy="hipaa_safe_harbor",
)
result = redacted.compute(scheduler="threads")
```

The accessor calls OpenMed's batch de-identification once per Dask partition,
preserving row order, indexes, divisions, and the original column metadata. Use
the same helper directly when a functional style fits better:

```python
from openmed.integrations.dask_accessor import map_partitions_deidentify

redacted = map_partitions_deidentify(
    records,
    target_columns="clinical_note",
    policy="strict_no_leak",
)
```

Dask Series are supported through the same accessor:

```python
notes = records["clinical_note"].deid.deidentify(policy="hipaa_safe_harbor")
```

The integration is intended for local and ordinary Dask DataFrame workflows.
Distributed cluster deployment guidance and Dask-cuDF/GPU execution are outside
this integration's current scope.
