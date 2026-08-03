# PySpark De-identification UDFs

OpenMed can register a `pandas_udf` so data engineers can redact free-text
columns at warehouse scale on a Spark cluster, using OpenMed's first-party
models. The integration is optional: importing `openmed` or `openmed.interop`
does not import PySpark.

Install the optional extra:

```bash
pip install "openmed[spark]"
```

The extra installs PySpark together with the pandas and PyArrow runtimes
required by `pandas_udf`.

Redact one or more columns on a static DataFrame:

```python
from openmed.interop.spark_udf import deidentify_columns

redacted = deidentify_columns(
    df,
    columns=["clinical_note", "comment"],
    policy="hipaa_safe_harbor",
)
```

For finer control, build the UDF directly:

```python
from openmed.interop.spark_udf import make_deidentify_udf

redact = make_deidentify_udf(policy="hipaa_safe_harbor")
df = df.withColumn("clinical_note_redacted", redact(df["clinical_note"]))

spark.udf.register("openmed_deidentify", redact)
```

## Executor model-loading

Spark reuses one Python worker process across many batches within a
partition. The OpenMed model loads lazily, once per worker process, and is
reused for every batch that process handles. It is not reloaded per batch
or per row. The model is never broadcast from the driver: broadcasting a
loaded model would force the driver to hold and pickle a copy of it, which
can exhaust driver memory or fail for models that do not serialize cleanly.
Only the small `policy`/keyword-argument closure is shipped to executors;
each worker loads its own model instance locally on first use.

## No raw PHI in driver logs

Task failures propagate their exception message back to the driver, which
often forwards into centralized log aggregation outside this process's
control. The adapter does not interpolate raw input text into exception
messages or log statements.

## Scope

This adapter targets static/batch DataFrames (`df.withColumn`, Spark SQL).
For redacting a Structured Streaming `foreachBatch` sink, use
`openmed.integrations.spark_streaming` instead.
