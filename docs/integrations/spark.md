# Spark Redaction Transform Contract

OpenMed provides a dependency-light contract for redacting selected nullable
string columns in Spark DataFrames. Install the optional Spark runtime when
running the DataFrame operation:

```bash
pip install "openmed[spark]"
```

The contract is available from `openmed.interop.spark` and does not import
PySpark when the module is imported. Configure the columns explicitly and
apply the transform to a DataFrame:

```python
from openmed.interop.spark import SparkRedactionTransform

redact = SparkRedactionTransform(
    columns=["clinical_note", "comment"],
    method="mask",
    policy="hipaa_safe_harbor",
)
redacted = redact.apply(df)
```

Only the configured columns are changed. Other fields are copied unchanged,
and null values remain null. A configured column must contain strings or nulls;
the transform fails before returning a result for any other value.

## Partition and retry behavior

`SparkRedactionTransform` contains only an immutable `SparkRedactionConfig`.
Spark serializes that configuration with the partition function; it never
serializes a loaded model, surrogate vault, or mutable redaction mapping from
the driver. Each partition attempt constructs its own worker-local
deidentifier. Spark retries and speculative attempts therefore receive fresh
worker state rather than sharing state across executors.

Additional de-identification options are snapshotted as bounded value data:
nulls, booleans, finite numbers, strings, bytes, paths, and nested lists,
tuples, or string-keyed dictionaries. Stateful hooks, recognizers, and request
budgets are rejected from the serialized configuration. Each partition gets a
fresh copy of the accepted values, without executable deserialization.

The default worker uses OpenMed's cache-only configuration. It does not make a
mandatory network request, so model artifacts must already be available on
each worker. `mask`, `hash`, and seeded replacement/date-shift configurations
are deterministic for the same input and configuration. A partition replay
recreates the worker from the same serialized settings; callers supplying a
custom worker factory must keep that factory deterministic as well.

For offline contract tests, call `redact_partition` with a synthetic worker
factory. The factory runs once per partition attempt. A factory used with a
real DataFrame must be serializable and must not capture driver-owned mutable
state.

## Failure and privacy boundary

Worker failures are converted to `SparkRedactionError` with stable contract
metadata. Source text and the original exception message are not included,
because Spark commonly forwards worker exception text to driver logs. This
module returns redacted rows only; it does not emit raw-value reports or
persist surrogate state. It is an interoperability helper, not a compliance
certification or clinical decision guarantee.

The existing `openmed.interop.spark_udf` pandas UDF remains available for
static `withColumn` workflows. Use this transform when an explicit
partition-local `mapPartitions` contract is the boundary you need.
