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

For a local contract test, use `redact_partition` with a synthetic worker
factory. The factory is called once per partition attempt and can be tested
without a Spark installation:

```python
from openmed.interop.spark import SparkRedactionConfig, redact_partition

config = SparkRedactionConfig(columns=["note"], method="mask")

def worker_factory():
    def redact(text, **kwargs):
        return text.replace("synthetic-person-001@example.test", "[EMAIL]")

    return redact

rows = [{"note": "synthetic-person-001@example.test", "kind": "fixture"}]
redacted_rows = list(
    redact_partition(rows, config, deidentifier_factory=worker_factory)
)
```

`deidentifier_factory` is an explicit test or advanced-runtime seam. If it is
used with a real Spark DataFrame, it must be serializable by the Spark worker
runtime and must not capture driver-owned mutable state.

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
