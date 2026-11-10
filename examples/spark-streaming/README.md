# Spark Structured Streaming de-identification

This example wires OpenMed into a Spark Structured Streaming `foreachBatch`
sink. Each micro-batch is redacted in Python partitions, tagged with a Spark
batch ID, and appended to a target table only when that batch ID has not already
been written.

Install the optional Spark dependencies in your application environment:

```bash
uv pip install --upgrade "openmed[spark]"
```

```python
from openmed.integrations.spark_streaming import (
    SparkDeidentifyColumn,
    deidentify_write_stream,
)

raw_notes = (
    spark.readStream.table("raw_clinical_notes")
    .select("encounter_id", "note_text", "triage_comment", "ingested_at")
)

query = (
    deidentify_write_stream(raw_notes)
    .columns(
        SparkDeidentifyColumn(
            name="note_text",
            policy="hipaa_safe_harbor",
        ),
        SparkDeidentifyColumn(
            name="triage_comment",
            output_column="triage_comment_redacted",
            policy="strict_no_leak",
            deidentify_kwargs={"use_safety_sweep": True},
        ),
    )
    .target_table("redacted_clinical_notes")
    .checkpoint("/secure/checkpoints/openmed/redacted_clinical_notes")
    .query_name("openmed_redacted_clinical_notes")
    .trigger(processingTime="30 seconds")
    .start()
)

query.awaitTermination()
```

By default the sink adds `_openmed_batch_id` to the output. On replay, the sink
checks the target table for that batch ID and skips the write when it is already
present. Keep the checkpoint path and target table on durable storage managed by
your Spark deployment.

The Spark integration imports PySpark lazily. Importing `openmed` or
`openmed.integrations.spark_streaming` does not require a Spark installation.
