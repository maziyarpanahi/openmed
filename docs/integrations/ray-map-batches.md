# Ray Data map-batches de-identification

OpenMed provides a stateful `Dataset.map_batches` stage for distributed,
columnar de-identification. Ray runs the callable class in an actor pool, and
each actor loads one OpenMed model pipeline in its constructor and reuses that
pipeline for every batch it processes.

Install OpenMed's model dependencies and Ray Data:

```bash
pip install "openmed[hf]" "ray[data]"
```

## Apply the stage

Choose the free-text column explicitly. Non-target columns, null values, batch
row counts, and the dataset's total row count pass through unchanged.

```python
import ray

from openmed.integrations.ray_map_batches import map_batches_deidentify

ray.init()

records = ray.data.from_items(
    [
        {"record_id": "a", "clinical_note": "Patient Jane Roe called."},
        {"record_id": "b", "clinical_note": "No identifiers here."},
    ]
)

redacted = map_batches_deidentify(
    records,
    column="clinical_note",
    policy_profile="hipaa_safe_harbor",
    batch_size=256,
    batch_format="pyarrow",
    concurrency=4,
)

redacted.write_parquet("/secure/output/redacted-notes")
```

The returned dataset is lazy. A terminal operation such as `write_parquet`,
`materialize`, or `take_all` starts execution.

## Actor pool and batch format

`concurrency=4` creates a fixed pool of four model actors. Use `(minimum,
maximum)` for an autoscaling pool, or `(minimum, maximum, initial)` to set its
initial size as well:

```python
redacted = map_batches_deidentify(
    records,
    column="clinical_note",
    policy_profile="strict_no_leak",
    batch_size=512,
    batch_format="pandas",
    concurrency=(1, 8, 2),
    num_cpus=2,
)
```

Supported batch formats are `"pyarrow"` and `"pandas"`. The stage copies the
batch before replacing the target column, leaving Ray's zero-copy input buffers
untouched. `num_cpus`, `num_gpus`, and other Ray worker resource arguments are
forwarded to `Dataset.map_batches`.

## Use the callable class directly

For custom Ray Data plans, pass `RayDeidentifyBatch` to `map_batches`. A class
UDF is important here: a function UDF is stateless and would not retain the
loaded model between calls.

```python
from ray.data import ActorPoolStrategy

from openmed.integrations.ray_map_batches import RayDeidentifyBatch

redacted = records.map_batches(
    RayDeidentifyBatch,
    batch_size=256,
    batch_format="pyarrow",
    compute=ActorPoolStrategy(size=4),
    fn_constructor_kwargs={
        "column": "clinical_note",
        "policy_profile": "hipaa_safe_harbor",
    },
)
```

Ray's object store and worker processes temporarily hold the input batches.
Deploy this stage only on a trusted, access-controlled cluster appropriate for
the sensitivity of the source data, and write results only to approved storage.
