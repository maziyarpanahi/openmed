# Prefect Batch De-identification

OpenMed ships a reusable Prefect task and flow that wrap the local dataset
redaction runner, so Prefect users get observability, retries, and scheduling
for de-identification jobs without re-implementing orchestration glue. The
integration is optional: importing `openmed` or `openmed.interop` does not
import Prefect, and `prefect` is only loaded when the adapter module is
imported.

Install the optional extra:

```bash
pip install "openmed[prefect]"
```

The integration supports Prefect 3.7 and later 3.x releases.

Run the flow over a list of local dataset files:

```python
from openmed.interop.prefect_tasks import deidentify_dataset_flow

result = deidentify_dataset_flow(
    input_paths=["notes-2026-01.csv", "notes-2026-02.csv"],
    text_columns=["note"],
    policy="hipaa_safe_harbor",
    method="mask",
    confidence_threshold=0.7,
)
print(result["files_processed"], result["spans_redacted"])
```

`deidentify_dataset_flow` fans `deidentify_file_task` over `input_paths`, one
task run per file, and aggregates the per-file summaries. Each file is
processed with `openmed.processing.batch.redact_dataset`, so `.csv`,
`.jsonl`/`.ndjson`, and `.parquet` inputs are supported. By default the
redacted copy is written next to its input as `<stem>.redacted<suffix>`; pass
`output_dir` to collect the redacted files in one directory instead. Inputs
must have unique basenames when `output_dir` is set so one task cannot
overwrite another task's output.

The single-file task can also be used directly inside your own flows:

```python
from prefect import flow

from openmed.interop.prefect_tasks import deidentify_file_task

@flow
def nightly_deid(path: str) -> dict:
    return deidentify_file_task(path, text_columns=["note"])
```

Both the task and the flow return PHI-free summaries containing counts only:
`files_processed`, `rows_processed`, `cells_redacted`, and `spans_redacted`.
The flow also returns the count-only per-file summaries under `files`, in the
same order as `input_paths`, so downstream tasks can branch on the results
without copying dataset contents into result state.

Prefect records task parameters for orchestration. Use opaque dataset paths
and column names, and never put PHI in filenames or directory names. Prefect
runtime options such as retries stay configurable through standard Prefect
APIs, for example `deidentify_file_task.with_options(retries=2)`.

Additional keyword arguments (for example `lang`, `keep_year`,
`date_shift_days`, or `model_name`) are forwarded to
`openmed.processing.batch.redact_dataset` unchanged.
