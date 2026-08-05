---
name: deidentify-a-dataset
description: "De-identify selected free-text columns in a local CSV, JSONL, or Parquet dataset with OpenMed and produce a separate redacted dataset plus a PHI-free aggregate summary. Use when an agent must prepare a clinical dataset for analysis or sharing without overwriting the source or exposing cell values in logs."
---

# De-identify a dataset

Keep the source local, name the free-text columns explicitly, and write to a
different destination. Never infer columns or print source and redacted cell
values.

## Procedure

1. Confirm that the input is CSV, JSONL/NDJSON, or Parquet.
2. Confirm which columns contain free text. Do not scan or log values to guess.
3. Choose a policy and language. Prefer `strict_no_leak` when recall is the
   governing safety requirement.
4. Write to a new path; never overwrite the input.
5. Inspect only `result.summary`, which contains aggregate counts and rates.
6. Validate recall and residual leakage on representative synthetic or
   approved evaluation fixtures before releasing the output.

## Runnable synthetic example

Install the model runtime first with `python -m pip install "openmed[hf]"`.

```python
import csv
from pathlib import Path

from openmed import redact_dataset

source = Path("synthetic-notes.csv")
destination = Path("synthetic-notes.redacted.csv")

with source.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["record_id", "note"])
    writer.writeheader()
    writer.writerows(
        [
            {
                "record_id": "SYNTH-001",
                "note": (
                    "Taylor Example called 212-555-0198 about a "
                    "metformin refill."
                ),
            },
            {
                "record_id": "SYNTH-002",
                "note": (
                    "Send the synthetic follow-up to "
                    "demo.patient@example.test."
                ),
            },
        ]
    )

result = redact_dataset(
    source,
    text_columns=["note"],
    output_path=destination,
    policy="strict_no_leak",
    lang="en",
)

print(result.output_path)
print(result.summary.to_dict())  # Aggregate counts only; no cell contents.
```

Use the equivalent CLI for an existing dataset:

```bash
openmed redact-dataset notes.csv \
  --text-columns note,comment \
  --policy strict_no_leak \
  --output notes.redacted.csv
```

## Safety checks

- Keep model inference and files on infrastructure the user controls.
- Do not print input rows, detected entity surfaces, reversible mappings, or
  exception payloads that may contain source text.
- Keep source and output paths separate and access-controlled.
- Treat the aggregate summary as evidence, not as proof of compliance.
- Never commit real clinical data or restricted evaluation corpora.

## Repository example

Read and run
[the offline dataset walkthrough](../../examples/datasets_walkthrough.py) when
you need a bundled synthetic fixture and first-run download controls.
