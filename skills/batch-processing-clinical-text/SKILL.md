---
name: batch-processing-clinical-text
description: "Run large-scale batch NER, PII extraction, or de-identification over many clinical notes on-device with OpenMed, with sharding, checkpointing, resumability, and append-only JSONL output. Use when the user needs to process a corpus or folder of notes, de-identify a dataset, run NER over thousands of documents, build a resumable batch pipeline, or stream results to JSONL without holding everything in memory. Covers process_batch / BatchProcessor / BatchItem / BatchResult, the operation= selector (analyze_text | extract_pii | deidentify), iter_process streaming, the PHI-safe on_progress callback, chunking long documents, and no-PHI logging. Produces a resumable batch runner over an OpenMed model."
license: Apache-2.0
metadata:
  project: OpenMed
  category: deployment-ops
  pairs: adjacent
  version: "1.0"
---

# Batch processing clinical text

`openmed.processing` runs OpenMed over many documents efficiently, with progress
tracking, per-item error isolation, and streaming. It runs **fully on-device**:
the corpus, the model, and the output never leave the host. This skill shows a
**resumable** runner — sharded, checkpointed, append-only JSONL — that you can
restart without reprocessing.

## When to use this skill

For corpora, folders, or datasets — anything beyond a handful of notes. For a
single note, just call `openmed.analyze_text` / `deidentify` directly
(`extracting-clinical-entities`, `deidentifying-clinical-text`). For an
always-on HTTP service, see `serving-openmed-rest-api`.

## Quick start

```python
from openmed import process_batch

texts = ["Patient has type 2 diabetes.", "No acute distress. BP 120/80."]
result = process_batch(texts, model_name="disease_detection_superclinical")

print(result.summary())          # PHI-safe counts + timing
print(result.successful_items, "/", result.total_items)
for item in result.get_successful_results():
    print(item.id, item.result.to_dict()["entities"])   # spans only; avoid raw text in logs
```

`process_batch(...)` is a thin wrapper over `BatchProcessor`. Real signatures
(`openmed/processing/batch.py`):

- `process_batch(texts, model_name="disease_detection_superclinical", ids=None, config=None, progress_callback=None, on_progress=None, **kwargs) -> BatchResult`
- `BatchProcessor(model_name=..., operation="analyze_text", batch_size=8, continue_on_error=True, **analyze_kwargs)`
  with `operation ∈ {"analyze_text", "extract_pii", "deidentify"}`.
- `BatchItem(id, text, source=None, metadata=None)`
- `BatchResult` — `.items`, `.total_items`, `.successful_items`, `.failed_items`,
  `.success_rate`, `.average_processing_time`, `.summary()`, `.to_dict()`,
  `.get_successful_results()`, `.get_failed_results()`.
- `BatchItemResult` — `.id`, `.result` (a `PredictionResult`/`DeidentificationResult`),
  `.error`, `.processing_time`, `.source`, `.success`, `.to_dict()`.

## Choosing the operation

```python
from openmed import BatchProcessor

# NER (default)
ner = BatchProcessor(model_name="disease_detection_superclinical")          # operation="analyze_text"
# Detect PHI spans
pii = BatchProcessor(operation="extract_pii", model_name="OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1")
# De-identify (rewrites text)
deid = BatchProcessor(operation="deidentify", method="mask", confidence_threshold=0.7)
```

`BatchProcessor` reuses one model loader across items (and a cached privacy
filter for PII ops), so a single processor over many texts is far cheaper than
many one-off calls.

## Streaming + PHI-safe progress

```python
from openmed.processing import BatchProcessor, BatchProgress

proc = BatchProcessor(operation="deidentify", method="mask", batch_size=16)

def on_progress(p: BatchProgress) -> None:   # frozen record: completed/total/current_index/elapsed
    if p.completed % 500 == 0:
        print(f"{p.completed}/{p.total} ({p.elapsed:.1f}s)")   # NO PHI

for item in proc.iter_process(texts, ids=doc_ids, on_progress=on_progress):
    write_jsonl(item)   # one result at a time — constant memory for huge corpora
```

`on_progress` receives only counts/timing (never text), so it is safe to log.
`iter_process` yields `BatchItemResult`s without holding the whole corpus.

## Workflow

1. **De-identify upstream** if the corpus has PHI, or key everything by stable
   internal `ids` so logs/output never carry identifiers.
2. **Pick the operation + model** — one `BatchProcessor` per
   `operation` (`analyze_text` / `extract_pii` / `deidentify`); reuse it across
   all items so the loader is shared.
3. **Shard the corpus** into independent partitions writing to separate JSONL
   files; run shards as separate processes for parallelism.
4. **Stream with `iter_process`** and append each `BatchItemResult` to JSONL
   (flush per item) — that append-only file is your checkpoint.
5. **Track progress** with the PHI-safe `on_progress` callback (counts/timing
   only).
6. **Re-run to resume:** skip ids already present in the output; with
   `continue_on_error=True`, failures are recorded, not raised.
7. **Reconcile** via `result.get_failed_results()` / a failures pass, then hand
   JSONL to the downstream consumer.

## A resumable batch runner

```python
import json
from pathlib import Path
from openmed.processing import BatchProcessor

def run_resumable(texts, ids, out_path: Path, *, operation="deidentify", **kw):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 1) checkpoint = ids already written (append-only JSONL is the source of truth)
    done = set()
    if out_path.exists():
        with out_path.open() as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    todo = [(i, t) for i, t in zip(ids, texts) if i not in done]
    if not todo:
        return
    pending_ids, pending_texts = zip(*todo)

    proc = BatchProcessor(operation=operation, continue_on_error=True, **kw)
    # 2) append each result as it completes -> safe to kill/restart anytime
    with out_path.open("a") as f:
        for item in proc.iter_process(list(pending_texts), ids=list(pending_ids)):
            record = {"id": item.id, "ok": item.success,
                      "result": item.result.to_dict() if item.success else None,
                      "error": item.error}            # item.error is a message, keep PHI out
            f.write(json.dumps(record) + "\n")
            f.flush()
```

Re-running skips finished ids (`continue_on_error=True` keeps one bad doc from
killing the run; failures are recorded, not raised). Shard a corpus by writing
to `out/shard_000.jsonl`, `shard_001.jsonl`, … and run shards in separate
processes.

## Chunking long documents

`BatchProcessor` does not split single documents. For notes longer than the
model's max sequence length, pre-split into sentences/windows
(`openmed.processing.sentences`) — or for `analyze_text`, rely on built-in
sentence handling — then re-stitch entities by adding the chunk offset back to
each entity's `start`/`end` so spans point into the original document.

## Hand-off to / from OpenMed

- **Per-item engine:** each operation calls the same `openmed.analyze_text` /
  `extract_pii` / `deidentify` you'd call directly — same results, batched.
- **Downstream:** JSONL feeds `building-patient-timelines`, `etl-to-omop-cdm`,
  and `exporting-to-fhir`. De-id JSONL feeds `evaluating-with-leakage-gates`.
- **Service vs batch:** for request/response use `serving-openmed-rest-api`; for
  corpora use this batch path.

## Edge cases & gotchas

- **No PHI in logs/JSONL keys.** Use stable `ids` (`BatchItem.id`), offsets, and
  labels. `BatchItemResult.error` is a message — keep raw text out of inputs to
  exceptions you log.
- **`continue_on_error=True` is the default** and recommended for corpora; check
  `result.get_failed_results()` afterward. Set `False` only when one failure
  should abort everything.
- **`batch_size` is a throughput dial**, not correctness — tune to memory/CPU.
  Larger isn't always faster on CPU.
- **Append-only is the checkpoint.** Don't buffer results in memory and write at
  the end; you lose progress on a crash. Append + flush per item.
- **`process_files`/`process_directory`** read files for you and set
  `BatchItem.source`; unreadable files become failed items (not crashes) under
  `continue_on_error`.
- **Mixed languages:** pass `lang=` per run for PII ops; don't run an English
  PII model across other languages (`deidentifying-multilingual-text`).

## Standards & references

- JSON Lines (newline-delimited JSON): https://jsonlines.org/
- HIPAA de-identification (when batching PHI removal), 45 CFR 164.514(b):
  https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html
- OpenMed source: `openmed/processing/batch.py` (`process_batch`,
  `BatchProcessor`, `BatchItem`, `BatchItemResult`, `BatchResult`,
  `BatchProgress`, `iter_process`).
