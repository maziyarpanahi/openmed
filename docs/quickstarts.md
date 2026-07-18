# Persona Quickstarts

The [Quick Start](getting-started.md) walks through installation in one linear path.
This page splits that path into three short, copy-paste tracks so you land on the
right first command for your role, then hands you off to the deeper guide.

Pick the track that matches how you'll use OpenMed:

| Persona | Goal | Jump to |
| --- | --- | --- |
| **Researcher** | De-identify a corpus and measure PHI leakage | [Researcher](#researcher) |
| **App developer** | Call de-identification from a service or agent | [App developer](#app-developer) |
| **Data engineer** | Redact files, directories, and datasets at scale | [Data engineer](#data-engineer) |

!!! warning "Not a medical device"
    OpenMed is a research and engineering toolkit for text de-identification and
    named-entity recognition. It is **not** a medical device and does not provide
    clinical decisions, diagnosis, or treatment advice. De-identification is
    probabilistic: review residual-risk output and validate against your own
    compliance requirements before releasing data. Every example below uses
    **synthetic, non-PHI** text.

All tracks assume a working install. The minimal setup is:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv (skip if present)
uv venv --python 3.11
source .venv/bin/activate
uv pip install "openmed[hf]"                      # core + Hugging Face models
```

The first model-backed call may download model artifacts. Clinical text is still
processed locally; after warming the cache, use
[local-only offline mode](configuration.md#local-only-offline-mode) to block
outbound connections during inference.

---

## Researcher

**Goal:** run entity extraction and de-identification over clinical notes, then
quantify how much PHI leaked through.

### 1. Extract entities from a synthetic note

`extract_pii()` returns a `PredictionResult` whose `entities` carry the detected
text, canonical label, confidence, and character offsets.

```python
from openmed import extract_pii

note = "Patient Casey Example (MRN 00123) called from 555-0100 on 01/15/2020."

result = extract_pii(note)
for entity in result.entities:
    print(f"{entity.label:>10}  {entity.text!r}  [{entity.start}:{entity.end}]")
```

Reach for `analyze_text()` when you want the clinical NER models (diseases,
medications, findings) instead of PII labels:

```python
from openmed import analyze_text

clinical = analyze_text(
    "Metastatic breast cancer treated with paclitaxel and trastuzumab.",
    model_name="disease_detection_superclinical",
)
print(clinical.entities[0])
```

### 2. De-identify the note

```python
from openmed import deidentify

redacted = deidentify(note, method="mask")
print(redacted.deidentified_text)
# Detected spans are replaced with labels such as [first_name] and [phone_number].
```

### 3. Measure PHI leakage

For a de-identification study, compare your gold PHI spans against the spans the
model predicted. `compute_leakage_rate()` reports the character-weighted fraction
of gold PHI not covered by a same-label prediction — the lower the better.

```python
from openmed.eval import compute_leakage_rate

gold = [
    {"start": 8, "end": 21, "label": "NAME"},    # "Casey Example"
    {"start": 46, "end": 54, "label": "PHONE"},   # "555-0100"
]
predicted = [
    {"start": 8, "end": 21, "label": "NAME"},     # phone missed on purpose
]

metrics = compute_leakage_rate(gold, predicted, source_text=note)
print(f"Overall PHI leakage: {metrics.overall:.1%}")
print(f"Missed phone leakage: {metrics.by_label['PHONE']:.1%}")
```

**Next steps:** the full method matrix (`mask`, `remove`, `replace`, `hash`,
`shift_dates`), reversible mappings, and longitudinal patient-keyed date shifting
live in [PII Anonymization](anonymization.md#quickstart-choosing-a-method).
For the metric bundle and release gates, see
[Eval Harness & Metrics](eval-harness.md).

---

## App developer

**Goal:** de-identify text from an application or agent without embedding the
models in your own request path.

### Option A — call the library directly

If your service is already Python, one call is enough:

```python
from openmed import deidentify

redacted = deidentify("Contact Jordan Sample at jordan.sample@example.org.")
print(redacted.deidentified_text)
# Contact [first_name] [last_name] at [email].
```

### Option B — run the REST service

Install the service extra and start the API:

```bash
uv pip install "openmed[hf,service]"
uvicorn openmed.service.app:app --host 127.0.0.1 --port 8080
```

Call `POST /pii/deidentify` with any HTTP client:

```bash
curl -s http://127.0.0.1:8080/pii/deidentify \
  -H "Content-Type: application/json" \
  -d '{"text": "Contact Jordan Sample at jordan.sample@example.org.", "method": "mask"}'
```

Or use the typed Python client bundled with the `service` extra:

```python
from openmed.service.client import OpenMedClient

with OpenMedClient("http://127.0.0.1:8080", timeout=300.0) as client:
    redacted = client.deidentify(
        "Contact Jordan Sample at jordan.sample@example.org.",
        method="mask",
    )
    print(redacted["deidentified_text"])
```

### Option C — expose OpenMed to an AI agent over MCP

Install the MCP extra and start the Model Context Protocol server. It exposes an
`openmed_deidentify` tool (plus `openmed_extract_pii` and `openmed_analyze_text`)
over stdio, so agent frameworks can redact PHI as a tool call:

```bash
uv pip install "openmed[hf,mcp]"
python -m openmed.mcp.server                       # stdio transport (default)
```

For a local agent runtime that connects over HTTP, keep the server on loopback:

```bash
python -m openmed.mcp.server --transport streamable-http --host 127.0.0.1 --port 8081
```

These service examples deliberately bind to loopback. Do not expose them to an
untrusted network without an authenticated, TLS-terminating boundary.

**Next steps:** health checks, model warm-pool controls, request batching, the
privacy gateway, and the full endpoint reference are in
[REST Service](rest-service.md).

---

## Data engineer

**Goal:** de-identify many files or dataset rows with progress reporting and
PHI-free aggregate summaries — no per-record boilerplate.

### 1. Process a directory of notes

`BatchProcessor` reuses one warmed model across the batch and keeps going when a
single record fails. It returns de-identified text in memory and does not
overwrite the source files:

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from openmed import BatchProcessor

processor = BatchProcessor(
    operation="deidentify",
    model_name="pii_detection",
    method="mask",
    batch_size=16,
    confidence_threshold=0.7,
    continue_on_error=True,
)

with TemporaryDirectory() as temp_dir:
    notes_dir = Path(temp_dir)
    (notes_dir / "note-1.txt").write_text(
        "Patient Casey Example called from 555-0100.", encoding="utf-8"
    )
    (notes_dir / "note-2.txt").write_text(
        "Email Jordan Sample at jordan.sample@example.org.", encoding="utf-8"
    )

    result = processor.process_directory(
        notes_dir, pattern="*.txt", recursive=True
    )

    redacted_texts = [
        item.result.deidentified_text for item in result.get_successful_results()
    ]
    print(f"Processed {len(redacted_texts)}/{result.total_items} files")
    print(f"Failures: {result.failed_items}")
```

### 2. Redact specific columns of a dataset

For CSV or JSONL input, `redact_dataset()` de-identifies only the free-text
columns you name and returns an aggregate, PHI-free audit summary. Parquet uses
the same API after installing the `columnar` extra with
`uv pip install "openmed[hf,columnar]"`:

```python
from openmed import redact_dataset

result = redact_dataset(
    "notes.csv",
    text_columns=["note", "comment"],
    output_path="notes.redacted.csv",
    policy="strict_no_leak",
)

# Aggregate counts only — total spans, per-label counts, residual-leakage estimate.
print(result.summary.to_dict())
```

The source dataset is never overwritten: `output_path` receives the redacted
rows, while the summary remains in memory unless you explicitly persist it.

The same path is available from the console entry point:

```bash
openmed redact-dataset notes.csv \
  --text-columns note,comment \
  --policy strict_no_leak \
  --output notes.redacted.csv
```

**Next steps:** streaming iteration, progress callbacks, batch PII extraction,
and result export are covered in [Batch Processing](batch-processing.md).
