# Build with OpenMed from an agent

Use this guide when a coding agent is building an application that imports
OpenMed. The repository `AGENTS.md` explains how to contribute to OpenMed; this
page explains how to consume its public APIs safely.

All examples below use synthetic values. Model inference runs locally after the
first model download.

## Install

Install the core library and the local model runtime:

```bash
python -m pip install "openmed[hf]"
```

Add optional surfaces only when the application needs them:

```bash
python -m pip install "openmed[hf,mcp,service]"
```

## Three primary calls

### Analyze clinical text

Use `analyze_text` for clinical or biomedical named-entity recognition.

```python
from openmed import analyze_text

result = analyze_text(
    "Synthetic assessment: type 2 diabetes treated with metformin.",
    model_name="disease_detection_superclinical",
    confidence_threshold=0.5,
)

for entity in result.entities:
    print(entity.label, entity.text, entity.start, entity.end)
```

The result is a typed `AnalyzeResult`. Its `entities` expose text, label,
confidence, and character offsets; `result.to_dict()` produces a JSON-ready
mapping.

### Extract PII without changing the text

Use `extract_pii` when the application needs identifier spans and offsets.

```python
from openmed import extract_pii

synthetic_note = (
    "Synthetic patient Taylor Example, record SYNTH-0007, called "
    "212-555-0198."
)
result = extract_pii(
    synthetic_note,
    lang="en",
    confidence_threshold=0.5,
)

for entity in result.entities:
    print(entity.label, entity.start, entity.end, entity.confidence)
```

Do not print `entity.text` for real records. Keep raw surfaces inside the
trusted process and use labels, offsets, hashes, and confidence values for
diagnostics.

### De-identify text

Use `deidentify` to mask, remove, replace, hash, or consistently shift detected
identifiers.

```python
from openmed import deidentify

synthetic_note = (
    "Synthetic patient Taylor Example can be reached at "
    "demo.patient@example.test."
)
result = deidentify(
    synthetic_note,
    method="mask",
    lang="en",
    confidence_threshold=0.5,
)

print(result.deidentified_text)
print(result.num_entities_redacted)
```

`DeidentificationResult` also carries detected entities, metadata, an optional
audit report, and an optional reversible mapping. Do not log or return
`original_text`, entity surfaces, or mappings across an untrusted boundary.

## Public API map

| Task | Public surface | Result or purpose |
|---|---|---|
| Clinical NER | `openmed.analyze_text` | `AnalyzeResult` with typed entity spans |
| PII detection | `openmed.extract_pii` | PII predictions without rewriting text |
| De-identification | `openmed.deidentify` | `DeidentificationResult` and rewritten text |
| Re-identification | `openmed.reidentify` | Restore only from an explicitly retained mapping |
| Batch text processing | `openmed.BatchProcessor`, `openmed.process_batch` | Reuse model state across many inputs |
| Dataset redaction | `openmed.redact_dataset` | Separate CSV, JSONL, or Parquet output plus aggregate summary |
| Model discovery | `openmed.get_model_info`, `openmed.get_models_by_category`, `openmed.get_pii_models_by_language` | Offline registry metadata |
| Model loading | `openmed.ModelLoader`, `openmed.load_model` | Explicit model lifecycle and cache reuse |
| FHIR export | `openmed.clinical.exporters.fhir` | Deterministic FHIR R4 helpers |
| Evaluation | `openmed.eval` | Recall, leakage, calibration, and release gates |
| Agent tools | `openmed.mcp.tool_registry` | Typed tool definitions and validation |
| Local service | `openmed.service.app:app` | REST endpoints for self-hosted use |
| Command line | `openmed` | Analyze, PII, dataset, model, and benchmark commands |

Use the [API reference](api-reference.md) for signatures and the
[feature map](feature-map.md) for the source module behind each capability.

## Privacy invariants

An agent that builds with OpenMed must preserve these rules:

1. Keep inference local. A first run may download model artifacts; steady-state
   inference must not require uploading text or enabling telemetry.
2. Use offline mode after approved models are cached:

   ```python
   from openmed import OpenMedConfig, extract_pii

   config = OpenMedConfig(local_only=True)
   result = extract_pii(
       "Synthetic contact: sample.user@example.test.",
       config=config,
   )
   ```

3. Never place raw PHI in logs, traces, cache keys, temporary filenames,
   exception messages, analytics, or audit artifacts.
4. Keep audit evidence PHI-free: store offsets, canonical labels, thresholds,
   provenance, counts, and keyed hashes rather than identifier text.
5. Use synthetic data in examples and committed fixtures. Never commit
   DUA-gated corpora or restricted terminology assets.
6. Validate direct-identifier recall, critical leakage, span integrity, and
   quantized-model deltas. Aggregate F1 alone is not a privacy gate.
7. Treat de-identification as a technical control, not a compliance guarantee
   or an automated clinical decision.
8. Keep reversible mappings encrypted and separate, or leave
   `keep_mapping=False`.

See [No-Raw-PHI Logging](security/no-raw-phi-logging.md),
[No-Telemetry Guarantee](security/no-telemetry.md), and the
[redactor threat model](security/threat-model.md) for the full controls.

## MCP server and tool registry

Install and launch the local stdio server:

```bash
python -m pip install "openmed[mcp]"
openmed-mcp --transport stdio
```

The server exposes `openmed_analyze_text`, `openmed_extract_pii`,
`openmed_deidentify`, model discovery and lifecycle tools, and a PHI-safe
workflow runner. The source of truth for their input and output schemas is
`openmed/mcp/tool_registry.py`.

Use the [MCP client guide](mcp-clients.md) for client configuration and
authenticated remote deployment. Prefer stdio or loopback. Do not bind an
unauthenticated server to a public interface.

Applications that need tool definitions without running an MCP transport can
import the registry directly:

```python
from openmed.mcp.tool_registry import TOOL_REGISTRY

for tool in TOOL_REGISTRY.latest_specs():
    print(tool.name)
```

## Command-line interface

Discover models without sending clinical text anywhere:

```bash
openmed models list
openmed models info disease_detection_superclinical
```

Run the three primary paths:

```bash
openmed analyze \
  --text "Synthetic assessment: asthma treated with albuterol."

openmed pii extract \
  --text "Synthetic contact: 212-555-0198."

openmed pii deidentify \
  --text "Synthetic contact: demo.patient@example.test." \
  --method mask
```

For files and datasets, prefer `--input-file`, `openmed deid`, or
`openmed redact-dataset` so shell history does not capture sensitive values.
See [Batch Processing](batch-processing.md) for streaming and aggregate
PHI-free progress reporting.

## Repository skills

The repository-root `skills/` directory contains portable procedures an agent
can load directly. Start with these task recipes:

- `skills/deidentify-a-dataset/SKILL.md`
- `skills/extract-clinical-entities-to-fhir/SKILL.md`
- `skills/pick-a-pii-model/SKILL.md`
- `skills/benchmark-pii-recall/SKILL.md`

Each skill includes a synthetic runnable snippet and points to a checked-in
repository example.
