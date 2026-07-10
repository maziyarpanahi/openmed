# Examples & Copy/Paste Recipes

This page curates the most useful samples already in the repository so you can
jump straight to runnable notebooks or scripts. The v1.6, v1.7, and v1.8
examples use synthetic data and are safe to run during release review.

## Notebooks (`examples/notebooks/`)

| Notebook | Highlights |
| --- | --- |
| `getting_started.ipynb` | Mirrors the Quick Start guide with step-by-step installation, registry exploration, and a first call to `analyze_text`. |
| `Sentence_Detection_Batching.ipynb` | Demonstrates pySBD-based segmentation, batching, and how to align predictions back to the original paragraphs. |
| `ZeroShot_NER_Tour.ipynb` | Walks through GLiNER indexing, domain defaults, inference API usage, and the adapter that converts spans into BIO/BILOU schemes. |

Run them with VS Code, Jupyter, or Google Colab—each relies on the same `uv pip install ".[hf]"` baseline.

## Scripts & tools

| Path | What it does |
| --- | --- |
| `examples/pii_model_comparison.py` | Compares multiple PII models across shared sample text and summarizes extraction quality. |
| `examples/pii_batch_processing.py` | Runs batch PII extraction and de-identification with `BatchProcessor(operation=...)`. |
| `examples/pii_multilingual_new_languages.py` | Exercises Dutch, Hindi, Telugu, Portuguese, Arabic, Japanese, and Turkish registry entries, locale-specific regex matches, and optional live extraction with the new public checkpoints. |
| `examples/gradio_deid_app.py` | Interactive Gradio UI to paste synthetic text, pick a `mask`/`replace`/`hash` method, and view the de-identified output plus detected entities (optional `pip install gradio`). |
| `examples/v16_policy_audit_release_gates.py` | Demonstrates v1.6 policy profiles, canonical spans, signed audit reports, review bundles, redaction previews, leakage heatmaps, and k-anonymity metrics without model downloads. |
| `examples/v17_multimodal_browser_interop.py` | Demonstrates v1.7 multimodal and interop surfaces: AsciiDoc offset projection, OCR contracts, chat JSONL, CSV manifests, FHIR, HL7 v2, and Transformers.js browser bundle checks. |
| `examples/privacy_gateway_quickstart.py` | Shows redaction before an external model call and safe re-identification after the protected boundary. |
| `examples/dbt-deidentify/` | Demonstrates the v1.8 warehouse transformation package for table redaction macros and redacted staging models. |
| `examples/spark-streaming/` | Demonstrates Spark structured-streaming de-identification against synthetic records. |
| `examples/first_five_minutes_redact_extract_fhir.py` | Walks through synthetic redaction, deterministic clinical extraction, and FHIR Bundle assembly. |
| `scripts/smoke_gliner.py` | Runs a bounded set of GLiNER models/texts to confirm zero-shot dependencies are installed before releasing. |
| `tests/run-tests.sh` | Convenience runner that stitches together unit, integration, and smoke tests; extend it to include docs builds and API smoke checks. |

Run the v1.6 and v1.7 release examples:

```bash
uv run python examples/v16_policy_audit_release_gates.py
uv run python examples/v17_multimodal_browser_interop.py
```

For the full coverage map, see
[OpenMed v1.6-v1.7 Feature Coverage](./release/v1.6-v1.7-feature-coverage.md).

## v1.7 multimodal, interop, and browser recipes

The v1.7 examples are grouped around the main new public surfaces:

- Multimodal text contracts: `ExtractedDocument`, `SourceSpan`, OCR result
  projection, Markdown/AsciiDoc extraction, and metadata-safe source mapping.
- Structured health data: FHIR `$de-identify`, FHIR Bulk NDJSON, deterministic
  FHIR Bundles, `OperationOutcome`, HL7 v2 field redaction, and CSV/TSV
  PHI-column manifests.
- Browser deployment: ONNX/WebGPU artifacts packaged for Transformers.js, with
  the expected `transformersjs/` file layout checked before publishing.

## v1.8 runtime, deployment, and warehouse recipes

The v1.8 examples and guides focus on cross-platform runtime and production
deployment paths:

- Android/Kotlin and Swift-Kotlin parity: [Android Span Parity](./android-parity.md), [Android ONNX Export](./export-onnx-android.md), and [Swift-Kotlin API Parity](./swift-kotlin-parity.md).
- Browser and mobile JavaScript: [ONNX Runtime Web Loader](./runtimes/onnxruntime-web.md), [Transformers.js Export](./export-transformersjs.md), and the React Native bridge under `js/openmedkit-react-native/`.
- Service operations: [REST Authentication](./serving/authentication.md), [gRPC Service](./serving/grpc.md), [Async REST Jobs & Webhooks](./serving/async-jobs.md), [Serving Resilience](./serving/resilience.md), and [REST Tracing](./serving/tracing.md).
- Structured-data jobs: [Columnar Redactor](./integrations/columnar-redactor.md), [Lakehouse Table Redaction](./integrations/lakehouse-redaction.md), [Dask DataFrame De-identification](./integrations/dask.md), [DuckDB De-identification UDFs](./duckdb-deidentification.md), and `examples/dbt-deidentify/`.

## Apple Silicon & Swift recipes

OpenMed `1.8.1` includes release-critical Apple, Android, browser, and service entry points:

- [MLX Backend](./mlx-backend.md) for Python on Apple Silicon Macs, including Privacy Filter, OpenMed Multilingual Privacy Filter, and experimental GLiNER-family artifacts
- [OpenMedKit (Swift Package)](./swift-openmedkit.md) for macOS, iOS, and iPadOS apps
- [Android ONNX Export](./export-onnx-android.md) and [Android Span Parity](./android-parity.md) for the Kotlin OpenMedKit surface
- [Transformers.js Export](./export-transformersjs.md) for browser token classification through ONNX/WebGPU artifacts

Python MLX quick check:

```bash
uv pip install ".[mlx]"
uv run python -c "from openmed.core.backends import get_backend; print(type(get_backend()).__name__)"
```

Swift MLX quick check:

```swift
import OpenMedKit

let modelDirectory = try await OpenMedModelStore.downloadMLXModel(
    repoID: "OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1-mlx"
)

let openmed = try OpenMed(backend: .mlx(modelDirectoryURL: modelDirectory))
let entities = try openmed.extractPII("Patient John Doe, DOB 1990-05-15")
```

Browser bundle smoke check:

```bash
uv run python -m openmed.onnx.transformersjs \
  --onnx-export-dir dist/example-onnx
```

## Copy-ready snippets

You can find these directly in the docs:

- [Analyze Text Helper](./analyze-text.md) — dict/JSON/HTML/CSV outputs with metadata.
- [REST Service (MVP)](./rest-service.md) — FastAPI endpoints and Docker runbook.
- [LangChain Redaction Wrapper](./integrations-langchain.md) — RAG context redaction before model calls.
- [MLX Backend](./mlx-backend.md) — Apple Silicon Python runtime and artifact packaging.
- [OpenMedKit (Swift)](./swift-openmedkit.md) — native app runtime for macOS, iOS, and iPadOS.
- [Transformers.js Export](./export-transformersjs.md) — browser/WebGPU packaging for token classification.
- [ModelLoader & Pipelines](./model-loader.md) — caching, token helpers, multi-model setups.
- [Advanced NER & Output Formatting](./output-formatting.md) — span filtering and conversions.
- [Zero-shot Toolkit](./zero-shot-ner.md) — indexing, label defaults, and inference APIs.

## Sample automation pipeline

```bash
#!/usr/bin/env bash
set -euo pipefail

uv pip install ".[hf,docs]"
python examples/pii_model_comparison.py > artifacts/result.txt
uv run python examples/v16_policy_audit_release_gates.py > artifacts/v16.json
uv run python examples/v17_multimodal_browser_interop.py > artifacts/v17.json
uv run mkdocs build --strict
python scripts/smoke_gliner.py --limit 1 --threshold 0.5
```

Use this pattern in CI to guarantee models, docs, and zero-shot flows stay healthy before publishing.
