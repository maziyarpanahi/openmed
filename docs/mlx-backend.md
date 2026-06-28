# MLX Backend (Apple Silicon)

OpenMed v1.5.5 expands native Apple Silicon acceleration via [Apple MLX](https://github.com/ml-explore/mlx), including preconverted Arabic, Japanese, and Turkish PII token-classification artifacts.

That MLX story now has two surfaces:

- **Python MLX** through `openmed[mlx]` on Apple Silicon Macs
- **Swift MLX** through `OpenMedKit` on Apple Silicon macOS and real iPhone/iPad hardware

## Installation

```bash
# From the repository root
pip install -e ".[mlx]"
```

This installs `mlx`, `mlx-lm`, `huggingface-hub`, `transformers`,
`tokenizers`, and `safetensors`.

## Quick Start

```python
from openmed import analyze_text
from openmed.core.config import OpenMedConfig

# MLX is auto-detected on Apple Silicon — no config needed
result = analyze_text(
    "Patient John Doe, DOB 1990-05-15, SSN 123-45-6789",
    model_name="pii_detection",
)
print(result.entities)
```

### Python MLX-LM Quick Start

OpenMed also exposes MLX-LM causal language models through the same
`openmed[mlx]` extra. The first supported model is the private
`OpenMed/laneformer-2b-it-q4-mlx` conversion of
`kogai/laneformer-2b-it`.

```python
from openmed import generate_text

response = generate_text(
    messages=[
        {
            "role": "user",
            "content": "Explain why local clinical language models matter.",
        }
    ],
    model_name="OpenMed/laneformer-2b-it-q4-mlx",
    max_tokens=128,
)
print(response)
```

Use `OpenMed/laneformer-2b-it-q4-mlx` to request the preconverted OpenMed MLX
artifact explicitly. The resolver also accepts these aliases:

- `kogai/laneformer-2b-it`
- `laneformer-2b-it`
- a local directory containing the converted MLX-LM artifact

For explicit reuse across several prompts, keep the model loaded:

```python
from openmed.mlx import OpenMedMLXLanguageModel

runner = OpenMedMLXLanguageModel("OpenMed/laneformer-2b-it-q4-mlx")
print(runner.generate("Define delayed tensor parallelism.", max_tokens=128))
```

To force a specific backend:

```python
config = OpenMedConfig(backend="mlx")   # Force MLX
config = OpenMedConfig(backend="hf")    # Force HuggingFace/PyTorch
config = OpenMedConfig(backend=None)    # Auto-detect (default)
```

## How It Works

1. **Auto-detection**: On Apple Silicon Macs with `mlx` installed, OpenMed automatically selects the Python MLX backend.
2. **Artifact packaging**: Supported conversions now produce a self-contained MLX artifact with:
   - `openmed-mlx.json`
   - `config.json`
   - `id2label.json`
   - tokenizer assets
   - `weights.safetensors` by default
   - `weights.npz` as a fallback when needed
3. **Shared contract**: That same MLX artifact shape is now the contract for both Python MLX and Swift MLX.
4. **Identical output shape**: MLX produces the same entity format as the HuggingFace backend, so downstream entity merging and PII handling stay consistent.

The public runtime focuses on automatic preparation at first use. OpenMed's broader cross-architecture conversion work is still being generalized privately across the full model collection.

Quantized export certification is documented in
[MLX Quantized Export Certification](export-mlx-quant.md), including INT4
recall-delta reports and the `certified` manifest field.

## Architecture Coverage

As of May 4, 2026, the current public MLX path covers these families:

- `bert`
- `distilbert`
- `roberta`
- `xlm-roberta`
- `electra`
- `deberta-v2` / DeBERTa-v3-backed experimental GLiNER-family artifacts
- `openai-privacy-filter`
- `privacy-filter-nemotron` / `privacy-filter-multilingual` artifacts through the OpenAI Privacy Filter runtime

Python MLX and Swift MLX now share the same artifact contract for OpenMed PII, Privacy Filter, OpenAI Nemotron Privacy Filter, OpenMed Multilingual Privacy Filter, and experimental GLiNER-family tasks. The Arabic/Japanese/Turkish PII rollout adds 28 supported `-mlx` repos now; unsupported ModernBERT, Qwen3, and Longformer PII checkpoints remain deferred until those architectures land in the converter.

MLX-LM text generation is a separate Python-only artifact contract. It uses
MLX-LM files such as `model.safetensors`, tokenizer assets, `config.json`, and
custom `model_file` implementations when needed. Laneformer support is
available through `OpenMed/laneformer-2b-it-q4-mlx`.

Architectures still in active rollout:

- `modernbert`
- `longformer`
- `eurobert`
- `qwen3`

That rollout is about making the converter universal and repeatable across the whole OpenMed collection, not just a single pilot checkpoint.

## Fallback Behavior

If MLX is not available (non-Apple hardware, or `mlx` not installed), OpenMed automatically falls back to the HuggingFace/PyTorch backend. No code changes required.

That automatic fallback applies to the token-classification backend. MLX-LM
text generation requires `mlx-lm` and a supported MLX runtime.

## MLX and Swift Apps

OpenMedKit can now load supported OpenMed MLX artifacts directly in Swift.

- Use Python MLX when you are running OpenMed from Python on Apple Silicon.
- Use Swift MLX when you want the same supported MLX artifact to run in an Apple app on:
  - Apple Silicon macOS
  - a real iPhone/iPad device
- Use CoreML when you already have a bundled Apple model package or need a fallback path outside Swift MLX.

Swift MLX does **not** target iOS Simulator.

### Swift MLX Quick Start

```swift
import OpenMedKit

let modelDirectory = try await OpenMedModelStore.downloadMLXModel(
    repoID: "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-mlx"
)

let openmed = try OpenMed(
    backend: .mlx(modelDirectoryURL: modelDirectory)
)

let entities = try openmed.analyzeText(
    "Patient John Doe, DOB 1990-05-15, SSN 123-45-6789"
)
```

See [OpenMedKit (Swift)](swift-openmedkit.md) for the full Swift setup flow.
