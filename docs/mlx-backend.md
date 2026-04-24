# MLX Backend (Apple Silicon)

OpenMed v1.1.0 expands native Apple Silicon acceleration via [Apple MLX](https://github.com/ml-explore/mlx).

That MLX story now has two surfaces:

- **Python MLX** through `openmed[mlx]` on Apple Silicon Macs
- **Swift MLX** through `OpenMedKit` on Apple Silicon macOS and real iPhone/iPad hardware

## Installation

```bash
# From the repository root
pip install -e ".[mlx]"
```

This installs `mlx`, `huggingface-hub`, `transformers`, `tokenizers`, and `safetensors`.

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

## Architecture Coverage

As of April 24, 2026, the current public MLX path covers these families:

- `bert`
- `distilbert`
- `roberta`
- `xlm-roberta`
- `electra`
- `deberta-v2` / DeBERTa-v3-backed experimental GLiNER-family artifacts
- `openai-privacy-filter`

Python MLX and Swift MLX now share the same artifact contract for OpenMed PII, Privacy Filter, and experimental GLiNER-family tasks.

Architectures still in active rollout:

- `modernbert`
- `longformer`
- `eurobert`
- `qwen3`

That rollout is about making the converter universal and repeatable across the whole OpenMed collection, not just a single pilot checkpoint.

## Fallback Behavior

If MLX is not available (non-Apple hardware, or `mlx` not installed), OpenMed automatically falls back to the HuggingFace/PyTorch backend. No code changes required.

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
