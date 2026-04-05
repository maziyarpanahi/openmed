# MLX Backend (Apple Silicon)

OpenMed v1.0.0 introduces native Apple Silicon acceleration via [Apple MLX](https://github.com/ml-explore/mlx). On Macs with M1/M2/M3/M4 chips, NER and PII inference runs directly on the GPU while preserving the same entity output format as the Hugging Face backend.

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

1. **Auto-detection**: On Apple Silicon Macs with `mlx` installed, OpenMed automatically selects the MLX backend.
2. **Internal model packaging**: The first time you use a supported model with MLX, OpenMed prepares an MLX-compatible copy in `~/.cache/openmed/mlx/`.
   It now writes `weights.safetensors` by default and falls back to `weights.npz` only when that is the path that works for a given export environment.
3. **Identical output**: MLX produces the same entity format as the HuggingFace backend — all downstream processing (entity merging, quality gates, PII detection) works identically.

The public runtime focuses on automatic preparation at first use. OpenMed's broader cross-architecture conversion work is still being generalized privately across the full model collection.

## Architecture Coverage

As of April 4, 2026, the current MLX runtime code path covers these families:

- `bert`
- `deberta-v2` / DeBERTa-v3-style checkpoints
- `distilbert`
- `roberta`
- `xlm-roberta`
- BERT-tagged/BERT-compatible checkpoints such as the current BioMedELECTRA representative config

Architectures still in active rollout:

- `modernbert`
- `longformer`
- `eurobert`
- `qwen3`

That rollout is about making the converter universal and repeatable across the whole OpenMed collection, not just a single pilot checkpoint.

## Fallback Behavior

If MLX is not available (non-Apple hardware, or `mlx` not installed), OpenMed automatically falls back to the HuggingFace/PyTorch backend. No code changes required.

## MLX and Swift Apps

OpenMed's MLX artifacts are for the **Python/macOS** runtime.

- Use MLX when you are running OpenMed from Python on Apple Silicon.
- Use CoreML + `OpenMedKit` when you are shipping a Swift app on macOS, iOS, or iPadOS.
- Do not try to load MLX weight files such as `weights.safetensors` or `weights.npz` directly from Swift in 1.0.0.
