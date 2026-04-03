# MLX Backend (Apple Silicon)

OpenMed v1.0.0 introduces native Apple Silicon acceleration via [Apple MLX](https://github.com/ml-explore/mlx). On Macs with M1/M2/M3/M4 chips, NER and PII inference runs directly on the GPU — up to 10x faster than CPU-only PyTorch.

## Installation

```bash
pip install "openmed[mlx]"
```

This installs `mlx`, `huggingface-hub`, `tokenizers`, and `safetensors`.

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
2. **On-the-fly conversion**: The first time you use a model with MLX, it's automatically converted from HuggingFace format and cached in `~/.cache/openmed/mlx/`.
3. **Identical output**: MLX produces the same entity format as the HuggingFace backend — all downstream processing (entity merging, quality gates, PII detection) works identically.

## Model Conversion

### Automatic (recommended)

Models are converted automatically on first use. No manual step needed.

### Manual conversion

For pre-converting models (e.g., for offline deployment):

```bash
python -m openmed.mlx.convert \
    --model OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1 \
    --output ./mlx-models/pii-small
```

With 8-bit quantization (reduces model size by ~4x):

```bash
python -m openmed.mlx.convert \
    --model OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1 \
    --output ./mlx-models/pii-small-q8 \
    --quantize 8
```

The output directory contains:
- `weights.npz` — Model weights in MLX/NumPy format
- `config.json` — Model architecture configuration
- `id2label.json` — Label ID to entity name mapping

## Supported Models

Currently, the MLX backend supports **BERT-based** token classification models:

| Model | Parameters | Status |
|-------|-----------|--------|
| OpenMed-PII-SuperClinical-Small-44M-v1 | 44M | Supported |
| OpenMed-PII-SuperClinical-Base-110M-v1 | 110M | Supported |
| Other BERT-based NER models | Varies | Supported |

DeBERTa, ModernBERT, and ELECTRA architectures will be added in future releases.

## Fallback Behavior

If MLX is not available (non-Apple hardware, or `mlx` not installed), OpenMed automatically falls back to the HuggingFace/PyTorch backend. No code changes required.

## Conversion Without MLX

You can convert models on any machine (even Linux) — the converter falls back to NumPy format:

```bash
# On Linux CI — produces NumPy .npz (no MLX needed for conversion)
python -m openmed.mlx.convert \
    --model OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1 \
    --output ./mlx-models/pii-small
```

The NumPy `.npz` files are fully compatible with the MLX backend.
