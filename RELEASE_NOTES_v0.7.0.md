# OpenMed v0.7.0 — MLX Hardware-Accelerated Inference for macOS & iOS

**Release date:** 2026-04-03

OpenMed v0.7.0 is a major release that brings **native Apple Silicon acceleration** to clinical NLP.
This makes OpenMed the first open-source clinical NLP library with MLX support for Python and a
Swift package for iOS/macOS app development.

---

## Highlights

### Apple MLX Inference Backend

Hardware-accelerated NER and PII detection on Apple Silicon Macs:

```bash
pip install "openmed[mlx]"
```

```python
from openmed import analyze_text
from openmed.core.config import OpenMedConfig

config = OpenMedConfig(backend="mlx")
result = analyze_text(
    "Patient John Doe, SSN 123-45-6789",
    model_name="pii_detection",
    config=config,
)
```

- Pure-MLX BERT implementation with token-classification head
- Automatic model conversion from HuggingFace format (one-time, cached)
- BIO tag decoding with simple/first/average/max aggregation
- Output format identical to HuggingFace — all downstream code works unchanged
- Auto-detection: prefers MLX on Apple Silicon, falls back to PyTorch

### Model Conversion Tools

**HuggingFace → MLX:**
```bash
python -m openmed.mlx.convert \
  --model OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1 \
  --output ./mlx-models/pii-small \
  --quantize 8
```

**HuggingFace → CoreML (for iOS/macOS):**
```bash
python -m openmed.coreml.convert \
  --model OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1 \
  --output ./OpenMedPII.mlpackage \
  --precision float16
```

### Swift Package: OpenMedKit

Drop-in NER for iOS 16+ and macOS 13+ apps:

```swift
// Package.swift dependency
.package(url: "https://github.com/maziyarpanahi/openmed.git", from: "0.7.0")

// Usage
let openmed = try OpenMed(
    modelURL: Bundle.main.url(forResource: "OpenMedPII", withExtension: "mlmodelc")!,
    id2labelURL: Bundle.main.url(forResource: "id2label", withExtension: "json")!
)
let entities = try openmed.analyzeText("Patient John Doe, SSN 123-45-6789")
```

Components:
- `NERPipeline` — CoreML inference with softmax → BIO decoding
- `PostProcessing` — Entity grouping with first/average/max strategies
- `EntityPrediction` — Swift struct matching Python's dataclass
- Uses `swift-transformers` for HuggingFace-compatible tokenization

### Backend Abstraction

New pluggable backend system:

| Backend | Install | Platform | Auto-detect |
|---------|---------|----------|-------------|
| HuggingFace/PyTorch | `pip install openmed[hf]` | Any | Fallback |
| Apple MLX | `pip install openmed[mlx]` | macOS (Apple Silicon) | Preferred |

```python
# Explicit backend selection
config = OpenMedConfig(backend="mlx")   # Force MLX
config = OpenMedConfig(backend="hf")    # Force HuggingFace
config = OpenMedConfig(backend=None)    # Auto-detect (default)
```

---

## Pilot Model

**`OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1`** is the recommended model for MLX/CoreML:
- 44M parameters (BERT architecture)
- ~80MB quantized to 8-bit
- Covers all 18 HIPAA Safe Harbor identifiers
- Extensive test coverage in the existing suite

---

## New Files

| Component | Files | Purpose |
|-----------|-------|---------|
| MLX backend | `openmed/mlx/models/bert_tc.py` | Pure-MLX BERT-TC model |
| | `openmed/mlx/inference.py` | MLX NER pipeline |
| | `openmed/mlx/convert.py` | HF → MLX conversion CLI |
| CoreML export | `openmed/coreml/convert.py` | HF → CoreML conversion CLI |
| Backend layer | `openmed/core/backends.py` | InferenceBackend protocol |
| Swift package | `swift/OpenMedKit/` | iOS/macOS NER library |
| Tests | 37 new tests | Backends, conversion, inference, CoreML |

---

## Next Steps

After tagging v0.7.0:

1. **Convert and upload pilot model** to HuggingFace Hub as MLX format
2. **Convert and publish CoreML model** for Swift developers
3. **Add more architectures** — DeBERTa, ModernBERT (currently BERT-only)
4. **Performance benchmarks** — MLX vs PyTorch CPU vs PyTorch MPS

---

## Test Summary

| Suite | Tests | Status |
|-------|-------|--------|
| Backend abstraction | 11 | All pass |
| MLX conversion | 15 | All pass |
| MLX inference | 4 | All pass |
| CoreML module | 3 | All pass |
| Swift PostProcessing | 7 | Ready (requires Xcode) |
| **Full Python suite** | **697** | **All pass** |

---

**Full Changelog:** https://github.com/maziyarpanahi/openmed/compare/v0.6.4...v0.7.0
