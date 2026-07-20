---
name: running-openmed-ondevice
description: "Run OpenMed models fully on-device with the MLX (Apple Silicon), CoreML (iOS/macOS), or ONNX/WebGPU (cross-platform/browser) backends, including convert-quantize-run workflows. Use when the user wants to deploy OpenMed at the edge, run NER/de-id on Apple Silicon, target iPhone/iPad/Mac, export to ONNX or WebGPU, quantize a clinical model to int8/4-bit, run with no network, or pick between MLX/CoreML/ONNX. Covers the mlx/coreml/onnx extras, the convert() functions and python -m convert CLIs, quantization, loading a local MLX artifact through analyze_text, OpenMedMLXLanguageModel/generate_text, and the on-device-only PHI guarantee (nothing leaves the host)."
license: Apache-2.0
metadata:
  project: OpenMed
  category: deployment-ops
  pairs: adjacent
  version: "1.0"
---

# Running OpenMed on-device

OpenMed runs **fully on-device** by design. These three backends let you take it
further at the edge: **MLX** (Apple Silicon acceleration), **CoreML** (iOS/macOS
/ Neural Engine), and **ONNX / WebGPU** (cross-platform and in-browser). The
flow is the same: **convert → (quantize) → run locally**. Because inference is
local, raw PHI never leaves the device — the strongest privacy posture OpenMed
offers.

## When to use this skill

When you need OpenMed where there is no server: an iOS/macOS app (CoreML),
fast NER/de-id on an Apple Silicon Mac (MLX), or a portable/browser deployment
(ONNX/WebGPU). For a hosted endpoint use `serving-openmed-rest-api`; for an
agent tool use `deploying-openmed-mcp`; for corpora use
`batch-processing-clinical-text`.

## Pick a backend

| Backend | Extra | Best for | Quantization |
| --- | --- | --- | --- |
| **MLX** | `openmed[mlx]` | Apple Silicon Macs; fastest local NER/de-id; on-device LLMs | 4-bit / 8-bit weights |
| **CoreML** | `openmed[coreml]` | iOS/iPadOS/macOS apps, Neural Engine | int8 palettization |
| **ONNX / WebGPU** | `openmed[onnx]` | cross-platform runtimes, browser (transformers.js) | fp16 (WebGPU); int8 via ORT |

## Quick start — MLX (Apple Silicon)

```bash
pip install "openmed[mlx]"

# Convert a HF token-classification model to an OpenMed MLX artifact, 8-bit:
python -m openmed.mlx.convert --model OpenMed/<some-ner-model> --output ./mlx_ner --quantize 8
```

```python
import openmed

# Run NER/de-id through the normal API — pass the local artifact dir as model_name.
# The loader auto-detects the MLX backend from the artifact (or set backend explicitly).
result = openmed.analyze_text(
    "Patient received 75mg clopidogrel for NSTEMI.",
    model_name="./mlx_ner",          # local MLX artifact directory
    output_format="dict",
)

# Force MLX via config if you prefer to be explicit:
from openmed.core.config import OpenMedConfig
cfg = OpenMedConfig(backend="mlx")   # None=auto-detect, "mlx", or "hf"
```

`convert()` is also importable: `openmed.mlx.convert.convert(model_id, output_dir,
quantize_bits=8)`. The CLI accepts `--quantize {4,8}`, `--quantize-group-size`,
`--cache-dir`, and an optional `--eval-suite` to **certify quantized recall**
against the full-precision parent (recommended for clinical models — quantization
can drop recall on rare entities).

### On-device LLM generation (MLX)

```python
from openmed.mlx.lm import generate_text, OpenMedMLXLanguageModel

text = generate_text(
    messages=[{"role": "user", "content": "Summarize: chest pain, troponin elevated."}],
    model_name="OpenMed/laneformer-2b-it-q4-mlx",   # resolves to a local MLX-LM artifact
    max_tokens=128,
)

llm = OpenMedMLXLanguageModel("OpenMed/laneformer-2b-it-q4-mlx")
out = llm.generate(prompt="...", max_tokens=64, temp=0.0)
```

## Quick start — CoreML (iOS/macOS)

```bash
pip install "openmed[coreml]"
python -m openmed.coreml.convert --model OpenMed/<some-ner-model> --output model.mlpackage --quantize int8
```

```python
from openmed.coreml.convert import convert
convert(
    "OpenMed/<some-ner-model>",
    "model.mlpackage",
    compute_units="cpuAndNeuralEngine",   # "all" | "cpuAndNeuralEngine" | "cpuOnly"
    compute_precision="float16",          # float16 for Neural Engine, float32 for CPU
    quantize="int8",                      # emits an int8-palettized sibling .mlpackage
)
```

Bundle the `.mlpackage` in your Xcode app and run it with Core ML; the converter
writes the `id2label` map so your app can decode token labels. Use `float16` +
`cpuAndNeuralEngine` for the Neural Engine; `int8` shrinks the model for
storage-constrained devices.

## Quick start — ONNX / WebGPU

```bash
pip install "openmed[onnx]"
python -m openmed.onnx.convert --model OpenMed/<some-ner-model> --output ./onnx_out
```

```python
from openmed.onnx.convert import convert
res = convert("OpenMed/<some-ner-model>", "./onnx_out", include_webgpu=True, opset=18)
# Emits model.onnx (fp32) and model.webgpu.onnx (fp16) + an export manifest.
```

Run `model.onnx` with ONNX Runtime on any platform, or ship `model.webgpu.onnx`
to the browser via transformers.js for in-page, zero-upload inference. Use
`--no-webgpu` to skip the fp16 artifact.

## Workflow

1. **Pick the backend** for the target (table above).
2. **Convert** the HF/OpenMed model with the matching `convert()` /
   `python -m openmed.<backend>.convert`.
3. **Quantize** if size/latency demands it (MLX 4/8-bit, CoreML int8, WebGPU
   fp16). For clinical de-id/NER, **certify recall** — MLX's `--eval-suite`
   writes a recall-delta report so you don't silently lose rare entities.
4. **Run locally:** MLX artifacts go straight through `analyze_text` /
   `deidentify`; CoreML/ONNX artifacts run in their native runtimes (Core ML,
   ONNX Runtime, transformers.js).
5. **Verify** outputs against the full-precision model before shipping
   (`evaluating-with-leakage-gates` for de-id).

## Hand-off to / from OpenMed

- **Same API surface:** an MLX artifact path is a drop-in `model_name` for
  `openmed.analyze_text` / `deidentify` — downstream skills
  (`building-patient-timelines`, `exporting-to-fhir`) are unchanged.
- **From the catalog:** start from a model chosen via `choosing-openmed-models` /
  `loading-openmed-models`, then convert it here.
- **Eval gate:** pipe quantized de-id output into
  `evaluating-with-leakage-gates` before release.

## Edge cases & gotchas

- **Quantization can hurt clinical recall.** A dropped rare PHI entity is a
  breach. Always benchmark the quantized model vs. full precision (MLX
  `--eval-suite`/recall-delta; manual eval for CoreML/ONNX) and gate on leakage,
  not just F1.
- **MLX is Apple-Silicon only.** On non-Apple hardware the MLX backend isn't
  available and OpenMed falls back to PyTorch; convert/quantize steps that need
  `mlx` will skip quantization with a warning.
- **CoreML compute units matter.** `float16` targets the Neural Engine but some
  ops fall back to CPU; validate latency on a real device, not just the
  simulator.
- **ONNX dynamic axes / opset.** Keep `opset>=18` and verify the model with
  `onnx.checker` (the converter does). token-classification only — these
  converters wrap `AutoModelForTokenClassification`.
- **On-device ≠ no responsibility.** Local inference removes network exposure,
  but the model and any cached output still live on the device — encrypt at
  rest and keep raw PHI out of logs.
- **No license bundling.** Convert your own permissively-licensed models; don't
  embed restricted terminologies in shipped artifacts.

## Standards & references

- Apple MLX: https://github.com/ml-explore/mlx · MLX-LM: https://github.com/ml-explore/mlx-lm
- Core ML Tools: https://apple.github.io/coremltools/
- ONNX: https://onnx.ai/ · ONNX Runtime: https://onnxruntime.ai/
- WebGPU in the browser via transformers.js: https://huggingface.co/docs/transformers.js
- OpenMed source: `openmed/mlx/convert.py` & `openmed/mlx/lm.py`
  (`convert`, `generate_text`, `OpenMedMLXLanguageModel`),
  `openmed/coreml/convert.py` (`convert`), `openmed/onnx/convert.py`
  (`convert`, `export_onnx`, `export_webgpu`), `openmed/core/backends.py`
  (auto-detect), `openmed/core/config.py` (`backend`).
