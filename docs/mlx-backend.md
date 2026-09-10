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
`tokenizers`, `safetensors`, and `Pillow`.

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

### Privacy Filter decoding and compilation

The Python MLX Privacy Filter keeps eager execution as the default. Kernel
compilation is an explicit performance experiment because fused kernels can
produce unit-in-the-last-place floating-point differences:

```python
from openmed.mlx.inference import PrivacyFilterMLXPipeline

pipeline = PrivacyFilterMLXPipeline(
    "/path/to/openai-privacy-filter-mlx",
    compile_forward=True,
)
```

Alternatively, set `OPENMED_MLX_COMPILE=1`. The values `1`, `true`, `yes`, and
`on` are accepted case-insensitively. An explicit `compile_forward=True` or
`compile_forward=False` always overrides the environment variable. Leaving the
argument unset and the variable absent selects eager mode.

BIOES transition tables retain only the most recent bias configuration, so
varying biases cannot grow an unbounded matrix cache. The NumPy decoder and its
pure-Python fallback reject NaN and positive-infinite emissions, state scores,
or biases; negative infinity remains the supported representation for an
impossible emission. Confidence reconstruction is rounded to binary32, matching
the previous MLX probability tensor. Labels, text, and offsets must match
exactly; allow an absolute score tolerance of `1e-6` across MLX devices or
compiled/eager kernels.

#### Reference latency measurement

The following steady-state measurement was collected on August 23, 2026, on an
Apple M2 Max MacBook Pro with 96 GB RAM and macOS 26.5.2. Both revisions used
Python 3.11.10, MLX 0.32.1, NumPy 1.26.4, tiktoken 0.14.0, eager mode, and the
same locally cached `OpenMed/privacy-filter-mlx` artifact at revision
`833fa7ea3fd36148900deea2d55bdadd4b90efa9`. Model loading was excluded. Each
cell is the median of 12 synchronized pipeline calls after three warmups over
synthetic text encoded to the exact sequence length.

| Batch | Tokens | Before, ms | Optimized, ms | Change |
|---:|---:|---:|---:|---:|
| 1 | 32 | 21.947 | 21.704 | -1.1% |
| 1 | 64 | 38.357 | 37.131 | -3.2% |
| 2 | 32 | 38.141 | 37.408 | -1.9% |
| 2 | 64 | 70.745 | 67.745 | -4.2% |
| 4 | 32 | 69.887 | 67.300 | -3.7% |
| 4 | 64 | 132.328 | 127.491 | -3.7% |

“Before” is upstream revision `55232ecf`; “Optimized” is the repaired #2946
working tree based on `61e318b2`. To reproduce the method, load one pipeline per
revision, generate text by decoding the first `N` tokens of a repeated synthetic
sentence and assert that re-encoding returns exactly `N` tokens, run each batch
and sequence-length pair three times, then time 12 further calls with
`time.perf_counter()`. A synthetic person/email/phone fixture produced identical
entity JSON and confidence values before and after the optimization.

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

### Python Vision-Language Quick Start

OpenMed includes a native Cohere Compass runtime for the North Micro Vision
MLX family. It consumes data-only model artifacts directly:

```python
from openmed.mlx import OpenMedMLXVisionLanguageModel

model = OpenMedMLXVisionLanguageModel(
    "OpenMed/North-Micro-Vision-Instruct-6bit-mlx"
)

text = model.generate(
    "Explain why local processing can improve clinical-document privacy.",
    max_tokens=96,
)
document = model.generate(
    "List the visible medication and dose.",
    image="synthetic-clinical-note.png",
    max_tokens=128,
)
```

The first load downloads the selected artifact from Hugging Face. Pass a local
model directory instead for an air-gapped deployment. Images and prompts are
processed locally; OpenMed does not add telemetry or a cloud inference
fallback. Treat generated text as sensitive and validate it against the source
image before consequential use.

The compatible repositories use one naming family:

- `OpenMed/North-Micro-Vision-Instruct-4bit-mlx`
- `OpenMed/North-Micro-Vision-Instruct-5bit-mlx`
- `OpenMed/North-Micro-Vision-Instruct-6bit-mlx`
- `OpenMed/North-Micro-Vision-Instruct-8bit-mlx`
- `OpenMed/North-Micro-Vision-Instruct-bf16-mlx`

### Maple Preview: structured clinical tasks

OpenMed resolves `maple`, `maple-preview`, and the original
`deepgrove/maple-preview` id to the pinned
`deepgrove/maple-preview-2bit-mlx` artifact. The wrapper supports validated PII
spans and redaction, clinical entities, directed relations, and note-grounded
answers:

```python
from openmed import MapleClinicalAssistant, MapleTask

assistant = MapleClinicalAssistant("maple")
note = "Synthetic patient Alex Doe takes metformin 500 mg daily."

masked = assistant.complete_task(MapleTask.PII, note)
relations = assistant.complete_task(MapleTask.RELATIONS, note)
answer = assistant.complete_task(
    MapleTask.REASONING,
    masked.redacted_text or "",
    question="Which medication and dose are documented?",
)
```

The model is loaded lazily. Custom executable model code is pinned to an
immutable Hub revision, and the task parser rejects out-of-bounds spans,
unknown relation endpoints, extra schema keys, and malformed JSON. See
[Maple Preview on device](maple-on-device.md) for native iOS, Android, browser,
and mixed-bit export workflows.

### Paged KV Cache for Long Notes

Long clinical-note prompts can opt into OpenMed's paged KV-cache planning for
MLX-LM generation. The plan uses a fixed page pool, chunked prompt prefill, and
a sliding in-memory window when a prompt exceeds the configured budget:

```python
from openmed.mlx import OpenMedMLXLanguageModel, PagedKVCacheConfig

runner = OpenMedMLXLanguageModel("OpenMed/laneformer-2b-it-q4-mlx")
cache = PagedKVCacheConfig(
    memory_budget_bytes=512 * 1024 * 1024,
    page_size_tokens=128,
    chunk_size_tokens=512,
    # Set this from the loaded model's KV footprint when known.
    bytes_per_token=65_536,
)

response = runner.generate(
    long_note_prompt,
    max_tokens=256,
    paged_kv_cache=cache,
)
print(runner.last_paged_kv_cache_plan.to_dict())
```

The exact dense-cache context supported by a budget is:

```text
floor(memory_budget_bytes / (page_size_tokens * bytes_per_token)) * page_size_tokens
```

For example, with 128-token pages:

| Budget | Bytes per cached token | Exact context before eviction |
|---:|---:|---:|
| 256 MiB | 65,536 | 4,096 tokens |
| 512 MiB | 65,536 | 8,192 tokens |
| 1 GiB | 65,536 | 16,384 tokens |

Prompts at or below that exact context keep byte-identical generation inputs to
the dense-cache path while using chunked prefill. Longer prompts degrade
gracefully by bounding resident KV pages to the configured window and recording
the older tokens that require recompute/eviction accounting; tokens inside the
resident window remain exact.

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
- `cohere-compass` for native-resolution text-and-image generation through
  `OpenMedMLXVisionLanguageModel`

Python MLX and Swift MLX now share the same artifact contract for OpenMed PII, Privacy Filter, OpenAI Nemotron Privacy Filter, OpenMed Multilingual Privacy Filter, and experimental GLiNER-family tasks. The Arabic/Japanese/Turkish PII rollout adds 28 supported `-mlx` repos now; unsupported ModernBERT, Qwen3, and Longformer PII checkpoints remain deferred until those architectures land in the converter.

MLX-LM causal text generation is a separate Python-only artifact contract. It
uses MLX-LM files such as `model.safetensors`, tokenizer assets, `config.json`,
and custom `model_file` implementations when needed. Laneformer support is
available through `OpenMed/laneformer-2b-it-q4-mlx`. Maple Preview support uses
the pinned `deepgrove/maple-preview-2bit-mlx` custom MLX-LM artifact and a
native OpenMedKit implementation for Swift. Cohere Compass vision-language
generation instead uses the shared OpenMed Python/OpenMedKit contract described
above.

Architectures still in active rollout:

- `modernbert`
- `longformer`
- `eurobert`
- `qwen3`

That rollout is about making the converter universal and repeatable across the whole OpenMed collection, not just a single pilot checkpoint.

## Fallback Behavior

If MLX is not available (non-Apple hardware, or `mlx` not installed), OpenMed automatically falls back to the HuggingFace/PyTorch backend. No code changes required.

That automatic fallback applies to the token-classification backend. MLX-LM
text generation and the Compass vision-language runtime require Apple Silicon,
the `openmed[mlx]` dependencies, and a supported OpenMed MLX artifact.

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
