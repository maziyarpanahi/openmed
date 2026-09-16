# Maple Preview on device

OpenMed supports DeepGrove's
[`maple-preview`](https://huggingface.co/deepgrove/maple-preview) through one
privacy-first task contract and three native demo surfaces. The preferred Apple
artifact is the published
[`maple-preview-2bit-mlx`](https://huggingface.co/deepgrove/maple-preview-2bit-mlx)
checkpoint. Model weights are intentionally not committed to OpenMed.

Maple Preview is a research model with limited post-training, not a clinical
model or medical device. It can omit identifiers, extract the wrong entity,
link unrelated concepts, or generate unsupported text. Every workflow marks
the result for human review and must not trigger diagnosis, treatment,
disclosure, or another consequential decision automatically.

## Pinned artifacts

OpenMed resolves the aliases `maple`, `maple-preview`, and
`deepgrove/maple-preview` to the 2-bit MLX repository. Executable custom model
code is pinned rather than loaded from a mutable branch.

| Artifact | Immutable revision | Use |
| --- | --- | --- |
| `deepgrove/maple-preview` | `ac1ddd79d2b5cb4406f5d2bebdf95406ce505a07` | BF16 source for audited conversion |
| `deepgrove/maple-preview-2bit-mlx` | `361db5da5e74ff6fcdd852d478e1f266ce11013a` | Python MLX-LM and native Swift MLX |

The 2-bit repository is about 5.3 GB. Plan for additional working memory and KV
cache, use a recent high-memory device, and test thermal behavior and peak RAM
on every target class before release.

## Python MLX tasks

Install the optional backend, then let OpenMed download the pinned snapshot on
first use:

```bash
python -m pip install --upgrade "openmed[mlx]"
```

```python
from openmed import MapleClinicalAssistant, MapleTask

note = "Synthetic patient Alex Doe takes metformin 500 mg daily."
maple = MapleClinicalAssistant("maple")

pii = maple.complete_task(MapleTask.PII, note)
print(pii.redacted_text)

entities = maple.complete_task(MapleTask.ENTITIES, note)
relations = maple.complete_task(MapleTask.RELATIONS, note)
answer = maple.complete_task(
    MapleTask.REASONING,
    pii.redacted_text or "",
    question="Which medication and dose are documented?",
)
```

Structured results retain validated offsets and labels but omit copied source
surfaces by default. PII replacement is deterministic after span validation.
Reasoning exposes a concise answer, uncertainty, and evidence offsets—not
hidden chain-of-thought. Do not log or persist prompts when they may contain
protected information.

## Reproducible 4-bit and 8-bit MLX variants

The official 2-bit artifact is used directly. OpenMed's exporter can produce
separate 4-bit and 8-bit variants from the immutable BF16 source without
overwriting an existing export:

```bash
python -m openmed.mlx.maple_export \
  --output "$HOME/Developer/openmed-mlx-export/artifacts" \
  --bits 4 8 \
  --dry-run
```

Remove `--dry-run` on an Apple Silicon host with enough disk and memory. Each
output includes `openmed-maple-export.json` with source and runtime-code
revisions, quantization parameters, and an explicit unvalidated status. Before
publication, run synthetic task-contract checks, direct-identifier recall and
critical-leakage evaluation, and quantized-vs-source quality comparisons.

Development smoke tests on Apple Silicon with the revisions above produced the
following result. These are task-contract checks, not clinical-quality claims:

| MLX variant | Result on synthetic PII, entities, relations, reasoning, and chat |
| --- | --- |
| Published 2-bit | Passed all task contracts with source-derived offsets |
| Exported 4-bit, group 128 | Converted and loaded, but exhausted 4,096 tokens before final JSON; do not deploy |
| Exported 8-bit, group 128 | Passed all task contracts and emitted grounded visible chat |

The failed 4-bit result is deliberately retained as a release warning: a
successful tensor conversion is not evidence of useful or safe generation.

## iOS and iPadOS scanning studio

[`swift/OpenMedScanDemo/`](https://github.com/maziyarpanahi/openmed/tree/master/swift/OpenMedScanDemo)
now offers Maple Preview in the existing document-scanning flow:

1. VisionKit scans the page and Vision performs OCR.
2. Maple performs PII removal and the UI presents masked spans for review.
3. The same local runtime extracts clinical entities and directed relations.
4. Maple Insights creates a document-grounded brief and streams chat answers
   over the masked note after suppressing Maple's private reasoning segment.
   Prompt-driven entity and relation JSON remains buffered until its spans,
   label vocabulary, and relation endpoints validate.

The OpenMedKit package contains the native `OpenMedMaple` runtime and Maple MLX
architecture implementation. The demo downloads the three exact-head shards at
the pinned revision, rejects a partial model directory, and reuses the local
cache for disconnected inference. Run it on real Apple hardware; the MLX path
does not target iOS Simulator.

See the [scan demo README](https://github.com/maziyarpanahi/openmed/blob/master/swift/OpenMedScanDemo/README.md) and
[OpenMedKit guide](swift-openmedkit.md#maple-preview-local-generative-tasks).

## Android clinical studio

[`android/OpenMedMapleDemo/`](https://github.com/maziyarpanahi/openmed/tree/master/android/OpenMedMapleDemo)
is a responsive Compose demo for PII removal, entity extraction, relation
extraction, and reasoning/chat. It declares no internet permission. A model is
selected through Android's document picker and streamed into protected app
storage only after every declared size and SHA-256 checksum passes.

Maple's MLX tensors are not ONNX graphs. OpenMed's reproducible exporter starts
from the immutable BF16 source and emits one cached decoder graph. The same
`model.onnx` is used for prompt prefill with empty caches and token decode with
the preceding `present.*` outputs. It uses the following contract:

- `input_ids`: INT64 `[batch, sequence]`;
- `attention_mask`: INT64 `[batch, total_sequence]`;
- `past_key_values.{0..23}.{key,value}`: FLOAT32
  `[batch, 4, past_sequence, 128]`;
- `present.{0..23}.{key,value}`: FLOAT32
  `[batch, 4, total_sequence, 128]`; and
- `logits`: FLOAT32 `[batch, sequence, 151936]`.

There is no `position_ids` graph input. GroupQueryAttention owns rotary
positions on each three-layer sliding-attention run, while every fourth global
attention layer intentionally uses NoPE. Android creates zero-length
`[1, 4, 0, 128]` caches for prefill and feeds each `present.*` tensor back to
its matching `past_key_values.*` input during decode.

### Reproducible portable export

Install the exact conversion stack printed by the exporter. The pinned set is
`accelerate==1.11.0`, `huggingface-hub==0.35.3`, `numpy==2.2.6`,
`onnx==1.21.0`, `onnx-ir==0.2.0`, `onnxruntime==1.25.1`,
`onnxruntime-genai==0.12.0`, `safetensors==0.6.2`,
`tokenizers==0.22.1`, `torch==2.9.1`, `tqdm==4.67.1`, and
`transformers==4.57.1`:

```bash
python -m openmed.onnx.maple_export requirements \
  > /tmp/openmed-maple-onnx-requirements.txt
uv pip install --requirement /tmp/openmed-maple-onnx-requirements.txt
```

Acquire the exact source revision, inspect a dry-run plan, run the conversion,
and package its integrity-bound receipt:

```bash
python -m openmed.onnx.maple_export download /path/to/maple-bf16
python -m openmed.onnx.maple_export export \
  /path/to/maple-bf16 /path/to/maple-onnx-mobile \
  --target mobile --dry-run
python -m openmed.onnx.maple_export export \
  /path/to/maple-bf16 /path/to/maple-onnx-mobile \
  --target mobile
python -m openmed.onnx.maple_export bundle \
  /path/to/maple-onnx-mobile /path/to/maple-android.ommaple.zip
```

The export is deliberately non-overwriting. Its
`openmed-maple-onnx-export.json` records the source SHA, exact toolchain,
artifact sizes and SHA-256 hashes, structural graph checks, and uncompleted
runtime/parity/device gates. The bundle validator checks those files again:

```bash
python -c 'from openmed.onnx import validate_maple_onnx_bundle as v; print(v("/path/to/maple-android.ommaple.zip").manifest["source_revision"])'
```

The same Python validator can audit a received bundle without extracting it:

```python
from openmed.onnx import validate_maple_onnx_bundle

validated = validate_maple_onnx_bundle("maple-android.ommaple.zip")
print(validated.manifest["source_revision"])
```

No converted Android weights are distributed in the source repository. The
full BF16 snapshot is roughly 40 GB, so conversion is a high-memory release
operation rather than a unit-test job. See the
[Android demo contract](https://github.com/maziyarpanahi/openmed/blob/master/android/OpenMedMapleDemo/README.md).

### Why Android portable ONNX remains four-bit

The Apple MLX artifact remains DeepGrove's published two-bit checkpoint. The
Android/mobile ONNX path is separately quantized from BF16 to symmetric
four-bit experts with 128-value blocks. It uses the official
[`com.microsoft.QMoE`](https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftqmoe)
operator with interleaved gate/up rows, fused clamped SwiGLU, no separate FC3,
top-8 normalized routing, and FP32 router MatMuls.

Although the current QMoE schema describes a two-bit attribute value, the
ONNX Runtime 1.25.1 CPU/mobile kernel tested here rejects two-bit experts and
accepts four- or eight-bit experts. It also rejects the unfused separate-FC3
form for this kernel. Consequently, OpenMed keeps the Android artifact at four
bits and does not claim that its CPU/mobile runtime supports the browser's
two-bit graph. The browser path is a separate two-bit export using the patched
ONNX Runtime 1.26 native WebGPU provider described below.

The checked-in synthetic probe validates that the exact four-bit fused QMoE
form loads and executes on ONNX Runtime 1.25.1 CPU without downloading Maple:

```bash
python -m openmed.onnx.maple_qmoe_smoke
```

That probe establishes operator compatibility only. It does not establish a
successful full-checkpoint conversion, source parity, target-device execution,
quality, memory use, or useful generation.

A development conversion of the complete pinned BF16 checkpoint also passed
the stricter graph contract on Apple Silicon. Both generated graphs lowered
RMS normalization to standard ONNX arithmetic, loaded in raw ONNX Runtime
1.25.1, and completed a real one-token prefill plus cached decode with finite
`[1, 1, 151936]` logits and KV growth from one to two tokens:

| Export target | I/O | Full graph result |
| --- | --- | --- |
| `mobile` | FLOAT32 | 24 QMoE + 24 GQA layers; CPU prefill/decode passed |
| `webgpu` | FLOAT16 | 24 QMoE + 24 GQA layers; CPU prefill/decode passed |

This is conversion and CPU-runtime evidence, not source-logit parity or an
Android target-device claim. A corrected two-bit browser graph has since passed
a real Chrome WebGPU run as documented below; physical Android execution
remains a separate release gate.

## Web browser clinical studio

[`docs/demo/web/`](/docs/demo/web/) is a standalone, responsive browser experience
with Maple PII, entity, and relation workflows plus persistent streamed chat
beneath their results. Its bottom conversation picker can use either the loaded
Maple runtime or a separate LiquidAI LFM2.5 2.6B Q4F16 runtime while preserving
the completed extraction. It includes keyboard-accessible task tabs,
model-cache controls, and synthetic UI previews. Serve the repository locally:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000/docs/demo/web/`. The page accepts audited,
same-origin model packs only. Its Content Security Policy has no cloud-inference
fallback, and notes and outputs are never placed in browser storage.

Stock Transformers.js cannot load Maple's custom MLX checkpoint by itself. The
browser adapter uses the same unified cache names and dimensions as Android,
but the `webgpu` export uses FLOAT16 caches and logits. OpenMed replaces all 96
attention projection tensors and all 24 MoE expert tensors with the official
coherent 2-bit MLX ternary weights without a lossy requantization step.
Embeddings and the language-model head retain their audited four-bit base-export
representation, while routers remain FP32 and norms remain FP16:

```bash
python -m openmed.onnx.maple_export export \
  /path/to/maple-bf16 /path/to/maple-onnx-webgpu \
  --target webgpu --dry-run
python -m openmed.onnx.maple_export export \
  /path/to/maple-bf16 /path/to/maple-onnx-webgpu \
  --target webgpu
python -m openmed.onnx.maple_export repack-2bit-webgpu \
  /path/to/maple-onnx-webgpu /path/to/maple-preview-2bit-mlx \
  /path/to/maple-2bit-onnx-webgpu-coherent
python -m openmed.onnx.maple_export write-web-manifest \
  /path/to/maple-2bit-onnx-webgpu-coherent
```

The runtime adapter and model pack must implement the contract in the
[browser demo README](https://github.com/maziyarpanahi/openmed/blob/master/docs/demo/web/README.md). Browser applications normally serve
the validated ONNX, external-data, tokenizer, and configuration files
individually rather than ask the page to unzip a multi-gigabyte bundle. The
corrected pack is approximately 5.0 GB and contains `model.onnx` plus 13
external-data shards, `model.onnx.data.000` through `model.onnx.data.012`. The
2-bit graph requires the pinned OpenMed ONNX Runtime 1.26 native WebGPU build
with Asyncify WASM support and the narrow QMoE patch; the demo rejects an
unmarked stock runtime. This path does not use ONNX Runtime's JSEP build.

### Selectable LFM2.5 chat

The optional conversation runtime pins
[`LiquidAI/LFM2.5-2.6B-ONNX`](https://huggingface.co/LiquidAI/LFM2.5-2.6B-ONNX)
at revision `66826372fd4fa166f53be0371c9315745c07cace` and uses its official
Q4F16 graph. The graph and two external-data files total exactly
1,534,153,680 bytes; the configuration, chat template, tokenizer, and license
bring the same-origin pack to 1,552,083,193 bytes. Download the exact nine-file
layout with the pinned command in the
[browser demo README](https://github.com/maziyarpanahi/openmed/blob/master/docs/demo/web/README.md#optional-lfm25-conversation-model).

Select **LFM2.5 · 2.6B Q4F16** at the bottom of the page, or use
`?chat_model=lfm25&lfm_model=./models/lfm2.5-2.6b-onnx-q4f16/`. Its dedicated
module worker uses vendored Transformers.js 4.2.0 and ONNX Runtime Web with
remote model loading disabled. Model-asset caching follows the explicit UI
toggle; prompts and outputs are never persisted. Loading it releases Maple's
GPU session so both models are not resident simultaneously; the extraction
remains visible. Real token deltas stream from the worker, then the page
validates the grounded answer, exact evidence quotes, and uncertainty.
Hidden-reasoning markers are rejected from visible output.

LFM2.5 is distributed under the custom **LFM Open License v1.0**, not a
permissive open-source license. Commercial use is licensed only below the
US$10-million annual-revenue threshold. Redistribution must include the license,
mark modifications, and retain applicable notices and attribution. Do not
mirror the pack under OpenMed without explicit license review.

A development validation in actual Chrome 151 loaded the full corrected graph
through the native WebGPU provider (`apple · metal-3`). Direct streamed PII
returned three exact identifier surfaces in 12.9 seconds. Prompted entity
extraction returned the exact medication, dosage, condition, and symptom
surfaces in 19.1 seconds (125 tokens at 7.0 tokens/second), after which the
browser derived offsets locally. Grounded chat reached its first visible token
in 946 ms and completed 108 tokens in 16.2 seconds at 7.1 tokens/second with two
exact evidence quotes and explicit uncertainty. This is one development-machine
coherence and runtime measurement, not a performance or clinical-quality
guarantee. Peak memory and the full release suite still need to pass on every
supported browser and device.

## Release gates

Do not call any Maple artifact production-ready until all of these pass on the
specific quantization and platform:

- structured JSON and span-integrity tests for every task;
- direct-identifier recall, critical leakage, and Unicode/grapheme fixtures;
- relation endpoint and evidence-offset validation;
- source-versus-quantized quality deltas on synthetic or approved eval data;
- cache/no-cache generation parity where both graphs exist;
- peak memory, time-to-first-token, sustained throughput, and thermal tests;
- offline reruns after model acquisition; and
- human review of hallucination, uncertainty, and prompt-injection behavior.

Committed fixtures must remain synthetic. Keep restricted clinical datasets
and all raw protected information outside model bundles, logs, caches, and
test artifacts.
