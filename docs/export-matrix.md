# Export Format Support Matrix

Which combinations of architecture family, export backend, and precision tier
does OpenMed actually support today? This page is the single answer, kept
honest against the code by a coherence test in
`tests/unit/test_export_matrix.py`.

Use it before you plan a conversion so you know up front whether your
architecture reaches your target runtime.

## Support levels

- ✅ **Supported** — a first-class path with a documented converter, unit
  tests, and (where relevant) a certification gate.
- ⚠️ **Partial / conditional** — path exists but has a documented restriction
  (for example a family allowlist, or "certified only when recall holds").
- ❌ **Not supported** — no converter, or explicitly rejected by the backend.

The matrix rows are the Hugging Face model-type strings and OpenMed family
names most contributors search for. The **MLX aliases** column lists the
token-classification aliases understood by
`openmed.mlx.models._SUPPORTED_TOKEN_CLASSIFICATION_MODEL_TYPES` (or the MLX
custom-family set). The coherence test asserts every entry in that allowlist
appears here as a backtick-wrapped alias, so the matrix cannot drift silently
when a new family is added to the code.

## Matrix

| Family | MLX aliases | MLX fp | MLX INT8 | MLX INT4 | CoreML fp16 | CoreML INT8 | ONNX (default) | ONNX Android | Transformers.js | GGUF (embedding) |
|---|---|---|---|---|---|---|---|---|---|---|
| BERT | `bert` | ✅ | ✅ | ⚠️ INT4-if-recall-holds | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ backbone only (see limitations) |
| DistilBERT | `distilbert` | ✅ (via bert) | ✅ | ⚠️ INT4-if-recall-holds | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ backbone only |
| Electra | `electra` | ✅ (via bert) | ✅ | ⚠️ INT4-if-recall-holds | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ backbone only |
| RoBERTa | `roberta` | ✅ (via bert) | ✅ | ⚠️ INT4-if-recall-holds | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ backbone only |
| XLM-RoBERTa | `xlm-roberta`, `xlm_roberta` | ✅ (via bert) | ✅ | ⚠️ INT4-if-recall-holds | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ backbone only |
| DeBERTa-v2 | `deberta`, `deberta-v2` | ✅ | ✅ | ⚠️ INT4-if-recall-holds | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ backbone only |
| ModernBERT | `modernbert`, `modern-bert` | ✅ | ✅ | ⚠️ INT4-if-recall-holds | ⚠️ qualification-gated, static buckets | ⚠️ INT8-if-recall-holds | ⚠️ qualification-gated (CPU provider; WebGPU fp16 is a CPU proxy) | ⚠️ qualification-gated (CPU provider) | ❌ rejected, no qualified path | ❌ |
| Longformer | `longformer` | ✅ | ✅ | ⚠️ INT4-if-recall-holds | ⚠️ qualification-gated, attention-window buckets | ⚠️ INT8-if-recall-holds | ⚠️ qualification-gated (CPU provider, explicit global attention mask; WebGPU fp16 is a CPU proxy) | ⚠️ qualification-gated (CPU provider, two inputs, global attention off) | ❌ rejected, no qualified path | ❌ |
| OpenAI Privacy Filter | `openai-privacy-filter`, `privacy-filter`, `privacy-filter-nemotron`, `nemotron-privacy-filter`, `privacy-filter-multilingual`, `multilingual-privacy-filter` | ✅ | ✅ | ⚠️ INT4-if-recall-holds | ⚠️ qualification-gated, declared checkpoints, pinned revision, static buckets | ⚠️ INT8-if-recall-holds | ⚠️ qualification-gated (CPU provider; WebGPU fp16 is a CPU proxy; unoptimized graph only) | ⚠️ qualification-gated; batch converts only declared, qualified variants | ❌ rejected, no qualified path | ❌ |
| GLiNER (span) | `gliner-uni-encoder-span` | ✅ (MLX custom family) | ✅ | ⚠️ INT4-if-recall-holds | ❌ typed refusal (`unsupported_runtime`) | ❌ | ⚠️ qualification-gated profile `gliner-span-uni-encoder-v1` (CPU provider, fp32, bounded) | ❌ not qualified | ❌ not qualified | ❌ |
| GLiClass | `gliclass-uni-encoder` | ✅ (MLX custom family) | ✅ | ⚠️ INT4-if-recall-holds | ❌ typed refusal (`unsupported_runtime`) | ❌ | ⚠️ qualification-gated profile `gliclass-uni-encoder-v1` (CPU provider, fp32, bounded) | ❌ not qualified | ❌ not qualified | ❌ |
| GLiNER (token-relex) | `gliner-uni-encoder-token-relex` | ✅ (MLX custom family) | ✅ | ⚠️ INT4-if-recall-holds | ❌ rejected (`unsupported_task`) | ❌ | ❌ rejected (`unsupported_task`) | ❌ | ❌ | ❌ |

## Known limitations

The cells above encode a handful of load-bearing constraints; the source of
truth is the backend code, but the essentials are:

- **GGUF is embedding-only.** `openmed.gguf.convert` deliberately rejects
  token-classification heads: llama.cpp can run the BERT-family encoder
  embedding path but does not expose the token classifier head required to
  produce OpenMed entity labels. Export the encoder backbone (for example a
  checkpoint whose `config.json` names `BertModel` rather than
  `BertForTokenClassification`) instead. See
  [GGUF Embedding Export](export-gguf.md).
- **CoreML is token-classification-only and enforces a family allowlist.**
  `openmed.coreml.convert.SUPPORTED_TOKEN_CLASSIFICATION_MODEL_TYPES` accepts
  `bert`, `distilbert`, `electra`, `roberta`, `xlm-roberta`, `deberta-v2`,
  `modernbert`, `longformer`, and `openai-privacy-filter`. Any other
  architecture fails before tracing with the `unsupported_family` reason and
  an error that lists the supported families. GLiNER and GLiClass checkpoints
  declare a DeBERTa backbone, so they are refused first with
  `unsupported_runtime` (relation extraction with `unsupported_task`) instead
  of being traced as a token classifier. See
  [CoreML Packaging](coreml-export.md).
- **GLiNER span and GLiClass exports are label-conditioned and bounded.** They
  export through `openmed.onnx.zero_shot_export`, not the token-classification
  converter, under versioned profiles with declared label, sequence, word, and
  span-width bounds; see
  [GLiNER span and GLiClass runtime support](#gliner-span-and-gliclass-runtime-support).
- **ModernBERT exports are qualification-gated.** ModernBERT is traced through
  a family adapter and every exported variant is qualified against the PyTorch
  reference checkpoint before it is claimed; see
  [ModernBERT runtime support](#modernbert-runtime-support).
- **Longformer exports are bounded and qualification-gated.** There is no
  unbounded long-context claim: every export declares attention-window-aligned
  sequence buckets, a memory ceiling, a global attention policy, and a
  long-document windowing bound; see
  [Longformer runtime support](#longformer-runtime-support).
- **Privacy Filter exports are qualification-gated and declared.** Only the
  checkpoint families listed in
  [Privacy Filter runtime support](#privacy-filter-runtime-support) export,
  always at a pinned source revision, with the canonical Viterbi decoder and
  threshold profile shipped next to the graph. The MLX-only aliases above do
  not reach the Core ML or ONNX exporters.
- **MLX enforces a family allowlist.** The MLX dispatcher resolves architectures
  through `_SUPPORTED_TOKEN_CLASSIFICATION_MODEL_TYPES` (BERT-family,
  DeBERTa-v2, ModernBERT, and Longformer) plus the MLX-custom set
  (`gliner-uni-encoder-span`, `gliclass-uni-encoder`,
  `gliner-uni-encoder-token-relex`) and the privacy filter family.
  Architectures outside the allowlist raise a clear
  "unsupported family" error rather than falling through silently. See
  [MLX Backend](mlx-backend.md).
- **INT4 is certified only when recall holds.** `openmed.mlx.convert`
  will emit an INT4 artifact for any allowlisted family, but the artifact is
  marked `certified: false` in `recall_delta.json` when the measured per-label
  recall delta exceeds the INT4 limit. Release gates block over-budget INT4
  formats. See [MLX Quantized Export Certification](export-mlx-quant.md).
- **WebGPU fp16 is gated on parity.** The ONNX (default) column includes
  `model.webgpu.onnx` only when the fp16 graph passes the export-time gate
  against its fp32 parent on ONNX Runtime CPU: logit deltas, token and span
  agreement, per-label recall loss on G1/G2 labels and an fp16 activation-range
  probe. A failed gate fails the conversion, or omits the artifact with a
  recorded reason under `--webgpu-gate-policy omit`. See
  [fp16 WebGPU Parity Gate](export-onnx-webgpu.md#fp16-webgpu-parity-gate).

## ModernBERT runtime support

ModernBERT alternates full and bidirectional sliding-window attention and uses
rotary position embeddings. The stock Transformers forward pass does not trace
to Core ML, so both converters route ModernBERT through the family adapter in
`openmed.torch.export_adapters`. The adapter selects eager attention, builds
the full and sliding-window masks with plain tensor arithmetic (finite in
float16), passes them through the per-layer-type mask mapping Transformers 5
accepts, and, only while tracing, replaces the run-time `shape // 2` slice in
`rotate_half` that coremltools cannot fold. The traced graph keeps the
reference attention, positional, padding, and classifier-head semantics; the
logit gate below proves it for every export.

The adapter declares this support table. Any other runtime, precision,
quantization, provider, shape mode, or a sequence longer than
`max_position_embeddings` fails before weights load or inference with a stable
reason code. Transformers 4.x fails with `unsupported_converter_environment`
because it rebuilds ModernBERT masks internally.

| Runtime | Precisions | Quantizations | Providers | Shape modes | Target |
|---|---|---|---|---|---|
| coreml | float32, float16 | none, int8, int4 | all, cpuAndNeuralEngine, cpuOnly | static | iOS16 |
| onnx | float32, float16 | none, int8 | CPUExecutionProvider | dynamic | opset>=18 |

Core ML packages always use one static `(1, max_seq_length)` bucket; export
one package per bucket you ship. ONNX graphs keep dynamic `batch` and
`sequence` axes; qualification evaluates every declared bucket the suite
reaches, and the runtime loader rejects lengths above the qualified maximum.
The ONNX default profile qualifies `model.onnx` and qualifies the WebGPU fp16
graph on the CPU provider only as a numeric proxy (`claim: cpu_proxy`); that is
not a WebGPU support claim. The Android profile qualifies its fp32, fp16,
INT8, and `.ort` graphs on the CPU provider. OpenVINO IR and Transformers.js
bundles are rejected with `unsupported_runtime`.

### Qualification gates

`openmed.eval.export_qualification` is family-neutral; later families reuse
it. For each variant it runs the reference checkpoint and the exported
runtime on the same padded feeds from a committed synthetic PHI suite
(`openmed/eval/fixtures/export_qualification_phi.jsonl` by default) and
records:

| Gate | Float variants | Quantized variants |
|---|---|---|
| Logit parity | float32 atol/rtol `1e-4`; float16 atol `5e-2`, rtol `1e-2` | recorded, not gated |
| Span parity | decoded spans identical | recorded, not gated |
| Direct-identifier recall | no per-label loss | INT8 `0.005`, INT4 `0.010` per-label delta |
| Critical leakage | no high-risk span the reference caught may leak | same |
| Span integrity | spans in bounds, non-empty, labelled, non-overlapping | same |
| Per-language recall | recorded for every suite language; gates when the family binds it | same limit as direct-identifier recall |

A family may bind its canonical span decoder, decoder profile, and default
suite through `QualificationBinding`; the reference and every candidate are
then decoded the same way, and the profile is recorded in the report.

A variant that fails any gate, raises, or returns the wrong shape fails
closed. The converter writes the evidence, then raises
`ExportQualificationError`; unqualified artifacts are never published.
Core ML qualification needs Core ML prediction, so it runs on macOS; elsewhere
the manifest records `status: not_run` with `runtime_unavailable`.

The report (`openmed-export-qualification.json` for ONNX,
`<stem>_qualification.json` for Core ML) and the manifest summary record the
source revision (Hub commit, or a content digest for a local checkpoint
directory instead of its path), tokenizer and label-map identity digests, the
offset contract, input and output names, sequence buckets, precision,
provider, converter versions, operator inventory, latency, peak memory,
artifact sizes, and SHA-256 digests of every artifact file. It never contains
fixture text, decoded values, token ids, logits, or local paths.
`verify_qualification_report()` re-checks every digest offline, and
`check_qualified_request()` lets a runtime reject an unqualified variant,
provider, or sequence length before inference. `OnnxModel` applies it
automatically when a bundle carries a report.

Reason codes: `unsupported_family`, `unsupported_runtime`,
`unsupported_provider`, `unsupported_precision`, `unsupported_quantization`,
`unsupported_sequence_length`, `unsupported_sequence_shape`,
`bucket_unavailable`, `sequence_too_long`, `memory_budget_exceeded`,
`unsupported_global_attention`, `unsupported_operator`,
`unsupported_converter_environment`, `runtime_unavailable`,
`missing_label_map`, `missing_offset_mapping`, `missing_evidence`,
`source_revision_unavailable`, `source_revision_mismatch`,
`tokenizer_identity_mismatch`, `label_map_mismatch`, `io_contract_mismatch`,
`artifact_missing`, `artifact_digest_mismatch`, `report_invalid`,
`variant_not_qualified`, `inference_failed`, `logit_parity_failed`,
`span_parity_failed`, `sequence_parity_failed`, `long_document_parity_failed`,
`direct_identifier_recall_failed`, `critical_leakage_detected`,
`quantization_delta_exceeded`, `unsupported_configuration`,
`unsupported_remote_code`, `adapter_version_mismatch`,
`span_integrity_failed`, `multilingual_recall_failed`.

### Offline reproduction

Download the pinned checkpoint once, then export from the local directory with
the network disabled:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m openmed.coreml.convert --model ./checkpoints/modernbert-pii \
  --output ./dist/ModernBertPII.mlpackage --max-seq-length 512 \
  --precision float16 --compute-units cpuAndNeuralEngine --quantize int8 \
  --qualify --expected-source-revision sha256:<content digest>
python -m openmed.onnx.convert --model ./checkpoints/modernbert-pii \
  --output ./dist/modernbert-pii-onnx --max-seq-length 512 \
  --no-onnx-optimization --expected-source-revision sha256:<content digest>
```

The committed tests build a tiny ModernBERT in-test, so they need no
download: `tests/unit/coreml/test_coreml_modernbert.py` (macOS with
`coremltools`), `tests/unit/onnx/test_onnx_modernbert.py`, and
`tests/unit/torch/test_export_adapters.py`. They also prove that export,
qualification, loading, and inference open no network connection and write no
synthetic sentinel to logs, output, or artifacts. Release evidence for a
pinned production ModernBERT checkpoint is produced separately with the
commands above; until it is published the cells stay ⚠️.

## Longformer runtime support

The stock Longformer forward pads to a multiple of the attention window,
gathers global tokens with data-dependent `nonzero` indices, and reads
`.item()` to pick a code path; none of that traces faithfully. Both converters
therefore route Longformer through the family adapter in
`openmed.torch.export_adapters`. The adapter reuses every checkpoint weight
(embeddings, local and global query/key/value projections, output,
feed-forward, and classifier) and evaluates each layer's attention with
explicit masks: a local query attends to keys inside its own layer's window
plus every global key with the local projections, a global query attends to
every non-padding key with the global projections, and padding queries
produce zeros, exactly as in the reference. Masks use a finite `-1e4` so
float16 never produces `NaN`, and nothing reads a shape at run time, so the
same graph traces to static Core ML buckets and to a bounded dynamic ONNX
axis. Transformers 4.x and 5.x both work because only module weights are read.

The adapter declares this support table. Any other runtime, precision,
quantization, provider, or shape mode fails before weights load or inference
with a stable reason code.

| Runtime | Precisions | Quantizations | Providers | Shape modes | Target |
|---|---|---|---|---|---|
| coreml | float32, float16 | none, int8, int4 | all, cpuAndNeuralEngine, cpuOnly | static | iOS16 |
| onnx | float32, float16 | none, int8 | CPUExecutionProvider | dynamic | opset>=18 |

An ONNX graph that still contains `NonZero`, `If`, or `Loop` (the stock
data-dependent global gathers and branches the adapter removes) fails with
`unsupported_operator` before qualification runs any inference.

Graph inputs are `input_ids`, `attention_mask`, and `global_attention_mask`
for Core ML and the default ONNX profile. The Android ONNX profile keeps the
two-input mobile contract (`input_ids`, `attention_mask`), so global attention
is fixed off in that graph; its INT8 graph is the only ONNX quantization.
`--global-attention` names the fill policy qualification uses and the
manifests record: `none` (the default, the stock token-classification
behavior; every token uses local attention) or `first_token` (the leading
special token is global). Padding is never global. Any other policy, or
`first_token` on the Android profile, fails with
`unsupported_global_attention`.

### Sequence bounds and memory

Sequence buckets are every multiple of the largest per-layer attention window
up to the position limit (`max_position_embeddings - pad_token_id - 1`, so
4096 tokens for a `longformer-base-4096` shaped checkpoint). The export graph
evaluates local and global attention densely, so the declared estimate is the
per-layer attention working set of one row: 4 tensors (local and global
scores and probabilities) of `heads x S x S` values. A bucket is supported only
when that estimate fits the runtime ceiling. A length past the position limit
fails with `unsupported_sequence_length`, a length that is not a declared
bucket with `bucket_unavailable`, and a bucket over the ceiling with
`memory_budget_exceeded`, all before weights load. Core ML packages use one
static `(1, max_seq_length)` bucket each; ONNX graphs bound the dynamo
exporter's `sequence` dimension at `max_seq_length` (the Android TorchScript
export cannot record bounds, so qualification and the loader enforce them).
For a `longformer-base-4096` shaped configuration (12 heads, window 512):

| Runtime | Precision | Memory ceiling (MiB) | Supported buckets |
|---|---|---|---|
| coreml | float16 | 512 | 512, 1024, 1536, 2048 |
| coreml | float32 | 512 | 512, 1024, 1536 |
| onnx | float32 | 2048 | 512, 1024, 1536, 2048, 2560, 3072 |

These are declared estimates for the attention working set, not device
measurements; qualification additionally records the measured peak process
memory of every variant.

### Long documents

Documents longer than `max_seq_length` are never truncated. They are split
into overlapping windows of the complete encoding (`overlap_tokens` defaults
to `min(96, (window - 2) // 4)`); each token keeps the logits of the window
where it has the most surrounding context, and spans are decoded once in
document order with document-global character offsets. The bound is
`max_document_tokens` (64 windows by default); a longer document fails with
`sequence_too_long` and no partial result. `OnnxModel` applies the qualified
window, overlap, document bound, and global attention policy automatically,
and `check_qualified_request(..., document_tokens=..., global_attention=...)`
lets other runtimes apply the same contract.

### Qualification gates

Longformer reuses every gate above and adds two:

| Gate | Float variants | Quantized variants |
|---|---|---|
| Sequence sweep | logits within the float tolerance at minimum, typical, attention-window boundary (each half window and window, plus one either side), bucket boundary, and maximum lengths, with the policy fill and a first/middle/last global-attention probe | recorded, not gated |
| Long documents | windowed document-global spans identical to the windowed reference across overlap boundaries | recorded; no new critical leaks |

Reports and manifests record the input names, global attention policy,
windowing contract, declared and supported buckets, the selected bucket, the
memory ceiling and estimate, and per-variant sweep and long-document evidence
(lengths, counts, maxima, and reason codes only).

### Offline reproduction

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m openmed.coreml.convert --model ./checkpoints/longformer-pii \
  --output ./dist/LongformerPII-2048.mlpackage --max-seq-length 2048 \
  --precision float16 --compute-units cpuAndNeuralEngine --quantize int8 \
  --qualify --expected-source-revision sha256:<content digest>
python -m openmed.onnx.convert --model ./checkpoints/longformer-pii \
  --output ./dist/longformer-pii-onnx --max-seq-length 3072 \
  --no-onnx-optimization --expected-source-revision sha256:<content digest>
```

The committed tests build a tiny Longformer (per-layer windows 8, 16, and 8;
buckets 16 to 64) in-test: `tests/unit/coreml/test_coreml_longformer.py`
(macOS with `coremltools`), `tests/unit/onnx/test_onnx_longformer.py`, and
`tests/unit/torch/test_export_adapters.py`. They check reference parity at
minimum, typical, window-boundary, bucket-boundary, and maximum lengths with
and without global attention, overlap-boundary span reconstruction, every
rejection reason, and that export, qualification, loading, and inference open
no network connection and write no synthetic sentinel anywhere. Release
evidence for a pinned production Longformer checkpoint and device runs are
produced separately; until they are published the cells stay ⚠️.

## Privacy Filter runtime support

The OpenAI Privacy Filter is a bidirectional token classifier with grouped
query attention over a banded window (`|i - j| <= sliding_window`) with
learned attention sinks, interleaved YaRN rotary embeddings, and a top-k
mixture of clamped SwiGLU experts. The Transformers expert loop is data
dependent, so the family adapter in `openmed.torch.privacy_filter_export`
maps the checkpoint (fused QKV, flattened expert projections, optional
classifier bias) onto a trace-friendly module. The router runs exactly as in
the reference (top-k, softmax, `1 / k` scaling); every expert is then
evaluated densely with zero routing weight for unselected experts, which
keeps the graph free of control flow. Because dense experts hold
`tokens x experts x 2 x intermediate` activations, qualified exports are
bounded to 512 tokens.

The adapter declares this support table. Anything else fails before weights
load or inference with a stable reason code.

| Runtime | Precisions | Quantizations | Providers | Shape modes | Target |
|---|---|---|---|---|---|
| coreml | float32, float16 | none, int8 | all, cpuAndNeuralEngine, cpuOnly | static | iOS16 |
| onnx | float32, float16 | none, int8 | CPUExecutionProvider | dynamic | opset>=18 |

### Eligible checkpoints

Only these Transformers-format checkpoint families export. A Hub id must be
listed and match its label space; a local directory must carry one of these
label spaces. Every request must pin `--expected-source-revision` (the Hub
commit, or the `sha256:` content digest of a local directory), otherwise it
fails with `source_revision_unavailable`. No revision is reviewed or
release-qualified yet, so the matrix cells stay ⚠️.

| Checkpoint | Label space | Labels | Reviewed revision | Release evidence |
|---|---|---|---|---|
| `openai/privacy-filter` | `openai-privacy-filter-v1` | 33 | none yet | none yet |
| `OpenMed/privacy-filter-nemotron` | `nemotron_fine_v1` | 221 | none yet | none yet |
| `OpenMed/privacy-filter-nemotron-v2` | `nemotron_fine_v1` | 221 | none yet | none yet |
| `OpenMed/privacy-filter-multilingual` | `openmed_pii54_multi_v1` | 217 | none yet | none yet |
| `OpenMed/privacy-filter-multilingual-v2` | `openmed_pii54_multi_v1` | 217 | none yet | none yet |

Fail-closed rules:

- Configurations that declare `auto_map` or custom pipelines fail with
  `unsupported_remote_code`, and the exporter and reference loader never pass
  `trust_remote_code`. Transformers releases older than 5.6, which only load
  the family through repository code, fail with
  `unsupported_converter_environment`.
- The original OPF format (`privacy_filter`) and the MLX aliases are not
  export sources and fail with `unsupported_family`.
- Unsupported shapes (heads not divisible by key-value heads, odd head size,
  more experts per token than experts, a negative window, rotary types other
  than `yarn` and `default`) fail with `unsupported_configuration`. So do
  missing, unexpected, or mis-shaped weights.
- ONNX graphs may only use the adapter's operator set
  (`PRIVACY_FILTER_ONNX_OPERATORS`); control flow, data-dependent shape
  operators, and fused or contrib operators fail with `unsupported_operator`.
  Graph optimization is therefore rejected up front: export with
  `--no-onnx-optimization`.
- Float16 Core ML packages keep the attention softmax (`exp`, `real_div`,
  `reduce_sum`) in float32, as the reference does; the package metadata
  records `float32_ops`.

### Decoder and threshold profile

Every export writes the canonical decoder and threshold profile:
`openmed-decoder-profile.json` in ONNX bundles and
`<stem>_decoder_profile.json` next to Core ML packages. The profile names the
BIOES Viterbi decoder over log-softmax emissions with the six learned
transition biases (read offline from the checkpoint's
`viterbi_calibration.json`, else zero), the threshold profile (mean token
probability at least `0.5`, the `extract_pii` default), the span construction
(grapheme snapping, whitespace trim, `refine_privacy_filter_span`), and the
label space and label-map digest. Its digest is recorded in the manifest, the
qualification report, and the Core ML metadata.

Qualification decodes the reference and every candidate with that decoder,
maps the upstream labels (`private_person`, `private_email`, and so on) to
canonical direct identifiers for the recall and leakage gates, and gates
per-language recall on the multilingual synthetic suite
`openmed/eval/fixtures/export_qualification_privacy_filter.jsonl` (12
languages across Latin, Devanagari, Arabic, Han, kana, and Hangul scripts).
`OnnxModel` verifies the profile against the report and the label map before
it loads a bundle and then decodes with it; an unknown family or another
adapter version fails with `unsupported_family` or
`adapter_version_mismatch`.

### Android batch conversion

`scripts/onnx/batch_android_convert_publish.py` still excludes every Privacy
Filter row unless its declaration pins a reviewed revision and lists
published evidence for every Android variant the batch emits
(`onnx-android-fp32`, `onnx-android-fp16`, and `onnx-int8` unless `--no-int8`).
Included rows convert at that revision and must still pass the gates, and
existing artifacts whose qualification did not pass are never reused.

### Offline reproduction

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m openmed.coreml.convert --model ./checkpoints/privacy-filter \
  --output ./dist/PrivacyFilter.mlpackage --max-seq-length 512 \
  --precision float16 --compute-units cpuAndNeuralEngine --quantize int8 \
  --qualify --expected-source-revision sha256:<content digest>
python -m openmed.onnx.convert --model ./checkpoints/privacy-filter \
  --output ./dist/privacy-filter-onnx --max-seq-length 512 \
  --no-onnx-optimization --expected-source-revision sha256:<content digest>
```

The committed tests build a tiny native Privacy Filter in-test:
`tests/unit/torch/test_privacy_filter_export.py`,
`tests/unit/onnx/test_onnx_privacy_filter.py`, and
`tests/unit/coreml/test_coreml_privacy_filter.py` (macOS with
`coremltools`). They prove mapping and graph parity, decoder and threshold
equivalence, multilingual spans, and that export, qualification, loading,
and inference open no network connection and write no synthetic sentinel to
logs, output, or artifacts. Release evidence for the declared checkpoints is
produced separately with the commands above.

## GLiNER span and GLiClass runtime support

GLiNER span detection and GLiClass zero-shot classification take their
candidate labels as part of the model input, so their graphs do not fit the
`(input_ids, attention_mask) -> logits` token-classification contract. OpenMed
exports them through `openmed.onnx.zero_shot_export` under two versioned
profiles defined in `openmed.ner.families.zero_shot_export`. The source is a
local OpenMed MLX artifact directory (`config.json`, `weights.safetensors`,
`tokenizer.json`); Hub ids are never resolved and repository code is never
run. The weights are loaded into the PyTorch reference graph in
`openmed.torch.zero_shot_export`, which re-expresses the MLX runtime
computation parameter for parameter, and that graph is traced to ONNX.

| Profile | Family | Task | Graph inputs | Graph output |
|---|---|---|---|---|
| `gliner-span-uni-encoder-v1` | `gliner-uni-encoder-span` | `zero-shot-ner` | `input_ids`, `attention_mask`, `words_mask`, `text_lengths`, `span_idx`, `span_mask` | `logits` `[batch, words, max_width, labels]` |
| `gliclass-uni-encoder-v1` | `gliclass-uni-encoder` | `zero-shot-sequence-classification` | `input_ids`, `attention_mask` | `logits` `[batch, labels]` |

The GLiNER tensor names follow the portable GLiNER ONNX convention, and the
GLiClass graph keeps the contract that `GLiClassOnnxRunner` in
`openmed.structured.decision_gliclass` already consumes.

Both profiles declare this support table. Every other runtime, precision,
quantization, provider, or shape mode fails before weights load with a stable
reason code:

| Runtime | Precisions | Quantizations | Providers | Shape modes | Target |
|---|---|---|---|---|---|
| onnx | float32 | none | CPUExecutionProvider | dynamic | opset>=18 |

Core ML is explicitly unsupported for both profiles: the Core ML converter and
the profile's `support_for("coreml")` refuse with `unsupported_runtime`. No
Core ML package was traced or gated for a tiny model, so no bounded-shape,
operator-coverage, memory, parity, or recall evidence exists for one. Mobile
(ONNX Runtime Mobile, NNAPI, Core ML EP) and browser (WebGPU, WASM) providers
are likewise not qualified: requesting one fails with `unsupported_provider`,
and the bundle manifest records every provider other than the CPU provider as
unsupported.

**Unsupported variants.** Relation extraction (`gliner-uni-encoder-token-relex`
or the `zero-shot-relation-extraction` task) fails with `unsupported_task`.
Quantized sources fail with `unsupported_quantization`; checkpoints that need
repository code (`auto_map`, `trust_remote_code`) fail with
`unsupported_remote_code`. GLiNER span modes other than `markerV0`, more than
one LSTM layer, convolutional DeBERTa layers, non-GELU activations, GLiClass
scorers other than `mlp`, `extract_text_features`, `use_lstm`,
`squeeze_layers`, pooling other than `first` or `mean`, an explicit
first-token `class_token_pooling` (the OpenMed reference span-averages
labels), and marker tokens that do not match the tokenizer all fail with
`unsupported_model_variant`. The token-classification ONNX converter refuses
GLiNER and GLiClass checkpoints with `unsupported_task`, and `OnnxModel`
refuses a zero-shot bundle the same way.

### Prompt, label, and offset contract

- **GLiNER span:** the prompt is `<<ENT>> label` per candidate label in caller
  order, then `<<SEP>>`, then the source words, encoded pre-tokenized. Words
  come from the regex `\w+(?:[-_]\w+)*|\S` shared with the MLX runtime.
  `words_mask` marks the first token of each source word with its 1-based
  index; `span_idx` enumerates every start word and every width below the
  checkpoint `max_width`; `span_mask` masks spans past the text or wider than
  the requested width. Span offsets are half-open Unicode code-point offsets
  from the first word's start to the last word's end.
- **GLiClass:** the input string is `<<LABEL>>label` per candidate label, then
  `<<SEP>>`, an optional instruction, and the text (or the text first for
  checkpoints with `prompt_first: false`). Few-shot examples are not part of
  the profile.
- **Labels:** caller order is preserved end to end. Labels must be non-blank,
  unique, free of control characters and reserved markers (`<<ENT>>`,
  `<<SEP>>`, `<<REL>>`, `<<LABEL>>`, `<<EXAMPLE>>`), and at most
  `max_label_chars` code points. Text containing a reserved GLiClass marker is
  refused with `unsupported_control_marker`; the encoder also checks that the
  number of marker tokens equals the number of labels. Inputs are never
  truncated.
- **Decoder:** scores are `sigmoid(logit)` and a candidate is kept when
  `score >= threshold` (`0.5` by default, recorded in the manifest). GLiNER
  flat NER keeps the highest-scoring non-overlapping spans, breaking ties by
  start, end, then caller label order, and returns spans ordered by start,
  end, label order; nested mode keeps every span above the threshold.
  GLiClass ranks labels by descending score with ties in caller label order.

### Bounds

Bounds are declared at export, recorded in the manifest and the
qualification report, and enforced by the runtime before inference. A
request outside them returns a typed `ZeroShotUnsupported` result (`status`,
`reason`, `dimension`, `limit`) with no text, labels, or token ids.

| Bound | Default | Hard limit | Reason when exceeded |
|---|---|---|---|
| Candidate labels | 32 | 64 | `unsupported_label_count` |
| Tokens, prompt included | `min(512, max_position_embeddings)` | `max_position_embeddings` | `unsupported_sequence_length` |
| Source words (GLiNER) | checkpoint `max_len` (384 if unset) | the token bound | `unsupported_word_count` |
| Span width in words (GLiNER) | checkpoint `max_width` | checkpoint `max_width` | `unsupported_span_width` |
| Label length | 128 code points | set at export (`max_label_chars`) | `unsupported_label` |
| Activation memory | none | caller `memory_budget_bytes` | `unsupported_memory_budget` |

The memory estimate is a deterministic float32 activation figure for the
encoder, disentangled attention scores, and the span or label head; the
manifest records it at the declared bounds. Only batch size 1 is qualified.

### Qualification gates

`openmed.eval.zero_shot_qualification` extends the family-neutral harness.
The PyTorch reference and the exported graph on `CPUExecutionProvider` receive
identical feeds built by the contract, once in the suite's candidate-label
order and once in reversed order, so a graph that froze its tracing example's
label count or label positions fails. The default suites are the packaged
synthetic PHI suite (GLiNER; each canonical label is asked for with a
natural-language prompt such as `person` or `identifier`) and
`openmed/eval/fixtures/zero_shot_classification.jsonl` (GLiClass).

| Gate | GLiNER span | GLiClass |
|---|---|---|
| Logit parity | float32 atol/rtol `1e-4` on valid span and label cells | float32 atol/rtol `1e-4` |
| Decoded output | spans identical (labels, code-point offsets, overlap resolution, threshold) | label ranking identical (`label_order_parity_failed`); accepted labels identical (`decision_parity_failed`) |
| Direct-identifier recall | no per-label loss against the reference | not applicable |
| Critical leakage | no high-risk span the reference caught may leak | not applicable |

The report (`openmed-export-qualification.json`) uses the same format as the
token-classification reports, with `task` set to the profile's task, the
word-splitter offset contract, the decoder contract, and a label-schema digest
in place of a label map (candidate labels are caller-supplied, so no label
text is recorded). The bundle manifest (`openmed-zero-shot.json`) records the
profile, source revision, adapter version, architecture facts, prompt and
label schema, tokenizer identity, tensor contract, bounds, memory estimate,
decoder and threshold, offset contract, precision, runtime compatibility,
unsupported surfaces, file digests, the qualification summary, and that the
export does not change the model's clinical claim or trigger decisions.

`ZeroShotOnnxModel` in `openmed.onnx.zero_shot_inference` loads a bundle only
when the report verifies against every file digest, the tokenizer identity
matches, and the requested provider is the qualified one; it needs only
`numpy`, `onnxruntime`, and `tokenizers`. A bundle exported with
`--no-qualify` claims nothing and is refused with `missing_evidence`.

Additional reason codes for these profiles: `unsupported_task`,
`unsupported_model_variant`, `unsupported_remote_code`,
`unsupported_label_count`, `unsupported_label`, `unsupported_control_marker`,
`unsupported_span_width`, `unsupported_word_count`,
`unsupported_memory_budget`, `label_order_parity_failed`,
`decision_parity_failed`.

### Reproduction

Convert the pinned checkpoint to an OpenMed MLX artifact once, then export
from the local directory with the network disabled:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m openmed.onnx.zero_shot_export --source ./artifacts/gliner-span-mlx \
  --output ./dist/gliner-span-onnx --max-labels 16 --max-sequence-length 384 \
  --expected-source-revision sha256:<content digest>
python -m openmed.onnx.zero_shot_export --source ./artifacts/gliclass-mlx \
  --output ./dist/gliclass-onnx --max-labels 16
```

The committed tests build tiny GLiNER-shaped and GLiClass-shaped checkpoints
in-test (`tests/fixtures/zero_shot_export_stub.py`), so they need no
download: `tests/unit/ner/test_zero_shot_export_contract.py` and
`tests/unit/eval/test_zero_shot_qualification.py` run everywhere;
`tests/unit/onnx/test_onnx_zero_shot_export.py` exports, qualifies, and loads
both profiles with PyTorch and ONNX Runtime, proves that no network
connection is opened and no synthetic sentinel reaches logs, output, or
artifacts, and on Apple silicon checks that the MLX runtime, the PyTorch
reference, and the ONNX graph agree. Release evidence for a pinned
production GLiNER span checkpoint and a pinned GLiClass checkpoint is
produced separately with the commands above; until it is published the cells
stay ⚠️.

## See also

- [OpenMedKit Support Matrix (v4.0)](openmedkit-support-matrix.md), which
  reuses these families per kit surface, OS, runtime and precision
- [MLX Backend](mlx-backend.md)
- [MLX Quantized Export Certification](export-mlx-quant.md)
- [CoreML Packaging](coreml-export.md)
- [ONNX and WebGPU Export](export-onnx-webgpu.md)
- [Android ONNX Export](export-onnx-android.md)
- [Transformers.js Export](export-transformersjs.md)
- [GGUF Embedding Export](export-gguf.md)

## Producer provenance and offline verification

MLX, ONNX and CoreML record versioned `provenance`: source ID, resolved remote
revision or local content digest, available source license, toolchain versions,
and sorted relative POSIX `{path, size_bytes, sha256}` records. Unknown license/version metadata
is `null`; OpenMed's license does not imply the source license.

Remote model, config and tokenizer loaders share one pinned snapshot. Local inputs
must stay unchanged and are rechecked before finalization. Local input/output
symlinks are rejected; normal remote HF-cache links are allowed. Private load
paths, credentials and CLI/environment values are excluded. Direct writers
record unavailable remote revisions as `null`; use converters for resolution.

Call `openmed.core.export_provenance.verify_export_provenance(root, name)`
offline with `openmed-mlx.json`, `openmed-onnx.json` or `openmed-coreml.json`.
It checks safe paths, exact membership, sizes and digests, including nested
files and ONNX sidecars. It rejects symlinks and excludes only the root
manifest, not nested namesakes. Verification is explicit, not loader-enforced. Legacy
readers and segmenter checks remain; manifests without provenance cannot pass
this check. Unsigned consistency requires a trusted manifest: it cannot
authenticate publishers or prevent replacing manifest and files together.
Signing and bundle schemas remain separate.

### CoreML package schema v2

Each `.mlpackage/openmed-coreml.json` uses `schema_version: 2` and describes
only its own movable package: variant path `.`, `size_bytes` excluding the root
manifest, and package-relative inventory paths. `validate_coreml_bundle` retains v1/v2 segmenter checks. Consumers
must distinguish the package manifest from the all-variant v1 conversion report.

Sibling `<model>_coreml_manifest.json` remains v1 with timestamps, latency,
residency and parity; residency display paths are relative. Observation paths,
including custom paths, must stay outside every package. Sibling
`<model>_id2label.json` files stay outside and unbound by package inventory.
Retain/distribute these siblings separately; publishing one package omits them.

### MLX and ONNX observations

Runtime layouts and certification fields remain, except ONNX
`validation.latency` moves to sibling `<output-directory>_onnx_observations.json`;
stable validation stays canonical. Bundled `recall_delta.json` is a stable
projection; timestamps live in sibling
`<output-directory>_mlx_observations.json` or
`<output-directory>_onnx_observations.json`. The report API still returns
observations. MLX artifact paths become `.` and evaluation suites become
content digests.

External reports retain an in-artifact projection at the referenced relative
filename. Runtime/control-file collisions reject; complete certification reports
can be updated. Standalone certification rechecks an available original local
reference before/after evaluation; a custom runner without it owns source
matching. Certification and Transformers.js updates refresh inventories. With
provenance and `update_manifest=True`, JS bundles must be strict subdirectories
of the source export root; `False` permits separate external bundles without
changing the parent. Legacy exports retain prior behavior.

Identical source bytes, options, toolchain and generated payload bytes yield
identical canonical manifests; numerical backends/serializers need not be
deterministic. Tiny synthetic/mock conversions exercise real writers, not
real-model parity or clinical performance.

- [Export Toolchains, Artifact Licenses and Redistribution](security/export-redistribution.md)
