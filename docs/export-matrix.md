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
| OpenAI Privacy Filter | `openai-privacy-filter`, `privacy-filter`, `privacy-filter-nemotron`, `nemotron-privacy-filter`, `privacy-filter-multilingual`, `multilingual-privacy-filter` | ✅ | ✅ | ⚠️ INT4-if-recall-holds | ❌ family not in CoreML allowlist | ❌ | ❌ no ONNX converter path today | ❌ | ❌ | ❌ |
| GLiNER (span) | `gliner-uni-encoder-span` | ✅ (MLX custom family) | ✅ | ⚠️ INT4-if-recall-holds | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| GLiClass | `gliclass-uni-encoder` | ✅ (MLX custom family) | ✅ | ⚠️ INT4-if-recall-holds | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| GLiNER (token-relex) | `gliner-uni-encoder-token-relex` | ✅ (MLX custom family) | ✅ | ⚠️ INT4-if-recall-holds | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

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
  `modernbert`, and `longformer`. Any other architecture fails before tracing
  with the `unsupported_family` reason and an error that lists the supported
  families. See [CoreML Packaging](coreml-export.md).
- **ModernBERT exports are qualification-gated.** ModernBERT is traced through
  a family adapter and every exported variant is qualified against the PyTorch
  reference checkpoint before it is claimed; see
  [ModernBERT runtime support](#modernbert-runtime-support).
- **Longformer exports are bounded and qualification-gated.** There is no
  unbounded long-context claim: every export declares attention-window-aligned
  sequence buckets, a memory ceiling, a global attention policy, and a
  long-document windowing bound; see
  [Longformer runtime support](#longformer-runtime-support).
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
`quantization_delta_exceeded`.

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

## See also

- [MLX Backend](mlx-backend.md)
- [MLX Quantized Export Certification](export-mlx-quant.md)
- [CoreML Packaging](coreml-export.md)
- [ONNX and WebGPU Export](export-onnx-webgpu.md)
- [Android ONNX Export](export-onnx-android.md)
- [Transformers.js Export](export-transformersjs.md)
- [GGUF Embedding Export](export-gguf.md)
