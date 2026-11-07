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
| Longformer | `longformer` | ✅ | ✅ | ⚠️ INT4-if-recall-holds | ❌ family not in CoreML allowlist | ❌ | ⚠️ upstream ONNX support required | ⚠️ same as ONNX (default) | ⚠️ same as ONNX (default) | ❌ |
| OpenAI Privacy Filter | `openai-privacy-filter`, `privacy-filter`, `privacy-filter-nemotron`, `nemotron-privacy-filter`, `privacy-filter-multilingual`, `multilingual-privacy-filter` | ✅ | ✅ | ⚠️ INT4-if-recall-holds | ❌ family not in CoreML allowlist | ❌ | ❌ no ONNX converter path today | ❌ | ❌ | ❌ |
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
  `bert`, `distilbert`, `electra`, `roberta`, `xlm-roberta`, `deberta-v2`, and
  `modernbert`. Any other architecture fails before tracing with the
  `unsupported_family` reason and an error that lists the supported families.
  GLiNER and GLiClass checkpoints declare a DeBERTa backbone, so they are
  refused first with `unsupported_runtime` (relation extraction with
  `unsupported_task`) instead of being traced as a token classifier. See
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
`unsupported_operator`, `unsupported_converter_environment`,
`runtime_unavailable`, `missing_label_map`, `missing_offset_mapping`,
`missing_evidence`, `source_revision_unavailable`, `source_revision_mismatch`,
`tokenizer_identity_mismatch`, `label_map_mismatch`, `io_contract_mismatch`,
`artifact_missing`, `artifact_digest_mismatch`, `report_invalid`,
`variant_not_qualified`, `inference_failed`, `logit_parity_failed`,
`span_parity_failed`, `direct_identifier_recall_failed`,
`critical_leakage_detected`, `quantization_delta_exceeded`.

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

- [MLX Backend](mlx-backend.md)
- [MLX Quantized Export Certification](export-mlx-quant.md)
- [CoreML Packaging](coreml-export.md)
- [ONNX and WebGPU Export](export-onnx-webgpu.md)
- [Android ONNX Export](export-onnx-android.md)
- [Transformers.js Export](export-transformersjs.md)
- [GGUF Embedding Export](export-gguf.md)
