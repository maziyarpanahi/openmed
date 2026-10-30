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
| ModernBERT | `modernbert`, `modern-bert` | ✅ | ✅ | ⚠️ INT4-if-recall-holds | ❌ family not in CoreML allowlist | ❌ | ⚠️ upstream ONNX support required | ⚠️ same as ONNX (default) | ⚠️ same as ONNX (default) | ❌ |
| Longformer | `longformer` | ✅ | ✅ | ⚠️ INT4-if-recall-holds | ❌ family not in CoreML allowlist | ❌ | ⚠️ upstream ONNX support required | ⚠️ same as ONNX (default) | ⚠️ same as ONNX (default) | ❌ |
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
  `bert`, `distilbert`, `electra`, `roberta`, `xlm-roberta`, and `deberta-v2`.
  Any other architecture fails before tracing with an error that lists the
  supported families. See [CoreML Packaging](coreml-export.md).
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

## See also

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
