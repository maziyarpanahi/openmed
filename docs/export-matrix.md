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

The existing MLX, ONNX and CoreML producers record a common versioned
`provenance` object. It contains the source model ID, resolved revision (remote
sources) or content digest (local directories), an available source license,
explicit toolchain distribution versions, and a sorted list of relative POSIX
file paths with `size_bytes` and SHA-256 digests. Missing license or distribution
metadata is `null`; the package's own license does not imply the source license.

A converter resolves the remote snapshot once and uses that same local snapshot
for its model, config and tokenizer loaders. Local source identities are based
on file contents rather than the directory name or a mutable branch. Input and
output symlinks are rejected for local directories; normal Hugging Face cache
symlinks remain valid remote snapshot inputs. Credentials, local loading paths
and command-line/environment details are not producer metadata. Local input
directories must remain unchanged throughout conversion; a content recheck
before finalization rejects a changed source. Direct calls to
low-level manifest writers cannot resolve an unavailable remote revision and
record it as `null`; use the converter entry point for resolved provenance.

Verification is explicit and offline:

```python
from openmed.core.export_provenance import verify_export_provenance

verify_export_provenance("converted-model", "openmed-onnx.json")
verify_export_provenance("converted-model.mlpackage", "openmed-coreml.json")
```

For MLX use `openmed-mlx.json`. The verifier checks the exact file set, safe
paths, sizes and digests, including nested files and ONNX external tensor
sidecars. It rejects changed, missing or extra files and any symlink in the
artifact tree. Only the named manifest at the artifact root is excluded from
its own inventory; a nested file with the same name is not excluded. Existing
optional-resource validators retain their segmenter checks. Older manifests
remain readable, but an old manifest without provenance cannot establish
integrity through this new explicit verifier. No loader is forced to run the
new check. Unsigned provenance verifies consistency against a trusted manifest;
it does not authenticate a publisher or prevent simultaneous replacement of
both the manifest and its files. Signing and bundle schemas are separate work.

### CoreML package schema v2

Each `.mlpackage/openmed-coreml.json` now has `schema_version: 2` and describes
only that package. Its variant path is `.`; its `size_bytes` is the payload size
excluding this root manifest. Inventory paths are relative to the package, so
a package can be moved and verified without sibling variants. The existing
`validate_coreml_bundle` reader accepts both v1 and v2 and still checks declared
segmenter resources for both versions. Package v2 consumers must not assume
that the package manifest equals the sibling conversion report or that it
lists all conversion variants.

The sibling `<model>_coreml_manifest.json` remains a schema v1 conversion report
with timestamps, measured latency, residency and parity observations. Residency
artifact paths are relative display paths. These observations are outside the
canonical package and its inventory: retaining wall-clock or measured timings
inside a file would prevent byte-for-byte reproducibility. Explicit observation
paths must also be outside the packages. Sibling `<model>_id2label.json` files
remain outside the packages and are neither copied inside nor covered by the
package inventory. Consumers needing these reports or labels must retain them
separately; publishing a single package does not publish all siblings.

### MLX and ONNX observations

The existing MLX and ONNX manifest layouts retain their runtime fields and gain
provenance. ONNX optimization latency observations are retained in sibling
`<output-directory>_onnx_observations.json`, rather than `validation.latency`
inside the canonical artifact. Consumers of that timing field should read the
observation report. Stable validation outcomes remain in the manifest.

Recall-certification fields and relative report references remain available.
For provenance-producing conversions, the in-artifact `recall_delta.json` is a
stable certification projection; actual timestamps live in the sibling
`<output-directory>_mlx_observations.json` or
`<output-directory>_onnx_observations.json`. MLX's local artifact path becomes
`.` and its evaluation suite is identified by a content digest. The report API
continues returning observation metadata. Standalone certification rechecks an
available original local source before and after evaluation; a custom runner
without that reference is responsible for matching the declared source. A requested external report path is
retained with an in-artifact projection at the referenced relative filename.
Certification and Transformers.js updates refresh an existing provenance
inventory after their final file changes. Custom certification report paths must
not replace runtime files or control manifests; an existing complete certification report can
be updated. With existing provenance and `update_manifest=True`, a standalone
Transformers.js bundle must be a strict subdirectory of the source export root.
Use `update_manifest=False` for a separate external bundle without adding it to
the parent manifest. Legacy manifests retain their previous export behavior.

Reproducibility means identical source bytes, options, toolchain and generated
payload bytes produce identical canonical manifests. This is not a promise that
every numerical backend, runtime measurement or serializer is deterministic.
The regression fixtures are tiny synthetic checkpoints and mocked converter
backends; these checks do not establish real-model export parity or clinical
performance.
