# Portable Tokenizer Conformance

Bundle digests and final-span parity can show that two runtimes disagree, but
not whether the drift began in normalization, special-token insertion,
truncation windows or model execution. OpenMed therefore pins the
intermediate tokenizer outputs every runtime must reproduce, and each runtime
checks them before any inference result is compared.

The contract version is `openmed.tokenizer-conformance.v1`.

## Fixture Layout

All files live in `tests/fixtures/parity/tokenizer/` and are synthetic:

| File | Role |
| --- | --- |
| `synthetic-wordpiece-v1/tokenizer.json` | Tiny uncased WordPiece tokenizer built from invented text |
| `synthetic-wordpiece-v1/tokenizer_config.json` | Tokenizer class and special tokens for runtimes that need it |
| `synthetic-wordpiece-v1/config.json` | Minimal model config so offline loaders need no download |
| `tokenizer_conformance_v1.json` | Versioned vectors, one case per line |
| `tokenizer_runtime_support_v1.json` | Hand-reviewed runtime declarations |
| `manifest.json` | SHA-256 and byte size of every file plus the contract digest |

The tokenizer uses `BertNormalizer` (clean text, CJK padding, accent stripping
and lowercasing), `BertPreTokenizer`, `WordPiece` with `##` continuations and a
`[CLS] $A [SEP]` template. Its vocabulary comes from the synthetic corpus in
`openmed.eval.tokenizer_conformance`; no model vocabulary is downloaded or
copied.

## Vector Fields

Each case records its synthetic text, controlled feature tags and the expected
value of every field. Offsets are half-open Unicode scalar offsets into the
original text.

| Field | Meaning |
| --- | --- |
| `normalized` | Normalizer output for the whole text |
| `tokens` | Token strings, including template special tokens |
| `input_ids` | Ids fed to the model, with special tokens and without truncation |
| `attention_mask` | One per token; vectors never pad |
| `token_type_ids` | Segment ids from the template |
| `special_tokens_mask` | `1` for template special tokens |
| `offsets` | Source span per token; template special tokens use `[0, 0]` |
| `truncated_input_ids` | Single-pass truncation at `max_length`, keeping `[CLS]` and `[SEP]` |
| `window_ranges` | Sequence-token ranges of right-side truncation windows |
| `window_input_ids` | Ids of each window after the template is applied |

Window cases carry `max_length` and `stride`. The sequence capacity is
`max_length` minus the two template special tokens, consecutive windows start
`capacity - stride` tokens apart, and the last window ends at the final
sequence token, as in Rust `Encoding::truncate`. Single-pass truncation equals
the first window with the overflow dropped.

The cases cover precomposed, decomposed and leading combining marks, control
and format characters, Unicode whitespace and CRLF, Arabic and Hebrew with
diacritics and bidi controls, CJK including a supplementary-plane ideograph and
full-width punctuation, Devanagari, Bengali and Tamil, single, ZWJ, modifier and
flag emoji, empty and whitespace-only text, unknown and over-long words, a
literal special token in the text, and exact-fit, overflow-by-one, strided,
stride-zero, subword-boundary and CJK windows.

## Reference Semantics

The declared reference is the Rust `tokenizers` crate, which Python
`transformers`, DJL, swift-transformers and Transformers.js tokenizers aim to
reproduce. `openmed.eval.tokenizer_reference` implements the supported subset
with the standard library, so fixture generation never needs an optional
dependency. Components outside that subset, such as BPE or Unigram models,
`Metaspace`, embedded truncation or padding, normalized added tokens, and
inputs that need cross-character canonical reordering, fail closed with
`UnsupportedTokenizerFeature` and a reason code.

## Runtime Runners

Every runner loads the same tokenizer offline, runs the kit's existing adapter
and compares each supported field exactly. It fails before any inference
comparison.

| Runtime | Library | Adapter under test | Unsupported fields |
| --- | --- | --- | --- |
| `python-tokenizers` | Rust `tokenizers` via Python | `openmed.onnx.inference` tokenizer adapter and `encode_windows` | none |
| `swift-transformers` | swift-transformers 1.0.0 | `OpenMed.encodeForModel`, `buildOffsets`, `makeTokenChunks` | `normalized`, `special_tokens_mask` |
| `djl-tokenizers` | DJL tokenizers 0.33.0 | `Encoding.toTokenOffsets` and DJL overflow windows | `normalized`, `window_ranges` |
| `tokenizers-js` | `@huggingface/tokenizers` 0.2.0 | `alignTokenOffsets` | `special_tokens_mask`, `truncated_input_ids`, `window_ranges`, `window_input_ids` |

Known divergences are declared per feature in
`tokenizer_runtime_support_v1.json` with a reason code. A diverging comparison
on a case tagged with a declared feature is reported as a declared divergence;
any other divergence fails. A declaration that no case triggers is stale and
also fails, so declarations stay accurate when libraries change. The current
declarations record these findings:

- swift-transformers keeps control and format characters, strips only the
  U+0300 to U+036F combining block, and matches WordPiece pieces on grapheme
  clusters, so marks and emoji modifiers cannot split into separate pieces.
- The Swift kit truncates model input with `prefix(maxSeqLength)`, so text
  longer than the limit loses its trailing `[SEP]`.
- The Swift kit offset heuristic cannot place `[UNK]`, extends spans over
  removed trailing marks, maps literal special-token text to `[0, 0]`, and
  re-tokenizes chunk text, which changes the first id of a window that starts
  on a `##` continuation.
- Transformers.js tokenizers pad CJK characters by UTF-16 code unit, so a
  supplementary-plane ideograph next to Latin text is not split.
- The web `alignTokenOffsets` adapter refuses `[UNK]`, mark-only pieces and
  words split by a removed format character, extends spans over removed
  trailing marks, and skips literal special tokens.
- DJL matches every supported field, including scalar offsets. Its overflow
  windows are padded to the longest window by default, so the runner disables
  padding.

If a runtime cannot load the synthetic tokenizer offline it reports
`TOKENIZER_CONFORMANCE runtime=<id> status=unsupported reason=<code>` and skips
instead of passing. The DJL runner does this on JVMs without a bundled native
library, for example an x86_64 JVM on Apple silicon, and the Swift runner does
it on platforms without swift-transformers.

## Diagnostics

Reports and failures carry only case ids, field names, indices, feature names,
reason codes and digests, for example
`TOKENIZER_FIELD_VALUE_MISMATCH case=ascii-basic-01 field=input_ids index=0`.

| Code | Meaning |
| --- | --- |
| `TOKENIZER_FIELD_VALUE_MISMATCH` | First differing element at `index` |
| `TOKENIZER_FIELD_LENGTH_MISMATCH` | Lengths differ after `index` shared elements |
| `TOKENIZER_FIELD_MISSING` | A supported field was not produced |
| `TOKENIZER_RUNTIME_ERROR` | The runtime raised; its message is dropped |
| `TOKENIZER_DECLARED_DIVERGENCE_STALE` | A declared divergence no longer occurs |
| `TOKENIZER_RUNTIME_UNDECLARED` | The runtime has no support declaration |
| `TOKENIZER_CONTRACT_VERSION_MISMATCH` | Vectors or declarations use another contract |

In Python, `require_tokenizer_conformance` raises `TokenizerConformanceError`
with these codes; call it before comparing inference results.

## Running the Runners

```bash
python -m pytest tests/unit/eval/test_tokenizer_conformance.py -q
swift test --filter TokenizerConformanceTests
cd android && ./gradlew :openmedkit:testDebugUnitTest --tests '*TokenizerConformanceTest'
cd js/openmedkit-web && npm test
```

The Python `tokenizers` runner needs the `onnx-runtime` or `hf` extra and
skips with an explicit unsupported status without it; the low-resource CI lane
runs it before its inference benchmark.

## Updating the Vectors

Edit the cases or corpus in `openmed/eval/tokenizer_conformance.py`, update
the hand-reviewed runtime declarations if needed, then regenerate and check:

```bash
python scripts/eval/generate_tokenizer_conformance.py
python scripts/eval/generate_tokenizer_conformance.py --check
```

`--check` fails when any committed file differs from a fresh generation or a
declaration names an unknown field or feature. An intentional change to
normalization, special-token insertion or windowing changes the vectors and
the contract digest in the same diff, and every runner fails until its adapter
matches or its declaration explains the divergence.

## Bundle Binding

A model bundle records the tokenizer contract its runtime adapters were
qualified against in a `tokenizer_contract` field:

```json
{
  "tokenizer_contract": {
    "version": "openmed.tokenizer-conformance.v1",
    "digest": "sha256:<contract_digest from manifest.json>"
  }
}
```

`tokenizer_contract_reference(manifest)` returns this value after checking
that the manifest digest matches its file entries. The portable bundle
manifest can reference the field unchanged, and the registry rollback check
already compares `tokenizer_contract` values and blocks with
`TOKENIZER_CONTRACT_MISMATCH` when they differ. Readers must fail closed on an
unknown version or a digest they cannot verify.

See [Swift-Kotlin API Parity](swift-kotlin-parity.md) for the shared span
fixtures that run after tokenizer conformance passes.
