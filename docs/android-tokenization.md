# Android Tokenization Strategy (Decision)

Android token-classification inference needs WordPiece/BPE tokenization on
device, with token-to-character offset mapping so detected tokens can be mapped
back to character spans and turned into OpenMedSpan records. This page records
the decision on how OpenMedKit Android tokenizes, the options considered, and
the rationale.

**Status:** Accepted. The decision below is embodied in the shipping
`android/openmedkit` module.

## Constraints

Tokenization on Android must satisfy the same local-first guarantees as the
rest of OpenMedKit:

- **On-device only.** Tokenization runs entirely on the handset. No network
  access is performed after the model and its tokenizer assets are downloaded;
  there is no remote tokenization path.
- **No PHI in logs.** The tokenizer must not log raw input text, token strings,
  or character offsets. De-identification input is clinical text and must never
  reach application or dependency logs.
- **Permissive license only.** The tokenizer dependency must be permissively
  licensed (Apache-2.0/MIT-style). No GPL/LGPL or proprietary tokenizer.
- **Span-offset fidelity.** Char-level offset mapping must be faithful enough to
  recover exact character spans for every detected entity.
- **Python parity.** On-device tokenization must match OpenMed's Python
  tokenization so span offsets and labels agree across platforms.

## Options Considered

Three realistic approaches were evaluated.

| Criterion | DJL HuggingFace tokenizers AAR | Pure-Kotlin WordPiece | ORT-Extensions in-graph BertTokenizer |
| --- | --- | --- | --- |
| License | Apache-2.0 (DJL wrapper and the Rust `tokenizers` crate) | Project code (permissive) | MIT (`onnxruntime-extensions`) |
| Offset mapping | Native char offsets from the Rust tokenizer `Encoding` — high fidelity | Must be re-implemented by hand; correctness is the author's burden | Limited; in-graph tokenizers do not cleanly expose char offsets for span recovery |
| Binary / asset size | Adds a native `.so` per ABI to the APK; tokenizer asset is `tokenizer.json` | Smallest — no native library; tokenizer asset is `tokenizer.json`/`vocab.txt` | Tokenization baked into the `.ort` graph; no separate tokenizer runtime, larger graph |
| Min SDK | 26 (matches the OpenMedKit module) | Unconstrained | Tied to the ONNX Runtime + Extensions build |
| Python parity | Exact — consumes the same `tokenizer.json` as OpenMed's Python fast tokenizer | Approximate — must replicate normalization, pre-tokenization, and offsets | WordPiece/BERT only; no BPE; graph-encoded rules can drift from Python |

### DJL HuggingFace tokenizers AAR

`ai.djl.huggingface:tokenizers` wraps the same Rust `tokenizers` library that
backs Hugging Face's Python fast tokenizers. It loads a `tokenizer.json` and
returns an `Encoding` whose char-offset table maps every token back to the
source string. Because it consumes the identical `tokenizer.json` that OpenMed's
Python pipeline uses, WordPiece/BPE behavior, normalization, and offsets match
by construction — the strongest possible parity guarantee. The cost is a native
library per ABI in the APK and a minimum SDK of 26.

### Pure-Kotlin WordPiece

A hand-written WordPiece tokenizer has the smallest footprint and no native or
copyleft dependency. The trade-off is implementation burden and parity risk: the
normalization, pre-tokenization, and offset-mapping rules must be re-implemented
to match the Python fast tokenizer exactly, and any drift silently corrupts span
recovery. It also does not cover BPE-based checkpoints without additional work.

### ORT-Extensions in-graph BertTokenizer

ONNX Runtime Extensions can bake a `BertTokenizer` into the `.ort` graph, so the
model accepts raw strings and no separate tokenizer runtime is shipped. However,
in-graph tokenizers do not cleanly surface char-level offsets, which are required
for span recovery; the approach is WordPiece/BERT-only (no BPE); and encoding
rules frozen into the graph can drift from the Python tokenizer.

## Decision

- **Primary strategy: the DJL HuggingFace tokenizers AAR.** It is Apache-2.0
  (permissive, no GPL), provides native high-fidelity char offsets for span
  recovery, targets min SDK 26 (the module's floor), and gives exact parity with
  OpenMed's Python tokenization because both consume the same `tokenizer.json`.
  Span-offset fidelity and Python parity are the decisive factors, and they
  outweigh the per-ABI native-library footprint. This is what the
  `android/openmedkit` module ships today (`ai.djl.huggingface:tokenizers` with
  `HuggingFaceTokenizer.newInstance(...)` and encoding offset mapping in
  `Runtime.kt`).
- **Fallback strategy: pure-Kotlin WordPiece.** If the AAR's native footprint
  becomes unacceptable for a target deployment, a pure-Kotlin WordPiece
  implementation is the fallback: it removes the native dependency at the cost of
  significant implementation work and a parity-verification burden against the
  Python tokenizer. Implementing it is a separate Android-module task.

ORT-Extensions in-graph tokenization is **not** adopted, primarily because its
weak char-offset support is incompatible with the span-recovery requirement.

## Tokenizer Assets To Bundle

The primary strategy consumes the exported Hugging Face fast-tokenizer assets
from the model directory:

- `tokenizer.json` — the fast-tokenizer definition (vocabulary, merges,
  normalization, and pre-tokenization rules). Required by the DJL tokenizer.
- `tokenizer_config.json` — special-token and configuration metadata.
- `id2label.json` — the label map used to decode token-classification logits
  into entity labels.

These are emitted alongside the ONNX graph by the Android export task
(`openmed/onnx/convert.py`; see [Android ONNX Export](export-onnx-android.md)),
which writes the repository's tokenizer and label assets next to the exported
model. Bundling the assets into the app or downloading them with the model is a
separate task.

## References

- [Android ONNX Export](export-onnx-android.md) — the export matrix and the task
  that emits the tokenizer and label assets.
- [Android Span Parity](android-parity.md) — the cross-platform span guarantees
  that offset fidelity must preserve.
- [Swift-Kotlin API Parity](swift-kotlin-parity.md) — cross-platform API
  alignment.
- [Model Manifest](model-manifest.md) — model catalog and reproducibility
  metadata.
