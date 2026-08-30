# Maple Clinical Studio for Android

This Compose demo runs a user-exported
[`deepgrove/maple-preview`](https://huggingface.co/deepgrove/maple-preview)
decoder with ONNX Runtime Mobile. It provides four workflows in one polished,
responsive interface:

- PII detection and redaction with a reviewable list of identifiers;
- clinical entity extraction with source evidence;
- relation extraction constrained to an explicit relation vocabulary; and
- source-grounded reasoning/chat with uncertainty surfaced in the final answer.

The app has **no `INTERNET` permission**. Clinical text, prompts, tokens, and
model outputs stay inside the app process. The included note and preview outputs
are synthetic and visibly labeled; preview mode never pretends that Maple ran.
Streaming output stays hidden until Maple closes its private reasoning section;
an incomplete final answer is blocked for review. PII redaction ignores any
model-authored rewritten note: the app verifies that every proposed entity is
an exact, unique source surface and applies bracketed replacements itself.

## Run the UI immediately

Install JDK 11 and Android SDK Platform 33, then run:

```bash
cd android
./gradlew :OpenMedMapleDemo:app:assembleDebug
```

Open `OpenMedMapleDemo/app/build/outputs/apk/debug/app-debug.apk` in an Android
emulator or device. Until a model is imported, all four tasks use deterministic
synthetic UI previews, which makes the complete demo reviewable without adding
multi-gigabyte weights to git.

## Export contract

The Hugging Face MLX checkpoint is not an ONNX model and cannot be opened by
ONNX Runtime. The supported route starts from the pinned BF16 source and emits
one integrity-bound, unified cached graph:

```bash
python -m openmed.onnx.maple_export requirements \
  > /tmp/openmed-maple-onnx-requirements.txt
uv pip install --requirement /tmp/openmed-maple-onnx-requirements.txt
python -m openmed.onnx.maple_export download \
  "$HOME/Developer/openmed-android-onnx/maple-preview-bf16"
python -m openmed.onnx.maple_export export \
  "$HOME/Developer/openmed-android-onnx/maple-preview-bf16" \
  "$HOME/Developer/openmed-android-onnx/maple-preview-4bit-onnx-mobile" \
  --target mobile
python -m openmed.onnx.maple_export bundle \
  "$HOME/Developer/openmed-android-onnx/maple-preview-4bit-onnx-mobile" \
  "$HOME/Developer/openmed-android-onnx/maple-preview-android.ommaple.zip"
```

The importer can also consume a compatible custom export shaped as either:

1. one unified cached `decoder_model.ort` graph (recommended), with both
   `prefill_path` and `decode_path` pointing to that file;
2. separate prefill and cached-decode graphs, with KV-cache outputs named
   `present.*` and matching decode inputs named `past_key_values.*`; or
3. one stateless `model.onnx`/`model.ort` graph. In that case set `decode_path`
   and `cache` to JSON `null`; the demo will re-run the full context for every
   token and will be much slower.

The unified graph uses batch size 1 and this tensor contract:

```text
inputs   input_ids: int64[1, sequence]
         attention_mask: int64[1, total_sequence]
         past_key_values.N.{key,value}: float32[1, 4, past_sequence, 128]
outputs  logits: float32[1, sequence, 151936]
         present.N.{key,value}: float32[1, 4, total_sequence, 128]
```

For prefill, the app reads each cache input's ONNX Runtime `NodeInfo` and
supplies an empty tensor with shape `[1, 4, 0, 128]` in the graph-declared
floating-point dtype. The past-sequence dimension must therefore be dynamic or
zero-capable. After the first token, each `present.N.*` output is passed to its
matching `past_key_values.N.*` input. When both manifest paths name the same
file, the app opens one shared inference session and the bundle declares and
stores that graph only once.

The export must lower Maple's custom ternary/MoE implementation to operators
supported by the selected ONNX Runtime Mobile build. The MLX artifact remains
2-bit, but ONNX Runtime 1.25.1 CPU rejects QMoE `expert_weight_bits=2`. The
validated portable Android path is blockwise-128 QMoE 4-bit with fused
interleaved SwiGLU and no separate FC3 input. QMoE requires ONNX Runtime 1.23.1
or newer; this project pins 1.25.1. Merely renaming or zipping the 2-bit MLX
safetensors is not sufficient. The source model is a 20B-A1B preview, so test on
a high-memory ARM64 device. A cached, packed export is strongly recommended;
the app allows at most 12 GiB of uncompressed payload and keeps 512 MiB of
free-storage headroom.

The `bundle` command writes `maple-bundle.json` first, stores already-packed
model data without recompressing it, and binds every payload to its exact byte
size and lowercase SHA-256. [`maple-bundle.example.json`](maple-bundle.example.json)
documents the schema for custom exporters; its placeholders are deliberately
rejected. Keep the source revision pinned to a commit SHA, never `main`. The
importer:

- rejects absolute paths, traversal, duplicate entries, undeclared files,
  floating revisions, placeholder hashes, and oversized archives;
- streams into `noBackupFilesDir` rather than memory;
- verifies the declared size and SHA-256 of every file before activation; and
- opens ONNX Runtime only after the entire bundle is verified.

The checksums provide bundle integrity, not publisher authentication. Obtain the
export and its manifest through a channel you trust; the app intentionally does
not contact Hugging Face or any signing service to establish provenance.

In the app, tap **Import model**, choose the ZIP through Android's system file
picker, and wait for verification. Inference remains offline after import.

The complete pinned FLOAT32 graph has passed ONNX Runtime 1.25.1 CPU prefill
and cached decode with finite logits on Apple Silicon. This validates the graph
contract, not Android-device memory, performance, thermal behavior, or clinical
quality; those remain physical-device release gates.

## Validate

```bash
cd android
./gradlew :OpenMedMapleDemo:app:testDebugUnitTest
./gradlew :OpenMedMapleDemo:app:assembleDebug
```

Before treating an export as releasable, also run synthetic parity cases for
all four prompts, direct-identifier recall and critical leakage, structured JSON
validity, cache/no-cache generation parity, peak RAM, time-to-first-token, and
thermal behavior on each supported device class. Never commit clinical records
or model binaries as fixtures.

## Safety scope

Maple-Preview has limited post-training and is not a clinical model. This demo
is research software, not a medical device. Its output can omit, misclassify, or
invent information. A qualified reviewer must compare every output to the
source record before clinical, privacy, or disclosure decisions.
