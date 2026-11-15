# OpenMedKit Flutter

`openmedkit_flutter` is a local-first Flutter FFI plugin for OpenMed ONNX
token-classification exports. It exposes typed `extractPii` and `deidentify`
operations without adding matched source text to span records or logs.

The package does not download models. A model directory must already contain
`id2label.json`, `tokenizer.json`, and one of `model_int8.onnx`, `model.onnx`,
or `model_fp16.onnx`. Platform registration validates that directory before the
Dart bindings create the native runtime.

```dart
final openmed = await OpenMedKit.loadModel(
  modelDirectory: localModelDirectory,
  tokenizer: localTokenizer.encode,
);

final result = await openmed.deidentify(syntheticNote);
try {
  useRedactedText(result.deidentifiedText);
} finally {
  openmed.close();
}
```

## Native session contract

`native/openmed_ffi.h` defines a narrow `openmed_session_api`. Android builds
bind it to ONNX Runtime 1.25.1 from Maven Central, the version OpenMedKit
Android uses. Apple builds bind it to ONNX Runtime 1.20.0 from CocoaPods. Other
desktop builds can register the same vtable or set `OPENMED_ONNXRUNTIME_ROOT`
to a local ONNX Runtime package. The adapter receives a local model path plus
tokenized `input_ids`, `attention_mask`, and offsets. The shared C shim owns
logits decoding, typed span allocation, and masking, so Dart sees the same
contract on every target.

Tokenizer adapters must operate from local assets and return Unicode-scalar
offsets. Long-running inference should be invoked away from the UI isolate by
applications processing large notes.

Android 15+ devices with 16 KB memory pages only load 64-bit libraries whose
ELF `PT_LOAD` segments are 16 KB aligned. The Android build links
`libopenmed_ffi.so` with `-Wl,-z,max-page-size=16384`, which keeps NDK r27 and
older compatible, and ONNX Runtime 1.25.1 ships 16 KB-aligned libraries. ONNX
Runtime 1.20.0, the version used before, has a 4 KB-aligned
`libonnxruntime4j_jni.so`. Check a consumer APK with
`python3 scripts/android/check_elf_page_alignment.py --require-native-libraries
<app.apk>`.

The package is intentionally not configured for pub.dev publication.
