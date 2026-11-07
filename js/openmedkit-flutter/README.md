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

`native/openmed_ffi.h` defines a narrow `openmed_session_api`. Android and Apple
builds bind it to ONNX Runtime 1.20.0 from Maven Central and CocoaPods. Other
desktop builds can register the same vtable or set `OPENMED_ONNXRUNTIME_ROOT`
to a local ONNX Runtime package. The adapter receives a local model path plus
tokenized `input_ids`, `attention_mask`, and offsets. The shared C shim owns
logits decoding, typed span allocation, and masking, so Dart sees the same
contract on every target.

Tokenizer adapters must operate from local assets and return Unicode-scalar
offsets. Long-running inference should be invoked away from the UI isolate by
applications processing large notes.

The package is intentionally not configured for pub.dev publication.
