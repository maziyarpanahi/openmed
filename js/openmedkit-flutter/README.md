# OpenMedKit Flutter

`openmedkit_flutter` is a local-first Flutter FFI plugin for OpenMed ONNX
token-classification exports. It exposes typed `extractPii` and `deidentify`
operations without adding matched source text to span records or logs.

The package does not download models. A model loads only from a signed
[portable bundle](../../docs/models/portable-bundle.md) that passes
verification first.

## Load a verified bundle

```dart
final bundle = await OpenMedVerifiedBundle.open(
  directory: localBundleDirectory,
  trustPolicy: PortableBundleTrustPolicy.fromJson(myTrustPolicyDocument),
  context: PortableBundleContext(
    now: portableBundleTimestamp(DateTime.now()),
    target: PortableBundleTarget.flutter(platform: 'android', osVersion: '34'),
    sequenceFloors: {'my.bundle.id': lastAcceptedSequence},
  ),
);

final openmed = await OpenMedKit.loadBundle(
  bundle: bundle,
  tokenizer: localTokenizer.encode,
);
try {
  final result = await openmed.deidentify(syntheticNote);
  useRedactedText(result.deidentifiedText);
} finally {
  openmed.close();
}
```

`OpenMedVerifiedBundle.open` runs the same checks, in the same order, as the
Python, Swift, Kotlin, and TypeScript readers: size limits, the envelope,
Ed25519 signatures against the caller's trust policy, the canonical manifest,
structure, validity and rollback, compatibility, and the digest of every file.
It never follows symlinks and fails with an `OpenMedBundleException` whose
`code` is a portable-bundle reason code. The Dart reader passes the shared
golden vectors in `tests/fixtures/portable_bundle/vectors.json`.

The first eligible `onnx` (or, on Android and iOS, `ort`) runtime is selected
unless `runtimeId` names another. Labels come from the verified label map.
Tokenizer assets should be read with `bundle.readVerifiedFile(path)`, which
checks the digest again. `OpenMedKit.loadBundle` passes the model digest to
the native layer, which hashes the file before any session exists and creates
the ONNX Runtime session from exactly the bytes it hashed. A model changed
after verification fails with `OpenMedFfiException` code `integrity_failed`.
Graphs that keep weights in external data files are not supported by this
loader.

## Lifecycle

- `close()` frees the native runtime exactly once. Repeated calls do nothing.
- `replaceBundle(bundle)` creates the new runtime first. If that fails, the
  partial native allocations are released and the current model keeps serving.
  On success the previous runtime is freed once.
- A request still waiting for its tokenizer when the kit is closed, replaced,
  or cancelled through an `OpenMedCancellationToken` never reaches native code.
  It fails with `OpenMedCancelledException` (`closed`, `model_replaced`, or
  `cancelled`). A native call that is already running is not interrupted.
- Each kit is bound to the isolate that loaded it. To supervise a worker
  isolate that may be killed, create an `OpenMedRuntimeOwner` in a long-lived
  isolate, pass `owner.id` to the worker's `loadBundle(ownerId: ...)`, and call
  `owner.release()` from the worker's exit listener. Runtimes the worker did
  not close are then freed once; runtimes it closed are not freed again.
- `OpenMedKit.liveRuntimeCount()` reports how many native runtimes exist.

Errors are value-free: argument errors, native messages, and bundle errors
never include note text, file paths, digests, or adapter-provided text.

## Platforms

Android, iOS, macOS, Linux, and Windows load models through `dart:ffi`.
Android and Apple builds bind ONNX Runtime 1.20.0 from Maven Central and
CocoaPods. Other desktop builds can register an `openmed_session_api` adapter
or set `OPENMED_ONNXRUNTIME_ROOT` to a local ONNX Runtime package.

Platforms without `dart:io` and `dart:ffi`, such as the web, compile against
the same API. `OpenMedKit.platformSupport` reports `supported: false` with
reason `platform_unsupported`, loading throws
`OpenMedUnsupportedPlatformException`, and `PortableBundle.verify` returns a
refused result with `signature_verifier_unavailable`. Browser apps can use the
`openmed` npm package instead.

## Native contract

`native/openmed_ffi.h` defines ABI version 2. `openmed_runtime_create` takes
the verified `sha256:` digest of the model file and an optional owner id. The
shared C shim owns hashing, logits decoding, typed span allocation, masking,
and the runtime registry, so Dart sees the same contract on every target.
`openmed_runtime_session_kind` reports whether a runtime uses the synthetic
fixture session, the linked ONNX Runtime, or a registered adapter.

Tokenizer adapters must operate from local assets and return Unicode-scalar
offsets. Long-running inference should be invoked away from the UI isolate by
applications processing large notes.

## Tests

The default lane is synthetic. It runs the native checks under
AddressSanitizer, then the Dart tests against the fixture session:

```bash
cmake -S native -B build/checks -DOPENMED_FFI_BUILD_TESTS=ON
cmake --build build/checks && ctest --test-dir build/checks
cmake -S native -B build/ffi -DOPENMED_FFI_ENABLE_TEST_SESSION=ON
cmake --build build/ffi
OPENMED_FFI_LIBRARY=$PWD/build/ffi/libopenmed_ffi.so OPENMED_FFI_TEST_SESSION=1 \
  flutter test test/
```

The Dart suites cover the portable-bundle vectors, verified loading,
cancellation, repeated close, model replacement, isolate teardown, the shared
offset contract in `tests/fixtures/parity/offset_contract.json`, and the
unsupported web build. The signed fixture bundle is generated by
`tests/mobile/fixtures/generate_flutter_ffi_bundle.py` with a TEST-ONLY key.
Fixture sessions report `OpenMedSessionKind.fixture` and never support a real
runtime claim.

The real-artifact lane is opt-in (`workflow_dispatch` with `real_artifact`)
and runs `test/ffi_real_artifact_test.dart` on a runner that already holds an
extracted ONNX Runtime package and a signed bundle. It refuses the fixture
session and requires `OpenMedSessionKind.onnxRuntime`. Its vectors file is
pre-tokenized synthetic text:

```json
{
  "schema": "openmed.flutter_ffi.real_vectors",
  "version": 1,
  "score_tolerance": 0.001,
  "cases": [
    {
      "id": "synthetic-1",
      "text": "Patient Jane Roe called.",
      "input_ids": [101, 5310, 4869, 20996, 2170, 1012, 102],
      "attention_mask": [1, 1, 1, 1, 1, 1, 1],
      "offsets": [[0, 0], [0, 7], [8, 12], [13, 16], [17, 23], [23, 24], [0, 0]],
      "threshold": 0.5,
      "expected_spans": [{"start": 8, "end": 16, "label": "NAME", "score": 0.97}],
      "expected_redacted": "Patient [NAME] called."
    }
  ]
}
```

It writes value-free evidence (content id, runtime, session kind, and case
counts) when `OPENMED_FLUTTER_REAL_EVIDENCE` is set.

The package is intentionally not configured for pub.dev publication.
