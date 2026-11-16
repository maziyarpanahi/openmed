// OpenMedKit over dart:ffi for Android, iOS, macOS, Linux, and Windows.

import 'dart:convert';
import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'crypto.dart';
import 'errors.dart';
import 'portable_bundle.dart';
import 'portable_bundle_types.dart';
import 'types.dart';

/// Offline portable-bundle verification for this platform.
class PortableBundle {
  PortableBundle._();

  /// Verifies manifest and envelope bytes, plus an optional in-memory tree.
  ///
  /// Throws [PortableBundleException] (`context_invalid`) for a malformed
  /// [context]. Every other failure is a value-free result.
  static PortableBundleVerification verify({
    required List<int> manifest,
    required List<int> envelope,
    required PortableBundleTrustPolicy trustPolicy,
    required PortableBundleContext context,
    Map<String, List<int>>? files,
  }) {
    return verifyPortableBundle(
      manifest: manifest,
      envelope: envelope,
      trustPolicy: trustPolicy,
      context: context,
      files: files == null ? null : PortableBundleMemoryTree(files),
    );
  }

  /// Verifies a bundle directory without following symlinks.
  static PortableBundleVerification verifyDirectory(
    String directory, {
    required PortableBundleTrustPolicy trustPolicy,
    required PortableBundleContext context,
  }) {
    return verifyPortableBundleDirectoryDetailed(
      directory,
      trustPolicy: trustPolicy,
      context: context,
    ).verification;
  }

  /// Returns the RFC 8785 canonical encoding of [value].
  static String canonicalJson(Object? value) =>
      portableBundleCanonicalJson(value);

  /// Parses and validates canonical manifest bytes without signatures.
  static Map<String, dynamic> parseManifest(
    List<int> manifest, {
    List<String> understoodExtensions = const <String>[],
  }) =>
      parsePortableBundleManifest(
        manifest,
        understoodExtensions: understoodExtensions,
      );

  /// Returns the `sha256:<hex>` content identity of [manifest].
  static String contentId(List<int> manifest) =>
      portableBundleContentId(manifest);

  /// Returns the DSSE pre-authentication encoding of [manifest].
  static Uint8List signingInput(List<int> manifest) =>
      portableBundleSigningInput(manifest);

  /// Verifies one Ed25519 signature with the bundled RFC 8032 verifier.
  static bool verifyEd25519(
    List<int> publicKey,
    List<int> signature,
    List<int> message,
  ) =>
      Ed25519Verifier.verify(publicKey, signature, message);
}

/// A bundle whose signature, manifest, and every file digest were verified.
///
/// Instances exist only through [open]. A runtime can be created only from a
/// verified bundle, and the native layer hashes the model file again before
/// any session exists, so bytes changed after verification are refused.
class OpenMedVerifiedBundle {
  OpenMedVerifiedBundle._({
    required this.verification,
    required this.bundleId,
    required this.sequence,
    required this.runtimeId,
    required this.runtimeFormat,
    required this.runtimePrecision,
    required this.labels,
    required String root,
    required String modelPath,
    required String modelDigest,
    required Map<String, String> fileDigests,
  })  : _root = root,
        _modelPath = modelPath,
        _modelDigest = modelDigest,
        _fileDigests = fileDigests;

  /// Value-free verification result.
  final PortableBundleVerification verification;

  /// `bundle.id`.
  final String bundleId;

  /// `bundle.sequence`, the anti-rollback counter to persist after activation.
  final int sequence;

  /// Selected runtime id.
  final String runtimeId;

  /// Selected runtime format (`onnx` or `ort`).
  final String runtimeFormat;

  /// Selected runtime precision.
  final String runtimePrecision;

  /// Labels in model output order, read from the verified label map.
  final List<String> labels;

  final String _root;
  final String _modelPath;
  final String _modelDigest;
  final Map<String, String> _fileDigests;

  /// `sha256:<hex>` content identity of the manifest.
  String get contentId => verification.contentId!;

  /// Verifies the bundle in [directory] and selects a loadable runtime.
  ///
  /// [context] must name a target, normally
  /// [PortableBundleTarget.flutter]. The first eligible `onnx` or `ort`
  /// runtime is selected unless [runtimeId] names another eligible one.
  /// Verification hashes every declared file, so for large bundles call this
  /// away from the UI isolate.
  ///
  /// Throws [OpenMedBundleException] with a value-free code.
  static Future<OpenMedVerifiedBundle> open({
    required String directory,
    required PortableBundleTrustPolicy trustPolicy,
    required PortableBundleContext context,
    String? runtimeId,
  }) async {
    final String trimmed = directory.trim();
    if (trimmed.isEmpty ||
        RegExp(r'^[a-z][a-z0-9+.-]*://').hasMatch(trimmed.toLowerCase())) {
      throw const OpenMedBundleException('bundle_directory_invalid');
    }
    if (context.target == null) {
      throw const OpenMedBundleException('context_target_required');
    }
    final String root = Directory(trimmed).absolute.path;
    PortableBundleVerificationDetail detail;
    try {
      detail = verifyPortableBundleDirectoryDetailed(
        root,
        trustPolicy: trustPolicy,
        context: context,
      );
    } on PortableBundleException catch (error) {
      throw OpenMedBundleException(error.reason);
    }
    final PortableBundleVerification verification = detail.verification;
    final PortableBundleManifestView? manifest = detail.manifest;
    if (!verification.eligibleForActivation || manifest == null) {
      throw OpenMedBundleException(
        verification.reason ?? 'file_missing',
        verification: verification,
      );
    }
    final Map<String, dynamic>? runtime =
        _selectRuntime(manifest, verification.eligibleRuntimes, runtimeId);
    if (runtime == null) {
      throw OpenMedBundleException(
        'runtime_unavailable',
        verification: verification,
      );
    }
    final Map<String, dynamic> value = manifest.value;
    final Map<String, dynamic> offsets =
        value['offsets'] as Map<String, dynamic>;
    if (offsets['unit'] != 'unicode_scalar') {
      throw OpenMedBundleException(
        'offset_unit_unsupported',
        verification: verification,
      );
    }
    final Object? labelSection = value['labels'];
    if (labelSection is! Map<String, dynamic>) {
      throw OpenMedBundleException(
        'labels_missing',
        verification: verification,
      );
    }
    final Map<String, String> digests = <String, String>{
      for (final Map<String, dynamic> item in manifest.files)
        item['path'] as String: item['sha256'] as String,
    };
    final String labelPath = labelSection['path'] as String;
    final Uint8List labelBytes = _readVerified(root, labelPath, digests);
    final List<String> labels = _parseLabels(
      labelBytes,
      labelSection['count'] as int,
      verification,
    );
    final String entrypoint = runtime['entrypoint'] as String;
    return OpenMedVerifiedBundle._(
      verification: verification,
      bundleId: manifest.bundleId,
      sequence: manifest.sequence,
      runtimeId: runtime['id'] as String,
      runtimeFormat: runtime['format'] as String,
      runtimePrecision: runtime['precision'] as String,
      labels: List<String>.unmodifiable(labels),
      root: root,
      modelPath: _nativePath(root, entrypoint),
      modelDigest: digests[entrypoint]!,
      fileDigests: Map<String, String>.unmodifiable(digests),
    );
  }

  /// Reads a declared bundle file and checks its digest again.
  ///
  /// Use this to load tokenizer assets from the same verified bundle.
  /// Throws [OpenMedBundleException] with `file_reference_undeclared` or
  /// `file_digest_mismatch`.
  Future<Uint8List> readVerifiedFile(String path) async =>
      _readVerified(_root, path, _fileDigests);

  @override
  String toString() =>
      'OpenMedVerifiedBundle(content_id: $contentId, runtime: $runtimeId)';

  static Map<String, dynamic>? _selectRuntime(
    PortableBundleManifestView manifest,
    List<int> eligible,
    String? runtimeId,
  ) {
    for (final int index in eligible) {
      final Map<String, dynamic> runtime = manifest.runtimes[index];
      final Object? format = runtime['format'];
      if (format != 'onnx' && format != 'ort') {
        continue;
      }
      if (runtimeId == null || runtime['id'] == runtimeId) {
        return runtime;
      }
    }
    return null;
  }

  static String _nativePath(String root, String relative) {
    final String separator = Platform.pathSeparator;
    final String base = root.endsWith(separator) ? root : '$root$separator';
    return '$base${relative.split('/').join(separator)}';
  }

  static Uint8List _readVerified(
    String root,
    String path,
    Map<String, String> digests,
  ) {
    final String? expected = digests[path];
    if (expected == null) {
      throw const OpenMedBundleException('file_reference_undeclared');
    }
    final String native = _nativePath(root, path);
    Uint8List bytes;
    try {
      if (FileSystemEntity.typeSync(native, followLinks: false) !=
          FileSystemEntityType.file) {
        throw const OpenMedBundleException('file_missing');
      }
      bytes = File(native).readAsBytesSync();
    } on FileSystemException {
      throw const OpenMedBundleException('file_missing');
    }
    if (sha256Identity(bytes) != expected) {
      throw const OpenMedBundleException('file_digest_mismatch');
    }
    return bytes;
  }

  static List<String> _parseLabels(
    Uint8List bytes,
    int count,
    PortableBundleVerification verification,
  ) {
    final OpenMedBundleException invalid = OpenMedBundleException(
      'labels_invalid',
      verification: verification,
    );
    Object? decoded;
    try {
      decoded = jsonDecode(utf8.decode(bytes));
    } on FormatException {
      throw invalid;
    }
    if (decoded is! Map<String, dynamic> ||
        decoded.length != count ||
        count < 2) {
      throw invalid;
    }
    final List<String> labels = List<String>.filled(count, '');
    for (int index = 0; index < count; ++index) {
      final Object? label = decoded['$index'];
      if (label is! String || label.isEmpty || label.contains('\u0000')) {
        throw invalid;
      }
      labels[index] = label;
    }
    return labels;
  }
}

/// Native runtime ownership for isolates that may be torn down.
///
/// Create an owner in a long-lived isolate, pass [id] to a worker isolate's
/// [OpenMedKit.loadBundle], and call [release] from the worker's exit
/// listener. Every runtime the worker did not close is then freed exactly
/// once; runtimes it closed itself are not freed again.
class OpenMedRuntimeOwner {
  OpenMedRuntimeOwner._(this._bindings, this.id);

  /// Creates a new owner identifier.
  factory OpenMedRuntimeOwner.create({String? nativeLibraryPath}) {
    final _OpenMedBindings bindings = _OpenMedBindings.open(nativeLibraryPath);
    return OpenMedRuntimeOwner._(bindings, bindings.ownerCreate());
  }

  final _OpenMedBindings _bindings;

  /// Opaque, sendable owner identifier.
  final int id;

  /// Frees every runtime still held by this owner and returns the count.
  ///
  /// Call it only after every isolate that used those runtimes has exited.
  int release() => _bindings.ownerRelease(id);
}

/// Local OpenMed token-classification and de-identification runtime.
///
/// Each instance owns exactly one native runtime per model generation. The
/// runtime is freed exactly once: by [close], by a successful
/// [replaceBundle], or by [OpenMedRuntimeOwner.release]. A request that is
/// waiting for its tokenizer when the kit is closed, replaced, or cancelled
/// never reaches native code and fails with [OpenMedCancelledException].
class OpenMedKit {
  OpenMedKit._(
    this._bindings,
    this._runtime,
    this._bundle,
    this._tokenizer,
    this._ownerId,
    this.defaultThreshold,
  );

  final _OpenMedBindings _bindings;
  ffi.Pointer<_NativeRuntime> _runtime;
  OpenMedVerifiedBundle _bundle;
  final OpenMedTokenizer _tokenizer;
  final int _ownerId;

  /// Threshold used when a call does not pass one.
  final double defaultThreshold;
  int _generation = 0;
  bool _closed = false;

  /// Whether this platform can load models through the FFI.
  static OpenMedPlatformSupport get platformSupport {
    final String platform = Platform.operatingSystem;
    switch (platform) {
      case 'android':
      case 'ios':
        return OpenMedPlatformSupport(
          platform: platform,
          supported: true,
          runtimes: const <String>['onnx', 'ort'],
        );
      case 'macos':
      case 'linux':
      case 'windows':
        return OpenMedPlatformSupport(
          platform: platform,
          supported: true,
          runtimes: const <String>['onnx'],
        );
      default:
        return OpenMedPlatformSupport(
          platform: platform,
          supported: false,
          runtimes: const <String>[],
          reason: 'platform_unsupported',
        );
    }
  }

  /// Number of native runtimes created and not yet freed in this process.
  static int liveRuntimeCount({String? nativeLibraryPath}) =>
      _OpenMedBindings.open(nativeLibraryPath).liveCount();

  /// Session kinds the native library can create.
  static Set<OpenMedSessionKind> nativeSessionKinds({
    String? nativeLibraryPath,
  }) {
    final int capabilities =
        _OpenMedBindings.open(nativeLibraryPath).capabilities();
    return <OpenMedSessionKind>{
      if (capabilities & 1 != 0) OpenMedSessionKind.onnxRuntime,
      if (capabilities & 2 != 0) OpenMedSessionKind.fixture,
      if (capabilities & 4 != 0) OpenMedSessionKind.registeredAdapter,
    };
  }

  /// Creates a local runtime from a verified bundle without network access.
  ///
  /// [tokenizer] must encode from already-local assets, for example through
  /// [OpenMedVerifiedBundle.readVerifiedFile]. [ownerId] comes from
  /// [OpenMedRuntimeOwner.id] when another isolate supervises this one.
  static Future<OpenMedKit> loadBundle({
    required OpenMedVerifiedBundle bundle,
    required OpenMedTokenizer tokenizer,
    double defaultThreshold = 0.5,
    String? nativeLibraryPath,
    int ownerId = 0,
  }) async {
    final OpenMedPlatformSupport support = platformSupport;
    if (!support.supported) {
      throw OpenMedUnsupportedPlatformException(support.platform);
    }
    _validateThreshold(defaultThreshold);
    if (ownerId < 0) {
      throw ArgumentError('ownerId must come from OpenMedRuntimeOwner');
    }
    final _OpenMedBindings bindings = _OpenMedBindings.open(nativeLibraryPath);
    final ffi.Pointer<_NativeRuntime> runtime =
        _createRuntime(bindings, bundle, ownerId);
    return OpenMedKit._(
      bindings,
      runtime,
      bundle,
      tokenizer,
      ownerId,
      defaultThreshold,
    );
  }

  /// The verified bundle the current generation was created from.
  OpenMedVerifiedBundle get bundle => _bundle;

  /// Increments each time the model is replaced or the kit is closed.
  int get generation => _generation;

  /// Whether [close] was called.
  bool get isClosed => _closed;

  /// Session implementation behind the current generation.
  OpenMedSessionKind get sessionKind {
    _ensureOpen();
    switch (_bindings.sessionKind(_runtime)) {
      case 1:
        return OpenMedSessionKind.fixture;
      case 2:
        return OpenMedSessionKind.onnxRuntime;
      default:
        return OpenMedSessionKind.registeredAdapter;
    }
  }

  /// Detects PII locally and returns typed spans without matched source text.
  Future<List<OpenMedSpan>> extractPii(
    String text, {
    double? threshold,
    OpenMedCancellationToken? cancellation,
  }) async {
    final int generation = _begin(text, threshold, cancellation);
    final double resolvedThreshold = threshold ?? defaultThreshold;
    final OpenMedTokenBatch batch = await _tokenizer(text);
    _ensureCurrent(generation, cancellation);
    _validateBatch(text, batch);
    final List<OpenMedSpan> spans = _withNativeBatch<List<OpenMedSpan>>(
      text,
      batch,
      (ffi.Pointer<Utf8> nativeText,
          ffi.Pointer<_NativeTokenBatch> nativeBatch) {
        final ffi.Pointer<_NativeSpanList> spans = calloc<_NativeSpanList>();
        try {
          final int status = _bindings.runtimeExtractPii(
            _runtime,
            nativeText,
            nativeBatch,
            resolvedThreshold,
            spans,
          );
          _bindings.checkStatus(status);
          return _copySpans(spans.ref);
        } finally {
          _bindings.spanListFree(spans);
          calloc.free(spans);
        }
      },
    );
    _ensureCurrent(generation, cancellation);
    return spans;
  }

  /// Detects and masks PII locally using `[LABEL]` replacements.
  Future<OpenMedDeidentificationResult> deidentify(
    String text, {
    double? threshold,
    OpenMedCancellationToken? cancellation,
  }) async {
    final int generation = _begin(text, threshold, cancellation);
    final double resolvedThreshold = threshold ?? defaultThreshold;
    final OpenMedTokenBatch batch = await _tokenizer(text);
    _ensureCurrent(generation, cancellation);
    _validateBatch(text, batch);
    final OpenMedDeidentificationResult result =
        _withNativeBatch<OpenMedDeidentificationResult>(
      text,
      batch,
      (ffi.Pointer<Utf8> nativeText,
          ffi.Pointer<_NativeTokenBatch> nativeBatch) {
        final ffi.Pointer<ffi.Pointer<Utf8>> output =
            calloc<ffi.Pointer<Utf8>>();
        final ffi.Pointer<_NativeSpanList> spans = calloc<_NativeSpanList>();
        try {
          final int status = _bindings.runtimeDeidentify(
            _runtime,
            nativeText,
            nativeBatch,
            resolvedThreshold,
            output,
            spans,
          );
          _bindings.checkStatus(status);
          return OpenMedDeidentificationResult(
            deidentifiedText: output.value.toDartString(),
            spans: _copySpans(spans.ref),
          );
        } finally {
          if (output.value != ffi.nullptr) {
            _bindings.stringFree(output.value);
          }
          _bindings.spanListFree(spans);
          calloc.free(spans);
          calloc.free(output);
        }
      },
    );
    _ensureCurrent(generation, cancellation);
    return result;
  }

  /// Replaces the model with another verified bundle.
  ///
  /// The new runtime is created first. If that fails, its partial native
  /// allocations are released, the current generation keeps serving, and the
  /// error is rethrown. On success the previous runtime is freed exactly once
  /// and requests still waiting for their tokenizer are dropped with
  /// `model_replaced`.
  Future<void> replaceBundle(OpenMedVerifiedBundle bundle) async {
    _ensureOpen();
    final ffi.Pointer<_NativeRuntime> replacement =
        _createRuntime(_bindings, bundle, _ownerId);
    final ffi.Pointer<_NativeRuntime> previous = _runtime;
    _runtime = replacement;
    _bundle = bundle;
    _generation += 1;
    _bindings.runtimeDestroy(previous);
  }

  /// Releases the native session. Repeated calls are no-ops.
  void close() {
    if (_closed) {
      return;
    }
    _closed = true;
    _generation += 1;
    _bindings.runtimeDestroy(_runtime);
    _runtime = ffi.nullptr;
  }

  int _begin(
    String text,
    double? threshold,
    OpenMedCancellationToken? cancellation,
  ) {
    _ensureOpen();
    _validateText(text);
    _validateThreshold(threshold ?? defaultThreshold);
    final int generation = _generation;
    _ensureCurrent(generation, cancellation);
    return generation;
  }

  void _ensureCurrent(int generation, OpenMedCancellationToken? cancellation) {
    if (_closed) {
      throw const OpenMedCancelledException('closed');
    }
    if (generation != _generation) {
      throw const OpenMedCancelledException('model_replaced');
    }
    if (cancellation != null && cancellation.isCancelled) {
      throw const OpenMedCancelledException('cancelled');
    }
  }

  static ffi.Pointer<_NativeRuntime> _createRuntime(
    _OpenMedBindings bindings,
    OpenMedVerifiedBundle bundle,
    int ownerId,
  ) {
    final List<String> labels = bundle.labels;
    final ffi.Pointer<ffi.Pointer<_NativeRuntime>> runtimeOut =
        calloc<ffi.Pointer<_NativeRuntime>>();
    final ffi.Pointer<Utf8> nativeModelPath = bundle._modelPath.toNativeUtf8();
    final ffi.Pointer<Utf8> nativeDigest = bundle._modelDigest.toNativeUtf8();
    final ffi.Pointer<ffi.Pointer<Utf8>> nativeLabels =
        calloc<ffi.Pointer<Utf8>>(labels.length);
    try {
      for (int index = 0; index < labels.length; ++index) {
        nativeLabels[index] = labels[index].toNativeUtf8();
      }
      final int status = bindings.runtimeCreate(
        nativeModelPath,
        nativeDigest,
        nativeLabels,
        labels.length,
        ownerId,
        runtimeOut,
      );
      bindings.checkStatus(status);
      return runtimeOut.value;
    } finally {
      for (int index = 0; index < labels.length; ++index) {
        if (nativeLabels[index] != ffi.nullptr) {
          calloc.free(nativeLabels[index]);
        }
      }
      calloc.free(nativeLabels);
      calloc.free(nativeDigest);
      calloc.free(nativeModelPath);
      calloc.free(runtimeOut);
    }
  }

  T _withNativeBatch<T>(
    String text,
    OpenMedTokenBatch batch,
    T Function(
      ffi.Pointer<Utf8> text,
      ffi.Pointer<_NativeTokenBatch> batch,
    )
        operation,
  ) {
    final ffi.Pointer<Utf8> nativeText = text.toNativeUtf8();
    final ffi.Pointer<ffi.Int64> inputIds = calloc<ffi.Int64>(batch.tokenCount);
    final ffi.Pointer<ffi.Int64> attentionMask =
        calloc<ffi.Int64>(batch.tokenCount);
    final ffi.Pointer<ffi.Int64> starts = calloc<ffi.Int64>(batch.tokenCount);
    final ffi.Pointer<ffi.Int64> ends = calloc<ffi.Int64>(batch.tokenCount);
    final ffi.Pointer<_NativeTokenBatch> nativeBatch =
        calloc<_NativeTokenBatch>();
    try {
      for (int index = 0; index < batch.tokenCount; ++index) {
        inputIds[index] = batch.inputIds[index];
        attentionMask[index] = batch.attentionMask[index];
        starts[index] = batch.offsets[index].start;
        ends[index] = batch.offsets[index].end;
      }
      nativeBatch.ref
        ..inputIds = inputIds
        ..attentionMask = attentionMask
        ..startOffsets = starts
        ..endOffsets = ends
        ..tokenCount = batch.tokenCount;
      return operation(nativeText, nativeBatch);
    } finally {
      calloc.free(nativeBatch);
      calloc.free(ends);
      calloc.free(starts);
      calloc.free(attentionMask);
      calloc.free(inputIds);
      calloc.free(nativeText);
    }
  }

  List<OpenMedSpan> _copySpans(_NativeSpanList native) {
    return List<OpenMedSpan>.unmodifiable(
      List<OpenMedSpan>.generate(native.count, (int index) {
        final _NativeSpan span = native.items[index];
        return OpenMedSpan(
          start: span.start,
          end: span.end,
          label: span.label.toDartString(),
          score: span.score,
        );
      }),
    );
  }

  void _ensureOpen() {
    if (_closed) {
      throw StateError('OpenMedKit has been closed');
    }
  }

  static void _validateText(String text) {
    if (text.isEmpty || text.contains('\u0000')) {
      throw ArgumentError('text must be non-empty and must not contain NUL');
    }
  }

  static void _validateThreshold(double value) {
    if (!(value >= 0.0 && value <= 1.0)) {
      throw ArgumentError('threshold must be between 0 and 1');
    }
  }

  static void _validateBatch(String text, OpenMedTokenBatch batch) {
    final int textLength = text.runes.length;
    for (final OpenMedTokenOffset offset in batch.offsets) {
      if (offset.start < 0 ||
          offset.end < offset.start ||
          offset.end > textLength) {
        throw ArgumentError(
          'token offsets must be Unicode-scalar offsets within the source text',
        );
      }
    }
  }
}

class _NativeRuntime extends ffi.Opaque {}

class _NativeSpan extends ffi.Struct {
  @ffi.Int64()
  external int start;

  @ffi.Int64()
  external int end;

  @ffi.Double()
  external double score;

  external ffi.Pointer<Utf8> label;
}

class _NativeSpanList extends ffi.Struct {
  external ffi.Pointer<_NativeSpan> items;

  @ffi.IntPtr()
  external int count;
}

class _NativeTokenBatch extends ffi.Struct {
  external ffi.Pointer<ffi.Int64> inputIds;
  external ffi.Pointer<ffi.Int64> attentionMask;
  external ffi.Pointer<ffi.Int64> startOffsets;
  external ffi.Pointer<ffi.Int64> endOffsets;

  @ffi.IntPtr()
  external int tokenCount;
}

typedef _Uint32Native = ffi.Uint32 Function();
typedef _Uint32Dart = int Function();
typedef _Uint64Native = ffi.Uint64 Function();
typedef _Uint64Dart = int Function();
typedef _OwnerReleaseNative = ffi.Uint64 Function(ffi.Uint64);
typedef _OwnerReleaseDart = int Function(int);
typedef _RuntimeCreateNative = ffi.Int32 Function(
  ffi.Pointer<Utf8>,
  ffi.Pointer<Utf8>,
  ffi.Pointer<ffi.Pointer<Utf8>>,
  ffi.IntPtr,
  ffi.Uint64,
  ffi.Pointer<ffi.Pointer<_NativeRuntime>>,
);
typedef _RuntimeCreateDart = int Function(
  ffi.Pointer<Utf8>,
  ffi.Pointer<Utf8>,
  ffi.Pointer<ffi.Pointer<Utf8>>,
  int,
  int,
  ffi.Pointer<ffi.Pointer<_NativeRuntime>>,
);
typedef _RuntimeDestroyNative = ffi.Void Function(
  ffi.Pointer<_NativeRuntime>,
);
typedef _RuntimeDestroyDart = void Function(ffi.Pointer<_NativeRuntime>);
typedef _SessionKindNative = ffi.Int32 Function(ffi.Pointer<_NativeRuntime>);
typedef _SessionKindDart = int Function(ffi.Pointer<_NativeRuntime>);
typedef _RuntimeExtractPiiNative = ffi.Int32 Function(
  ffi.Pointer<_NativeRuntime>,
  ffi.Pointer<Utf8>,
  ffi.Pointer<_NativeTokenBatch>,
  ffi.Double,
  ffi.Pointer<_NativeSpanList>,
);
typedef _RuntimeExtractPiiDart = int Function(
  ffi.Pointer<_NativeRuntime>,
  ffi.Pointer<Utf8>,
  ffi.Pointer<_NativeTokenBatch>,
  double,
  ffi.Pointer<_NativeSpanList>,
);
typedef _RuntimeDeidentifyNative = ffi.Int32 Function(
  ffi.Pointer<_NativeRuntime>,
  ffi.Pointer<Utf8>,
  ffi.Pointer<_NativeTokenBatch>,
  ffi.Double,
  ffi.Pointer<ffi.Pointer<Utf8>>,
  ffi.Pointer<_NativeSpanList>,
);
typedef _RuntimeDeidentifyDart = int Function(
  ffi.Pointer<_NativeRuntime>,
  ffi.Pointer<Utf8>,
  ffi.Pointer<_NativeTokenBatch>,
  double,
  ffi.Pointer<ffi.Pointer<Utf8>>,
  ffi.Pointer<_NativeSpanList>,
);
typedef _SpanListFreeNative = ffi.Void Function(
  ffi.Pointer<_NativeSpanList>,
);
typedef _SpanListFreeDart = void Function(ffi.Pointer<_NativeSpanList>);
typedef _StringFreeNative = ffi.Void Function(ffi.Pointer<Utf8>);
typedef _StringFreeDart = void Function(ffi.Pointer<Utf8>);
typedef _LastErrorNative = ffi.Pointer<Utf8> Function();
typedef _LastErrorDart = ffi.Pointer<Utf8> Function();

class _OpenMedBindings {
  _OpenMedBindings(ffi.DynamicLibrary library)
      : capabilities = library.lookupFunction<_Uint32Native, _Uint32Dart>(
          'openmed_ffi_capabilities',
        ),
        runtimeCreate =
            library.lookupFunction<_RuntimeCreateNative, _RuntimeCreateDart>(
          'openmed_runtime_create',
        ),
        runtimeDestroy =
            library.lookupFunction<_RuntimeDestroyNative, _RuntimeDestroyDart>(
          'openmed_runtime_destroy',
        ),
        sessionKind =
            library.lookupFunction<_SessionKindNative, _SessionKindDart>(
          'openmed_runtime_session_kind',
        ),
        liveCount = library.lookupFunction<_Uint64Native, _Uint64Dart>(
          'openmed_runtime_live_count',
        ),
        ownerCreate = library.lookupFunction<_Uint64Native, _Uint64Dart>(
          'openmed_runtime_owner_create',
        ),
        ownerRelease =
            library.lookupFunction<_OwnerReleaseNative, _OwnerReleaseDart>(
          'openmed_runtime_owner_release',
        ),
        runtimeExtractPii = library.lookupFunction<_RuntimeExtractPiiNative,
            _RuntimeExtractPiiDart>('openmed_runtime_extract_pii'),
        runtimeDeidentify = library.lookupFunction<_RuntimeDeidentifyNative,
            _RuntimeDeidentifyDart>('openmed_runtime_deidentify'),
        spanListFree =
            library.lookupFunction<_SpanListFreeNative, _SpanListFreeDart>(
          'openmed_span_list_free',
        ),
        stringFree = library.lookupFunction<_StringFreeNative, _StringFreeDart>(
          'openmed_string_free',
        ),
        lastError = library.lookupFunction<_LastErrorNative, _LastErrorDart>(
          'openmed_last_error',
        );

  factory _OpenMedBindings.open(String? explicitPath) {
    final ffi.DynamicLibrary library = _library(explicitPath);
    final int abi = library.lookupFunction<_Uint32Native, _Uint32Dart>(
        'openmed_ffi_abi_version')();
    if (abi != openMedKitFfiAbiVersion) {
      throw const OpenMedFfiException(
        3,
        'the native library ABI does not match these bindings',
      );
    }
    return _OpenMedBindings(library);
  }

  static ffi.DynamicLibrary _library(String? explicitPath) {
    if (explicitPath != null && explicitPath.isNotEmpty) {
      return ffi.DynamicLibrary.open(explicitPath);
    }
    final String? environmentPath = Platform.environment['OPENMED_FFI_LIBRARY'];
    if (environmentPath != null && environmentPath.isNotEmpty) {
      return ffi.DynamicLibrary.open(environmentPath);
    }
    if (Platform.isIOS || Platform.isMacOS) {
      return ffi.DynamicLibrary.process();
    }
    if (Platform.isWindows) {
      return ffi.DynamicLibrary.open('openmed_ffi.dll');
    }
    if (Platform.isAndroid || Platform.isLinux) {
      return ffi.DynamicLibrary.open('libopenmed_ffi.so');
    }
    throw OpenMedUnsupportedPlatformException(Platform.operatingSystem);
  }

  final _Uint32Dart capabilities;
  final _RuntimeCreateDart runtimeCreate;
  final _RuntimeDestroyDart runtimeDestroy;
  final _SessionKindDart sessionKind;
  final _Uint64Dart liveCount;
  final _Uint64Dart ownerCreate;
  final _OwnerReleaseDart ownerRelease;
  final _RuntimeExtractPiiDart runtimeExtractPii;
  final _RuntimeDeidentifyDart runtimeDeidentify;
  final _SpanListFreeDart spanListFree;
  final _StringFreeDart stringFree;
  final _LastErrorDart lastError;

  void checkStatus(int status) {
    if (status == 0) {
      return;
    }
    final ffi.Pointer<Utf8> message = lastError();
    throw OpenMedFfiException(
      status,
      message == ffi.nullptr
          ? 'native OpenMed operation failed'
          : message.toDartString(),
    );
  }
}
