library openmedkit_flutter;

import 'dart:async';
import 'dart:convert';
import 'dart:ffi' as ffi;
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:flutter/services.dart';

/// Name shared by the Dart API and every native platform registrar.
const String openMedKitFlutterChannelName =
    'org.openmed.openmedkit_flutter/platform';

/// Current native C ABI expected by these Dart bindings.
const int openMedKitFfiAbiVersion = 1;

/// One half-open token offset into the source text, measured in Unicode scalars.
class OpenMedTokenOffset {
  const OpenMedTokenOffset(this.start, this.end)
      : assert(start >= 0),
        assert(end >= start);

  final int start;
  final int end;
}

/// Tokenizer output consumed by the ONNX token-classification session.
class OpenMedTokenBatch {
  OpenMedTokenBatch({
    required List<int> inputIds,
    required List<OpenMedTokenOffset> offsets,
    List<int>? attentionMask,
  })  : inputIds = List<int>.unmodifiable(inputIds),
        attentionMask = List<int>.unmodifiable(
          attentionMask ?? List<int>.filled(inputIds.length, 1),
        ),
        offsets = List<OpenMedTokenOffset>.unmodifiable(offsets) {
    if (inputIds.isEmpty ||
        inputIds.length != this.attentionMask.length ||
        inputIds.length != offsets.length) {
      throw ArgumentError(
        'inputIds, attentionMask, and offsets must have the same non-zero length',
      );
    }
  }

  final List<int> inputIds;
  final List<int> attentionMask;
  final List<OpenMedTokenOffset> offsets;

  int get tokenCount => inputIds.length;
}

/// Offline tokenizer adapter supplied by the host application.
///
/// Android applications can reuse the same tokenizer directory consumed by
/// OpenMedKit Android. Desktop applications can bind an equivalent local fast
/// tokenizer. Implementations must not download assets while encoding.
typedef OpenMedTokenizer = FutureOr<OpenMedTokenBatch> Function(String text);

/// A privacy-safe typed span returned through the C ABI.
///
/// Span records intentionally omit the matched source substring. [start] and
/// [end] are half-open Unicode-scalar offsets into the caller's source text.
class OpenMedSpan {
  const OpenMedSpan({
    required this.start,
    required this.end,
    required this.label,
    required this.score,
  });

  final int start;
  final int end;
  final String label;
  final double score;

  Map<String, Object> toJson() => <String, Object>{
        'start': start,
        'end': end,
        'label': label,
        'score': score,
      };
}

/// Result of local de-identification.
class OpenMedDeidentificationResult {
  const OpenMedDeidentificationResult({
    required this.deidentifiedText,
    required this.spans,
  });

  final String deidentifiedText;
  final List<OpenMedSpan> spans;

  Map<String, Object> toJson() => <String, Object>{
        'deidentified_text': deidentifiedText,
        'spans': spans.map((OpenMedSpan span) => span.toJson()).toList(),
      };
}

/// Failure reported by the native FFI layer.
class OpenMedFfiException implements Exception {
  const OpenMedFfiException(this.status, this.message);

  final int status;
  final String message;

  @override
  String toString() =>
      'OpenMedFfiException(status: $status, message: $message)';
}

/// Local OpenMed token-classification and de-identification runtime.
class OpenMedKit {
  OpenMedKit._(
    this._bindings,
    this._runtime,
    this._tokenizer,
    this.defaultThreshold,
  );

  static const List<String> _autoModelNames = <String>[
    'model_int8.onnx',
    'model.onnx',
    'model_fp16.onnx',
  ];

  final _OpenMedBindings _bindings;
  final ffi.Pointer<_NativeRuntime> _runtime;
  final OpenMedTokenizer _tokenizer;
  final double defaultThreshold;
  bool _closed = false;

  /// Loads a local OpenMed ONNX export without performing network requests.
  ///
  /// [modelDirectory] must contain `id2label.json` plus the selected ONNX
  /// graph. [tokenizer] must encode from already-local assets. A native ONNX
  /// session adapter registers the session vtable declared in `openmed_ffi.h`.
  static Future<OpenMedKit> loadModel({
    required String modelDirectory,
    required OpenMedTokenizer tokenizer,
    String variant = 'auto',
    double defaultThreshold = 0.5,
    String? nativeLibraryPath,
    MethodChannel? platformChannel,
  }) async {
    if (modelDirectory.trim().isEmpty || _looksRemote(modelDirectory)) {
      throw ArgumentError.value(
        modelDirectory,
        'modelDirectory',
        'must be a non-empty local filesystem path',
      );
    }
    _validateThreshold(defaultThreshold);
    final MethodChannel channel =
        platformChannel ?? const MethodChannel(openMedKitFlutterChannelName);
    final String? preparedDirectory = await channel.invokeMethod<String>(
      'prepareModel',
      <String, Object>{'modelDirectory': modelDirectory},
    );
    if (preparedDirectory == null || preparedDirectory.isEmpty) {
      throw const OpenMedFfiException(
        1,
        'the platform registrar did not return a model directory',
      );
    }

    final String modelPath = _resolveModelPath(preparedDirectory, variant);
    final List<String> labels = _readLabels(preparedDirectory);
    final _OpenMedBindings bindings = _OpenMedBindings.open(
      nativeLibraryPath,
    );
    if (bindings.abiVersion() != openMedKitFfiAbiVersion) {
      throw OpenMedFfiException(
        1,
        'native ABI ${bindings.abiVersion()} does not match '
        '$openMedKitFfiAbiVersion',
      );
    }
    final ffi.Pointer<ffi.Pointer<_NativeRuntime>> runtimeOut =
        calloc<ffi.Pointer<_NativeRuntime>>();
    final ffi.Pointer<Utf8> nativeModelPath = modelPath.toNativeUtf8();
    final ffi.Pointer<ffi.Pointer<Utf8>> nativeLabels =
        calloc<ffi.Pointer<Utf8>>(labels.length);
    try {
      for (int index = 0; index < labels.length; ++index) {
        nativeLabels[index] = labels[index].toNativeUtf8();
      }
      final int status = bindings.runtimeCreate(
        nativeModelPath,
        nativeLabels,
        labels.length,
        runtimeOut,
      );
      bindings.checkStatus(status);
      return OpenMedKit._(
        bindings,
        runtimeOut.value,
        tokenizer,
        defaultThreshold,
      );
    } finally {
      for (int index = 0; index < labels.length; ++index) {
        calloc.free(nativeLabels[index]);
      }
      calloc.free(nativeLabels);
      calloc.free(nativeModelPath);
      calloc.free(runtimeOut);
    }
  }

  /// Detects PII locally and returns typed spans without matched source text.
  Future<List<OpenMedSpan>> extractPii(
    String text, {
    double? threshold,
  }) async {
    _ensureOpen();
    _validateText(text);
    final double resolvedThreshold = threshold ?? defaultThreshold;
    _validateThreshold(resolvedThreshold);
    final OpenMedTokenBatch batch = await _tokenizer(text);
    _validateBatch(text, batch);
    return _withNativeBatch<List<OpenMedSpan>>(
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
  }

  /// Detects and masks PII locally using `[LABEL]` replacements.
  Future<OpenMedDeidentificationResult> deidentify(
    String text, {
    double? threshold,
  }) async {
    _ensureOpen();
    _validateText(text);
    final double resolvedThreshold = threshold ?? defaultThreshold;
    _validateThreshold(resolvedThreshold);
    final OpenMedTokenBatch batch = await _tokenizer(text);
    _validateBatch(text, batch);
    return _withNativeBatch<OpenMedDeidentificationResult>(
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
  }

  /// Releases the native token-classification session.
  void close() {
    if (_closed) {
      return;
    }
    _bindings.runtimeDestroy(_runtime);
    _closed = true;
  }

  T _withNativeBatch<T>(
    String text,
    OpenMedTokenBatch batch,
    T Function(
      ffi.Pointer<Utf8> text,
      ffi.Pointer<_NativeTokenBatch> batch,
    ) operation,
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
    if (text.isEmpty) {
      throw ArgumentError.value(text, 'text', 'must not be empty');
    }
  }

  static void _validateThreshold(double value) {
    if (value < 0.0 || value > 1.0) {
      throw ArgumentError.value(value, 'threshold', 'must be between 0 and 1');
    }
  }

  static void _validateBatch(String text, OpenMedTokenBatch batch) {
    final int textLength = text.runes.length;
    for (final OpenMedTokenOffset offset in batch.offsets) {
      if (offset.end > textLength) {
        throw ArgumentError(
          'token offsets must be Unicode-scalar offsets within the source text',
        );
      }
    }
  }

  static bool _looksRemote(String value) {
    final String normalized = value.trim().toLowerCase();
    return normalized.startsWith('http://') ||
        normalized.startsWith('https://');
  }

  static String _resolveModelPath(String directory, String variant) {
    final List<String> candidates;
    switch (variant.trim().toLowerCase()) {
      case 'auto':
        candidates = _autoModelNames;
        break;
      case 'int8':
        candidates = const <String>['model_int8.onnx'];
        break;
      case 'fp32':
        candidates = const <String>['model.onnx'];
        break;
      case 'fp16':
        candidates = const <String>['model_fp16.onnx'];
        break;
      default:
        candidates = <String>[variant];
    }
    for (final String candidate in candidates) {
      final File file = File(_join(directory, candidate));
      if (file.existsSync()) {
        return file.absolute.path;
      }
    }
    throw FileSystemException(
      'no local ONNX graph found for variant $variant',
      directory,
    );
  }

  static List<String> _readLabels(String directory) {
    final File file = File(_join(directory, 'id2label.json'));
    if (!file.existsSync()) {
      throw FileSystemException('id2label.json is required', file.path);
    }
    final Object? decoded = jsonDecode(file.readAsStringSync());
    if (decoded is! Map<String, dynamic> || decoded.isEmpty) {
      throw const FormatException('id2label.json must contain a non-empty map');
    }
    final Map<int, String> indexed = <int, String>{};
    decoded.forEach((String key, dynamic value) {
      final int? index = int.tryParse(key);
      if (index == null || index < 0 || value is! String || value.isEmpty) {
        throw const FormatException(
            'id2label entries must map integer keys to labels');
      }
      indexed[index] = value;
    });
    final int largest = indexed.keys.reduce(
      (int left, int right) => left > right ? left : right,
    );
    return List<String>.generate(
      largest + 1,
      (int index) => indexed[index] ?? 'O',
      growable: false,
    );
  }

  static String _join(String directory, String name) {
    final String separator = Platform.pathSeparator;
    return directory.endsWith(separator)
        ? '$directory$name'
        : '$directory$separator$name';
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

typedef _AbiVersionNative = ffi.Uint32 Function();
typedef _AbiVersionDart = int Function();
typedef _RuntimeCreateNative = ffi.Int32 Function(
  ffi.Pointer<Utf8>,
  ffi.Pointer<ffi.Pointer<Utf8>>,
  ffi.IntPtr,
  ffi.Pointer<ffi.Pointer<_NativeRuntime>>,
);
typedef _RuntimeCreateDart = int Function(
  ffi.Pointer<Utf8>,
  ffi.Pointer<ffi.Pointer<Utf8>>,
  int,
  ffi.Pointer<ffi.Pointer<_NativeRuntime>>,
);
typedef _RuntimeDestroyNative = ffi.Void Function(
  ffi.Pointer<_NativeRuntime>,
);
typedef _RuntimeDestroyDart = void Function(ffi.Pointer<_NativeRuntime>);
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
      : abiVersion = library.lookupFunction<_AbiVersionNative, _AbiVersionDart>(
          'openmed_ffi_abi_version',
        ),
        runtimeCreate =
            library.lookupFunction<_RuntimeCreateNative, _RuntimeCreateDart>(
          'openmed_runtime_create',
        ),
        runtimeDestroy =
            library.lookupFunction<_RuntimeDestroyNative, _RuntimeDestroyDart>(
          'openmed_runtime_destroy',
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
    if (explicitPath != null && explicitPath.isNotEmpty) {
      return _OpenMedBindings(ffi.DynamicLibrary.open(explicitPath));
    }
    final String? environmentPath = Platform.environment['OPENMED_FFI_LIBRARY'];
    if (environmentPath != null && environmentPath.isNotEmpty) {
      return _OpenMedBindings(ffi.DynamicLibrary.open(environmentPath));
    }
    if (Platform.isIOS || Platform.isMacOS) {
      return _OpenMedBindings(ffi.DynamicLibrary.process());
    }
    if (Platform.isWindows) {
      return _OpenMedBindings(ffi.DynamicLibrary.open('openmed_ffi.dll'));
    }
    if (Platform.isAndroid || Platform.isLinux) {
      return _OpenMedBindings(ffi.DynamicLibrary.open('libopenmed_ffi.so'));
    }
    throw UnsupportedError('OpenMedKit Flutter does not support this platform');
  }

  final _AbiVersionDart abiVersion;
  final _RuntimeCreateDart runtimeCreate;
  final _RuntimeDestroyDart runtimeDestroy;
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
