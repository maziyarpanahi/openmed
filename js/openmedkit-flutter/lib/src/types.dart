// Platform-neutral value types shared by the native and unsupported builds.

import 'dart:async';

/// Name shared by the Dart API and every native platform registrar.
const String openMedKitFlutterChannelName =
    'org.openmed.openmedkit_flutter/platform';

/// Current native C ABI expected by these Dart bindings.
const int openMedKitFfiAbiVersion = 2;

/// One half-open token offset into the source text, measured in Unicode scalars.
class OpenMedTokenOffset {
  /// Creates an offset.
  const OpenMedTokenOffset(this.start, this.end)
      : assert(start >= 0),
        assert(end >= start);

  /// Inclusive start offset.
  final int start;

  /// Exclusive end offset.
  final int end;
}

/// Tokenizer output consumed by the ONNX token-classification session.
class OpenMedTokenBatch {
  /// Creates a batch. All lists must have the same non-zero length.
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

  /// Token ids.
  final List<int> inputIds;

  /// Attention mask; zero marks padding.
  final List<int> attentionMask;

  /// Unicode-scalar offsets per token.
  final List<OpenMedTokenOffset> offsets;

  /// Number of tokens.
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
  /// Creates a span.
  const OpenMedSpan({
    required this.start,
    required this.end,
    required this.label,
    required this.score,
  });

  /// Inclusive Unicode-scalar start offset.
  final int start;

  /// Exclusive Unicode-scalar end offset.
  final int end;

  /// Entity label without its BIO prefix.
  final String label;

  /// Mean token probability.
  final double score;

  /// Returns the shared serialization of this span.
  Map<String, Object> toJson() => <String, Object>{
        'start': start,
        'end': end,
        'label': label,
        'score': score,
      };
}

/// Result of local de-identification.
class OpenMedDeidentificationResult {
  /// Creates a result.
  const OpenMedDeidentificationResult({
    required this.deidentifiedText,
    required this.spans,
  });

  /// Source text with every span replaced by `[LABEL]`.
  final String deidentifiedText;

  /// Spans that were replaced, in source order.
  final List<OpenMedSpan> spans;

  /// Returns the shared serialization of this result.
  Map<String, Object> toJson() => <String, Object>{
        'deidentified_text': deidentifiedText,
        'spans': spans.map((OpenMedSpan span) => span.toJson()).toList(),
      };
}

/// Session implementation behind a loaded runtime.
///
/// Only [onnxRuntime] results are evidence for a real runtime claim.
/// [fixture] sessions produce synthetic logits for tests.
enum OpenMedSessionKind {
  /// Synthetic test logits.
  fixture,

  /// ONNX Runtime linked into the native library.
  onnxRuntime,

  /// A native adapter registered through `openmed_register_session_api`.
  registeredAdapter,
}

/// Whether this platform can load OpenMed models through the FFI.
class OpenMedPlatformSupport {
  /// Creates a support record.
  const OpenMedPlatformSupport({
    required this.platform,
    required this.supported,
    required this.runtimes,
    this.reason,
  });

  /// Platform name, for example `android`, `linux`, or `web`.
  final String platform;

  /// Whether models can be loaded here.
  final bool supported;

  /// Portable-bundle runtime formats this platform can load.
  final List<String> runtimes;

  /// Value-free reason code when unsupported, such as `platform_unsupported`.
  final String? reason;
}

/// Cooperative cancellation for one inference request.
///
/// Cancelling never interrupts a native call that is already running; it
/// drops the result before it is returned and releases its native buffers.
class OpenMedCancellationToken {
  bool _cancelled = false;

  /// Whether [cancel] was called.
  bool get isCancelled => _cancelled;

  /// Requests cancellation.
  void cancel() {
    _cancelled = true;
  }
}
