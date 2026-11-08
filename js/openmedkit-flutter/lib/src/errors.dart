// Value-free exceptions. Messages come from a fixed vocabulary and never
// contain source text, file paths, digests, or key material.

import 'portable_bundle_types.dart';

/// Base class for OpenMedKit Flutter failures.
abstract class OpenMedException implements Exception {
  /// Stable, value-free reason code.
  String get code;
}

/// Failure reported by the native FFI layer.
class OpenMedFfiException implements OpenMedException {
  /// Creates an exception for a native [status].
  const OpenMedFfiException(this.status, this.message);

  /// Native `openmed_status` value.
  final int status;

  /// Fixed native message; never contains source text or paths.
  final String message;

  @override
  String get code {
    switch (status) {
      case 1:
        return 'invalid_argument';
      case 2:
        return 'out_of_memory';
      case 3:
        return 'runtime_unavailable';
      case 4:
        return 'inference_failed';
      case 5:
        return 'invalid_output';
      case 6:
        return 'integrity_failed';
      default:
        return 'native_failure';
    }
  }

  @override
  String toString() =>
      'OpenMedFfiException(status: $status, message: $message)';
}

/// A bundle could not be verified or does not fit this runtime.
class OpenMedBundleException implements OpenMedException {
  /// Creates an exception for [code].
  const OpenMedBundleException(this.code, {this.verification});

  /// A portable-bundle reason code, or one of `bundle_directory_invalid`,
  /// `context_target_required`, `runtime_unavailable`, `labels_missing`,
  /// `labels_invalid`, or `offset_unit_unsupported`.
  @override
  final String code;

  /// The value-free verification result, when verification ran.
  final PortableBundleVerification? verification;

  @override
  String toString() => 'OpenMedBundleException($code)';
}

/// The current platform cannot load OpenMed models through the FFI.
class OpenMedUnsupportedPlatformException implements OpenMedException {
  /// Creates an exception for [platform].
  const OpenMedUnsupportedPlatformException(this.platform);

  /// Platform name, for example `web`.
  final String platform;

  @override
  String get code => 'platform_unsupported';

  @override
  String toString() => 'OpenMedUnsupportedPlatformException($platform)';
}

/// An in-flight request was dropped before its result was returned.
class OpenMedCancelledException implements OpenMedException {
  /// Creates an exception for [code].
  const OpenMedCancelledException(this.code);

  /// `cancelled`, `closed`, or `model_replaced`.
  @override
  final String code;

  @override
  String toString() => 'OpenMedCancelledException($code)';
}
