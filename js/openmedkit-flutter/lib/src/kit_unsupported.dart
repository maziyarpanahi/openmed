// OpenMedKit for platforms without dart:io and dart:ffi, such as Flutter web.
//
// Every entry point reports an explicit, typed unsupported result instead of
// failing to compile or falling back to a remote service. Browser apps use the
// `openmed` npm package, which runs the same verifier with WebCrypto.

import 'dart:typed_data';

import 'errors.dart';
import 'portable_bundle_types.dart';
import 'types.dart';

const String _platform = 'web';

/// Portable-bundle verification is unavailable without dart:ffi.
class PortableBundle {
  PortableBundle._();

  /// Returns a refused result with `signature_verifier_unavailable`.
  static PortableBundleVerification verify({
    required List<int> manifest,
    required List<int> envelope,
    required PortableBundleTrustPolicy trustPolicy,
    required PortableBundleContext context,
    Map<String, List<int>>? files,
  }) {
    context.validate();
    return const PortableBundleVerification(
      ok: false,
      reason: 'signature_verifier_unavailable',
      contentId: null,
    );
  }

  /// Returns a refused result with `signature_verifier_unavailable`.
  static PortableBundleVerification verifyDirectory(
    String directory, {
    required PortableBundleTrustPolicy trustPolicy,
    required PortableBundleContext context,
  }) {
    context.validate();
    return const PortableBundleVerification(
      ok: false,
      reason: 'signature_verifier_unavailable',
      contentId: null,
    );
  }

  /// Throws [OpenMedUnsupportedPlatformException].
  static String canonicalJson(Object? value) =>
      throw const OpenMedUnsupportedPlatformException(_platform);

  /// Throws [OpenMedUnsupportedPlatformException].
  static Map<String, dynamic> parseManifest(
    List<int> manifest, {
    List<String> understoodExtensions = const <String>[],
  }) =>
      throw const OpenMedUnsupportedPlatformException(_platform);

  /// Throws [OpenMedUnsupportedPlatformException].
  static String contentId(List<int> manifest) =>
      throw const OpenMedUnsupportedPlatformException(_platform);

  /// Throws [OpenMedUnsupportedPlatformException].
  static Uint8List signingInput(List<int> manifest) =>
      throw const OpenMedUnsupportedPlatformException(_platform);

  /// Throws [OpenMedUnsupportedPlatformException].
  static bool verifyEd25519(
    List<int> publicKey,
    List<int> signature,
    List<int> message,
  ) =>
      throw const OpenMedUnsupportedPlatformException(_platform);
}

/// Verified bundles cannot be opened without dart:ffi.
class OpenMedVerifiedBundle {
  OpenMedVerifiedBundle._();

  /// Throws [OpenMedUnsupportedPlatformException].
  static Future<OpenMedVerifiedBundle> open({
    required String directory,
    required PortableBundleTrustPolicy trustPolicy,
    required PortableBundleContext context,
    String? runtimeId,
  }) async =>
      throw const OpenMedUnsupportedPlatformException(_platform);

  /// Not reachable: no instance can be created on this platform.
  PortableBundleVerification get verification =>
      throw const OpenMedUnsupportedPlatformException(_platform);

  /// Not reachable: no instance can be created on this platform.
  String get contentId =>
      throw const OpenMedUnsupportedPlatformException(_platform);

  /// Not reachable: no instance can be created on this platform.
  String get bundleId =>
      throw const OpenMedUnsupportedPlatformException(_platform);

  /// Not reachable: no instance can be created on this platform.
  int get sequence =>
      throw const OpenMedUnsupportedPlatformException(_platform);

  /// Not reachable: no instance can be created on this platform.
  String get runtimeId =>
      throw const OpenMedUnsupportedPlatformException(_platform);

  /// Not reachable: no instance can be created on this platform.
  String get runtimeFormat =>
      throw const OpenMedUnsupportedPlatformException(_platform);

  /// Not reachable: no instance can be created on this platform.
  String get runtimePrecision =>
      throw const OpenMedUnsupportedPlatformException(_platform);

  /// Not reachable: no instance can be created on this platform.
  List<String> get labels =>
      throw const OpenMedUnsupportedPlatformException(_platform);

  /// Not reachable: no instance can be created on this platform.
  Future<Uint8List> readVerifiedFile(String path) async =>
      throw const OpenMedUnsupportedPlatformException(_platform);
}

/// Native runtime ownership is unavailable without dart:ffi.
class OpenMedRuntimeOwner {
  /// Throws [OpenMedUnsupportedPlatformException].
  factory OpenMedRuntimeOwner.create({String? nativeLibraryPath}) =>
      throw const OpenMedUnsupportedPlatformException(_platform);

  /// Not reachable: no instance can be created on this platform.
  int get id => throw const OpenMedUnsupportedPlatformException(_platform);

  /// Not reachable: no instance can be created on this platform.
  int release() => throw const OpenMedUnsupportedPlatformException(_platform);
}

/// OpenMedKit is unsupported without dart:ffi.
class OpenMedKit {
  OpenMedKit._();

  /// Reports that this platform cannot load models.
  static OpenMedPlatformSupport get platformSupport =>
      const OpenMedPlatformSupport(
        platform: _platform,
        supported: false,
        runtimes: <String>[],
        reason: 'platform_unsupported',
      );

  /// Always zero: no native runtime exists on this platform.
  static int liveRuntimeCount({String? nativeLibraryPath}) => 0;

  /// Always empty: no native library exists on this platform.
  static Set<OpenMedSessionKind> nativeSessionKinds({
    String? nativeLibraryPath,
  }) =>
      const <OpenMedSessionKind>{};

  /// Throws [OpenMedUnsupportedPlatformException].
  static Future<OpenMedKit> loadBundle({
    required OpenMedVerifiedBundle bundle,
    required OpenMedTokenizer tokenizer,
    double defaultThreshold = 0.5,
    String? nativeLibraryPath,
    int ownerId = 0,
  }) async =>
      throw const OpenMedUnsupportedPlatformException(_platform);

  /// Not reachable: no instance can be created on this platform.
  double get defaultThreshold =>
      throw const OpenMedUnsupportedPlatformException(_platform);

  /// Not reachable: no instance can be created on this platform.
  OpenMedVerifiedBundle get bundle =>
      throw const OpenMedUnsupportedPlatformException(_platform);

  /// Not reachable: no instance can be created on this platform.
  int get generation =>
      throw const OpenMedUnsupportedPlatformException(_platform);

  /// Not reachable: no instance can be created on this platform.
  bool get isClosed => true;

  /// Not reachable: no instance can be created on this platform.
  OpenMedSessionKind get sessionKind =>
      throw const OpenMedUnsupportedPlatformException(_platform);

  /// Throws [OpenMedUnsupportedPlatformException].
  Future<List<OpenMedSpan>> extractPii(
    String text, {
    double? threshold,
    OpenMedCancellationToken? cancellation,
  }) async =>
      throw const OpenMedUnsupportedPlatformException(_platform);

  /// Throws [OpenMedUnsupportedPlatformException].
  Future<OpenMedDeidentificationResult> deidentify(
    String text, {
    double? threshold,
    OpenMedCancellationToken? cancellation,
  }) async =>
      throw const OpenMedUnsupportedPlatformException(_platform);

  /// Throws [OpenMedUnsupportedPlatformException].
  Future<void> replaceBundle(OpenMedVerifiedBundle bundle) async =>
      throw const OpenMedUnsupportedPlatformException(_platform);

  /// No-op: nothing native to release.
  void close() {}
}
