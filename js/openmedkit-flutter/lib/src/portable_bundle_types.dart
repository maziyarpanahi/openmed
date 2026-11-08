// Platform-neutral types for the signed portable model-bundle contract.
//
// The contract is defined in docs/models/portable-bundle.md and implemented by
// the Python reference verifier in openmed/core/portable_bundle.py.

import 'dart:typed_data';

/// Manifest schema name.
const String portableBundleSchema = 'openmed.portable_bundle';

/// Signature envelope schema name.
const String portableBundleEnvelopeSchema =
    'openmed.portable_bundle.signatures';

/// Trust-policy document schema name.
const String portableBundleTrustPolicySchema =
    'openmed.portable_bundle.trust_policy';

/// File name of the canonical manifest inside a bundle directory.
const String portableBundleManifestFilename = 'openmed-bundle.json';

/// File name of the signature envelope inside a bundle directory.
const String portableBundleEnvelopeFilename = 'openmed-bundle.sig.json';

/// DSSE payload type that signatures cover.
const String portableBundlePayloadType =
    'application/vnd.openmed.portable-bundle.v1+json';

/// SDK name this package reports in a verification target.
const String openMedKitFlutterSdkName = 'openmedkit-flutter';

/// Portable-contract SDK version this package implements.
const String openMedKitFlutterSdkVersion = '4.0.0';

/// Stable, value-free reason codes in evaluation order.
const List<String> portableBundleReasons = <String>[
  'manifest_too_large',
  'envelope_too_large',
  'envelope_malformed',
  'envelope_version_unsupported',
  'content_id_mismatch',
  'bundle_revoked',
  'signature_missing',
  'signature_algorithm_unsupported',
  'untrusted_key',
  'key_revoked',
  'key_not_yet_valid',
  'key_expired',
  'signature_invalid',
  'signature_verifier_unavailable',
  'signature_threshold_not_met',
  'manifest_not_canonical',
  'schema_unsupported',
  'schema_version_unsupported',
  'unknown_critical_field',
  'manifest_malformed',
  'path_unsafe',
  'executable_file_forbidden',
  'runtime_combination_unsupported',
  'duplicate_path',
  'path_conflict',
  'file_order_invalid',
  'file_reference_undeclared',
  'fallback_invalid',
  'unknown_critical_extension',
  'size_limit_exceeded',
  'bundle_not_yet_valid',
  'bundle_expired',
  'rollback_rejected',
  'sdk_incompatible',
  'policy_incompatible',
  'platform_incompatible',
  'file_missing',
  'file_size_mismatch',
  'file_digest_mismatch',
  'file_undeclared',
  'trust_policy_invalid',
  'context_invalid',
  'signing_key_invalid',
];

/// Runtime formats and the platforms each one supports.
const Map<String, List<String>> portableBundleFormatPlatforms =
    <String, List<String>>{
  'coreml': <String>['ios', 'macos', 'watchos', 'visionos'],
  'mlx': <String>['ios', 'macos'],
  'onnx': <String>['android', 'ios', 'macos', 'linux', 'windows', 'web'],
  'ort': <String>['android', 'ios'],
  'transformersjs': <String>['web'],
  'pytorch': <String>['linux', 'macos', 'windows'],
};

/// Platforms a verification target may name.
const List<String> portableBundlePlatforms = <String>[
  'ios',
  'macos',
  'watchos',
  'visionos',
  'android',
  'web',
  'linux',
  'windows',
];

/// Raised for malformed caller inputs. Carries only a reason code.
class PortableBundleException implements Exception {
  /// Creates an exception for [reason].
  const PortableBundleException(this.reason);

  /// One of [portableBundleReasons].
  final String reason;

  @override
  String toString() => 'PortableBundleException($reason)';
}

final RegExp _kid = RegExp(r'^(?:[A-Za-z0-9][A-Za-z0-9._:-]{0,127})$');
final RegExp _sha256 = RegExp(r'^(?:sha256:[0-9a-f]{64})$');
final RegExp _timestamp = RegExp(
  r'^(?:[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])'
  r'T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z)$',
);
final RegExp _token = RegExp(r'^(?:[a-z0-9][a-z0-9._-]{0,63})$');
final RegExp _version =
    RegExp(r'^(?:(?:0|[1-9][0-9]{0,8})(?:\.(?:0|[1-9][0-9]{0,8})){0,3})$');
final RegExp _extension = RegExp(
  r'^(?:[a-z0-9][a-z0-9-]{0,31}(?:\.[a-z0-9][a-z0-9_-]{0,31}){1,7})$',
);
final RegExp _base64Url = RegExp(r'^(?:[A-Za-z0-9_-]+)$');

const int _maxSafeInteger = 9007199254740991;
const int _maxSignatures = 16;
const String _base64Alphabet =
    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_';

/// Returns whether [value] is a `YYYY-MM-DDTHH:MM:SSZ` timestamp.
bool isPortableBundleTimestamp(Object? value) =>
    value is String && _timestamp.hasMatch(value);

/// Returns whether [value] is a dotted numeric version with up to four parts.
bool isPortableBundleVersion(Object? value) =>
    value is String && _version.hasMatch(value);

/// Formats [instant] as a portable-bundle timestamp in UTC.
String portableBundleTimestamp(DateTime instant) {
  final DateTime utc = instant.toUtc();
  String two(int value) => value.toString().padLeft(2, '0');
  return '${utc.year.toString().padLeft(4, '0')}-${two(utc.month)}-'
      '${two(utc.day)}T${two(utc.hour)}:${two(utc.minute)}:'
      '${two(utc.second)}Z';
}

/// Encodes [bytes] as unpadded base64url.
String base64UrlEncodeUnpadded(List<int> bytes) {
  final StringBuffer output = StringBuffer();
  for (int index = 0; index < bytes.length; index += 3) {
    final int a = bytes[index];
    final int b = index + 1 < bytes.length ? bytes[index + 1] : 0;
    final int c = index + 2 < bytes.length ? bytes[index + 2] : 0;
    final int triple = (a << 16) | (b << 8) | c;
    final int remaining = bytes.length - index;
    output
      ..write(_base64Alphabet[(triple >> 18) & 63])
      ..write(_base64Alphabet[(triple >> 12) & 63]);
    if (remaining > 1) {
      output.write(_base64Alphabet[(triple >> 6) & 63]);
    }
    if (remaining > 2) {
      output.write(_base64Alphabet[triple & 63]);
    }
  }
  return output.toString();
}

/// Decodes canonical unpadded base64url that encodes exactly [length] bytes.
///
/// Returns `null` for any other input, including non-zero trailing bits.
Uint8List? base64UrlDecodeCanonical(Object? text, int length) {
  if (text is! String || !_base64Url.hasMatch(text) || text.length % 4 == 1) {
    return null;
  }
  final List<int> bytes = <int>[];
  int bits = 0;
  int accumulator = 0;
  for (int index = 0; index < text.length; ++index) {
    accumulator =
        ((accumulator << 6) | _base64Alphabet.indexOf(text[index])) & 0xffffff;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      bytes.add((accumulator >> bits) & 0xff);
    }
  }
  final Uint8List raw = Uint8List.fromList(bytes);
  if (raw.length != length || base64UrlEncodeUnpadded(raw) != text) {
    return null;
  }
  return raw;
}

bool _isSafeInt(Object? value, {int minimum = 0}) =>
    value is int && value >= minimum && value <= _maxSafeInteger;

bool _hasExactKeys(Map<dynamic, dynamic> value, List<String> keys) =>
    value.length == keys.length && keys.every(value.containsKey);

/// One caller-controlled Ed25519 trust anchor.
class PortableBundleTrustedKey {
  /// Creates a trusted key.
  const PortableBundleTrustedKey({
    required this.kid,
    required this.publicKey,
    required this.notBefore,
    required this.notAfter,
    required this.revoked,
  });

  /// Key identifier that signatures reference.
  final String kid;

  /// Raw 32-byte Ed25519 public key.
  final Uint8List publicKey;

  /// First instant the key is valid.
  final String notBefore;

  /// First instant the key is no longer valid.
  final String notAfter;

  /// Whether the key was revoked.
  final bool revoked;
}

/// Offline trust roots, rotation windows, and revocations.
///
/// Trust roots always come from the caller and are never bundled.
class PortableBundleTrustPolicy {
  PortableBundleTrustPolicy._(
      this.keys, this.threshold, this.revokedContentIds);

  /// Parses a `portable_bundle_trust_policy` JSON document.
  ///
  /// Throws [PortableBundleException] with `trust_policy_invalid` for any
  /// malformed input.
  factory PortableBundleTrustPolicy.fromJson(Object? value) {
    const PortableBundleException invalid =
        PortableBundleException('trust_policy_invalid');
    if (value is! Map ||
        !_hasExactKeys(value, const <String>[
          'schema',
          'schema_version',
          'threshold',
          'keys',
          'revoked_content_ids',
        ])) {
      throw invalid;
    }
    final Object? keys = value['keys'];
    final Object? threshold = value['threshold'];
    final Object? revoked = value['revoked_content_ids'];
    if (value['schema'] != portableBundleTrustPolicySchema ||
        value['schema_version'] != 1 ||
        value['schema_version'] is! int ||
        threshold is! int ||
        threshold < 1 ||
        threshold > _maxSignatures ||
        keys is! List ||
        keys.isEmpty ||
        keys.length > 64 ||
        revoked is! List ||
        !revoked.every(
          (Object? item) => item is String && _sha256.hasMatch(item),
        )) {
      throw invalid;
    }
    final List<PortableBundleTrustedKey> parsed = <PortableBundleTrustedKey>[];
    for (final Object? entry in keys) {
      if (entry is! Map ||
          !_hasExactKeys(entry, const <String>[
            'kid',
            'kty',
            'crv',
            'x',
            'not_before',
            'not_after',
            'status',
          ])) {
        throw invalid;
      }
      final Object? kid = entry['kid'];
      final Object? notBefore = entry['not_before'];
      final Object? notAfter = entry['not_after'];
      final Uint8List? publicKey = base64UrlDecodeCanonical(entry['x'], 32);
      if (kid is! String ||
          !_kid.hasMatch(kid) ||
          entry['kty'] != 'OKP' ||
          entry['crv'] != 'Ed25519' ||
          publicKey == null ||
          notBefore is! String ||
          !isPortableBundleTimestamp(notBefore) ||
          notAfter is! String ||
          !isPortableBundleTimestamp(notAfter) ||
          notBefore.compareTo(notAfter) >= 0 ||
          (entry['status'] != 'active' && entry['status'] != 'revoked')) {
        throw invalid;
      }
      parsed.add(PortableBundleTrustedKey(
        kid: kid,
        publicKey: publicKey,
        notBefore: notBefore,
        notAfter: notAfter,
        revoked: entry['status'] == 'revoked',
      ));
    }
    if (parsed.map((PortableBundleTrustedKey key) => key.kid).toSet().length !=
        parsed.length) {
      throw invalid;
    }
    return PortableBundleTrustPolicy._(
      List<PortableBundleTrustedKey>.unmodifiable(parsed),
      threshold,
      Set<String>.unmodifiable(revoked.cast<String>()),
    );
  }

  /// Trusted keys; overlapping windows support rotation.
  final List<PortableBundleTrustedKey> keys;

  /// Number of distinct valid trusted signatures required.
  final int threshold;

  /// Bundle content identities that must be refused.
  final Set<String> revokedContentIds;

  /// Returns the trusted key with identifier [kid], if any.
  PortableBundleTrustedKey? key(String kid) {
    for (final PortableBundleTrustedKey candidate in keys) {
      if (candidate.kid == kid) {
        return candidate;
      }
    }
    return null;
  }
}

/// Runtime environment a bundle must be compatible with.
class PortableBundleTarget {
  /// Creates a target.
  const PortableBundleTarget({
    required this.platform,
    required this.sdkName,
    required this.sdkVersion,
    required this.runtimes,
    this.osVersion,
    this.policyVersion,
    this.memoryBytes,
    this.storageBytes,
  });

  /// Target for this Flutter package on [platform].
  ///
  /// The FFI loads `onnx` graphs everywhere and `ort` graphs on Android and
  /// iOS. [osVersion] must be supplied when a runtime declares an OS range;
  /// otherwise that runtime is not eligible.
  factory PortableBundleTarget.flutter({
    required String platform,
    String? osVersion,
    int? policyVersion,
    int? memoryBytes,
    int? storageBytes,
  }) {
    return PortableBundleTarget(
      platform: platform,
      sdkName: openMedKitFlutterSdkName,
      sdkVersion: openMedKitFlutterSdkVersion,
      runtimes: platform == 'android' || platform == 'ios'
          ? const <String>['onnx', 'ort']
          : const <String>['onnx'],
      osVersion: osVersion,
      policyVersion: policyVersion,
      memoryBytes: memoryBytes,
      storageBytes: storageBytes,
    );
  }

  /// One of [portableBundlePlatforms].
  final String platform;

  /// Reading SDK name, for example [openMedKitFlutterSdkName].
  final String sdkName;

  /// Reading SDK version.
  final String sdkVersion;

  /// Runtime formats the reader can load.
  final List<String> runtimes;

  /// Dotted OS version, when known.
  final String? osVersion;

  /// Privacy-policy version the caller enforces, when any.
  final int? policyVersion;

  /// Available memory in bytes, when known.
  final int? memoryBytes;

  /// Available storage in bytes, when known.
  final int? storageBytes;
}

/// Verification instant, target, rollback floors, and limits.
class PortableBundleContext {
  /// Creates a context. [now] is a `YYYY-MM-DDTHH:MM:SSZ` timestamp.
  const PortableBundleContext({
    required this.now,
    this.target,
    this.sequenceFloors = const <String, int>{},
    this.understoodExtensions = const <String>[],
    this.maxManifestBytes = 1 << 20,
    this.maxEnvelopeBytes = 64 << 10,
    this.maxTotalBytes = 64 * 1024 * 1024 * 1024,
    this.maxFiles = 4096,
  });

  /// Verification instant.
  final String now;

  /// Runtime environment, or `null` to skip runtime selection.
  final PortableBundleTarget? target;

  /// Highest accepted sequence for each bundle id.
  final Map<String, int> sequenceFloors;

  /// Critical extensions this reader implements.
  final List<String> understoodExtensions;

  /// Largest accepted manifest in bytes.
  final int maxManifestBytes;

  /// Largest accepted envelope in bytes.
  final int maxEnvelopeBytes;

  /// Largest accepted declared tree size in bytes.
  final int maxTotalBytes;

  /// Largest accepted number of declared files.
  final int maxFiles;

  /// Throws [PortableBundleException] (`context_invalid`) when malformed.
  void validate() {
    const PortableBundleException invalid =
        PortableBundleException('context_invalid');
    if (!isPortableBundleTimestamp(now) ||
        !<int>[maxManifestBytes, maxEnvelopeBytes, maxTotalBytes, maxFiles]
            .every((int limit) => _isSafeInt(limit, minimum: 1)) ||
        !understoodExtensions.every(_extension.hasMatch) ||
        !sequenceFloors.values.every(_isSafeInt)) {
      throw invalid;
    }
    final PortableBundleTarget? resolved = target;
    if (resolved == null) {
      return;
    }
    bool optionalUint(int? value) => value == null || _isSafeInt(value);
    if (!portableBundlePlatforms.contains(resolved.platform) ||
        !_token.hasMatch(resolved.sdkName) ||
        !isPortableBundleVersion(resolved.sdkVersion) ||
        !resolved.runtimes.every(portableBundleFormatPlatforms.containsKey) ||
        !(resolved.osVersion == null ||
            isPortableBundleVersion(resolved.osVersion)) ||
        !optionalUint(resolved.policyVersion) ||
        !optionalUint(resolved.memoryBytes) ||
        !optionalUint(resolved.storageBytes)) {
      throw invalid;
    }
  }
}

/// Value-free verification result.
class PortableBundleVerification {
  /// Creates a result.
  const PortableBundleVerification({
    required this.ok,
    required this.reason,
    required this.contentId,
    this.validSignatures = 0,
    this.eligibleRuntimes = const <int>[],
    this.filesChecked = 0,
    this.filesVerified = false,
  });

  /// Whether every check passed.
  final bool ok;

  /// First failing reason code, or `null` when [ok].
  final String? reason;

  /// `sha256:<hex>` of the manifest bytes, once computed.
  final String? contentId;

  /// Number of distinct trusted keys with a valid signature.
  final int validSignatures;

  /// Indexes into the manifest `runtimes` that fit the target.
  final List<int> eligibleRuntimes;

  /// Number of declared files that matched.
  final int filesChecked;

  /// Whether a file tree was supplied and matched exactly.
  final bool filesVerified;

  /// Whether the bundle may be activated.
  bool get eligibleForActivation => ok && filesVerified;

  /// Returns the shared JSON shape of the result.
  Map<String, Object?> toJson() => <String, Object?>{
        'content_id': contentId,
        'eligible_runtimes': eligibleRuntimes,
        'files_checked': filesChecked,
        'files_verified': filesVerified,
        'ok': ok,
        'reason': reason,
        'valid_signatures': validSignatures,
      };
}
