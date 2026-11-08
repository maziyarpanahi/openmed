// Offline verifier for signed OpenMed portable model bundles (native build).
//
// This reader runs the checks of openmed/core/portable_bundle.py in the same
// order and returns the same value-free result for the golden vectors in
// tests/fixtures/portable_bundle/vectors.json. It performs no network access,
// executes no bundled content, follows no symlinks, and reports only reason
// codes, digests, counts, and indexes.

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'crypto.dart';
import 'portable_bundle_types.dart';

const int _maxSafeInteger = 9007199254740991;
const int _maxSignatures = 16;
const int _maxDepth = 32;
const int _hashChunk = 1 << 20;

const Map<String, List<String>> _formatPrecisions = <String, List<String>>{
  'coreml': <String>['fp32', 'fp16', 'int8', 'int4'],
  'mlx': <String>['fp32', 'fp16', 'bf16', 'int8', 'int4', 'int2'],
  'onnx': <String>['fp32', 'fp16', 'int8', 'int4'],
  'ort': <String>['fp32', 'fp16', 'int8'],
  'transformersjs': <String>['fp32', 'fp16', 'int8', 'int4'],
  'pytorch': <String>['fp32', 'fp16', 'bf16'],
};
const List<String> _precisions = <String>[
  'fp32',
  'fp16',
  'bf16',
  'int8',
  'int4',
  'int2',
];
const List<String> _fileRoles = <String>[
  'calibration',
  'config',
  'decoder',
  'evidence',
  'graph',
  'labels',
  'license',
  'metadata',
  'policy',
  'provenance',
  'segmenter',
  'tokenizer',
  'weights',
];
const List<String> _offsetUnits = <String>[
  'unicode_scalar',
  'utf16_code_unit',
  'utf8_byte',
];
const List<String> _reservedPaths = <String>[
  portableBundleManifestFilename,
  portableBundleEnvelopeFilename,
];
const List<String> _topKeys = <String>[
  'schema',
  'schema_version',
  'bundle',
  'model',
  'tokenizer',
  'labels',
  'offsets',
  'decoder',
  'segmenter',
  'policy',
  'calibration',
  'compatibility',
  'runtimes',
  'licenses',
  'provenance',
  'evidence',
  'files',
  'extensions',
  'critical_extensions',
];

RegExp _whole(String pattern) => RegExp('^(?:$pattern)\$');

final RegExp _token = _whole(r'[a-z0-9][a-z0-9._-]{0,63}');
final RegExp _bundleId = _whole(r'[a-z0-9][a-z0-9._-]{0,127}');
final RegExp _modelId = _whole(r'[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}');
final RegExp _revision = _whole(r'[0-9a-f]{40}');
final RegExp _language = _whole(r'[a-z]{2,3}(?:-[A-Za-z0-9]{1,8}){0,4}');
final RegExp _license = _whole(r'[a-z0-9][a-z0-9.+-]{0,63}');
final RegExp _sha256 = _whole(r'sha256:[0-9a-f]{64}');
final RegExp _mediaType = _whole(r'[a-z]+/[a-z0-9][a-z0-9.+-]{0,126}');
final RegExp _extension =
    _whole(r'[a-z0-9][a-z0-9-]{0,31}(?:\.[a-z0-9][a-z0-9_-]{0,31}){1,7}');
final RegExp _semver =
    _whole(r'(?:0|[1-9][0-9]{0,8})(?:\.(?:0|[1-9][0-9]{0,8})){2}');
final RegExp _timestamp = _whole(
  r'[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])'
  r'T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z',
);
final RegExp _kid = _whole(r'[A-Za-z0-9][A-Za-z0-9._:-]{0,127}');
final RegExp _pathSegment = _whole(r'[A-Za-z0-9_][A-Za-z0-9._+-]{0,127}');

class _Refusal implements Exception {
  const _Refusal(this.reason);

  final String reason;
}

// --------------------------------------------------------------------------
// Canonical JSON (RFC 8785 subset)
// --------------------------------------------------------------------------

/// Returns the RFC 8785 canonical encoding of a JSON value.
///
/// Non-integer numbers, integers outside the IEEE 754 safe range, unpaired
/// surrogates, and non-string keys throw [PortableBundleException] with
/// `manifest_not_canonical`.
String portableBundleCanonicalJson(Object? value) {
  final StringBuffer output = StringBuffer();
  try {
    _encode(value, output, 0);
  } on _Refusal {
    throw const PortableBundleException('manifest_not_canonical');
  }
  return output.toString();
}

void _encode(Object? value, StringBuffer output, int depth) {
  if (depth > _maxDepth) {
    throw const _Refusal('manifest_not_canonical');
  }
  if (value == null) {
    output.write('null');
  } else if (value is bool) {
    output.write(value ? 'true' : 'false');
  } else if (value is int) {
    if (value > _maxSafeInteger || value < -_maxSafeInteger) {
      throw const _Refusal('manifest_not_canonical');
    }
    output.write(value.toString());
  } else if (value is String) {
    _encodeString(value, output);
  } else if (value is List) {
    output.write('[');
    for (int index = 0; index < value.length; ++index) {
      if (index > 0) {
        output.write(',');
      }
      _encode(value[index], output, depth + 1);
    }
    output.write(']');
  } else if (value is Map) {
    if (!value.keys.every((Object? key) => key is String)) {
      throw const _Refusal('manifest_not_canonical');
    }
    // String.compareTo orders by UTF-16 code units, as RFC 8785 requires.
    final List<String> keys = value.keys.cast<String>().toList()..sort();
    output.write('{');
    for (int index = 0; index < keys.length; ++index) {
      if (index > 0) {
        output.write(',');
      }
      _encode(keys[index], output, depth + 1);
      output.write(':');
      _encode(value[keys[index]], output, depth + 1);
    }
    output.write('}');
  } else {
    throw const _Refusal('manifest_not_canonical');
  }
}

/// Writes [value] with the escapes of ECMAScript `JSON.stringify`.
void _encodeString(String value, StringBuffer output) {
  const String hex = '0123456789abcdef';
  output.write('"');
  for (int index = 0; index < value.length; ++index) {
    final int unit = value.codeUnitAt(index);
    if (unit >= 0xd800 && unit <= 0xdbff) {
      if (index + 1 >= value.length) {
        throw const _Refusal('manifest_not_canonical');
      }
      final int next = value.codeUnitAt(index + 1);
      if (next < 0xdc00 || next > 0xdfff) {
        throw const _Refusal('manifest_not_canonical');
      }
      output
        ..writeCharCode(unit)
        ..writeCharCode(next);
      ++index;
      continue;
    }
    if (unit >= 0xdc00 && unit <= 0xdfff) {
      throw const _Refusal('manifest_not_canonical');
    }
    switch (unit) {
      case 0x22:
        output.write(r'\"');
        break;
      case 0x5c:
        output.write(r'\\');
        break;
      case 0x08:
        output.write(r'\b');
        break;
      case 0x0c:
        output.write(r'\f');
        break;
      case 0x0a:
        output.write(r'\n');
        break;
      case 0x0d:
        output.write(r'\r');
        break;
      case 0x09:
        output.write(r'\t');
        break;
      default:
        if (unit < 0x20) {
          output
            ..write(r'\u00')
            ..write(hex[unit >> 4])
            ..write(hex[unit & 0x0f]);
        } else {
          output.writeCharCode(unit);
        }
    }
  }
  output.write('"');
}

Object? _parseCanonical(Uint8List data) {
  Object? value;
  try {
    value = jsonDecode(utf8.decode(data));
  } on FormatException {
    throw const _Refusal('manifest_not_canonical');
  }
  final StringBuffer output = StringBuffer();
  _encode(value, output, 0);
  if (!_bytesEqual(utf8.encode(output.toString()), data)) {
    throw const _Refusal('manifest_not_canonical');
  }
  return value;
}

bool _bytesEqual(List<int> left, List<int> right) {
  if (left.length != right.length) {
    return false;
  }
  for (int index = 0; index < left.length; ++index) {
    if (left[index] != right[index]) {
      return false;
    }
  }
  return true;
}

/// Returns the `sha256:<hex>` content identity of canonical manifest bytes.
String portableBundleContentId(List<int> manifest) => sha256Identity(manifest);

/// Returns the DSSE v1 pre-authentication encoding that signatures cover.
Uint8List portableBundleSigningInput(List<int> manifest) {
  final List<int> header = ascii.encode(
    'DSSEv1 ${portableBundlePayloadType.length} $portableBundlePayloadType '
    '${manifest.length} ',
  );
  return Uint8List(header.length + manifest.length)
    ..setAll(0, header)
    ..setAll(header.length, manifest);
}

// --------------------------------------------------------------------------
// Manifest structure
// --------------------------------------------------------------------------

/// Validated manifest fields that later checks and loaders need.
class PortableBundleManifestView {
  PortableBundleManifestView._(
    this.value,
    this.bundleId,
    this.sequence,
    this.issuedAt,
    this.expiresAt,
    this.sdks,
    this.policyMin,
    this.policyMax,
    this.runtimes,
    this.files,
  );

  /// The parsed canonical manifest value.
  final Map<String, dynamic> value;

  /// `bundle.id`.
  final String bundleId;

  /// `bundle.sequence`.
  final int sequence;

  /// `bundle.issued_at`.
  final String issuedAt;

  /// `bundle.expires_at`.
  final String expiresAt;

  /// `compatibility.sdks` as `[name, min, maxExclusive]`.
  final List<List<String?>> sdks;

  /// `policy.min_version`.
  final int policyMin;

  /// `policy.max_version_exclusive`.
  final int policyMax;

  /// `runtimes` entries.
  final List<Map<String, dynamic>> runtimes;

  /// `files` entries in manifest order.
  final List<Map<String, dynamic>> files;

  /// Returns the `files` entry for [path], if declared.
  Map<String, dynamic>? file(String path) {
    for (final Map<String, dynamic> entry in files) {
      if (entry['path'] == path) {
        return entry;
      }
    }
    return null;
  }
}

Map<String, dynamic> _object(Object? value, List<String> keys) {
  if (value is! Map<String, dynamic>) {
    throw const _Refusal('manifest_malformed');
  }
  if (!value.keys.every(keys.contains)) {
    throw const _Refusal('unknown_critical_field');
  }
  if (value.length != keys.length) {
    throw const _Refusal('manifest_malformed');
  }
  return value;
}

String _string(Object? value, RegExp pattern) {
  if (value is! String || !pattern.hasMatch(value)) {
    throw const _Refusal('manifest_malformed');
  }
  return value;
}

String _choice(Object? value, Iterable<String> choices) {
  if (value is! String || !choices.contains(value)) {
    throw const _Refusal('manifest_malformed');
  }
  return value;
}

int _uint(Object? value, [int minimum = 0]) {
  if (value is! int || value > _maxSafeInteger || value < minimum) {
    throw const _Refusal('manifest_malformed');
  }
  return value;
}

List<dynamic> _array(Object? value, int minimum, int maximum) {
  if (value is! List || value.length < minimum || value.length > maximum) {
    throw const _Refusal('manifest_malformed');
  }
  return value;
}

void _unique(List<Object?> items) {
  if (items.toSet().length != items.length) {
    throw const _Refusal('manifest_malformed');
  }
}

String _path(Object? value) {
  if (value is! String) {
    throw const _Refusal('manifest_malformed');
  }
  final List<String> segments = value.split('/');
  if (value.isEmpty ||
      value.length > 512 ||
      segments.length > 16 ||
      !segments.every(_pathSegment.hasMatch) ||
      _reservedPaths.contains(value.toLowerCase())) {
    throw const _Refusal('path_unsafe');
  }
  return value;
}

List<String> _pathList(Object? value, int minimum, int maximum) {
  final List<String> items =
      _array(value, minimum, maximum).map<String>(_path).toList();
  _unique(items);
  return items;
}

String _version(Object? value) {
  if (!isPortableBundleVersion(value)) {
    throw const _Refusal('manifest_malformed');
  }
  return value as String;
}

List<int> _versionKey(String value) {
  final List<int> parts = value.split('.').map<int>(int.parse).toList();
  while (parts.length < 4) {
    parts.add(0);
  }
  return parts;
}

int _compareVersions(String left, String right) {
  final List<int> a = _versionKey(left);
  final List<int> b = _versionKey(right);
  for (int index = 0; index < 4; ++index) {
    if (a[index] != b[index]) {
      return a[index] < b[index] ? -1 : 1;
    }
  }
  return 0;
}

PortableBundleManifestView _validateManifest(
  Object? value,
  Set<String> understood,
) {
  if (value is! Map<String, dynamic>) {
    throw const _Refusal('manifest_malformed');
  }
  if (value['schema'] != portableBundleSchema) {
    throw const _Refusal('schema_unsupported');
  }
  final Object? schemaVersion = value['schema_version'];
  if (schemaVersion is! int) {
    throw const _Refusal('manifest_malformed');
  }
  if (schemaVersion != 1) {
    throw const _Refusal('schema_version_unsupported');
  }
  final Map<String, dynamic> top = _object(value, _topKeys);
  final List<String> references = <String>[];

  final Map<String, dynamic> bundle = _object(
    top['bundle'],
    const <String>['id', 'version', 'sequence', 'issued_at', 'expires_at'],
  );
  final String bundleId = _string(bundle['id'], _bundleId);
  _string(bundle['version'], _semver);
  final int sequence = _uint(bundle['sequence']);
  final String issuedAt = _string(bundle['issued_at'], _timestamp);
  final String expiresAt = _string(bundle['expires_at'], _timestamp);
  if (issuedAt.compareTo(expiresAt) >= 0) {
    throw const _Refusal('manifest_malformed');
  }

  final Map<String, dynamic> model = _object(
    top['model'],
    const <String>[
      'id',
      'revision',
      'family',
      'architecture',
      'task',
      'languages',
    ],
  );
  _string(model['id'], _modelId);
  if (model['revision'] != null) {
    _string(model['revision'], _revision);
  }
  _string(model['family'], _token);
  _string(model['architecture'], _token);
  _string(model['task'], _token);
  _unique(_array(model['languages'], 1, 32)
      .map<String>((Object? item) => _string(item, _language))
      .toList());

  if (top['tokenizer'] != null) {
    final Map<String, dynamic> tokenizer = _object(
      top['tokenizer'],
      const <String>['kind', 'contract_version', 'files'],
    );
    _string(tokenizer['kind'], _token);
    _uint(tokenizer['contract_version'], 1);
    references.addAll(_pathList(tokenizer['files'], 1, 64));
  }
  if (top['labels'] != null) {
    final Map<String, dynamic> labels =
        _object(top['labels'], const <String>['path', 'count', 'scheme']);
    references.add(_path(labels['path']));
    _uint(labels['count'], 1);
    _string(labels['scheme'], _token);
  }
  final Map<String, dynamic> offsets =
      _object(top['offsets'], const <String>['unit', 'contract_version']);
  _choice(offsets['unit'], _offsetUnits);
  _uint(offsets['contract_version'], 1);
  final Map<String, dynamic> decoder =
      _object(top['decoder'], const <String>['kind', 'contract_version']);
  _string(decoder['kind'], _token);
  _uint(decoder['contract_version'], 1);
  if (top['segmenter'] != null) {
    final Map<String, dynamic> segmenter =
        _object(top['segmenter'], const <String>['id', 'files']);
    _string(segmenter['id'], _token);
    references.addAll(_pathList(segmenter['files'], 1, 64));
  }
  final Map<String, dynamic> policy = _object(
    top['policy'],
    const <String>['min_version', 'max_version_exclusive', 'path'],
  );
  final int policyMin = _uint(policy['min_version'], 1);
  final int policyMax = _uint(policy['max_version_exclusive'], 1);
  if (policyMin >= policyMax) {
    throw const _Refusal('manifest_malformed');
  }
  if (policy['path'] != null) {
    references.add(_path(policy['path']));
  }
  if (top['calibration'] != null) {
    final Map<String, dynamic> calibration =
        _object(top['calibration'], const <String>['profile_version', 'path']);
    _uint(calibration['profile_version'], 1);
    references.add(_path(calibration['path']));
  }

  final Map<String, dynamic> compatibility =
      _object(top['compatibility'], const <String>['sdks']);
  final List<List<String?>> sdks = <List<String?>>[];
  for (final Object? entry in _array(compatibility['sdks'], 1, 16)) {
    final Map<String, dynamic> sdk =
        _object(entry, const <String>['name', 'min', 'max_exclusive']);
    final String name = _string(sdk['name'], _token);
    final String minimum = _version(sdk['min']);
    final String? maximum =
        sdk['max_exclusive'] == null ? null : _version(sdk['max_exclusive']);
    if (maximum != null && _compareVersions(minimum, maximum) >= 0) {
      throw const _Refusal('manifest_malformed');
    }
    sdks.add(<String?>[name, minimum, maximum]);
  }
  _unique(sdks.map<String?>((List<String?> item) => item[0]).toList());

  final List<Map<String, dynamic>> runtimes = <Map<String, dynamic>>[];
  for (final Object? entry in _array(top['runtimes'], 1, 32)) {
    final Map<String, dynamic> runtime = _object(entry, const <String>[
      'id',
      'format',
      'precision',
      'entrypoint',
      'files',
      'platforms',
      'min_memory_bytes',
      'min_storage_bytes',
      'fallback',
    ]);
    _string(runtime['id'], _token);
    final String format =
        _choice(runtime['format'], portableBundleFormatPlatforms.keys);
    final String precision = _choice(runtime['precision'], _precisions);
    final String entrypoint = _path(runtime['entrypoint']);
    final List<String> files = _pathList(runtime['files'], 1, 1024);
    if (!files.contains(entrypoint)) {
      throw const _Refusal('manifest_malformed');
    }
    references.addAll(files);
    final List<String> platforms = <String>[];
    for (final Object? item in _array(runtime['platforms'], 1, 16)) {
      final Map<String, dynamic> platform = _object(
        item,
        const <String>['platform', 'min_os', 'max_os_exclusive'],
      );
      final String name =
          _choice(platform['platform'], portableBundlePlatforms);
      final String? minOs =
          platform['min_os'] == null ? null : _version(platform['min_os']);
      final String? maxOs = platform['max_os_exclusive'] == null
          ? null
          : _version(platform['max_os_exclusive']);
      if (minOs != null &&
          maxOs != null &&
          _compareVersions(minOs, maxOs) >= 0) {
        throw const _Refusal('manifest_malformed');
      }
      platforms.add(name);
    }
    _unique(platforms);
    _uint(runtime['min_memory_bytes']);
    _uint(runtime['min_storage_bytes']);
    if (runtime['fallback'] != null) {
      _string(runtime['fallback'], _token);
    }
    final List<String> supported =
        portableBundleFormatPlatforms[format] ?? const <String>[];
    if (!(_formatPrecisions[format] ?? const <String>[]).contains(precision) ||
        !platforms.every(supported.contains)) {
      throw const _Refusal('runtime_combination_unsupported');
    }
    runtimes.add(runtime);
  }

  for (final Object? entry in _array(top['licenses'], 1, 16)) {
    final Map<String, dynamic> license =
        _object(entry, const <String>['id', 'files']);
    _string(license['id'], _license);
    references.addAll(_pathList(license['files'], 0, 64));
  }
  final Map<String, dynamic> provenance = _object(
    top['provenance'],
    const <String>['source_digest', 'export_record'],
  );
  if (provenance['source_digest'] != null) {
    _string(provenance['source_digest'], _sha256);
  }
  if (provenance['export_record'] != null) {
    references.add(_path(provenance['export_record']));
  }
  for (final Object? entry in _array(top['evidence'], 0, 64)) {
    final Map<String, dynamic> evidence =
        _object(entry, const <String>['kind', 'path']);
    _string(evidence['kind'], _token);
    references.add(_path(evidence['path']));
  }

  final List<Map<String, dynamic>> files = <Map<String, dynamic>>[];
  for (final Object? entry in _array(top['files'], 1, 65536)) {
    final Map<String, dynamic> item = _object(entry, const <String>[
      'path',
      'media_type',
      'size_bytes',
      'sha256',
      'role',
      'executable',
    ]);
    _path(item['path']);
    _string(item['media_type'], _mediaType);
    _uint(item['size_bytes']);
    _string(item['sha256'], _sha256);
    _choice(item['role'], _fileRoles);
    final Object? executable = item['executable'];
    if (executable is! bool) {
      throw const _Refusal('manifest_malformed');
    }
    if (executable) {
      throw const _Refusal('executable_file_forbidden');
    }
    files.add(item);
  }

  final Object? extensions = top['extensions'];
  if (extensions is! Map<String, dynamic> || extensions.length > 64) {
    throw const _Refusal('manifest_malformed');
  }
  for (final String name in extensions.keys) {
    _string(name, _extension);
  }
  final List<String> critical = _array(top['critical_extensions'], 0, 64)
      .map<String>((Object? item) => _string(item, _extension))
      .toList();
  _unique(critical);
  if (!critical.every(extensions.containsKey)) {
    throw const _Refusal('manifest_malformed');
  }

  final List<String> paths = files
      .map<String>((Map<String, dynamic> item) => item['path'] as String)
      .toList();
  final List<String> folded =
      paths.map<String>((String item) => item.toLowerCase()).toList();
  final Set<String> foldedSet = folded.toSet();
  if (foldedSet.length != folded.length) {
    throw const _Refusal('duplicate_path');
  }
  for (final String item in folded) {
    final List<String> segments = item.split('/');
    for (int index = 1; index < segments.length; ++index) {
      if (foldedSet.contains(segments.sublist(0, index).join('/'))) {
        throw const _Refusal('path_conflict');
      }
    }
  }
  // Portable paths are ASCII, so UTF-16 and UTF-8 orders coincide.
  for (int index = 0; index + 1 < paths.length; ++index) {
    if (paths[index].compareTo(paths[index + 1]) >= 0) {
      throw const _Refusal('file_order_invalid');
    }
  }
  final Set<String> declared = paths.toSet();
  if (!references.every(declared.contains)) {
    throw const _Refusal('file_reference_undeclared');
  }

  _unique(runtimes
      .map<Object?>((Map<String, dynamic> item) => item['id'])
      .toList());
  final Map<String, Map<String, dynamic>> byId = <String, Map<String, dynamic>>{
    for (final Map<String, dynamic> item in runtimes)
      item['id'] as String: item,
  };
  for (final Map<String, dynamic> runtime in runtimes) {
    String? current = runtime['fallback'] as String?;
    if (current == null) {
      continue;
    }
    if (!byId.containsKey(current) || current == runtime['id']) {
      throw const _Refusal('fallback_invalid');
    }
    final Set<String> seen = <String>{runtime['id'] as String};
    while (current != null) {
      if (seen.contains(current)) {
        throw const _Refusal('fallback_invalid');
      }
      seen.add(current);
      final Map<String, dynamic>? next = byId[current];
      if (next == null) {
        throw const _Refusal('fallback_invalid');
      }
      current = next['fallback'] as String?;
    }
  }

  if (!critical.every(understood.contains)) {
    throw const _Refusal('unknown_critical_extension');
  }

  return PortableBundleManifestView._(
    value,
    bundleId,
    sequence,
    issuedAt,
    expiresAt,
    sdks,
    policyMin,
    policyMax,
    runtimes,
    files,
  );
}

/// Parses and validates canonical manifest bytes without checking signatures.
///
/// Throws [PortableBundleException] with the first failing reason code.
/// Unknown non-critical extensions are preserved, so
/// [portableBundleCanonicalJson] reproduces the input bytes exactly.
Map<String, dynamic> parsePortableBundleManifest(
  List<int> manifest, {
  List<String> understoodExtensions = const <String>[],
}) {
  try {
    return _validateManifest(
      _parseCanonical(Uint8List.fromList(manifest)),
      understoodExtensions.toSet(),
    ).value;
  } on _Refusal catch (refusal) {
    throw PortableBundleException(refusal.reason);
  }
}

// --------------------------------------------------------------------------
// Envelope and signatures
// --------------------------------------------------------------------------

class _Envelope {
  const _Envelope(this.contentId, this.signatures);

  final String contentId;
  final List<Map<String, dynamic>> signatures;
}

bool _hasExactKeys(Map<String, dynamic> value, List<String> keys) =>
    value.length == keys.length && keys.every(value.containsKey);

_Envelope _parseEnvelope(Uint8List data) {
  Object? value;
  try {
    value = _parseCanonical(data);
  } on _Refusal {
    throw const _Refusal('envelope_malformed');
  }
  if (value is! Map<String, dynamic> ||
      value['schema'] != portableBundleEnvelopeSchema) {
    throw const _Refusal('envelope_malformed');
  }
  final Object? schemaVersion = value['schema_version'];
  if (schemaVersion is! int) {
    throw const _Refusal('envelope_malformed');
  }
  if (schemaVersion != 1) {
    throw const _Refusal('envelope_version_unsupported');
  }
  if (!_hasExactKeys(value, const <String>[
    'schema',
    'schema_version',
    'content_id',
    'signatures',
  ])) {
    throw const _Refusal('envelope_malformed');
  }
  final Object? contentId = value['content_id'];
  final Object? signatures = value['signatures'];
  if (contentId is! String ||
      !_sha256.hasMatch(contentId) ||
      signatures is! List ||
      signatures.length > _maxSignatures) {
    throw const _Refusal('envelope_malformed');
  }
  for (final Object? entry in signatures) {
    if (entry is! Map<String, dynamic> ||
        !_hasExactKeys(entry, const <String>['kid', 'alg', 'sig']) ||
        entry['kid'] is! String ||
        !_kid.hasMatch(entry['kid'] as String) ||
        entry['alg'] is! String ||
        entry['sig'] is! String) {
      throw const _Refusal('envelope_malformed');
    }
  }
  return _Envelope(contentId, signatures.cast<Map<String, dynamic>>());
}

int _checkSignatures(
  List<Map<String, dynamic>> signatures,
  Uint8List manifest,
  PortableBundleTrustPolicy policy,
  String now,
) {
  if (signatures.isEmpty) {
    throw const _Refusal('signature_missing');
  }
  final Uint8List message = portableBundleSigningInput(manifest);
  final Set<String> valid = <String>{};
  String? firstFailure;
  for (final Map<String, dynamic> entry in signatures) {
    final String kid = entry['kid'] as String;
    if (valid.contains(kid)) {
      continue;
    }
    final String? reason = _signatureFailure(entry, kid, message, policy, now);
    if (reason == null) {
      valid.add(kid);
    } else {
      firstFailure ??= reason;
    }
  }
  if (valid.length >= policy.threshold) {
    return valid.length;
  }
  throw _Refusal(firstFailure ?? 'signature_threshold_not_met');
}

String? _signatureFailure(
  Map<String, dynamic> entry,
  String kid,
  Uint8List message,
  PortableBundleTrustPolicy policy,
  String now,
) {
  if (entry['alg'] != 'Ed25519') {
    return 'signature_algorithm_unsupported';
  }
  final PortableBundleTrustedKey? key = policy.key(kid);
  if (key == null) {
    return 'untrusted_key';
  }
  if (key.revoked) {
    return 'key_revoked';
  }
  if (now.compareTo(key.notBefore) < 0) {
    return 'key_not_yet_valid';
  }
  if (now.compareTo(key.notAfter) >= 0) {
    return 'key_expired';
  }
  final Uint8List? signature = base64UrlDecodeCanonical(entry['sig'], 64);
  if (signature == null ||
      !Ed25519Verifier.verify(key.publicKey, signature, message)) {
    return 'signature_invalid';
  }
  return null;
}

// --------------------------------------------------------------------------
// File trees
// --------------------------------------------------------------------------

/// Read-only view of the files a bundle verification checks.
abstract class PortableBundleTree {
  /// Every file path in the tree, relative and POSIX-style.
  List<String> paths();

  /// Size in bytes of [path], or `null` when absent.
  int? size(String path);

  /// `sha256:<hex>` of [path].
  String digest(String path);
}

/// In-memory tree keyed by relative path.
class PortableBundleMemoryTree implements PortableBundleTree {
  /// Creates a tree over [files].
  PortableBundleMemoryTree(Map<String, List<int>> files)
      : _files = Map<String, List<int>>.unmodifiable(files);

  final Map<String, List<int>> _files;

  @override
  List<String> paths() => _files.keys.toList()..sort();

  @override
  int? size(String path) => _files[path]?.length;

  @override
  String digest(String path) => sha256Identity(_files[path]!);
}

/// Directory tree that never follows symlinks.
class _DirectoryTree implements PortableBundleTree {
  _DirectoryTree(Directory root) {
    if (FileSystemEntity.isLinkSync(root.path) ||
        !FileSystemEntity.isDirectorySync(root.path)) {
      throw const _Refusal('file_missing');
    }
    final String prefix = root.path.endsWith(Platform.pathSeparator)
        ? root.path
        : '${root.path}${Platform.pathSeparator}';
    for (final FileSystemEntity entity
        in root.listSync(recursive: true, followLinks: false)) {
      final FileSystemEntityType type =
          FileSystemEntity.typeSync(entity.path, followLinks: false);
      if (type == FileSystemEntityType.link) {
        throw const _Refusal('path_unsafe');
      }
      if (type == FileSystemEntityType.directory) {
        continue;
      }
      if (type != FileSystemEntityType.file ||
          !entity.path.startsWith(prefix)) {
        throw const _Refusal('path_unsafe');
      }
      final String relative = entity.path
          .substring(prefix.length)
          .split(Platform.pathSeparator)
          .join('/');
      _paths[relative] = File(entity.path);
    }
  }

  final Map<String, File> _paths = <String, File>{};

  @override
  List<String> paths() => _paths.keys.toList()..sort();

  @override
  int? size(String path) => _paths[path]?.lengthSync();

  @override
  String digest(String path) {
    final RandomAccessFile handle = _paths[path]!.openSync();
    try {
      final Sha256 hasher = Sha256();
      final Uint8List chunk = Uint8List(_hashChunk);
      int count;
      while ((count = handle.readIntoSync(chunk)) > 0) {
        hasher.add(Uint8List.sublistView(chunk, 0, count));
      }
      return 'sha256:${hexEncode(hasher.close())}';
    } finally {
      handle.closeSync();
    }
  }
}

// --------------------------------------------------------------------------
// Verification
// --------------------------------------------------------------------------

/// Verification result plus the validated manifest, for loaders.
class PortableBundleVerificationDetail {
  const PortableBundleVerificationDetail._(this.verification, this.manifest);

  /// Value-free result.
  final PortableBundleVerification verification;

  /// Validated manifest, present when manifest validation passed.
  final PortableBundleManifestView? manifest;
}

/// Verifies a portable bundle offline.
///
/// Checks run in the normative order and the first failure is reported.
/// `filesVerified` is true only when [files] was supplied and matched the
/// manifest exactly. Throws [PortableBundleException] (`context_invalid`)
/// for a malformed [context].
PortableBundleVerification verifyPortableBundle({
  required List<int> manifest,
  required List<int> envelope,
  required PortableBundleTrustPolicy trustPolicy,
  required PortableBundleContext context,
  PortableBundleTree? files,
}) {
  return verifyPortableBundleDetailed(
    manifest: manifest,
    envelope: envelope,
    trustPolicy: trustPolicy,
    context: context,
    files: () => files,
  ).verification;
}

/// Verifies a bundle and also returns its validated manifest.
///
/// [files] is invoked only after the manifest and compatibility checks pass,
/// so a directory is not walked for a bundle that is already refused.
PortableBundleVerificationDetail verifyPortableBundleDetailed({
  required List<int> manifest,
  required List<int> envelope,
  required PortableBundleTrustPolicy trustPolicy,
  required PortableBundleContext context,
  PortableBundleTree? Function()? files,
}) {
  context.validate();
  final Uint8List manifestBytes = Uint8List.fromList(manifest);
  final Uint8List envelopeBytes = Uint8List.fromList(envelope);
  final String now = context.now;
  String? contentId;
  int validSignatures = 0;
  List<int> eligible = const <int>[];
  int filesChecked = 0;
  PortableBundleManifestView? view;
  bool treeSupplied = false;
  try {
    if (manifestBytes.length > context.maxManifestBytes) {
      throw const _Refusal('manifest_too_large');
    }
    contentId = sha256Identity(manifestBytes);
    if (envelopeBytes.length > context.maxEnvelopeBytes) {
      throw const _Refusal('envelope_too_large');
    }
    final _Envelope parsed = _parseEnvelope(envelopeBytes);
    if (parsed.contentId != contentId) {
      throw const _Refusal('content_id_mismatch');
    }
    if (trustPolicy.revokedContentIds.contains(contentId)) {
      throw const _Refusal('bundle_revoked');
    }
    validSignatures =
        _checkSignatures(parsed.signatures, manifestBytes, trustPolicy, now);
    final PortableBundleManifestView validated = _validateManifest(
      _parseCanonical(manifestBytes),
      context.understoodExtensions.toSet(),
    );
    int declaredBytes = 0;
    for (final Map<String, dynamic> item in validated.files) {
      declaredBytes += item['size_bytes'] as int;
    }
    if (validated.files.length > context.maxFiles ||
        declaredBytes > context.maxTotalBytes) {
      throw const _Refusal('size_limit_exceeded');
    }
    if (now.compareTo(validated.issuedAt) < 0) {
      throw const _Refusal('bundle_not_yet_valid');
    }
    if (now.compareTo(validated.expiresAt) >= 0) {
      throw const _Refusal('bundle_expired');
    }
    final int? floor = context.sequenceFloors[validated.bundleId];
    if (floor != null && validated.sequence < floor) {
      throw const _Refusal('rollback_rejected');
    }
    eligible = _eligible(validated, context.target);
    view = validated;
    final PortableBundleTree? tree = files == null ? null : files();
    if (tree != null) {
      treeSupplied = true;
      for (final Map<String, dynamic> item in validated.files) {
        final String path = item['path'] as String;
        final int? size = tree.size(path);
        if (size == null) {
          throw const _Refusal('file_missing');
        }
        if (size != item['size_bytes']) {
          throw const _Refusal('file_size_mismatch');
        }
        if (tree.digest(path) != item['sha256']) {
          throw const _Refusal('file_digest_mismatch');
        }
        filesChecked += 1;
      }
      final Set<String> declared = validated.files
          .map<String>((Map<String, dynamic> item) => item['path'] as String)
          .toSet();
      for (final String candidate in tree.paths()) {
        if (!declared.contains(candidate) &&
            !_reservedPaths.contains(candidate)) {
          throw const _Refusal('file_undeclared');
        }
      }
    }
  } on _Refusal catch (refusal) {
    return PortableBundleVerificationDetail._(
      PortableBundleVerification(
        ok: false,
        reason: refusal.reason,
        contentId: contentId,
        validSignatures: validSignatures,
        eligibleRuntimes: List<int>.unmodifiable(eligible),
        filesChecked: filesChecked,
      ),
      view,
    );
  } on FileSystemException {
    return PortableBundleVerificationDetail._(
      PortableBundleVerification(
        ok: false,
        reason: 'file_missing',
        contentId: contentId,
        validSignatures: validSignatures,
        eligibleRuntimes: List<int>.unmodifiable(eligible),
        filesChecked: filesChecked,
      ),
      view,
    );
  }
  return PortableBundleVerificationDetail._(
    PortableBundleVerification(
      ok: true,
      reason: null,
      contentId: contentId,
      validSignatures: validSignatures,
      eligibleRuntimes: List<int>.unmodifiable(eligible),
      filesChecked: filesChecked,
      filesVerified: treeSupplied,
    ),
    view,
  );
}

/// Verifies a bundle directory that holds its manifest and envelope.
///
/// The directory and every file in it are checked without following
/// symlinks. Failures are returned as value-free results.
PortableBundleVerificationDetail verifyPortableBundleDirectoryDetailed(
  String directory, {
  required PortableBundleTrustPolicy trustPolicy,
  required PortableBundleContext context,
}) {
  context.validate();
  final Directory root = Directory(directory);
  Uint8List manifest;
  Uint8List envelope;
  try {
    manifest = _readBounded(
      root,
      portableBundleManifestFilename,
      context.maxManifestBytes,
      'manifest_too_large',
    );
    envelope = _readBounded(
      root,
      portableBundleEnvelopeFilename,
      context.maxEnvelopeBytes,
      'envelope_too_large',
    );
  } on _Refusal catch (refusal) {
    return PortableBundleVerificationDetail._(
      PortableBundleVerification(
        ok: false,
        reason: refusal.reason,
        contentId: null,
      ),
      null,
    );
  }
  return verifyPortableBundleDetailed(
    manifest: manifest,
    envelope: envelope,
    trustPolicy: trustPolicy,
    context: context,
    files: () => _DirectoryTree(root),
  );
}

Uint8List _readBounded(
  Directory root,
  String name,
  int limit,
  String tooLarge,
) {
  final String path = '${root.path}${Platform.pathSeparator}$name';
  try {
    if (FileSystemEntity.isLinkSync(path) ||
        FileSystemEntity.typeSync(path, followLinks: false) !=
            FileSystemEntityType.file) {
      throw const _Refusal('file_missing');
    }
    final RandomAccessFile handle = File(path).openSync();
    try {
      final Uint8List data = handle.readSync(limit + 1);
      if (data.length > limit) {
        throw _Refusal(tooLarge);
      }
      return data;
    } finally {
      handle.closeSync();
    }
  } on FileSystemException {
    throw const _Refusal('file_missing');
  }
}

List<int> _eligible(
  PortableBundleManifestView manifest,
  PortableBundleTarget? target,
) {
  if (target == null) {
    return List<int>.generate(manifest.runtimes.length, (int index) => index);
  }
  List<String?>? sdk;
  for (final List<String?> item in manifest.sdks) {
    if (item[0] == target.sdkName) {
      sdk = item;
      break;
    }
  }
  if (sdk == null ||
      _compareVersions(target.sdkVersion, sdk[1]!) < 0 ||
      (sdk[2] != null && _compareVersions(target.sdkVersion, sdk[2]!) >= 0)) {
    throw const _Refusal('sdk_incompatible');
  }
  final int? policyVersion = target.policyVersion;
  if (policyVersion != null &&
      !(manifest.policyMin <= policyVersion &&
          policyVersion < manifest.policyMax)) {
    throw const _Refusal('policy_incompatible');
  }
  final List<int> indexes = <int>[];
  for (int index = 0; index < manifest.runtimes.length; ++index) {
    if (_runtimeFits(manifest.runtimes[index], target)) {
      indexes.add(index);
    }
  }
  if (indexes.isEmpty) {
    throw const _Refusal('platform_incompatible');
  }
  return indexes;
}

bool _runtimeFits(Map<String, dynamic> runtime, PortableBundleTarget target) {
  if (!target.runtimes.contains(runtime['format'])) {
    return false;
  }
  final int? memory = target.memoryBytes;
  if (memory != null && (runtime['min_memory_bytes'] as int) > memory) {
    return false;
  }
  final int? storage = target.storageBytes;
  if (storage != null && (runtime['min_storage_bytes'] as int) > storage) {
    return false;
  }
  for (final Object? entry in runtime['platforms'] as List<dynamic>) {
    final Map<String, dynamic> platform = entry as Map<String, dynamic>;
    if (platform['platform'] != target.platform) {
      continue;
    }
    final String? low = platform['min_os'] as String?;
    final String? high = platform['max_os_exclusive'] as String?;
    if (low == null && high == null) {
      return true;
    }
    final String? osVersion = target.osVersion;
    if (osVersion == null) {
      return false;
    }
    return (low == null || _compareVersions(osVersion, low) >= 0) &&
        (high == null || _compareVersions(osVersion, high) < 0);
  }
  return false;
}
