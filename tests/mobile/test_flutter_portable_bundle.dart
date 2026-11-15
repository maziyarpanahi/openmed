// The Dart portable-bundle verifier against the shared golden vectors in
// tests/fixtures/portable_bundle/vectors.json and the decision calibration
// vectors in tests/fixtures/portable_bundle_calibration/vectors.json (Python
// reference results).

import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:openmedkit_flutter/openmedkit.dart';
import 'package:openmedkit_flutter/src/portable_bundle.dart'
    show
        decisionCalibrationExtension,
        decisionCalibrationExtensionVersion,
        decisionCalibrationFailure;

import 'flutter_ffi_support.dart';

Map<String, dynamic> _vectors() => readRepoJson(
      <String>['tests', 'fixtures', 'portable_bundle', 'vectors.json'],
    );

Map<String, dynamic> _calibrationVectors() => readRepoJson(
      <String>[
        'tests',
        'fixtures',
        'portable_bundle_calibration',
        'vectors.json',
      ],
    );

/// Read-level calibration refusals that precede the extension content checks.
const List<String> _calibrationReadReasons = <String>[
  'calibration_bundle_unverified',
  'calibration_extension_missing',
];

String _manifestText(Map<String, dynamic> vectors, String key) {
  final Map<String, dynamic> entry = (vectors['manifests']
      as Map<String, dynamic>)[key] as Map<String, dynamic>;
  final Object? text = entry['text'];
  if (text is String) {
    return text;
  }
  final String base = _manifestText(vectors, entry['base'] as String);
  final int at = entry['at'] as int;
  final int delete = entry['delete'] as int;
  return base.substring(0, at) +
      (entry['insert'] as String) +
      base.substring(at + delete);
}

PortableBundleTarget _target(Map<String, dynamic> target) =>
    PortableBundleTarget(
      platform: target['platform'] as String,
      sdkName: target['sdk_name'] as String,
      sdkVersion: target['sdk_version'] as String,
      runtimes: (target['runtimes'] as List<dynamic>).cast<String>(),
      osVersion: target['os_version'] as String?,
      policyVersion: target['policy_version'] as int?,
      memoryBytes: target['memory_bytes'] as int?,
      storageBytes: target['storage_bytes'] as int?,
    );

PortableBundleContext _context(Map<String, dynamic> raw) {
  final Map<String, dynamic>? target = raw['target'] as Map<String, dynamic>?;
  return PortableBundleContext(
    now: raw['now'] as String,
    target: target == null ? null : _target(target),
    sequenceFloors:
        (raw['sequence_floors'] as Map<String, dynamic>).cast<String, int>(),
    understoodExtensions:
        (raw['understood_extensions'] as List<dynamic>).cast<String>(),
    maxManifestBytes: raw['max_manifest_bytes'] as int,
    maxEnvelopeBytes: raw['max_envelope_bytes'] as int,
    maxTotalBytes: raw['max_total_bytes'] as int,
    maxFiles: raw['max_files'] as int,
  );
}

Map<String, List<int>>? _tree(
  Map<String, dynamic> vectors,
  Map<String, dynamic> item,
) {
  final Object? name = item['tree'];
  if (name == null) {
    return null;
  }
  final Map<String, List<int>> files = <String, List<int>>{};
  for (final Object? entry
      in (vectors['trees'] as Map<String, dynamic>)[name] as List<dynamic>) {
    final Map<String, dynamic> file = entry as Map<String, dynamic>;
    files[file['path'] as String] = utf8.encode(file['text'] as String);
  }
  for (final Object? path in item['tree_remove'] as List<dynamic>) {
    files.remove(path);
  }
  for (final Object? entry in item['tree_set'] as List<dynamic>) {
    final Map<String, dynamic> file = entry as Map<String, dynamic>;
    files[file['path'] as String] = utf8.encode(file['text'] as String);
  }
  return files;
}

PortableBundleVerification _run(
  Map<String, dynamic> vectors,
  Map<String, dynamic> item,
) {
  return PortableBundle.verify(
    manifest: utf8.encode(_manifestText(vectors, item['manifest'] as String)),
    envelope: utf8.encode(
      (vectors['envelopes'] as Map<String, dynamic>)[item['envelope']]
          as String,
    ),
    trustPolicy: PortableBundleTrustPolicy.fromJson(
      (vectors['trust_policies'] as Map<String, dynamic>)[item['trust_policy']],
    ),
    context: _context(
      (vectors['contexts'] as Map<String, dynamic>)[item['context']]
          as Map<String, dynamic>,
    ),
    files: _tree(vectors, item),
  );
}

List<int> _hex(String text) => <int>[
      for (int index = 0; index < text.length; index += 2)
        int.parse(text.substring(index, index + 2), radix: 16),
    ];

void main() {
  final Map<String, dynamic> vectors = _vectors();

  test('vectors are synthetic and share the reason vocabulary', () {
    expect(vectors['synthetic_only'], isTrue);
    expect(vectors['test_only_keys'], isTrue);
    expect(vectors['payload_type'], portableBundlePayloadType);
    expect(vectors['reason_codes'], portableBundleReasons);
  });

  test('every golden case matches the Python reference result', () {
    final List<dynamic> cases = vectors['cases'] as List<dynamic>;
    expect(cases.length, greaterThan(50));
    for (final Object? entry in cases) {
      final Map<String, dynamic> item = entry as Map<String, dynamic>;
      expect(
        _run(vectors, item).toJson(),
        item['expected'],
        reason: item['id'] as String,
      );
    }
  });

  test('golden content identity and signing input match', () {
    final List<int> golden = utf8.encode(_manifestText(vectors, 'golden'));
    final Map<String, dynamic> expected =
        vectors['golden'] as Map<String, dynamic>;
    expect(PortableBundle.contentId(golden), expected['content_id']);
    expect(
      PortableBundle.contentId(PortableBundle.signingInput(golden)),
      'sha256:${expected['signing_input_sha256']}',
    );
  });

  test('canonicalization vectors match RFC 8785 output', () {
    for (final Object? entry in vectors['canonicalization'] as List<dynamic>) {
      final Map<String, dynamic> item = entry as Map<String, dynamic>;
      expect(
        PortableBundle.canonicalJson(jsonDecode(item['input'] as String)),
        item['canonical'],
        reason: item['id'] as String,
      );
    }
    for (final Object? entry
        in vectors['canonicalization_rejects'] as List<dynamic>) {
      final Map<String, dynamic> item = entry as Map<String, dynamic>;
      expect(
        () => PortableBundle.canonicalJson(jsonDecode(item['input'] as String)),
        throwsA(isA<PortableBundleException>().having(
          (PortableBundleException error) => error.reason,
          'reason',
          'manifest_not_canonical',
        )),
        reason: item['id'] as String,
      );
    }
  });

  test('unknown non-critical extensions round-trip byte for byte', () {
    final String text = _manifestText(vectors, 'golden');
    final Map<String, dynamic> value =
        PortableBundle.parseManifest(utf8.encode(text));
    expect(value['extensions'], <String, Object>{
      'org.openmed.test.note': <String, Object>{'kept': true, 'revision': 1},
    });
    expect(PortableBundle.canonicalJson(value), text);
    expect(
      () => PortableBundle.parseManifest(utf8.encode(
        _manifestText(vectors, 'valid_critical_extension_understood'),
      )),
      throwsA(isA<PortableBundleException>().having(
        (PortableBundleException error) => error.reason,
        'reason',
        'unknown_critical_extension',
      )),
    );
  });

  test('RFC 8032 vectors verify with the bundled Ed25519 verifier', () {
    for (final Object? entry in vectors['ed25519_rfc8032'] as List<dynamic>) {
      final Map<String, dynamic> item = entry as Map<String, dynamic>;
      final List<int> publicKey = _hex(item['public_hex'] as String);
      final List<int> signature = _hex(item['signature_hex'] as String);
      final List<int> message = _hex(item['message_hex'] as String);
      expect(
        PortableBundle.verifyEd25519(publicKey, signature, message),
        isTrue,
        reason: item['name'] as String,
      );
      final List<int> flipped = List<int>.of(signature)..[0] ^= 1;
      expect(
        PortableBundle.verifyEd25519(publicKey, flipped, message),
        isFalse,
        reason: item['name'] as String,
      );
    }
  });

  test('caller input errors carry only reason codes', () {
    expect(
      () => PortableBundleTrustPolicy.fromJson(<String, Object>{'schema': 'x'}),
      throwsA(isA<PortableBundleException>().having(
        (PortableBundleException error) => error.toString(),
        'message',
        'PortableBundleException(trust_policy_invalid)',
      )),
    );
    final List<int> golden = utf8.encode(_manifestText(vectors, 'golden'));
    expect(
      () => PortableBundle.verify(
        manifest: golden,
        envelope: utf8.encode(
          (vectors['envelopes'] as Map<String, dynamic>)['golden'] as String,
        ),
        trustPolicy: PortableBundleTrustPolicy.fromJson(
          (vectors['trust_policies'] as Map<String, dynamic>)['default'],
        ),
        context: const PortableBundleContext(now: 'yesterday'),
      ),
      throwsA(isA<PortableBundleException>().having(
        (PortableBundleException error) => error.reason,
        'reason',
        'context_invalid',
      )),
    );
  });

  test('directory verification matches the in-memory result', () async {
    final Directory root = await materializeBundle();
    try {
      final PortableBundleVerification result = PortableBundle.verifyDirectory(
        root.path,
        trustPolicy: fixtureTrustPolicy(),
        context: fixtureContext(),
      );
      expect(result.ok, isTrue, reason: result.reason);
      expect(result.filesVerified, isTrue);
      expect(result.contentId, bundleFixture()['content_id']);
      expect(result.filesChecked, 6);

      await File('${root.path}${Platform.pathSeparator}.DS_Store')
          .writeAsString('x');
      expect(
        PortableBundle.verifyDirectory(
          root.path,
          trustPolicy: fixtureTrustPolicy(),
          context: fixtureContext(),
        ).reason,
        'file_undeclared',
      );
    } finally {
      await root.delete(recursive: true);
    }
  });

  test('directory verification refuses symlinks inside the bundle', () async {
    if (Platform.isWindows) {
      return;
    }
    final Directory root = await materializeBundle();
    try {
      await Link('${root.path}${Platform.pathSeparator}linked.json')
          .create(bundleFile(root, 'config.json').path);
      expect(
        PortableBundle.verifyDirectory(
          root.path,
          trustPolicy: fixtureTrustPolicy(),
          context: fixtureContext(),
        ).reason,
        'path_unsafe',
      );
    } finally {
      await root.delete(recursive: true);
    }
  });

  test('decision calibration verification cases match the Python reference',
      () {
    final Map<String, dynamic> calibration = _calibrationVectors();
    expect(calibration['synthetic_only'], isTrue);
    expect(calibration['test_only_keys'], isTrue);
    expect(calibration['extension'], decisionCalibrationExtension);
    expect(
      calibration['extension_version'],
      decisionCalibrationExtensionVersion,
    );
    final PortableBundleTrustPolicy trustPolicy =
        PortableBundleTrustPolicy.fromJson(calibration['trust_policy']);
    final List<dynamic> cases =
        calibration['verification_cases'] as List<dynamic>;
    expect(cases.length, greaterThan(10));
    for (final Object? entry in cases) {
      final Map<String, dynamic> item = entry as Map<String, dynamic>;
      final Map<String, List<int>> files = <String, List<int>>{};
      for (final Object? file in <Object?>[
        ...calibration['tree'] as List<dynamic>,
        ...item['tree_set'] as List<dynamic>,
      ]) {
        final Map<String, dynamic> value = file as Map<String, dynamic>;
        files[value['path'] as String] = utf8.encode(value['text'] as String);
      }
      final PortableBundleVerification result = PortableBundle.verify(
        manifest: utf8.encode(
          _manifestText(calibration, item['manifest'] as String),
        ),
        envelope: utf8.encode(item['envelope'] as String),
        trustPolicy: trustPolicy,
        context: PortableBundleContext(
          now: calibration['now'] as String,
          target: _target(calibration['target'] as Map<String, dynamic>),
          understoodExtensions:
              (item['understood_extensions'] as List<dynamic>).cast<String>(),
        ),
        files: files,
      );
      expect(result.toJson(), item['expected'], reason: item['id'] as String);
    }
  });

  test('decision calibration content checks match the Python read reasons',
      () {
    final Map<String, dynamic> calibration = _calibrationVectors();
    int checked = 0;
    for (final Object? entry in calibration['read_cases'] as List<dynamic>) {
      final Map<String, dynamic> item = entry as Map<String, dynamic>;
      final Map<String, dynamic> expected =
          item['expected'] as Map<String, dynamic>;
      final Object? reason = expected['reason'];
      if (_calibrationReadReasons.contains(reason)) {
        continue;
      }
      final Map<String, dynamic> structural = PortableBundle.parseManifest(
        utf8.encode(_manifestText(calibration, item['manifest'] as String)),
        understoodExtensions: const <String>[decisionCalibrationExtension],
      );
      expect(
        decisionCalibrationFailure(structural),
        reason,
        reason: item['id'] as String,
      );
      expect(expected['ok'], reason == null, reason: item['id'] as String);
      checked += 1;
    }
    expect(checked, greaterThan(40));
  });
}
