// Shared helpers for the OpenMedKit Flutter FFI tests.
//
// The signed bundle comes from tests/mobile/fixtures/flutter_ffi_bundle.json,
// which generate_flutter_ffi_bundle.py builds with a TEST-ONLY key. Every
// model file is a synthetic placeholder that the fixture session hashes but
// never interprets.

import 'dart:convert';
import 'dart:io';

import 'package:openmedkit_flutter/openmedkit.dart';

/// Synthetic note shared with tests/mobile/fixtures/flutter_ffi_golden.json.
const String syntheticNote =
    'Patient Alice Nguyen was born on 1979-04-12. Email alice@example.org.';

/// Fixture-session token id that selects label `index` of the bundle labels.
int labelToken(int index) => 1000 + index;

/// Returns a path under the repository root.
String repoPath(List<String> segments) {
  final String separator = Platform.pathSeparator;
  return <String>[Directory.current.path, '..', '..', ...segments]
      .join(separator);
}

/// Decodes a repository JSON file.
Map<String, dynamic> readRepoJson(List<String> segments) =>
    jsonDecode(File(repoPath(segments)).readAsStringSync())
        as Map<String, dynamic>;

/// The committed Flutter bundle fixture document.
Map<String, dynamic> bundleFixture() => readRepoJson(
    <String>['tests', 'mobile', 'fixtures', 'flutter_ffi_bundle.json']);

/// The native fixture library built with the synthetic session.
String fixtureLibraryPath() {
  final String? path = Platform.environment['OPENMED_FFI_LIBRARY'];
  if (path == null || path.isEmpty) {
    throw StateError('OPENMED_FFI_LIBRARY must point at the fixture library');
  }
  return path;
}

/// Host platform as a portable-bundle platform name.
String hostPlatform() => Platform.operatingSystem;

/// Trust policy that trusts only the TEST-ONLY fixture key.
PortableBundleTrustPolicy fixtureTrustPolicy() =>
    PortableBundleTrustPolicy.fromJson(bundleFixture()['trust_policy']);

/// Verification context for the host platform at the fixture instant.
PortableBundleContext fixtureContext({
  Map<String, int> sequenceFloors = const <String, int>{},
}) {
  return PortableBundleContext(
    now: bundleFixture()['now'] as String,
    target: PortableBundleTarget.flutter(platform: hostPlatform()),
    sequenceFloors: sequenceFloors,
  );
}

/// Writes the signed fixture bundle into a new temporary directory.
Future<Directory> materializeBundle() async {
  final Map<String, dynamic> fixture = bundleFixture();
  final Directory root =
      await Directory.systemTemp.createTemp('openmed-flutter-bundle-');
  final String separator = Platform.pathSeparator;
  Future<void> write(String relative, String text) async {
    final File file =
        File('${root.path}$separator${relative.split('/').join(separator)}');
    await file.parent.create(recursive: true);
    await file.writeAsBytes(utf8.encode(text));
  }

  for (final Object? entry in fixture['tree'] as List<dynamic>) {
    final Map<String, dynamic> item = entry as Map<String, dynamic>;
    await write(item['path'] as String, item['text'] as String);
  }
  await write('openmed-bundle.json', fixture['manifest'] as String);
  await write('openmed-bundle.sig.json', fixture['envelope'] as String);
  return root;
}

/// Returns a file inside a materialized bundle.
File bundleFile(Directory root, String relative) {
  final String separator = Platform.pathSeparator;
  return File('${root.path}$separator${relative.split('/').join(separator)}');
}

/// Opens the materialized fixture bundle through full verification.
Future<OpenMedVerifiedBundle> openFixtureBundle(
  Directory root, {
  String? runtimeId,
}) {
  return OpenMedVerifiedBundle.open(
    directory: root.path,
    trustPolicy: fixtureTrustPolicy(),
    context: fixtureContext(),
    runtimeId: runtimeId,
  );
}

/// Tokenizer for [syntheticNote]: three entity tokens and two specials.
OpenMedTokenBatch syntheticTokenizer(String text) {
  if (text != syntheticNote) {
    throw ArgumentError(
        'the fixture tokenizer only encodes the synthetic note');
  }
  return OpenMedTokenBatch(
    inputIds: <int>[101, labelToken(1), labelToken(2), labelToken(3), 102],
    attentionMask: const <int>[1, 1, 1, 1, 1],
    offsets: const <OpenMedTokenOffset>[
      OpenMedTokenOffset(0, 0),
      OpenMedTokenOffset(8, 20),
      OpenMedTokenOffset(33, 43),
      OpenMedTokenOffset(51, 68),
      OpenMedTokenOffset(0, 0),
    ],
  );
}
