// Opt-in real-artifact lane: a signed local bundle run through ONNX Runtime.
//
// This suite is skipped unless OPENMED_FLUTTER_REAL_BUNDLE_DIR is set. It
// refuses to run against the fixture session, so a mock-only run can never
// qualify a real runtime claim. Inputs are local files supplied by the
// operator; nothing is downloaded.
//
//   OPENMED_FFI_LIBRARY               libopenmed_ffi built with ONNX Runtime
//   OPENMED_FLUTTER_REAL_BUNDLE_DIR   signed portable bundle directory
//   OPENMED_FLUTTER_REAL_TRUST_POLICY trust-policy JSON for its signing key
//   OPENMED_FLUTTER_REAL_VECTORS      pre-tokenized synthetic cases (README)
//   OPENMED_FLUTTER_REAL_OS_VERSION   optional dotted OS version
//   OPENMED_FLUTTER_REAL_NOW          optional YYYY-MM-DDTHH:MM:SSZ instant
//   OPENMED_FLUTTER_REAL_EVIDENCE     optional path for value-free evidence

import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:openmedkit_flutter/openmedkit.dart';

String? _env(String name) {
  final String? value = Platform.environment[name];
  return value == null || value.isEmpty ? null : value;
}

String _required(String name) {
  final String? value = _env(name);
  if (value == null) {
    throw StateError('$name is required for the real-artifact lane');
  }
  return value;
}

Map<String, dynamic> _json(String path) =>
    jsonDecode(File(path).readAsStringSync()) as Map<String, dynamic>;

void main() {
  final String? bundleDirectory = _env('OPENMED_FLUTTER_REAL_BUNDLE_DIR');
  final Object skip = bundleDirectory == null
      ? 'set OPENMED_FLUTTER_REAL_BUNDLE_DIR to run the real-artifact lane'
      : false;

  test('a signed local bundle matches its vectors through ONNX Runtime',
      () async {
    expect(
      Platform.environment['OPENMED_FFI_TEST_SESSION'],
      isNot('1'),
      reason: 'a fixture session cannot qualify a real runtime claim',
    );
    final String library = _required('OPENMED_FFI_LIBRARY');
    final Set<OpenMedSessionKind> kinds =
        OpenMedKit.nativeSessionKinds(nativeLibraryPath: library);
    expect(kinds, contains(OpenMedSessionKind.onnxRuntime));
    expect(kinds, isNot(contains(OpenMedSessionKind.fixture)));

    final OpenMedPlatformSupport support = OpenMedKit.platformSupport;
    expect(support.supported, isTrue);
    final OpenMedVerifiedBundle bundle = await OpenMedVerifiedBundle.open(
      directory: bundleDirectory!,
      trustPolicy: PortableBundleTrustPolicy.fromJson(
        _json(_required('OPENMED_FLUTTER_REAL_TRUST_POLICY')),
      ),
      context: PortableBundleContext(
        now: _env('OPENMED_FLUTTER_REAL_NOW') ??
            portableBundleTimestamp(DateTime.now()),
        target: PortableBundleTarget.flutter(
          platform: support.platform,
          osVersion: _env('OPENMED_FLUTTER_REAL_OS_VERSION'),
        ),
      ),
    );

    final Map<String, dynamic> vectors =
        _json(_required('OPENMED_FLUTTER_REAL_VECTORS'));
    expect(vectors['schema'], 'openmed.flutter_ffi.real_vectors');
    expect(vectors['version'], 1);
    final double tolerance =
        (vectors['score_tolerance'] as num? ?? 0.001).toDouble();
    final List<Map<String, dynamic>> cases =
        (vectors['cases'] as List<dynamic>).cast<Map<String, dynamic>>();
    expect(cases, isNotEmpty);
    final Map<String, OpenMedTokenBatch> batches = <String, OpenMedTokenBatch>{
      for (final Map<String, dynamic> item in cases)
        item['text'] as String: OpenMedTokenBatch(
          inputIds: (item['input_ids'] as List<dynamic>).cast<int>(),
          attentionMask: (item['attention_mask'] as List<dynamic>).cast<int>(),
          offsets: <OpenMedTokenOffset>[
            for (final Object? pair in item['offsets'] as List<dynamic>)
              OpenMedTokenOffset(
                (pair as List<dynamic>)[0] as int,
                pair[1] as int,
              ),
          ],
        ),
    };

    final OpenMedKit kit = await OpenMedKit.loadBundle(
      bundle: bundle,
      tokenizer: (String text) => batches[text]!,
      nativeLibraryPath: library,
    );
    int passed = 0;
    try {
      expect(kit.sessionKind, OpenMedSessionKind.onnxRuntime);
      for (final Map<String, dynamic> item in cases) {
        final String id = item['id'] as String;
        final OpenMedDeidentificationResult result = await kit.deidentify(
          item['text'] as String,
          threshold: (item['threshold'] as num? ?? 0.5).toDouble(),
        );
        final List<dynamic> expected = item['expected_spans'] as List<dynamic>;
        expect(result.spans, hasLength(expected.length), reason: id);
        for (int index = 0; index < expected.length; ++index) {
          final Map<String, dynamic> span =
              expected[index] as Map<String, dynamic>;
          expect(result.spans[index].start, span['start'], reason: id);
          expect(result.spans[index].end, span['end'], reason: id);
          expect(result.spans[index].label, span['label'], reason: id);
          expect(
            result.spans[index].score,
            closeTo((span['score'] as num).toDouble(), tolerance),
            reason: id,
          );
        }
        expect(result.deidentifiedText, item['expected_redacted'], reason: id);
        passed += 1;
      }
    } finally {
      kit.close();
    }

    final String? evidence = _env('OPENMED_FLUTTER_REAL_EVIDENCE');
    if (evidence != null) {
      File(evidence).writeAsStringSync(jsonEncode(<String, Object>{
        'bundle_id': bundle.bundleId,
        'cases': cases.length,
        'content_id': bundle.contentId,
        'passed': passed,
        'platform': support.platform,
        'runtime_format': bundle.runtimeFormat,
        'runtime_id': bundle.runtimeId,
        'runtime_precision': bundle.runtimePrecision,
        'schema': 'openmed.flutter_ffi.real_artifact_evidence',
        'sequence': bundle.sequence,
        'session_kind': 'onnx_runtime',
        'version': 1,
      }));
    }
  }, skip: skip);
}
