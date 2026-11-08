// OpenMedKit Flutter FFI behavior through a verified bundle and the native
// fixture session.

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:openmedkit_flutter/openmedkit.dart';

import 'flutter_ffi_support.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  late Directory bundleRoot;

  setUp(() async {
    bundleRoot = await materializeBundle();
  });

  tearDown(() async {
    await bundleRoot.delete(recursive: true);
  });

  test('deidentifies a synthetic note and matches the committed golden',
      () async {
    final Map<String, dynamic> golden = readRepoJson(
      <String>['tests', 'mobile', 'fixtures', 'flutter_ffi_golden.json'],
    );
    final String text = golden['text'] as String;
    expect(text, syntheticNote);

    final List<String> output = <String>[];
    final DebugPrintCallback previousDebugPrint = debugPrint;
    debugPrint = (String? message, {int? wrapWidth}) {
      if (message != null) {
        output.add(message);
      }
    };

    OpenMedKit? runtime;
    try {
      await runZoned(
        () async {
          final OpenMedVerifiedBundle bundle =
              await openFixtureBundle(bundleRoot);
          expect(bundle.contentId, bundleFixture()['content_id']);
          expect(bundle.runtimeId, 'onnx-int8');
          expect(bundle.labels, hasLength(6));
          runtime = await OpenMedKit.loadBundle(
            bundle: bundle,
            tokenizer: syntheticTokenizer,
            nativeLibraryPath: fixtureLibraryPath(),
          );
          expect(runtime!.sessionKind, OpenMedSessionKind.fixture);
          final List<OpenMedSpan> extracted = await runtime!.extractPii(text);
          final OpenMedDeidentificationResult result =
              await runtime!.deidentify(text);

          expect(result.deidentifiedText, golden['deidentified_text']);
          _expectSpansMatchGolden(
            extracted,
            golden['spans'] as List<dynamic>,
            golden['score_tolerance'] as double,
          );
          _expectSpansMatchGolden(
            result.spans,
            golden['spans'] as List<dynamic>,
            golden['score_tolerance'] as double,
          );

          final String serialized = jsonEncode(result.toJson());
          for (final String fragment in <String>[
            'Alice Nguyen',
            '1979-04-12',
            'alice@example.org',
          ]) {
            expect(serialized, isNot(contains(fragment)));
            expect(bundle.toString(), isNot(contains(fragment)));
          }
          expect(bundle.toString(), isNot(contains(bundleRoot.path)));
        },
        zoneSpecification: ZoneSpecification(
          print: (Zone self, ZoneDelegate parent, Zone zone, String line) {
            output.add(line);
          },
        ),
      );
      expect(
        output,
        isEmpty,
        reason: 'raw PHI must never be written to Dart logs',
      );
    } finally {
      runtime?.close();
      debugPrint = previousDebugPrint;
    }
  });

  test('rejects remote and empty bundle directories before any file access',
      () async {
    for (final String directory in <String>[
      'https://example.invalid/model',
      '   ',
    ]) {
      await expectLater(
        OpenMedVerifiedBundle.open(
          directory: directory,
          trustPolicy: fixtureTrustPolicy(),
          context: fixtureContext(),
        ),
        throwsA(_bundleError('bundle_directory_invalid')),
      );
    }
    await expectLater(
      OpenMedVerifiedBundle.open(
        directory: bundleRoot.path,
        trustPolicy: fixtureTrustPolicy(),
        context: PortableBundleContext(now: bundleFixture()['now'] as String),
      ),
      throwsA(_bundleError('context_target_required')),
    );
  });

  test('refuses tampered, unsigned, and incompatible bundles before a session',
      () async {
    final int baseline =
        OpenMedKit.liveRuntimeCount(nativeLibraryPath: fixtureLibraryPath());

    await bundleFile(bundleRoot, 'onnx/model_int8.onnx')
        .writeAsString('tampered synthetic graph\n');
    await expectLater(
      openFixtureBundle(bundleRoot),
      throwsA(_bundleError('file_size_mismatch')),
    );

    final Directory unsigned = await materializeBundle();
    final Directory rolledBack = await materializeBundle();
    try {
      await bundleFile(unsigned, 'openmed-bundle.sig.json').delete();
      await expectLater(
        openFixtureBundle(unsigned),
        throwsA(_bundleError('file_missing')),
      );
      await expectLater(
        OpenMedVerifiedBundle.open(
          directory: rolledBack.path,
          trustPolicy: fixtureTrustPolicy(),
          context: fixtureContext(
            sequenceFloors: <String, int>{'openmed.synthetic.flutter-ffi': 4},
          ),
        ),
        throwsA(_bundleError('rollback_rejected')),
      );
      await expectLater(
        OpenMedVerifiedBundle.open(
          directory: rolledBack.path,
          trustPolicy: fixtureTrustPolicy(),
          context: PortableBundleContext(
            now: bundleFixture()['now'] as String,
            target: const PortableBundleTarget(
              platform: 'linux',
              sdkName: 'openmedkit-android',
              sdkVersion: '4.0.0',
              runtimes: <String>['onnx'],
            ),
          ),
        ),
        throwsA(_bundleError('sdk_incompatible')),
      );
    } finally {
      await unsigned.delete(recursive: true);
      await rolledBack.delete(recursive: true);
    }
    expect(
      OpenMedKit.liveRuntimeCount(nativeLibraryPath: fixtureLibraryPath()),
      baseline,
    );
  });

  test('native digest check refuses a model swapped after verification',
      () async {
    final int baseline =
        OpenMedKit.liveRuntimeCount(nativeLibraryPath: fixtureLibraryPath());
    final OpenMedVerifiedBundle bundle = await openFixtureBundle(bundleRoot);
    final File model = bundleFile(bundleRoot, 'onnx/model_int8.onnx');
    final String original = await model.readAsString();
    await model.writeAsString(original.replaceFirst('int8', 'INT8'));

    try {
      await OpenMedKit.loadBundle(
        bundle: bundle,
        tokenizer: syntheticTokenizer,
        nativeLibraryPath: fixtureLibraryPath(),
      );
      fail('a swapped model must not load');
    } on OpenMedFfiException catch (error) {
      expect(error.code, 'integrity_failed');
      expect(error.message, isNot(contains(bundleRoot.path)));
      expect(error.message, isNot(contains('sha256:')));
      expect(error.toString(), isNot(contains(bundleRoot.path)));
    }
    expect(
      OpenMedKit.liveRuntimeCount(nativeLibraryPath: fixtureLibraryPath()),
      baseline,
    );
  });

  test('a modified label map is refused before labels reach native code',
      () async {
    await bundleFile(bundleRoot, 'id2label.json')
        .writeAsString('{"0":"O","1":"B-OTHER"}\n');
    await expectLater(
      openFixtureBundle(bundleRoot),
      throwsA(_bundleError('file_size_mismatch')),
    );
  });

  test('argument errors never echo source text', () async {
    final OpenMedVerifiedBundle bundle = await openFixtureBundle(bundleRoot);
    final OpenMedKit runtime = await OpenMedKit.loadBundle(
      bundle: bundle,
      tokenizer: (String text) => OpenMedTokenBatch(
        inputIds: const <int>[101],
        offsets: <OpenMedTokenOffset>[
          OpenMedTokenOffset(0, text.startsWith('Patient Alice') ? 99 : 0),
        ],
      ),
      nativeLibraryPath: fixtureLibraryPath(),
    );
    try {
      for (final String text in <String>[
        'Patient Alice Nguyen',
        'Patient Alice\u0000Nguyen',
      ]) {
        try {
          await runtime.extractPii(text);
          fail('invalid input must be rejected');
        } on ArgumentError catch (error) {
          expect(error.toString(), isNot(contains('Alice')));
          expect(error.invalidValue, isNull);
        }
      }
      try {
        await runtime.extractPii('Patient Alice Nguyen', threshold: 2);
        fail('an out-of-range threshold must be rejected');
      } on ArgumentError catch (error) {
        expect(error.toString(), isNot(contains('Alice')));
      }
    } finally {
      runtime.close();
    }
  });

  test('selects only an eligible runtime the FFI can load', () async {
    await expectLater(
      openFixtureBundle(bundleRoot, runtimeId: 'coreml-fp16'),
      throwsA(_bundleError('runtime_unavailable')),
    );
    final OpenMedVerifiedBundle fp32 =
        await openFixtureBundle(bundleRoot, runtimeId: 'onnx-fp32');
    expect(fp32.runtimeId, 'onnx-fp32');
    expect(fp32.runtimeFormat, 'onnx');
    expect(fp32.runtimePrecision, 'fp32');
    expect(
      utf8.decode(await fp32.readVerifiedFile('tokenizer.json')),
      '{"synthetic":true,"version":"1.0"}\n',
    );
    await expectLater(
      fp32.readVerifiedFile('missing.json'),
      throwsA(_bundleError('file_reference_undeclared')),
    );
    await bundleFile(bundleRoot, 'tokenizer.json')
        .writeAsString('{"synthetic":false,"version":"1.0"}\n');
    await expectLater(
      fp32.readVerifiedFile('tokenizer.json'),
      throwsA(_bundleError('file_digest_mismatch')),
    );
  });

  test('does not return spans for masked tokens', () async {
    final OpenMedVerifiedBundle bundle = await openFixtureBundle(bundleRoot);
    final OpenMedKit runtime = await OpenMedKit.loadBundle(
      bundle: bundle,
      tokenizer: (String text) {
        final OpenMedTokenBatch batch = syntheticTokenizer(text);
        return OpenMedTokenBatch(
          inputIds: batch.inputIds,
          attentionMask: const <int>[1, 0, 1, 1, 1],
          offsets: batch.offsets,
        );
      },
      nativeLibraryPath: fixtureLibraryPath(),
    );
    try {
      final List<OpenMedSpan> spans = await runtime.extractPii(syntheticNote);
      expect(spans, hasLength(2));
      expect(spans[0].start, 33);
      expect(spans[0].end, 43);
      expect(spans[0].label, 'DATE_OF_BIRTH');
      expect(spans[1].start, 51);
      expect(spans[1].end, 68);
      expect(spans[1].label, 'EMAIL');
    } finally {
      runtime.close();
    }
  });

  test('reports explicit platform support and native session kinds', () {
    final OpenMedPlatformSupport support = OpenMedKit.platformSupport;
    expect(support.supported, isTrue);
    expect(support.platform, hostPlatform());
    expect(support.runtimes, contains('onnx'));
    expect(support.reason, isNull);
    expect(
      OpenMedKit.nativeSessionKinds(nativeLibraryPath: fixtureLibraryPath()),
      contains(OpenMedSessionKind.fixture),
    );
  });
}

Matcher _bundleError(String code) => isA<OpenMedBundleException>().having(
      (OpenMedBundleException error) => error.code,
      'code',
      code,
    );

void _expectSpansMatchGolden(
  List<OpenMedSpan> actual,
  List<dynamic> expected,
  double tolerance,
) {
  expect(actual, hasLength(expected.length));
  for (int index = 0; index < expected.length; ++index) {
    final Map<String, dynamic> golden = expected[index] as Map<String, dynamic>;
    expect(actual[index].start, golden['start']);
    expect(actual[index].end, golden['end']);
    expect(actual[index].label, golden['label']);
    expect(actual[index].score, closeTo(golden['score'] as double, tolerance));
  }
}
