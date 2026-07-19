import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:openmedkit_flutter/openmedkit.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  const MethodChannel channel = MethodChannel(openMedKitFlutterChannelName);
  late Directory modelDirectory;

  setUp(() async {
    modelDirectory = await Directory.systemTemp.createTemp(
      'openmed-flutter-ffi-model-',
    );
    await File(
      '${modelDirectory.path}${Platform.pathSeparator}model_int8.onnx',
    ).writeAsBytes(<int>[0]);
    await File(
      '${modelDirectory.path}${Platform.pathSeparator}id2label.json',
    ).writeAsString(
      jsonEncode(<String, String>{
        '0': 'O',
        '1': 'B-PERSON',
        '2': 'B-DATE_OF_BIRTH',
        '3': 'B-EMAIL',
      }),
    );
    channel.setMockMethodCallHandler((MethodCall call) async {
      expect(call.method, 'prepareModel');
      return modelDirectory.absolute.path;
    });
  });

  tearDown(() async {
    channel.setMockMethodCallHandler(null);
    await modelDirectory.delete(recursive: true);
  });

  test('deidentifies a synthetic note and matches the committed golden', () async {
    final File goldenFile = File(
      '${Directory.current.path}${Platform.pathSeparator}..${Platform.pathSeparator}..'
      '${Platform.pathSeparator}tests${Platform.pathSeparator}mobile'
      '${Platform.pathSeparator}fixtures${Platform.pathSeparator}flutter_ffi_golden.json',
    );
    final Map<String, dynamic> golden =
        jsonDecode(await goldenFile.readAsString()) as Map<String, dynamic>;
    final String text = golden['text'] as String;
    final String? libraryPath = Platform.environment['OPENMED_FFI_LIBRARY'];
    expect(
      libraryPath,
      isNotNull,
      reason: 'the native fixture library is required',
    );

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
          runtime = await OpenMedKit.loadModel(
            modelDirectory: modelDirectory.path,
            tokenizer: _fixtureTokenizer,
            nativeLibraryPath: libraryPath,
            platformChannel: channel,
          );
          final List<OpenMedSpan> extracted = await runtime!.extractPii(text);
          final OpenMedDeidentificationResult result = await runtime!
              .deidentify(text);

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
          }
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

  test('rejects remote model paths before invoking platform code', () async {
    expect(
      () => OpenMedKit.loadModel(
        modelDirectory: 'https://example.invalid/model',
        tokenizer: _fixtureTokenizer,
        platformChannel: channel,
      ),
      throwsArgumentError,
    );
  });
}

OpenMedTokenBatch _fixtureTokenizer(String text) {
  expect(
    text,
    'Patient Alice Nguyen was born on 1979-04-12. Email alice@example.org.',
  );
  return OpenMedTokenBatch(
    inputIds: const <int>[101, 1001, 1002, 1003, 102],
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
