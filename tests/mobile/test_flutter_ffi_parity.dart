// Shared offset, label, action, and serialization vectors through the FFI.
//
// tests/fixtures/parity/offset_contract.json is consumed by pytest, XCTest,
// and the Android unit tests. Here each case's grapheme-snapped span is fed to
// the native fixture session as B-NAME/I-NAME tokens, and the FFI must return
// the same Unicode-scalar offsets, the NAME label, the `[NAME]` replacement,
// and a source-free serialization. Snapping decoder offsets to grapheme
// boundaries happens before tokens reach the FFI and is not exercised here.

import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:openmedkit_flutter/openmedkit.dart';

import 'flutter_ffi_support.dart';

const int _nameBegin = 4;
const int _nameInside = 5;

OpenMedTokenBatch _caseTokens(int start, int end, {required bool split}) {
  final List<int> ids = <int>[101];
  final List<OpenMedTokenOffset> offsets = <OpenMedTokenOffset>[
    const OpenMedTokenOffset(0, 0),
  ];
  if (split && end - start >= 2) {
    ids
      ..add(labelToken(_nameBegin))
      ..add(labelToken(_nameInside));
    offsets
      ..add(OpenMedTokenOffset(start, start + 1))
      ..add(OpenMedTokenOffset(start + 1, end));
  } else {
    ids.add(labelToken(_nameBegin));
    offsets.add(OpenMedTokenOffset(start, end));
  }
  ids.add(102);
  offsets.add(const OpenMedTokenOffset(0, 0));
  return OpenMedTokenBatch(inputIds: ids, offsets: offsets);
}

void main() {
  late Directory bundleRoot;
  late OpenMedKit kit;
  final Map<String, OpenMedTokenBatch> batches = <String, OpenMedTokenBatch>{};
  final Map<String, dynamic> fixture = readRepoJson(
    <String>['tests', 'fixtures', 'parity', 'offset_contract.json'],
  );
  final List<Map<String, dynamic>> cases =
      (fixture['cases'] as List<dynamic>).cast<Map<String, dynamic>>();

  setUpAll(() async {
    bundleRoot = await materializeBundle();
    kit = await OpenMedKit.loadBundle(
      bundle: await openFixtureBundle(bundleRoot),
      tokenizer: (String text) => batches[text]!,
      nativeLibraryPath: fixtureLibraryPath(),
    );
    expect(kit.bundle.labels[_nameBegin], 'B-NAME');
    expect(kit.bundle.labels[_nameInside], 'I-NAME');
  });

  tearDownAll(() async {
    kit.close();
    await bundleRoot.delete(recursive: true);
  });

  test('shared offset fixture keeps its coverage', () {
    expect(fixture['version'], 1);
    expect(fixture['offset_unit'], 'unicode_scalar');
    expect(cases.length, greaterThanOrEqualTo(40));
    expect(
      cases.map((Map<String, dynamic> item) => item['category']).toSet(),
      containsAll(<String>['cjk', 'indic', 'joiner', 'mixed']),
    );
    expect(
      cases.where((Map<String, dynamic> item) {
        final String text = item['text'] as String;
        return text.length > text.runes.length;
      }),
      isNotEmpty,
      reason: 'supplementary-plane scalars must be covered',
    );
  });

  for (final bool split in <bool>[false, true]) {
    test(
        'every case keeps offsets, label, replacement, and serialization '
        '(${split ? 'B+I tokens' : 'single token'})', () async {
      for (final Map<String, dynamic> item in cases) {
        final String id = item['id'] as String;
        final String text = item['text'] as String;
        final int start = item['expected_start'] as int;
        final int end = item['expected_end'] as int;
        expect(item['replacement'], '[NAME]', reason: id);
        batches[text] = _caseTokens(start, end, split: split);

        final List<OpenMedSpan> extracted = await kit.extractPii(text);
        final OpenMedDeidentificationResult result = await kit.deidentify(text);
        for (final List<OpenMedSpan> spans in <List<OpenMedSpan>>[
          extracted,
          result.spans,
        ]) {
          expect(spans, hasLength(1), reason: id);
          expect(spans.single.start, start, reason: id);
          expect(spans.single.end, end, reason: id);
          expect(spans.single.label, 'NAME', reason: id);
        }
        expect(result.deidentifiedText, item['expected_redacted'], reason: id);
        expect(
          utf8.encode(result.deidentifiedText),
          utf8.encode(item['expected_redacted'] as String),
          reason: id,
        );

        final String entity =
            String.fromCharCodes(text.runes.toList().sublist(start, end));
        final String spans = jsonEncode(
          result.spans.map((OpenMedSpan span) => span.toJson()).toList(),
        );
        expect(spans, isNot(contains(entity)), reason: id);
        final Map<String, dynamic> roundTrip =
            jsonDecode(jsonEncode(result.toJson())) as Map<String, dynamic>;
        expect(roundTrip.keys, <String>['deidentified_text', 'spans']);
        expect(
          (roundTrip['spans'] as List<dynamic>).single,
          <String, Object>{
            'start': start,
            'end': end,
            'label': 'NAME',
            'score': result.spans.single.score,
          },
          reason: id,
        );
      }
    });
  }

  test('a threshold above every score keeps the source text unchanged',
      () async {
    final Map<String, dynamic> item = cases.first;
    final String text = item['text'] as String;
    batches[text] = _caseTokens(
      item['expected_start'] as int,
      item['expected_end'] as int,
      split: false,
    );
    final OpenMedDeidentificationResult result =
        await kit.deidentify(text, threshold: 0.999);
    expect(result.spans, isEmpty);
    expect(result.deidentifiedText, text);
  });
}
