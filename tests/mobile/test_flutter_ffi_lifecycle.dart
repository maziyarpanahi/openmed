// Lifecycle qualification for the OpenMedKit Flutter FFI: cancellation,
// repeated close, model replacement, runtime ownership, and isolate teardown.
// Native allocations are counted by openmed_runtime_live_count.

import 'dart:async';
import 'dart:io';
import 'dart:isolate';

import 'package:flutter_test/flutter_test.dart';
import 'package:openmedkit_flutter/openmedkit.dart';

import 'flutter_ffi_support.dart';

int _live() =>
    OpenMedKit.liveRuntimeCount(nativeLibraryPath: fixtureLibraryPath());

/// Tokenizer that waits until the test releases it.
class _GatedTokenizer {
  final List<Completer<void>> _gates = <Completer<void>>[];

  int get pending => _gates.length;

  Future<OpenMedTokenBatch> call(String text) async {
    final Completer<void> gate = Completer<void>();
    _gates.add(gate);
    await gate.future;
    return syntheticTokenizer(text);
  }

  void releaseAll() {
    for (final Completer<void> gate in _gates) {
      if (!gate.isCompleted) {
        gate.complete();
      }
    }
  }
}

Future<void> _settle() => Future<void>.delayed(Duration.zero);

void main() {
  late Directory bundleRoot;

  setUp(() async {
    bundleRoot = await materializeBundle();
  });

  tearDown(() async {
    await bundleRoot.delete(recursive: true);
  });

  Future<OpenMedKit> load(OpenMedTokenizer tokenizer, {int ownerId = 0}) async {
    return OpenMedKit.loadBundle(
      bundle: await openFixtureBundle(bundleRoot),
      tokenizer: tokenizer,
      nativeLibraryPath: fixtureLibraryPath(),
      ownerId: ownerId,
    );
  }

  test('repeated close frees the native runtime exactly once', () async {
    final int baseline = _live();
    final OpenMedKit kit = await load(syntheticTokenizer);
    expect(_live(), baseline + 1);
    kit
      ..close()
      ..close()
      ..close();
    expect(kit.isClosed, isTrue);
    expect(_live(), baseline);
    expect(() => kit.sessionKind, throwsStateError);
    await expectLater(kit.extractPii(syntheticNote), throwsStateError);
    await expectLater(
      kit.replaceBundle(await openFixtureBundle(bundleRoot)),
      throwsStateError,
    );
    expect(_live(), baseline);
  });

  test('cancellation drops a pending result and keeps the runtime usable',
      () async {
    final int baseline = _live();
    final _GatedTokenizer tokenizer = _GatedTokenizer();
    final OpenMedKit kit = await load(tokenizer.call);
    try {
      final OpenMedCancellationToken token = OpenMedCancellationToken();
      final Future<OpenMedDeidentificationResult> pending =
          kit.deidentify(syntheticNote, cancellation: token);
      await _settle();
      expect(tokenizer.pending, 1);
      token.cancel();
      tokenizer.releaseAll();
      await expectLater(pending, throwsA(_cancelled('cancelled')));

      final OpenMedCancellationToken already = OpenMedCancellationToken()
        ..cancel();
      await expectLater(
        kit.extractPii(syntheticNote, cancellation: already),
        throwsA(_cancelled('cancelled')),
      );
      expect(tokenizer.pending, 1, reason: 'no tokenizer call when cancelled');

      final Future<List<OpenMedSpan>> next = kit.extractPii(syntheticNote);
      await _settle();
      tokenizer.releaseAll();
      expect(await next, hasLength(3));
    } finally {
      kit.close();
    }
    expect(_live(), baseline);
  });

  test('close while tokenizing drops the late result without native access',
      () async {
    final int baseline = _live();
    final _GatedTokenizer tokenizer = _GatedTokenizer();
    final OpenMedKit kit = await load(tokenizer.call);
    final Future<List<OpenMedSpan>> extract = kit.extractPii(syntheticNote);
    final Future<OpenMedDeidentificationResult> deidentify =
        kit.deidentify(syntheticNote);
    await _settle();
    expect(tokenizer.pending, 2);
    kit.close();
    expect(_live(), baseline);
    tokenizer.releaseAll();
    await expectLater(extract, throwsA(_cancelled('closed')));
    await expectLater(deidentify, throwsA(_cancelled('closed')));
    expect(_live(), baseline);
  });

  test('replacement frees the previous generation once and drops its requests',
      () async {
    final int baseline = _live();
    final _GatedTokenizer tokenizer = _GatedTokenizer();
    final OpenMedKit kit = await load(tokenizer.call);
    try {
      expect(kit.bundle.runtimeId, 'onnx-int8');
      final int generation = kit.generation;
      final Future<List<OpenMedSpan>> stale = kit.extractPii(syntheticNote);
      await _settle();

      await kit.replaceBundle(
        await openFixtureBundle(bundleRoot, runtimeId: 'onnx-fp32'),
      );
      expect(kit.generation, generation + 1);
      expect(kit.bundle.runtimeId, 'onnx-fp32');
      expect(_live(), baseline + 1);

      tokenizer.releaseAll();
      await expectLater(stale, throwsA(_cancelled('model_replaced')));

      final Future<OpenMedDeidentificationResult> fresh =
          kit.deidentify(syntheticNote);
      await _settle();
      tokenizer.releaseAll();
      expect(
        (await fresh).deidentifiedText,
        'Patient [PERSON] was born on [DATE_OF_BIRTH]. Email [EMAIL].',
      );
    } finally {
      kit.close();
    }
    expect(_live(), baseline);
  });

  test('failed replacement releases its allocations and keeps serving',
      () async {
    final int baseline = _live();
    final OpenMedKit kit = await load(syntheticTokenizer);
    try {
      final int generation = kit.generation;
      final OpenMedVerifiedBundle replacement =
          await openFixtureBundle(bundleRoot, runtimeId: 'onnx-fp32');
      final File model = bundleFile(bundleRoot, 'onnx/model.onnx');
      final String original = await model.readAsString();
      await model.writeAsString(original.replaceFirst('fp32', 'FP32'));

      await expectLater(
        kit.replaceBundle(replacement),
        throwsA(isA<OpenMedFfiException>().having(
          (OpenMedFfiException error) => error.code,
          'code',
          'integrity_failed',
        )),
      );
      expect(kit.generation, generation);
      expect(kit.bundle.runtimeId, 'onnx-int8');
      expect(_live(), baseline + 1);
      expect(await kit.extractPii(syntheticNote), hasLength(3));

      await expectLater(
        openFixtureBundle(bundleRoot, runtimeId: 'onnx-fp32'),
        throwsA(isA<OpenMedBundleException>().having(
          (OpenMedBundleException error) => error.code,
          'code',
          'file_digest_mismatch',
        )),
      );
      expect(_live(), baseline + 1);
    } finally {
      kit.close();
    }
    expect(_live(), baseline);
  });

  test('owner release frees only runtimes that were not closed', () async {
    final int baseline = _live();
    final OpenMedRuntimeOwner owner =
        OpenMedRuntimeOwner.create(nativeLibraryPath: fixtureLibraryPath());
    final OpenMedKit closed = await load(syntheticTokenizer, ownerId: owner.id);
    final OpenMedKit orphaned =
        await load(syntheticTokenizer, ownerId: owner.id);
    expect(_live(), baseline + 2);
    closed.close();
    expect(owner.release(), 1);
    expect(owner.release(), 0);
    expect(_live(), baseline);
    // The orphaned kit's handle is gone natively; closing it is a no-op.
    orphaned.close();
    expect(_live(), baseline);
  });

  test('worker isolate results match the root isolate and teardown is clean',
      () async {
    final int baseline = _live();
    final OpenMedKit kit = await load(syntheticTokenizer);
    final Map<String, Object> direct;
    try {
      direct = (await kit.deidentify(syntheticNote)).toJson();
    } finally {
      kit.close();
    }

    final OpenMedRuntimeOwner owner =
        OpenMedRuntimeOwner.create(nativeLibraryPath: fixtureLibraryPath());
    final _WorkerRun run = await _spawnWorker(bundleRoot, owner.id, 'complete');
    await run.exited;
    expect(run.messages.last, 'closed');
    expect(run.messages, anyElement(equals(direct)));
    expect(owner.release(), 0, reason: 'the worker closed its own runtime');
    expect(_live(), baseline);
  });

  test('killing a worker isolate mid-request drops its result and frees once',
      () async {
    final int baseline = _live();
    final OpenMedRuntimeOwner owner =
        OpenMedRuntimeOwner.create(nativeLibraryPath: fixtureLibraryPath());
    final _WorkerRun run = await _spawnWorker(bundleRoot, owner.id, 'hang');
    await run.waiting.future;
    expect(run.messages, contains('waiting'), reason: 'worker is tokenizing');
    expect(_live(), baseline + 1);
    run.isolate.kill(priority: Isolate.immediate);
    await run.exited;
    expect(run.messages, isNot(contains('late_result')));
    expect(owner.release(), 1);
    expect(owner.release(), 0);
    expect(_live(), baseline);
  });
}

Matcher _cancelled(String code) => isA<OpenMedCancelledException>().having(
      (OpenMedCancelledException error) => error.code,
      'code',
      code,
    );

class _WorkerRun {
  _WorkerRun(this.isolate, this.messages, this.waiting, this.exited);

  final Isolate isolate;
  final List<Object?> messages;
  final Completer<void> waiting;
  final Future<void> exited;
}

Future<_WorkerRun> _spawnWorker(
  Directory bundleRoot,
  int ownerId,
  String mode,
) async {
  final ReceivePort messages = ReceivePort();
  final ReceivePort exit = ReceivePort();
  final ReceivePort errors = ReceivePort();
  final List<Object?> received = <Object?>[];
  final Completer<void> waiting = Completer<void>();
  final Completer<void> exited = Completer<void>();
  final List<Object?> failures = <Object?>[];
  messages.listen((Object? message) {
    received.add(message);
    if (message == 'waiting' && !waiting.isCompleted) {
      waiting.complete();
    }
  });
  errors.listen(failures.add);
  exit.listen((Object? _) {
    messages.close();
    errors.close();
    exit.close();
    if (!waiting.isCompleted) {
      waiting.complete();
    }
    exited.complete();
  });
  final Isolate isolate = await Isolate.spawn<Map<String, Object?>>(
    _worker,
    <String, Object?>{
      'directory': bundleRoot.path,
      'library': fixtureLibraryPath(),
      'mode': mode,
      'owner': ownerId,
      'reply': messages.sendPort,
    },
    onExit: exit.sendPort,
    onError: errors.sendPort,
  );
  return _WorkerRun(
    isolate,
    received,
    waiting,
    exited.future.then((_) {
      if (failures.isNotEmpty) {
        throw StateError('worker failed: $failures');
      }
    }),
  );
}

Future<void> _worker(Map<String, Object?> message) async {
  final SendPort reply = message['reply']! as SendPort;
  final String library = message['library']! as String;
  final bool hang = message['mode'] == 'hang';
  final OpenMedVerifiedBundle bundle = await OpenMedVerifiedBundle.open(
    directory: message['directory']! as String,
    trustPolicy: fixtureTrustPolicy(),
    context: fixtureContext(),
  );
  final Completer<OpenMedTokenBatch> never = Completer<OpenMedTokenBatch>();
  final OpenMedKit kit = await OpenMedKit.loadBundle(
    bundle: bundle,
    tokenizer: (String text) {
      if (!hang) {
        return syntheticTokenizer(text);
      }
      reply.send('waiting');
      return never.future;
    },
    nativeLibraryPath: library,
    ownerId: message['owner']! as int,
  );
  try {
    final OpenMedDeidentificationResult result =
        await kit.deidentify(syntheticNote);
    reply.send(hang ? 'late_result' : result.toJson());
  } finally {
    kit.close();
    reply.send('closed');
  }
}
