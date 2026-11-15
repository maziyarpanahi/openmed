// The build used where dart:ffi is unavailable (Flutter web) must report an
// explicit, typed unsupported result instead of loading anything.

import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:openmedkit_flutter/openmedkit.dart'
    show
        OpenMedPlatformSupport,
        OpenMedUnsupportedPlatformException,
        PortableBundleContext,
        PortableBundleException,
        PortableBundleTarget,
        PortableBundleTrustPolicy,
        PortableBundleVerification;
import 'package:openmedkit_flutter/src/kit_unsupported.dart' as web;

import 'flutter_ffi_support.dart';

Matcher get _unsupported => isA<OpenMedUnsupportedPlatformException>()
    .having(
      (OpenMedUnsupportedPlatformException error) => error.code,
      'code',
      'platform_unsupported',
    )
    .having(
      (OpenMedUnsupportedPlatformException error) => error.platform,
      'platform',
      'web',
    );

void main() {
  final Map<String, dynamic> fixture = bundleFixture();
  final PortableBundleTrustPolicy policy = fixtureTrustPolicy();
  final PortableBundleContext context = PortableBundleContext(
    now: fixture['now'] as String,
    target: PortableBundleTarget.flutter(platform: 'web'),
  );

  test('platform support is explicit and empty', () {
    final OpenMedPlatformSupport support = web.OpenMedKit.platformSupport;
    expect(support.platform, 'web');
    expect(support.supported, isFalse);
    expect(support.runtimes, isEmpty);
    expect(support.reason, 'platform_unsupported');
    expect(web.OpenMedKit.liveRuntimeCount(), 0);
    expect(web.OpenMedKit.nativeSessionKinds(), isEmpty);
  });

  test('loading and ownership throw a typed unsupported error', () async {
    await expectLater(
      web.OpenMedVerifiedBundle.open(
        directory: '/synthetic/bundle',
        trustPolicy: policy,
        context: context,
      ),
      throwsA(_unsupported),
    );
    expect(() => web.OpenMedRuntimeOwner.create(), throwsA(_unsupported));
    expect(
      () => web.PortableBundle.canonicalJson(<String, Object>{}),
      throwsA(_unsupported),
    );
  });

  test('verification returns a refused result, never a pass', () {
    final PortableBundleVerification result = web.PortableBundle.verify(
      manifest: utf8.encode(fixture['manifest'] as String),
      envelope: utf8.encode(fixture['envelope'] as String),
      trustPolicy: policy,
      context: context,
    );
    expect(result.ok, isFalse);
    expect(result.reason, 'signature_verifier_unavailable');
    expect(result.eligibleForActivation, isFalse);
    expect(
      web.PortableBundle.verifyDirectory(
        '/synthetic/bundle',
        trustPolicy: policy,
        context: context,
      ).reason,
      'signature_verifier_unavailable',
    );
    expect(
      () => web.PortableBundle.verify(
        manifest: const <int>[],
        envelope: const <int>[],
        trustPolicy: policy,
        context: const PortableBundleContext(now: 'not-a-time'),
      ),
      throwsA(isA<PortableBundleException>()),
    );
  });
}
