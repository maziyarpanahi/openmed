# OpenMedKit React Native parity app

This app checks that the `OpenMedKitRN` bridge returns the same result as
calling OpenMedKit directly, on iOS and Android, with a real local model.

On launch it replays the shared synthetic corpora twice per case and operation
(`analyzeText`, `extractPii`, `deidentify`):

1. through the public `@openmed/openmedkit-react-native` API, which crosses the
   React Native boundary into the `OpenMedKitRN` native module;
2. through `OpenMedKitParityProbe` (`probe/`), an example-only native module
   that calls OpenMedKit (Swift or Kotlin) directly, offline, on the same
   verified model bundle, and returns JSON.

`src/parity/` then compares the two results field by field: offsets, labels,
policy labels, actions, replacements, ordering, text hashes, de-identified
text, and scores within the manifest tolerance. It writes a report with only
platform, runtime, device class, bundle digest, counts, lifecycle outcomes,
and mismatch paths with value digests. The report never contains corpus text,
matched spans, file paths, or the per-run hash key, and the app checks this
before it writes the report.

## Guarantees the harness enforces

- **No JavaScript doubles in release.** `setOpenMedKitNativeModuleForTests`
  throws in a release bundle (`__DEV__ === false`), so the app checks the
  guard itself, and a report counts as evidence only when the bridge resolved
  a real native module from a release build.
- **Pinned local model.** CI stages a published model before the build
  (`scripts/stage-model-fixture.ts`, pins in `parity/model-sources.json`). The
  probe re-hashes every file natively and refuses missing, extra, or changed
  files. The app never downloads anything, and CI never converts or publishes
  models. A maintainer must publish and pin the iOS Core ML export locally
  (see `docs/security/hf-token-policy.md`). Until that happens, iOS staging
  fails closed.
- **Offline run.** The harness blocks and counts JavaScript `fetch`, XHR,
  WebSocket and EventSource calls. On Android, CI also switches the emulator
  to airplane mode.
- **Pinned corpora.** `parity/corpus.manifest.json` pins each corpus by
  canonical digest. Any drift stops the run.

## Adding parity vectors

The manifest declares where each file keeps its case list, ids, texts, and
optional policy names. The kit-parity (#3642) and portable-policy (#3643)
vector files are listed under `pending` with their selectors. To enable one:
move its entry to `corpora`, add its `canonical_sha256`, and add one import
line in `src/parity/corpusRegistry.ts`. Expected values are never forked by
platform: the harness compares the bridge with direct native calls.

## Running it

The parity logic is tested in Node with the bridge's own tooling:

```bash
cd js/openmedkit-react-native
npm ci
npm test
```

The native app needs the bridge's autolinking packaging from #4478.
`scripts/check-packaging.sh` reports what is still missing. Once that
packaging is present, `.github/workflows/react-native-parity.yml` runs the app
on an iOS Simulator (Core ML) and an Android emulator (ONNX). For now those
jobs run only on manual dispatch:

```bash
cd js/openmedkit-react-native/example
npm ci
scripts/prepare-native-app.sh ios   # or android
```

The host `ios/` and `android/` projects come from the pinned React Native
template and are not committed.

## Not covered yet

Cancellation, bridge invalidation, background expiration, memory pressure, and
model-generation isolation are reported as `not_covered` until the runtime
lifecycle APIs from #3646 and #3647 exist.
