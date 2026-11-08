# Electron In-Process De-identification

OpenMed's Electron reference integration keeps inference off the main and
renderer threads without sending a clinical note to a server. The renderer
sends text over one typed IPC channel, the main process forwards it to one
Electron utility process, and the renderer receives only this projection of an
`OpenMedSpan`:

```ts
type RendererOpenMedSpan = Pick<
  OpenMedSpan,
  | "schema_version"
  | "start"
  | "end"
  | "entity_type"
  | "canonical_label"
  | "policy_label"
  | "score"
>;
```

Raw text, hashes, detector evidence, metadata, and model errors are not returned
to the renderer. Redaction is applied locally from the returned offsets and
canonical labels. Offsets are JavaScript string (UTF-16) indices into the exact
text the renderer sent.

## Integration

The reference package lives in `js/openmedkit-electron`. Its main pieces are:

- `ElectronDeidentifyService`, the main-process owner of one utility process
- `registerElectronDeidentifyIpc`, the main-process handlers for the
  `openmed:deidentify` and `openmed:deidentify:cancel` channels
- `createElectronDeidentifyClient`, the preload/renderer contract
- `utility-process`, the default worker entrypoint to bundle with the app, and
  `utility-runtime`, for apps that build their own entrypoint
- `createHardenedWebPreferences`, `hardenOpenMedWebContents` and
  `hardenOpenMedSession`, the window and session guards

Create exactly one service after `app.whenReady()` and register it with the
application's `ipcMain`. Supply a dedicated, non-persistent Electron session;
the service installs a fail-closed request interceptor on that session. Point
`workerPath` at the bundled utility entrypoint. See the
[complete copy/paste example](https://github.com/maziyarpanahi/openmed/blob/master/examples/electron/redact-app.md).

Registration requires an `isTrustedSender` predicate. Check both the sending
`webContents` and its main frame; do not authorize subframes or a renderer by
URL text alone. Renderer and preload bundles should import the Node-free
`@openmed/openmedkit-electron/renderer` entrypoint.

## Verified model bundles

Pass `bundle` instead of `modelPath` to run a signed
[portable model bundle](../models/portable-bundle.md). Every field comes from
the host process:

```ts
const service = new ElectronDeidentifyService({
  utilityProcess,
  offlineSession,
  workerPath,
  bundle: {
    root: join(process.resourcesPath, "models", "my-pii-model"),
    trustPolicy: myTrustPolicyDocument,
    target: {
      platform: "web",
      sdkName: "openmedkit-web",
      sdkVersion: "4.0.0",
      runtimes: ["onnx", "transformersjs"],
    },
    sequenceFloors: { "org.example.my-model": lastAcceptedSequence },
    expectedContentId: "sha256:<hex>",
  },
});
const handle = await service.whenReady();
```

The utility process reads the bundle directory without following symbolic
links, verifies it with the `openmed` TypeScript verifier (`verifyPortableBundle`)
and refuses to serve any request until the manifest, signatures, trust policy,
validity window, anti-rollback floor, compatibility target and every file
digest pass. It then loads the first eligible runtime it supports. The
directory reader is also exported as `verifyPortableBundleDirectory` and
reproduces the shared golden vectors in
`tests/fixtures/portable_bundle/vectors.json`.

The default loader hands the bundle directory to Transformers.js, which reads
the files again after verification. Keep the bundle in an app-owned, read-only
location such as the app's resources. A custom loader passed to
`startUtilityProcess` (from `@openmed/openmedkit-electron/utility-runtime`)
receives the exact verified bytes in `bundle.files`.

A rejected bundle raises `ElectronBundleActivationError` with a value-free
`reason`: one of the portable verifier's reason codes, or
`bundle_source_invalid`, `content_id_unexpected`, `runtime_unsupported` or
`pipeline_load_failed`. Results carry the generation and content identity that
produced them in `response.bundle`.

### Generation changes

`service.activateBundle(source)` starts a new generation. Requests submitted
before the call finish on the generation they were submitted for; requests
submitted after it run only once the new bundle verifies, and fail with
`BUNDLE_UNAVAILABLE` if it is rejected. The utility process runs bundle loads
and inference through one queue, so results never mix generations. A result
whose generation does not match its request retires the utility process.

### Restart after a crash

If the utility process exits or crashes, every request it held rejects exactly
once with `UTILITY_PROCESS_EXITED`, and late messages from the dead process are
ignored. The next request starts a new utility process, which verifies the
current bundle again before serving.

## Renderer requests, cancellation and teardown

A renderer request contains exactly `schemaVersion`, `requestId` and `text`.
Any other member, for example a model path, bundle, runtime, generation or
policy, is refused with `INVALID_REQUEST` before inference. Renderers cannot
choose or widen what the utility process loads.

`client.submit(text)` returns a task with `requestId`, `result`, `outcome` and
`cancel()`; `client.deidentify(text, { signal })` accepts an `AbortSignal`.
Cancelling rejects the request with `CANCELLED` at once. The utility process
skips the job if it is still queued and discards its output if it is running.
Requests are scoped to the renderer that sent them, so one renderer cannot
cancel another's work. When a renderer's `webContents` is destroyed or its
process is gone, the service cancels that renderer's pending requests.

## Typed errors

Every failure that reaches a renderer is a typed response with a controlled
code from `ELECTRON_DEIDENTIFY_ERROR_CODES`, such as `UNTRUSTED_SENDER`,
`INVALID_REQUEST`, `CANCELLED`, `REQUEST_TIMEOUT`, `BUNDLE_UNAVAILABLE`,
`UTILITY_PROCESS_EXITED` or `INFERENCE_FAILED`. The client rejects with an
`ElectronDeidentifyError` whose `code` is that value. `contextBridge` copies
errors without custom properties, so a page behind the bridge should read
`task.outcome`, which never rejects and resolves to either the success response
or `{ ok: false, errorCode }`.

## Window hardening

Every window that can reach OpenMed IPC must be hardened, and the IPC handlers
enforce it: a sender whose resolved web preferences
(`webContents.getLastWebPreferences()`) disable `contextIsolation` or
`sandbox`, enable Node integration (including workers and subframes), turn off
`webSecurity`, allow insecure content, enable `<webview>`, experimental or
extra Blink features is refused with `UNTRUSTED_SENDER`, even if
`isTrustedSender` accepts it.

```ts
const window = new BrowserWindow({
  webPreferences: createHardenedWebPreferences({
    preload: join(__dirname, "preload.cjs"),
    partition: "openmed-renderer",
  }),
});
hardenOpenMedWebContents(window.webContents, { allowedOrigins: ["app://openmed"] });
hardenOpenMedSession(window.webContents.session, { allowedOrigins: ["app://openmed"] });
```

`hardenOpenMedWebContents` throws `ElectronHardeningError` for a relaxed window,
denies `window.open`, blocks `<webview>` attachment, and cancels navigation,
redirects and frame navigation away from the allowed origins. Allowed origins
must be local app-scheme origins; `http`, `https` and `file` origins are
refused. `hardenOpenMedSession` denies every permission request, cancels
off-origin requests and replaces app responses' CSP with `OPENMED_RENDERER_CSP`,
which allows scripts, styles, images and fonts from `'self'` only and refuses
inline scripts, `eval`, network connections, frames, workers and plugins. Serve
the renderer from a privileged custom scheme (for example `app://openmed`) and
set the same policy on the protocol handler's responses.

## Offline and cache behavior

The utility process starts with `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`. Its dedicated Electron session cancels Chromium-stack
requests; the worker blocks Fetch, WebSocket, EventSource, Node HTTP(S), HTTP/2,
DNS, TCP, TLS, UDP, and child-process entry points. It calls the OpenMed npm
loader with both `localFilesOnly: true` and `allowRemoteModels: false`.
Remote model identifiers and runtime downloads therefore fail closed.

With a legacy `modelPath`, pipelines are cached by absolute model path inside
the utility process. In both modes, the main-process service owns one worker for
the whole app, so all authorized windows reuse the same loaded pipeline.
Inference is serialized so a shared pipeline is never invoked concurrently. A
failed legacy load is evicted so a repaired local cache can be retried without
restarting the app.

Populate the model directory during installation or through an explicit,
non-PHI setup flow before inference. Model download and update UX is outside the
request path and should never receive clinical text.

## Logging and diagnostics

The utility process uses ignored standard streams. Its error responses contain
fixed error codes instead of raw exception text, which may include input or
local paths.

If a logger is supplied to `ElectronDeidentifyService`, it receives fixed event
names plus a span count, a safe error code, a bundle generation, a bundle
content identity or a bundle reason code. `service.diagnostics()` returns
value-free counters. Do not add request text, IPC payloads, model outputs, or
exception messages to main- or renderer-process logs. Do not start Electron's
`crashReporter` with uploads for apps that process clinical text: a minidump
can contain process memory.

Requests have bounded text, timeout, and pending-queue limits. A timeout,
malformed worker response, or IPC send failure retires the worker and rejects
all of its pending work with a fixed error. Returned spans must be ordered,
non-overlapping, inside the source-text bounds, and use known public labels.

## Qualification

The Node tests (`npm --prefix js/openmedkit-electron test`) use in-process fakes
and the shared golden vectors. They cover sender authorization and hardening
enforcement, refused request members, typed errors, queue and timeout recovery,
bundle verification, generation changes, rejected bundles, cancellation,
renderer teardown, crash and restart, and value-free diagnostics.

`js/openmedkit-electron/harness` is a minimal packaged app around the
integration. It is built locally with electron-builder as an unsigned
directory target (no signing or notarization) with the `RunAsNode`,
`NODE_OPTIONS` and inspector fuses disabled, and its runner
(`tests/web/test_electron_packaged_harness.mjs`) launches it offline:

```bash
npm ci --prefix js/openmedkit-electron/harness
npm --prefix js/openmedkit-electron/harness run package
npm --prefix js/openmedkit-electron/harness test
# or, without packaging:
npm --prefix js/openmedkit-electron/harness run test:unpackaged
```

The harness verifies the shared TEST-ONLY golden bundles in the real utility
process and checks the verification results against the vectors. It drives
every request through the production preload bridge in a sandboxed renderer:
every shared offset-contract case (`tests/fixtures/parity/offset_contract.json`)
and the npm golden must match the direct
in-process reference exactly, and the window must show no Node globals, an
active CSP, blocked popups and blocked remote navigation. It also exercises a
compromised renderer, cancellation, a bundle generation change, a tampered
bundle, renderer teardown, a real utility-process crash with restart, and a
relaxed window that must be refused. The runner plants a random synthetic name
and scans stdout, stderr, the report and the whole user-data directory
(including crash dumps) for it. The `Electron packaged harness` workflow runs
the packaged app headlessly on Linux under `xvfb`.

The harness model is synthetic: the golden bundles carry placeholder model
files, so inference uses a deterministic tagger that emits the shared
reference spans. The harness proves the packaged boundary preserves verified
bundles, policy and result semantics; it does not measure model quality.
