# Electron In-Process De-identification

OpenMed's Electron reference integration keeps inference off the main and
renderer threads without sending a clinical note to a server. The renderer
sends text over a single typed IPC channel, the main process forwards it to one
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
canonical labels.

## Integration

The reference package lives in `js/openmedkit-electron`. Its main pieces are:

- `ElectronDeidentifyService`, a lazily started utility-process owner
- `registerElectronDeidentifyIpc`, the main-process handler
- `createElectronDeidentifyClient`, the preload/renderer contract
- `utility-process`, the worker entrypoint to bundle with the Electron app

Create exactly one service after `app.whenReady()` and register it with the
application's `ipcMain`. Supply a dedicated, non-persistent Electron session;
the service installs a fail-closed request interceptor on that session. Point
`workerPath` at the bundled utility entrypoint and `modelPath` at an absolute,
pre-populated local model directory. See the
[complete copy/paste example](https://github.com/maziyarpanahi/openmed/blob/master/examples/electron/redact-app.md).

Registration requires an `isTrustedSender` predicate. Check both the sending
`webContents` and its main frame; do not authorize subframes or a renderer by
URL text alone. Renderer and preload bundles should import the Node-free
`@openmed/openmedkit-electron/renderer` entrypoint.

## Offline and cache behavior

The utility process starts with `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`. Its dedicated Electron session cancels Chromium-stack
requests; the worker blocks Fetch, WebSocket, EventSource, Node HTTP(S), HTTP/2,
DNS, TCP, TLS, UDP, and child-process entry points. It calls the OpenMed npm
loader with both `localFilesOnly: true` and `allowRemoteModels: false`.
Remote model identifiers and runtime downloads therefore fail closed.

Pipelines are cached by absolute model path inside the utility process. Because
the main-process service owns one worker for the whole app, all authorized
windows reuse the same loaded pipeline. Inference is serialized so a shared
pipeline is never invoked concurrently. A failed load is evicted so a repaired
local cache can be retried without restarting the app.

Populate the model directory during installation or through an explicit,
non-PHI setup flow before inference. Model download and update UX is outside the
request path and should never receive clinical text.

## Logging and renderer hardening

Use `contextIsolation: true` and `nodeIntegration: false`, and expose only the
typed client through the preload bridge. The utility process uses ignored
standard streams. Its error responses contain fixed error codes instead of raw
exception text, which may include input or local paths.

If a logger is supplied to `ElectronDeidentifyService`, it receives fixed event
names plus a span count or safe error code. Do not add request text, IPC payloads,
model outputs, or exception messages to main- or renderer-process logs.

Requests have bounded text, timeout, and pending-queue limits. A timeout,
malformed worker response, or IPC send failure retires the worker and rejects
all of its pending work with a fixed error. Returned spans must be ordered,
non-overlapping, inside the source-text bounds, and use known public labels.

The Node test harness uses the committed synthetic npm golden, exercises two
renderer clients against one service, checks both Electron- and Node-stack
network denial, verifies sender authorization, queue and timeout recovery,
pipeline reuse and serialization, and asserts that neither process log capture
contains any synthetic identifier.
