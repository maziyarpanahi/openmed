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
application's `ipcMain`. Point `workerPath` at the bundled utility entrypoint and
`modelPath` at an absolute, pre-populated local model directory. See the
[complete copy/paste example](https://github.com/maziyarpanahi/openmed/blob/master/examples/electron/redact-app.md).

## Offline and cache behavior

The utility process starts with `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`, disables Fetch, Node HTTP(S), and raw socket entry
points, and calls the OpenMed npm loader with both `localFilesOnly: true` and
`allowRemoteModels: false`.
Remote model identifiers and runtime downloads therefore fail closed.

Pipelines are cached by absolute model path inside the utility process. Because
the main-process service owns one worker for the whole app, all windows reuse
the same loaded pipeline. A failed load is evicted so a repaired local cache can
be retried without restarting the app.

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

The Node test harness uses the committed synthetic npm golden, exercises two
renderer clients against one service, traps all Fetch and HTTP(S) calls during
inference, verifies pipeline reuse, and asserts that neither process log capture
contains any synthetic identifier.
