# Synthetic Electron redaction app

This reference keeps the raw synthetic note on-device, sends it only over the
app's private IPC path to one shared Electron utility process, and returns only
span offsets, labels, and confidence scores. Use `contextIsolation`, keep Node
integration disabled in the renderer, and authorize every IPC sender.

Install the public de-identification package, build the repository integration,
and add it as a workspace or local package:

```bash
npm install openmed electron
npm --prefix ../openmed/js/openmedkit-electron ci
npm --prefix ../openmed/js/openmedkit-electron run build
npm install ../openmed/js/openmedkit-electron
```

Bundle `@openmed/openmedkit-electron/utility-process` as
`openmed-utility-process.cjs` with the app's main-process build. The model cache
must already contain a local Transformers.js export.

## Main process

```ts
import {
  app,
  BrowserWindow,
  ipcMain,
  protocol,
  session,
  utilityProcess,
} from "electron/main";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import {
  ElectronDeidentifyService,
  OPENMED_RENDERER_CSP,
  createHardenedWebPreferences,
  hardenOpenMedSession,
  hardenOpenMedWebContents,
  registerElectronDeidentifyIpc,
} from "@openmed/openmedkit-electron";

const APP_ORIGIN = "app://openmed";
protocol.registerSchemesAsPrivileged([
  { scheme: "app", privileges: { standard: true, secure: true } },
]);
await app.whenReady();

const offlineSession = session.fromPartition("openmed-offline-inference", {
  cache: false,
});
const service = new ElectronDeidentifyService({
  utilityProcess,
  offlineSession,
  workerPath: join(__dirname, "openmed-utility-process.cjs"),
  modelPath: join(
    app.getPath("userData"),
    "openmed",
    "models",
    "OpenMed-PII-ClinicalE5-Small-33M-v1-onnx-android",
  ),
});

// Serve the renderer from a local app scheme with the OpenMed CSP.
const rendererSession = session.fromPartition("openmed-renderer");
const rendererFiles = new Set(["/index.html", "/renderer.js"]);
rendererSession.protocol.handle("app", async (request) => {
  const { host, pathname } = new URL(request.url);
  if (host !== "openmed" || !rendererFiles.has(pathname)) {
    return new Response(null, { status: 404 });
  }
  return new Response(await readFile(join(__dirname, "renderer", pathname)), {
    headers: {
      "content-type": pathname.endsWith(".html") ? "text/html" : "text/javascript",
      "content-security-policy": OPENMED_RENDERER_CSP,
    },
  });
});
hardenOpenMedSession(rendererSession, { allowedOrigins: [APP_ORIGIN] });

const window = new BrowserWindow({
  webPreferences: createHardenedWebPreferences({
    preload: join(__dirname, "preload.cjs"),
    partition: "openmed-renderer",
  }),
});
hardenOpenMedWebContents(window.webContents, { allowedOrigins: [APP_ORIGIN] });
const unregister = registerElectronDeidentifyIpc(ipcMain, service, {
  isTrustedSender: (event) =>
    event.sender === window.webContents &&
    event.senderFrame === window.webContents.mainFrame,
});
await window.loadURL(`${APP_ORIGIN}/index.html`);

app.once("before-quit", () => {
  unregister();
  service.dispose();
});
```

`registerElectronDeidentifyIpc` refuses any sender whose web preferences are
relaxed, even when `isTrustedSender` accepts it, so always build windows with
`createHardenedWebPreferences`. To run a signed portable bundle instead of an
unverified model directory, pass `bundle` in place of `modelPath` (see
[Electron In-Process De-identification](../../docs/runtimes/electron.md)).

Create the service once, outside the window factory. Every window then shares
the same utility process and its in-memory pipeline cache.

## Preload

```ts
import { contextBridge, ipcRenderer } from "electron/renderer";
import { createElectronDeidentifyClient } from "@openmed/openmedkit-electron/renderer";

contextBridge.exposeInMainWorld(
  "openmed",
  createElectronDeidentifyClient(ipcRenderer),
);
```

Sandboxed preloads can only `require("electron")`, so bundle the preload into
one CommonJS file (for example with esbuild, `--platform=browser
--format=cjs --external:electron --external:node:*`; the renderer client never
reaches the `openmed` package's Node-only fallbacks).

## Renderer

```ts
import { redactTextWithSpans } from "@openmed/openmedkit-electron/renderer";

const note =
  "Patient Alice Nguyen, DOB 1979-04-12, email alice@example.org.";
const task = window.openmed.submit(note);
// `outcome` never rejects, so its error code survives the context bridge.
const outcome = await task.outcome;
if (outcome.ok) {
  document.querySelector("#redacted")!.textContent = redactTextWithSpans(
    note,
    outcome.spans,
  );
} else {
  document.querySelector("#status")!.textContent = outcome.errorCode;
}
```

Call `task.cancel()` to drop a request; it settles with `CANCELLED`.

The rendered output is:

```text
Patient [PERSON], DOB [DATE_OF_BIRTH], email [EMAIL].
```

Do not log the request, the source note, rejected inference values, or model
errors. The service logger emits only fixed event names, safe error codes, and
span counts. Keep `offlineSession` dedicated to this one service: its request
interceptor cancels every Electron-network request made by the utility process.
