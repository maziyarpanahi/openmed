# Synthetic Electron redaction app

This reference keeps the raw synthetic note inside the renderer that owns it,
runs inference in one shared Electron utility process, and sends only span
offsets, labels, and confidence scores back across IPC. Use `contextIsolation`
and keep Node integration disabled in the renderer.

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
import { app, BrowserWindow, ipcMain, utilityProcess } from "electron/main";
import { join } from "node:path";
import {
  ElectronDeidentifyService,
  registerElectronDeidentifyIpc,
} from "@openmed/openmedkit-electron";

await app.whenReady();

const service = new ElectronDeidentifyService({
  utilityProcess,
  workerPath: join(__dirname, "openmed-utility-process.cjs"),
  modelPath: join(
    app.getPath("userData"),
    "openmed",
    "models",
    "privacy-filter-transformersjs",
  ),
});
const unregister = registerElectronDeidentifyIpc(ipcMain, service);

const window = new BrowserWindow({
  webPreferences: {
    contextIsolation: true,
    nodeIntegration: false,
    preload: join(__dirname, "preload.cjs"),
  },
});
await window.loadFile("index.html");

app.once("before-quit", () => {
  unregister();
  service.dispose();
});
```

Create the service once, outside the window factory. Every window then shares
the same utility process and its in-memory pipeline cache.

## Preload

```ts
import { contextBridge, ipcRenderer } from "electron/renderer";
import { createElectronDeidentifyClient } from "@openmed/openmedkit-electron";

contextBridge.exposeInMainWorld(
  "openmed",
  createElectronDeidentifyClient(ipcRenderer),
);
```

## Renderer

```ts
import { redactTextWithSpans } from "@openmed/openmedkit-electron";

const note =
  "Patient Alice Nguyen, DOB 1979-04-12, email alice@example.org.";
const result = await window.openmed.deidentify(note);
document.querySelector("#redacted")!.textContent = redactTextWithSpans(
  note,
  result.spans,
);
```

The rendered output is:

```text
Patient [PERSON], DOB [DATE_OF_BIRTH], email [EMAIL].
```

Do not log the request, the source note, rejected inference values, or model
errors. The service logger emits only fixed event names, safe error codes, and
span counts.
