/**
 * Sandboxed preload of the packaged harness.
 *
 * `window.openmed` is exactly the production bridge: the renderer client with
 * `deidentify` and `submit`. `window.openmedHarness` is the test-only control
 * channel the main process uses to drive scenarios; it carries commands in
 * and value-free outcomes (offsets, labels, codes) out.
 */
import { contextBridge, ipcRenderer } from "electron";

import { createElectronDeidentifyClient } from "../../src/ipc";

const HARNESS_COMMAND = "openmed-harness:command";
const HARNESS_REPLY = "openmed-harness:reply";

contextBridge.exposeInMainWorld("openmed", createElectronDeidentifyClient(ipcRenderer));
contextBridge.exposeInMainWorld("openmedHarness", {
  isolation: {
    sandboxed: (process as NodeJS.Process & { sandboxed?: boolean }).sandboxed === true,
    contextIsolated: (process as NodeJS.Process & { contextIsolated?: boolean })
      .contextIsolated === true,
  },
  onCommand(listener: (command: unknown) => void): void {
    ipcRenderer.on(HARNESS_COMMAND, (_event, command: unknown) => listener(command));
  },
  reply(message: unknown): void {
    ipcRenderer.send(HARNESS_REPLY, message);
  },
});
