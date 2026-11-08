/**
 * Preload for the harness's "compromised renderer" window.
 *
 * It exposes raw `ipcRenderer.invoke` so the page can send hand-crafted
 * payloads on the OpenMed channels. The main process must refuse every
 * payload that names a model path, bundle, runtime or policy, and every
 * attempt to cancel another renderer's request.
 */
import { contextBridge, ipcRenderer } from "electron";

const HARNESS_COMMAND = "openmed-harness:command";
const HARNESS_REPLY = "openmed-harness:reply";

contextBridge.exposeInMainWorld("openmedAttack", {
  invoke(channel: string, payload: unknown): Promise<unknown> {
    return ipcRenderer.invoke(channel, payload);
  },
});
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
