/**
 * Renderer page script of the packaged harness.
 *
 * It only reaches OpenMed through `window.openmed` (the production bridge)
 * and reports value-free outcomes back to the main process: span offsets,
 * labels, bundle references and error codes. It never logs or stores text.
 */
import type { ElectronDeidentifyResponse, ElectronDeidentifyTask } from "../../src/ipc";

interface HarnessBridge {
  isolation: { sandboxed: boolean; contextIsolated: boolean };
  onCommand(listener: (command: HarnessCommand) => void): void;
  reply(message: unknown): void;
}

type HarnessCommand =
  | { id: string; op: "probe" }
  | { id: string; op: "submit"; text: string }
  | { id: string; op: "cancel"; target: string }
  | { id: string; op: "navigate" }
  | { id: string; op: "attack"; channel: string; payload: unknown };

declare global {
  interface Window {
    openmed?: { submit(text: string): ElectronDeidentifyTask };
    openmedHarness?: HarnessBridge;
    openmedAttack?: { invoke(channel: string, payload: unknown): Promise<unknown> };
    __openmedInlineRan?: boolean;
  }
}

const harness = window.openmedHarness;
const tasks = new Map<string, ElectronDeidentifyTask>();
let cspViolations = 0;
document.addEventListener("securitypolicyviolation", () => {
  cspViolations += 1;
});

function codeOf(error: unknown): string {
  const code = (error as { code?: unknown } | null)?.code;
  return typeof code === "string" ? code : "UNTYPED_ERROR";
}

function outcome(response: ElectronDeidentifyResponse) {
  return {
    ok: true,
    requestId: response.requestId,
    spans: response.spans,
    bundle: response.bundle,
  };
}

async function probe() {
  const scope = globalThis as unknown as Record<string, unknown>;
  let evalBlocked = false;
  try {
    (0, eval)("1");
  } catch {
    evalBlocked = true;
  }
  let functionBlocked = false;
  try {
    new Function("return 1")();
  } catch {
    functionBlocked = true;
  }
  const inline = document.createElement("script");
  inline.textContent = "window.__openmedInlineRan = true;";
  document.body.appendChild(inline);
  await new Promise<void>((resolve) => setTimeout(resolve, 50));
  const fetchBlocked = await fetch("https://example.invalid/model.onnx").then(
    () => false,
    () => true,
  );
  const popup = window.open("https://example.invalid/", "_blank");
  return {
    requireType: typeof scope.require,
    processType: typeof scope.process,
    moduleType: typeof scope.module,
    bufferType: typeof scope.Buffer,
    ipcRendererExposed: "ipcRenderer" in scope,
    bridgeKeys: Object.keys(window.openmed ?? {}).sort(),
    sandboxed: harness?.isolation.sandboxed === true,
    contextIsolated: harness?.isolation.contextIsolated === true,
    evalBlocked,
    functionBlocked,
    inlineScriptBlocked: window.__openmedInlineRan !== true,
    fetchBlocked,
    windowOpenBlocked: popup === null,
    cspViolations,
    origin: `${location.protocol}//${location.host}`,
  };
}

harness?.onCommand((command) => {
  void (async () => {
    switch (command.op) {
      case "probe":
        harness.reply({ id: command.id, op: "probe", probe: await probe() });
        return;
      case "submit": {
        const openmed = window.openmed;
        if (openmed === undefined) {
          harness.reply({ id: command.id, op: "result", ok: false, errorCode: "NO_BRIDGE" });
          return;
        }
        const task = openmed.submit(command.text);
        tasks.set(command.id, task);
        harness.reply({ id: command.id, op: "submitted", requestId: task.requestId });
        // `contextBridge` copies errors without their `code`; `outcome` keeps it.
        task.outcome.then(
          (response) =>
            harness.reply(
              response.ok
                ? { id: command.id, op: "result", ...outcome(response) }
                : { id: command.id, op: "result", ok: false, errorCode: response.errorCode },
            ),
          (error: unknown) =>
            harness.reply({ id: command.id, op: "result", ok: false, errorCode: codeOf(error) }),
        );
        return;
      }
      case "cancel": {
        const task = tasks.get(command.target);
        const acknowledged = task === undefined ? false : await task.cancel();
        harness.reply({ id: command.id, op: "cancelled", acknowledged });
        return;
      }
      case "navigate":
        harness.reply({ id: command.id, op: "navigating" });
        location.href = "https://example.invalid/";
        return;
      case "attack": {
        const attack = window.openmedAttack;
        if (attack === undefined) {
          harness.reply({ id: command.id, op: "attack", threw: true, response: null });
          return;
        }
        try {
          const response = (await attack.invoke(command.channel, command.payload)) as Record<
            string,
            unknown
          >;
          harness.reply({
            id: command.id,
            op: "attack",
            threw: false,
            response: {
              ok: response?.ok ?? null,
              errorCode: response?.errorCode ?? null,
              cancelled: response?.cancelled ?? null,
            },
          });
        } catch {
          harness.reply({ id: command.id, op: "attack", threw: true, response: null });
        }
        return;
      }
    }
  })();
});

harness?.reply({ id: "ready", op: "ready" });
