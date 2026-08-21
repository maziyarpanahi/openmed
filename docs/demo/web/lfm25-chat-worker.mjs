import { createOpenMedLfm25ChatRuntime } from "./lfm25-chat-web-adapter.mjs";

let runtime = null;
let generationController = null;

self.addEventListener("message", (event) => {
  const message = event.data ?? {};
  if (message.type === "load") void loadRuntime(message);
  if (message.type === "generate") void generate(message);
  if (message.type === "cancel") generationController?.abort("Generation stopped");
  if (message.type === "dispose") void dispose(message.requestId);
});

async function loadRuntime(message) {
  try {
    await releaseRuntime();
    const transformers = await import("./vendor/transformers.web.min.js");
    runtime = await createOpenMedLfm25ChatRuntime({
      ...message.options,
      onProgress: (progress) => postMessage({
        progress,
        requestId: message.requestId,
        type: "progress",
      }),
      transformers,
    });
    postMessage({
      details: runtime.details(),
      requestId: message.requestId,
      type: "loaded",
    });
  } catch (error) {
    await releaseRuntime();
    postFailure(message.requestId, error);
  }
}

async function generate(message) {
  if (!runtime) {
    postFailure(message.requestId, new Error("LFM2.5 is not loaded"));
    return;
  }
  if (generationController) {
    postFailure(message.requestId, new Error("LFM2.5 is already generating"));
    return;
  }
  generationController = new AbortController();
  try {
    const events = runtime.generate(message.messages, {
      ...message.generation,
      signal: generationController.signal,
    });
    for await (const value of events) {
      postMessage({ requestId: message.requestId, type: "delta", value });
    }
    postMessage({ requestId: message.requestId, type: "complete" });
  } catch (error) {
    postFailure(message.requestId, error);
  } finally {
    generationController = null;
  }
}

async function dispose(requestId) {
  await releaseRuntime();
  postMessage({ requestId, type: "disposed" });
  self.close();
}

async function releaseRuntime() {
  generationController?.abort("Runtime released");
  generationController = null;
  const candidate = runtime;
  runtime = null;
  await candidate?.dispose?.();
}

function postFailure(requestId, error) {
  postMessage({
    error: {
      message: error instanceof Error ? error.message : String(error),
      name: error?.name === "AbortError" ? "AbortError" : "Error",
    },
    requestId,
    type: "error",
  });
}
