// Synthetic same-origin adapter for the installable-playground browser spec.
// Detection runs in a module worker that reads a synthetic rule bundle and a
// tiny WebAssembly helper, so the offline profile exercises worker, WASM and
// model-bundle assets. It contains no model weights and no real identifiers.

export async function createOpenMedPipeline(options) {
  if (options?.task !== "token-classification" || !options.modelUrl) {
    throw new Error("unsupported-options");
  }
  const worker = new Worker(new URL("./synthetic-worker.js", import.meta.url), {
    type: "module",
  });
  const pending = new Map();
  let sequence = 0;
  worker.addEventListener("message", (event) => {
    const { error, id, result } = event.data ?? {};
    const entry = pending.get(id);
    if (!entry) {
      return;
    }
    pending.delete(id);
    if (error) {
      entry.reject(new Error(error));
    } else {
      entry.resolve(result);
    }
  });

  function call(message) {
    sequence += 1;
    const id = sequence;
    return new Promise((resolve, reject) => {
      pending.set(id, { reject, resolve });
      worker.postMessage({ ...message, id });
    });
  }

  await call({
    modelUrl: options.modelUrl,
    type: "load",
    wasmUrl: new URL("./synthetic-scorer.wasm", import.meta.url).href,
  });
  return (text) => call({ text, type: "detect" });
}
