// Synthetic inference stand-in: a hand-assembled WebAssembly `add` module, run on
// the main thread and in a same-origin worker. It touches no model or user text.
const WASM_ADD = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 7, 1, 96, 2, 127, 127, 1, 127, 3, 2, 1, 0, 7, 7, 1,
  3, 97, 100, 100, 0, 0, 10, 9, 1, 7, 0, 32, 0, 32, 1, 106, 11,
]);

// Violations are recorded as controlled codes only: no URLs, samples or payloads.
window.__violations = [];
document.addEventListener("securitypolicyviolation", (event) => {
  window.__violations.push(event.effectiveDirective);
});

async function wasmAdd() {
  try {
    const { instance } = await WebAssembly.instantiate(WASM_ADD);
    return { ok: instance.exports.add(2, 3) === 5 };
  } catch (error) {
    return { ok: false, error: error.name };
  }
}

function workerWasmAdd() {
  return new Promise((resolve) => {
    const worker = new Worker("/csp-profile/worker.js");
    worker.onmessage = (event) => resolve(event.data);
    worker.onerror = () => resolve({ ok: false, error: "WorkerError" });
    worker.postMessage(Array.from(WASM_ADD));
  });
}

async function crossOriginFetch() {
  try {
    await fetch("https://csp-blocked.example/collect", { method: "POST", body: "x" });
    return { refused: false };
  } catch {
    return { refused: true };
  }
}

window.__run = async () => {
  const main = await wasmAdd();
  const worker = await workerWasmAdd();
  const fetched = await crossOriginFetch();
  // Violation events are queued tasks; let them flush before reading.
  await new Promise((resolve) => setTimeout(resolve, 50));
  return { main, worker, fetch: fetched, violations: window.__violations.slice() };
};
