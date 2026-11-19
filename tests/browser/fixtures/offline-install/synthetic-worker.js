// Synthetic detection worker for the installable-playground browser spec.
// The span end offset is computed by a 41-byte WebAssembly "end" export
// (start + length) so the offline profile proves WASM assets load from cache.

let rules = [];
let scorer = null;

self.addEventListener("message", async (event) => {
  const { id, type } = event.data ?? {};
  try {
    if (type === "load") {
      const [model, wasm] = await Promise.all([
        fetch(new URL("rules.json", event.data.modelUrl)).then((response) => {
          if (!response.ok) {
            throw new Error("model-unavailable");
          }
          return response.json();
        }),
        fetch(event.data.wasmUrl).then((response) => {
          if (!response.ok) {
            throw new Error("wasm-unavailable");
          }
          return response.arrayBuffer();
        }),
      ]);
      rules = model.patterns.map(({ label, pattern }) => ({
        label,
        pattern: new RegExp(pattern, "g"),
      }));
      scorer = (await WebAssembly.instantiate(wasm)).instance.exports;
      self.postMessage({ id, result: { ready: true } });
      return;
    }
    if (type === "detect") {
      const spans = [];
      for (const rule of rules) {
        rule.pattern.lastIndex = 0;
        for (const match of String(event.data.text).matchAll(rule.pattern)) {
          spans.push({
            end: scorer.end(match.index, match[0].length),
            entity_group: rule.label,
            score: 1,
            start: match.index,
          });
        }
      }
      self.postMessage({ id, result: spans });
    }
  } catch {
    self.postMessage({ error: "synthetic-worker-failed", id });
  }
});
