// Worker-thread entry used by tests/web/test_inference_worker.spec.ts.
//
// It serves the synthetic parity vectors through the real worker server and
// can simulate a runtime crash that escapes the server (an uncaught exception
// whose message embeds the input text) to exercise host-side failure handling.
import { readFileSync } from "node:fs";
import { parentPort } from "node:worker_threads";

import { serveInferenceWorker } from "../../../js/openmedkit-web/dist/index.js";

// Keep in sync with CRASH_MARKER in test_inference_worker.spec.ts.
const CRASH_MARKER = "[crash-worker]";

const { vectors } = JSON.parse(
  readFileSync(new URL("./inference_worker_parity_vectors.json", import.meta.url), "utf8"),
);

if (parentPort === null) {
  throw new Error("inference worker fixture must run in a worker thread");
}

serveInferenceWorker({
  scope: parentPort,
  load(spec) {
    return {
      backend: "wasm-basic",
      localOnly: spec.localOnly,
      modelDigest: "sha256:synthetic-thread-fixture",
      pipeline(text) {
        if (text.startsWith(CRASH_MARKER)) {
          setTimeout(() => {
            throw new Error(`synthetic runtime crash while reading ${text}`);
          }, 0);
          return new Promise(() => undefined);
        }
        const vector = vectors.find((candidate) => candidate.text === text);
        return structuredClone(
          vector?.batches ?? vector?.tokens ?? [],
        );
      },
    };
  },
});
