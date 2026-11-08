/**
 * Utility-process entrypoint of the packaged harness.
 *
 * It runs the production OpenMed utility loop (`startUtilityProcess`) with the
 * production bundle verifier. Only the pipeline loader is synthetic: it
 * refuses to build a pipeline unless the verified bundle bytes contain the
 * label map and the selected runtime's entrypoint.
 */
import dns from "node:dns";
import net from "node:net";

import {
  startUtilityProcess,
  type UtilityParentPortLike,
  type VerifiedElectronBundle,
} from "../../src/utility-runtime";
import { createSyntheticPipeline } from "./synthetic-model";

function probeNetwork(): void {
  const attempts: Array<() => unknown> = [
    () => globalThis.fetch("https://example.invalid/model.onnx"),
    () => net.connect(443, "example.invalid"),
    () => dns.lookup("example.invalid", () => undefined),
  ];
  for (const attempt of attempts) {
    let blocked = false;
    try {
      attempt();
    } catch {
      blocked = true;
    }
    if (!blocked) {
      throw new Error("network probe was not blocked");
    }
  }
}

async function loadBundlePipeline(bundle: VerifiedElectronBundle) {
  const labels = bundle.files.get("id2label.json");
  const entrypoint = bundle.files.get(bundle.runtime.entrypoint);
  if (labels === undefined || entrypoint === undefined || entrypoint.length === 0) {
    throw new Error("verified bundle is incomplete");
  }
  JSON.parse(new TextDecoder().decode(labels));
  return createSyntheticPipeline({
    crash: () => {
      process.crash();
      throw new Error("unreachable");
    },
    probeNetwork,
  });
}

const parentPort = (process as NodeJS.Process & { parentPort?: UtilityParentPortLike })
  .parentPort;
if (parentPort) {
  startUtilityProcess(parentPort, { loadBundlePipeline });
}
