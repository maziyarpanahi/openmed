// Node.js example: de-identify a synthetic clinical note with the openmed
// npm package and print the redacted text plus the detected span labels.
//
// This file lives in the OpenMed repository, so it imports the built package
// from ../../js/openmedkit-web/dist/index.js (run `npm run build` in
// js/openmedkit-web first). In your own project, install the package with
// `npm install openmed` and import from "openmed" instead.
//
// Tests always pass a stub `pipeline`, so they run offline. Running this
// file directly loads a real model with loadOnnxModel, which downloads the
// model artifacts once (about 70 MB) before inferring locally.

import { pathToFileURL } from "node:url";

import {
  DEFAULT_MODEL_ID,
  deidentify,
  loadOnnxModel,
} from "../../js/openmedkit-web/dist/index.js";

export const SYNTHETIC_NOTE =
  "Patient Alice Nguyen, DOB 1979-04-12, email alice@example.org.";

/**
 * De-identify `text` with an injected token-classification pipeline.
 *
 * Prints the redacted text and the span labels (never the original
 * identifier values) and returns the full de-identification result.
 */
export async function run({ pipeline, text = SYNTHETIC_NOTE, print = console.log } = {}) {
  const result = await deidentify(text, { pipeline });
  print(result.deidentifiedText);
  print(result.spans.map((span) => span.canonical_label).join(", "));
  return result;
}

const invokedDirectly =
  process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;

if (invokedDirectly) {
  // Loads (and, on first run, downloads) the default OpenMed ONNX model.
  const pipeline = await loadOnnxModel(DEFAULT_MODEL_ID);
  await run({ pipeline });
}
