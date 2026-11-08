// De-identify one note with the OpenMed REST client and handle service errors.
//
// Build the client first (`npm run build`), then either import `run` and pass
// your own `fetch`, or run this file against a local OpenMed service:
//
//   OPENMED_BASE_URL=http://localhost:8080 node examples/deidentify.mjs
//
// Outside this repository, import from "@openmed/rest-client" instead of
// "../dist/index.js".
import { pathToFileURL } from "node:url";

import { OpenMedApiError, OpenMedClient } from "../dist/index.js";

// Temporary service conditions that are safe to retry later.
const RETRYABLE_CODES = new Set(["rate_limited", "service_busy", "timeout"]);

export async function run({
  fetch,
  baseUrl = "http://localhost:8080",
  text = "Patient Jane Doe, MRN 0000000, was seen on 2020-01-01.",
} = {}) {
  const client = new OpenMedClient({ baseUrl, fetch });
  try {
    const result = await client.deidentify({ text, method: "mask" });
    return {
      ok: true,
      deidentifiedText: result.deidentified_text,
      redactedCount: result.num_entities_redacted,
    };
  } catch (error) {
    if (!(error instanceof OpenMedApiError)) {
      throw error;
    }
    // Decide on the stable error code. The message is for people and may
    // change between releases, so never match on it.
    return {
      ok: false,
      code: error.code,
      status: error.status,
      retry: RETRYABLE_CODES.has(error.code),
    };
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const outcome = await run({
    fetch: globalThis.fetch,
    baseUrl: process.env.OPENMED_BASE_URL ?? "http://localhost:8080",
  });
  // Print only counts and codes, never the note itself.
  console.log(
    outcome.ok
      ? `redacted ${outcome.redactedCount} entities`
      : `failed with ${outcome.code} (HTTP ${outcome.status}), retry: ${outcome.retry}`,
  );
  if (!outcome.ok) process.exitCode = 1;
}
