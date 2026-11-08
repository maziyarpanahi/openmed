// Static registry of the shared parity corpora bundled into the app.
//
// Metro can only bundle statically imported JSON, so each corpus declared in
// `parity/corpus.manifest.json` needs one import line here. To add a newer
// vector file (for example the #3642 kit-parity or #3643 portable-policy
// vectors), move its entry from `pending` to `corpora` in the manifest with
// its canonical digest, then import it below under the same id.

import manifestJson from "../../parity/corpus.manifest.json";
import offsetContract from "../../../../../tests/fixtures/parity/offset_contract.json";

export const CORPUS_MANIFEST: unknown = manifestJson;

export const CORPUS_REGISTRY: Readonly<Record<string, unknown>> = Object.freeze({
  offset_contract: offsetContract,
});
