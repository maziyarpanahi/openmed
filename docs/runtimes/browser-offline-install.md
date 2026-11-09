# Installable offline browser profile

The [privacy playground](/docs/demo/privacy-playground/) can keep a verified
copy of itself for offline use. The profile is opt-in: nothing is registered or
stored until a user selects **Prepare offline use**, and a visitor who never
selects it gets the ordinary online page. The page ships a web app manifest
(`manifest.webmanifest`), so browsers that offer installation can also add it
as a standalone app.

The profile owns application packaging only. It does not replace the browser
inference runtime or the model cache, and it bundles no model weights.

## What is stored

`docs/demo/privacy-playground/offline-assets.json` is the versioned digest
manifest (schema `openmed.offline-assets.v1`). Each entry pins one static file
by relative URL, kind (`shell`, `runtime` or `model`), media type, byte length
and SHA-256 digest. The generation identifier is a SHA-256 over the profile
name, start URL, service-worker digest and every entry, so any changed byte
in an asset or in the worker logic starts a new generation.

The service worker `offline-sw.js` embeds the same table. It caches nothing
else:

- Only same-origin `GET` requests without a query string, whose URL exactly
  matches a manifest entry, are answered from the cache. The start URL is the
  only navigation it answers.
- `POST` and other methods, query-bearing URLs, range requests, cross-origin
  requests and every unlisted path are not intercepted at all.
- No request made by the page is ever written to a cache. Entered text,
  redaction results, adapter requests and documents with user data therefore
  never reach Cache Storage. Messages between the page and the worker carry
  fixed reason codes, counts and the generation digest only.

## Preparation and verification

Preparing the profile registers the worker, which then:

1. downloads `offline-assets.json` without the HTTP cache and requires its
   SHA-256 to equal the digest embedded in the worker, so a worker and a
   manifest from different deployments are never combined
   (`generation-mismatch`);
2. checks `navigator.storage.estimate()` for room for a complete copy;
3. downloads every asset (same origin, no redirects, no HTTP cache), checks its
   byte length and digest, and stores it in a cache named for the generation;
4. writes a completion marker and verifies the stored copy again.

Any failure deletes the partial cache and rejects installation. The page shows
**Preparation failed** with a reason code, such as `asset-fetch-failed`,
`asset-digest-mismatch`, `insufficient-storage` or `quota-exceeded`. The page
also asks for persistent storage on this explicit action; browsers may decline,
and the profile then reports best-effort storage.

After preparation, reopen the playground without a query string. It then runs
from the verified copy with no network connection.

## Updates switch generations atomically

A new deployment changes the worker bytes, so the browser installs it as a new
worker beside the active one. The new generation is downloaded and verified in
its own cache while the previous generation keeps serving. If the update is
interrupted, a file is corrupted or the deployment is mixed, the new worker is
discarded with its cache and the page reports **Update rejected; current copy
kept**.

A verified update waits until every playground tab has closed. **Switch to
verified update** activates it immediately only when a single tab is open; the
tab then reloads into the new generation, so no page ever loads files from two
generations. Activation deletes the previous generation. Each response the
worker serves carries an `X-OpenMed-Offline-Generation` header.

## Repair and eviction

Every navigation re-verifies the shell and runtime files, and every served file
is digest-checked before it is returned; model files are checked when served.
If a file is missing or corrupted, or the browser evicted the cached copy to
free storage, the start URL returns a self-contained repair page (HTTP 503 with
`X-OpenMed-Offline-State: repair`) instead of the playground:

| Code | Meaning |
| --- | --- |
| `offline-copy-evicted` | The browser removed the cached copy. |
| `offline-copy-incomplete` | Preparation did not finish. |
| `asset-missing` | A stored file is missing. |
| `asset-digest-mismatch` | A stored file no longer matches its digest. |
| `asset-size-mismatch` | A stored file has an unexpected size. |

**Repair offline copy** downloads and verifies the same generation again and
needs a connection to the publishing site. **Remove offline copy** deletes the
caches and unregisters the worker. A failure detected while the page is open
locks the redaction controls and shows the same repair actions.

## Add assets or a model bundle

Self-hosted deployments can pin additional files, such as a same-origin adapter
module, its worker and WASM files (`runtime`) or an approved model bundle
(`model`). Add each file to the `assets` list of `offline-assets.json` with a
`url` and `kind`, place the files under the site root, then regenerate the
digests:

```bash
python scripts/web/build_offline_manifest.py
python scripts/web/build_offline_manifest.py --check
```

Use `--root` and `--site-root` to build a profile for another deployment tree.
The tool rejects queries, fragments, schemes, encoded characters, inner dot
segments, symlinks and files outside the site root. Its messages name only the
relative URL and a fixed reason. `tests/browser/test_offline_asset_manifest.py`
fails when the committed profile is stale, so changes to any pinned file, such
as the shared stylesheets, need the regeneration step.

## Validation

```bash
.venv/bin/python -m pytest tests/browser/test_offline_asset_manifest.py -q
make docs-stage
npm ci --prefix tests/browser/brand
npm --prefix tests/browser/brand test -- installable-playground.spec.ts --project=chromium
```

The shared browser-test configuration starts the staged Pages preview, so stage
the site first; the spec itself serves its own private copy. It runs the
stdlib-only generator and egress checker with `python3`; set `OPENMED_PYTHON` to
use another interpreter.

The Playwright spec stages a private copy of the playground behind a
fault-injecting local server. It covers offline reopen after a browser restart
with the server stopped, interrupted, corrupted and mixed updates, at-rest
corruption, cache eviction, a Chromium quota override, and request interception
that runs the [network-egress checker](../demo/browser-egress.md) on the
inference trace. The inference path includes a synthetic adapter that runs a
module worker, a WASM helper and a synthetic rule bundle from the offline copy.
The spec runs on Chromium only, because the quota override and the
service-worker response provenance it asserts use Chromium protocols.

This profile is a demonstration packaging contract. It does not certify that
text is de-identified and must not be used to make clinical decisions.
