# Browser extension PHI guard

OpenMed PHI Guard is a Manifest V3 WebExtension under `web/extension/`. It
detects PHI in editable fields and page text, then lets a user mask an active
field before submitting it. Page text is visually masked in place. Detection
runs in the extension's private background context through the typed `openmed`
npm package; clinical text never leaves the browser. Chromium uses the Manifest
V3 service worker, while Firefox uses the same bundle as a non-persistent
background script.

## Install for development

Build the unpacked extension with Node.js 20 or newer:

```bash
cd web/extension
npm ci
npm run build
```

Load `web/extension/dist/` as an unpacked extension in Chromium-based browsers,
or as a temporary add-on in Firefox. Store listing and submission are outside
this package's scope.

The in-page panel provides:

- a **Mask PHI** action for the active editable field;
- a per-site enable/disable control stored in browser-local extension storage;
- HIPAA Safe Harbor, clinical minimal-redaction, and strict no-leak profiles
  sourced from OpenMed's bundled policy JSON files.

The content script also blocks form submission when an unmasked active field
has detected spans. Review the masked result before submitting it because no
automated detector guarantees complete recall.

## Offline and privacy guarantees

The manifest declares only the `storage` permission and an empty
`host_permissions` list. HTTP and HTTPS content-script matches allow the guard
to inspect editable text, but no code path sends that text over the network.
The background worker uses a compact local detector and the npm
de-identification API, and its content security policy sets
`connect-src 'none'`.

Only a site's enabled state and selected policy profile are persisted. Page
text, raw identifiers, span output, and hashes are not stored. Password fields
and non-text media are not inspected. See `web/extension/PRIVACY.md` for the
user-facing privacy notice.

## Test

Install the Playwright Chromium runtime once, then run the extension checks:

```bash
cd web/extension
npx playwright install chromium
npm test
```

The headless test loads the real unpacked extension on a synthetic page. It
asserts that editable and text-node spans match the bundled detector output,
masking produces the expected replacements, detection emits no HTTP(S)
requests, per-site disable persists across reloads, and profile selection
changes redaction behavior.

## Limitations

This first version handles text only. It does not redact images, PDFs rendered
to canvas, cross-origin frames, or browser-internal pages. It is a privacy
safeguard, not a medical device, and it must not trigger clinical decisions.
