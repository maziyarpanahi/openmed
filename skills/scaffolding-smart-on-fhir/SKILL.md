---
name: scaffolding-smart-on-fhir
description: "Scaffold a SMART-on-FHIR app (SMART App Launch v2 — EHR launch and standalone launch, OAuth2 PKCE, scopes, token handling, fhirContext) so an OpenMed-powered tool can run inside Epic or Cerner/Oracle Health. Covers the .well-known/smart-configuration discovery, authorize/token sequence, scopes like patient/DocumentReference.rs and launch/patient, and fetching clinical notes the app then de-identifies and runs NER on locally with OpenMed. Use when the user wants to embed OpenMed inside an EHR, mentions SMART on FHIR, OAuth2 launch, scopes, Epic/Cerner app, or clinician-facing FHIR app. Pairs adjacent."
license: Apache-2.0
metadata:
  project: OpenMed
  category: fhir-interop
  pairs: adjacent
  version: "1.0"
---

# Scaffolding SMART on FHIR

To put an OpenMed-powered tool *inside* a clinician's EHR (Epic, Cerner/Oracle
Health), you build a **SMART on FHIR** app: a web app the EHR launches with an
OAuth2 flow, granting scoped, time-limited access to the patient's FHIR data.
The app fetches the clinical notes, then runs OpenMed **on-device** (de-id +
NER) — so PHI is processed locally and only de-identified output, if anything,
leaves the browser/host.

## When to use

Reach for this when the deliverable is a clinician-facing app embedded in an
EHR, or a standalone app authorizing against an EHR's FHIR endpoint. Triggers:
"SMART on FHIR", "EHR launch", "OAuth2 scopes", "Epic/Cerner app",
"embed OpenMed in the chart". For pulling notes at *cohort* scale (no UI), use
`exporting-bulk-fhir` instead.

## Two launch flows

- **EHR launch** — clinician clicks your app in the chart. The EHR opens your
  `launch_uri?iss=<fhir-base>&launch=<opaque>`; you complete OAuth2 and inherit
  the current patient/encounter context.
- **Standalone launch** — user opens your app directly; it discovers the FHIR
  server and runs OAuth2, and the user/EHR picks the patient.

Both use **SMART App Launch v2**: OAuth2 authorization code flow with **PKCE**
(required in v2), discovered via `.well-known/smart-configuration`.

## Quick start: the launch sequence

```
1. EHR launch URL:
   GET https://app.example/launch?iss=https://ehr.example/fhir&launch=abc123

2. Discover endpoints:
   GET https://ehr.example/fhir/.well-known/smart-configuration
   -> { "authorization_endpoint": ".../authorize",
        "token_endpoint": ".../token",
        "code_challenge_methods_supported": ["S256"],
        "capabilities": ["launch-ehr","client-public","context-ehr-patient", ...] }

3. Redirect the browser to authorize (PKCE + the launch token):
   GET .../authorize?
       response_type=code&
       client_id=YOUR_CLIENT_ID&
       redirect_uri=https://app.example/callback&
       scope=launch openid fhirUser patient/DocumentReference.rs patient/Patient.r&
       state=RANDOM&
       aud=https://ehr.example/fhir&
       launch=abc123&
       code_challenge=BASE64URL(SHA256(verifier))&
       code_challenge_method=S256

4. Callback -> exchange code for token:
   POST .../token
       grant_type=authorization_code&code=...&redirect_uri=...&
       client_id=...&code_verifier=ORIGINAL_VERIFIER
   -> { "access_token": "...", "token_type": "Bearer", "expires_in": 3600,
        "scope": "patient/DocumentReference.rs ...",
        "patient": "Patient-123", "encounter": "Encounter-9",
        "id_token": "..." }

5. Call FHIR with the token:
   GET https://ehr.example/fhir/DocumentReference?patient=Patient-123&type=clinical-note
       Authorization: Bearer <access_token>
```

The token response carries the launch context (`patient`, sometimes
`encounter`, and in v2 a `fhirContext` array). Use `patient` to scope every
subsequent query.

## Scopes you actually need

SMART v2 scopes are `<level>/<Resource>.<permissions>` where permissions are a
subset of **`c r u d s`** (create/read/update/delete/search) — `.rs` = read +
search. Request the **minimum**:

| Scope | Why |
| --- | --- |
| `launch` | EHR launch context (omit for standalone; use `launch/patient`) |
| `openid fhirUser` | Identify the launching user |
| `patient/Patient.r` | The in-context patient demographics |
| `patient/DocumentReference.rs` | Read + search the patient's clinical notes |
| `patient/Condition.rs` | (optional) reconcile against existing problems |
| `offline_access` | (optional) refresh token for background work |

Prefer `patient/…` (current-patient) over `user/…` (everything the user can see)
to keep the blast radius small. Granular v2 scopes (`.rs`) are stricter than the
v1 `.read`/`.write` forms — use them.

## Where OpenMed runs

Notes arrive as `DocumentReference` → `content.attachment` (often base64 or a
`url` to a `Binary`). Decode, then process **locally**:

```python
import base64, openmed

note_b64 = document_reference["content"][0]["attachment"]["data"]
note = base64.b64decode(note_b64).decode("utf-8")

# De-identify on-device before anything else touches it
deid = openmed.deidentify(note, method="replace", policy="hipaa_safe_harbor")

# Clinical NER on the (de-identified or raw, per your IRB) text
entities = openmed.analyze_text(deid.text, model_name="disease_detection_superclinical")
# -> render highlights in the SMART app UI, or export FHIR (exporting-to-fhir)
```

OpenMed models run on-device after a one-time download — no note text is sent to
a third party by OpenMed. Keep the access token and any PHI in memory only; do
not log them.

## Hand-off to / from OpenMed

- **From the EHR to OpenMed:** fetched `DocumentReference` notes →
  `openmed.deidentify` → `openmed.analyze_text`.
- **From OpenMed back to the EHR:** built FHIR resources
  (`exporting-to-fhir`) → `to_bundle` (`assembling-fhir-bundles`) → write back
  with a write scope (e.g. `patient/Condition.c`) if your use case persists
  findings. Validate first (`validating-us-core`).
- **MCP option:** if the app calls a local OpenMed MCP server, the tools are
  `openmed_analyze_text` and `openmed_deidentify` — same on-device guarantees.

## Edge cases & gotchas

- **PKCE is mandatory in v2** and for public (browser) clients always. Generate
  a fresh `code_verifier` per launch; never reuse.
- **Validate `state` and `aud`.** Reject the callback if `state` does not match;
  set `aud` to the FHIR base or the EHR will reject the authorize request.
- **Tokens are short-lived.** Handle `expires_in`; use `offline_access` +
  refresh tokens only if you genuinely need background access, and store them
  securely (never client-side for confidential clients).
- **Scope down-grade is normal.** The EHR may grant fewer scopes than requested;
  read the returned `scope` and degrade gracefully.
- **Don't persist PHI in the browser.** Process in memory; if you must cache,
  cache the de-identified output only.
- **App registration is per-EHR.** Epic (fhir.epic.com) and Cerner each have
  their own developer portals, client registration, and sandbox FHIR endpoints;
  test against the sandbox before go-live.
- **OpenMed stays local.** The OAuth2 token authorizes *FHIR* calls to the EHR;
  it has nothing to do with OpenMed, which needs no network at inference time.

## Standards & references

- SMART App Launch v2: https://hl7.org/fhir/smart-app-launch/
- Scopes & launch context: https://hl7.org/fhir/smart-app-launch/scopes-and-launch-context.html
- `.well-known/smart-configuration`: https://hl7.org/fhir/smart-app-launch/conformance.html
- OAuth2 PKCE (RFC 7636): https://datatracker.ietf.org/doc/html/rfc7636
- Epic on FHIR: https://fhir.epic.com/
- Oracle Health (Cerner) FHIR: https://fhir.cerner.com/
