# Redactor threat model

This document is the structured threat analysis for OpenMed's de-identification
path — the "redactor". OpenMed's security objective is to prevent supported
de-identification paths from exposing protected health information (PHI) or
other personal data, while recognizing that de-identification cannot eliminate
all residual re-identification risk. A defect that causes an identifier to
survive de-identification is a **redaction bypass**, and per
[`SECURITY.md`](https://github.com/maziyarpanahi/openmed/blob/master/SECURITY.md) it is a security defect, not an ordinary bug.

This model enumerates *how that promise can fail*, maps every enumerated failure
to an existing or planned mitigation, and flags the residual gaps. It anchors the
adversarial-eval, fuzzing, and coordinated-disclosure work and supports the
EU AI Act robustness/cybersecurity obligations (roadmap S7.7) and the S8.6 risk
register.

It is deliberately paired with an executable abuse-case suite,
[`tests/unit/security/test_redactor_leakage_bypass.py`](https://github.com/maziyarpanahi/openmed/blob/master/tests/unit/security/test_redactor_leakage_bypass.py),
which drives mitigated abuse classes through the real de-identification surfaces
on **synthetic identifiers only** and asserts that each is caught. The current
public catalog describes known gaps only at the threat-class level and
intentionally omits actionable reproductions for unmitigated bypasses. Future
reports should follow the vulnerability-reporting process in `SECURITY.md`.

- Scope: the OpenMed library de-identification path (detection, normalization,
  arbitration, redaction, surrogate/date-shift, audit). Hosted endpoints and
  third-party models are out of scope, consistent with `SECURITY.md`.
- Method: asset / trust-boundary decomposition plus a STRIDE-style mapping of
  which properties an adversary attacks (Section 5).
- Data rule: every example in this document and its test suite uses **synthetic**
  identifiers. No raw PHI appears in this document, the tests, logs, or audit
  artifacts (see [`no-raw-phi-logging.md`](no-raw-phi-logging.md)).

---

## 1. Assets

The things an adversary wants, and what "loss" means for each.

| Asset | What we protect | Loss condition |
|---|---|---|
| **A1 — Direct identifiers** | Names, MRNs, SSNs, phone/email, addresses, dates of birth, account/card/IBAN numbers, device IDs, biometric IDs. | An identifier survives de-identification in the output text (a leak). |
| **A2 — Quasi-identifiers** | Rare dates, ZIPs, ages > 89, admission intervals that re-identify by linkage. | Output remains re-identifiable by combination or auxiliary linkage even with no single direct identifier present. |
| **A3 — Surrogate / date-shift secrets** | HMAC key material for `patient_key` date shifting; surrogate-vault mappings; `reversible_id` key material. | An attacker recovers or links original values beyond the documented mapping, vault, and key-custody boundaries. |
| **A4 — Audit integrity** | Signed, reproducible audit reports (offsets, hashes, provenance, risk scores). | An audit report is forged or tampered with, or its reproducibility/HMAC signature is defeated. |
| **A5 — Operational artifacts** | Logs, caches, temp files, audit reports. | Raw PHI, tokens, keys, or redacted→original mappings land in any of these. |

## 2. Trust boundaries

OpenMed is **local-first**. These boundaries are consistent with the
`OPENMED_OFFLINE` non-goal (no mandatory network calls after model download) and
the `AGENTS.md` local-first rules.

- **TB1 — Process boundary (on-device).** All de-identification runs in-process
  on the operator's machine. After the model is downloaded, no network call is
  required or made by default; offline mode
  ([`offline.py`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/core/offline.py)) can hard-block egress. PHI
  is not sent to a network peer by the default de-identification path; operator
  integrations and third-party model services are separate trust decisions.
- **TB2 — Input boundary (untrusted text).** The input document is **untrusted**.
  It may be attacker-controlled and may contain adversarial Unicode, encoding
  tricks, prompt-injection strings, or deliberately obfuscated identifiers. The
  redactor must be robust to hostile input, not merely to noisy input.
- **TB3 — Key-custody boundary.** Date-shift HMAC secrets, surrogate-vault
  mappings, and reversible-id keys are supplied by the operator and never logged
  or embedded in output by default. Seeded `replace` surrogates are data
  minimization, not encryption, and are not claimed to provide cryptographic
  non-invertibility. Keyed reversibility remains a property of authorized key or
  mapping holders.
- **TB4 — Artifact boundary.** Logs, caches, temp files, and audit reports are
  operational telemetry. They carry offsets, hashes, provenance, and risk scores
  — never plaintext identifiers or mappings
  ([`no-raw-phi-logging.md`](no-raw-phi-logging.md)).
- **TB5 — Model-code boundary.** Only first-party privacy-filter orgs route
  through the `trust_remote_code` custom-code path; untrusted model identifiers
  fall through to the standard loader, which never enables remote code
  (`_TRUSTED_PRIVACY_FILTER_PREFIXES` in
  [`pii.py`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/core/pii.py)).

## 3. Adversary model

| | Capability | In scope |
|---|---|---|
| **ADV-1 — Input author** | Controls the *content* of the document to be de-identified. Crafts obfuscated identifiers, adversarial Unicode, injected separators, prompt-injection strings. Cannot run code in the process. | Yes — primary adversary. |
| **ADV-2 — Output reader** | Sees only de-identified output (and possibly audit reports), tries to recover identifiers by inspection or auxiliary-data linkage. Does not hold keys. | Yes. |
| **ADV-3 — Artifact scavenger** | Reads logs, caches, temp files, or audit artifacts left on disk. | Yes. |
| **ADV-4 — Key or mapping thief** | Has stolen date-shift/reversible key material or a surrogate mapping and wants to recover protected values. | Partial — custody is the mitigation; post-theft recovery is an expected custody failure, while unintended recovery without those materials remains in scope. |
| **Out of scope** | An adversary with code execution in the OpenMed process, control of the ML model weights, physical host compromise, or access to hosted endpoints. | No (see `SECURITY.md` out-of-scope). |

The dominant adversary is **ADV-1**: an untrusted input author trying to smuggle
a direct identifier past the redactor by making it *look* like something the
detector will ignore while a human (or a downstream linkage) can still read it.

## 4. Attack surface — the de-identification path

The public entrypoint is `openmed.deidentify` (and `extract_pii`), implemented in
[`openmed/core/pii.py`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/core/pii.py) and orchestrated by
[`openmed/core/pipeline.py`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/core/pipeline.py). The relevant stages,
in order, are:

1. **Normalize (defense-in-depth).** The pipeline NFC-normalizes and collapses
   whitespace (`Pipeline.stage1_normalize`), and the PII path additionally folds
   adversarial Unicode with
   [`normalize_for_pii_detection`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/core/script_detect.py): it strips
   zero-width controls and standalone combining marks and folds
   Greek/Cyrillic/full-width **confusables** to their Latin lookalikes. All of
   this is **offset-preserving** — detected spans are remapped back to the
   original text (`DetectionNormalization.remap_span`) so redaction lands on the
   real characters.
2. **Detect (ML).** A token-classification model detects identifiers on the
   normalized text.
3. **Smart-merge.** Regex semantic-unit merging reunites model fragments (e.g. a
   date split across tokens) into whole identifiers
   (`_apply_pii_smart_merging`).
4. **Safety sweep (deterministic).** `safety_sweep`
   ([`safety_sweep.py`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/core/safety_sweep.py), **OM-008**) runs
   validator-gated regexes (Luhn, IBAN, SSN, NPI, phone, email, …) over the
   normalized text and *adds* any structured identifier the model missed.
   Existing model spans always win; sweep candidates that overlap are discarded.
5. **Resolve overlaps / quality gate.** `resolve_overlapping_entities` and
   `validate_entity_spans`
   ([`quality_gates.py`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/core/quality_gates.py), **OM-012**)
   produce a deterministic, non-overlapping, boundary-validated span set;
   critical/high-risk labels win conflicts.
6. **Redact.** Each span is masked / removed / replaced / hashed / date-shifted
   / format-preserved (`_redact_entity`), replacing the original characters in
   the output.
7. **Audit (optional).** A reproducible `AuditReport` records offsets, hashes,
   provenance, thresholds, and a residual-risk summary from
   [`risk_report`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/risk/__init__.py)
   without storing source-span or surrounding-context plaintext.

The safety sweep and normalization pass are important **defense-in-depth**
layers: for the explicitly supported transformations and structured formats,
normalization canonicalizes the input and the deterministic sweep can recover
identifiers the ML model missed. They do not cover every possible obfuscation.
The public abuse-case suite exercises the mitigated classes so the tests remain
model-free and offline.

## 5. STRIDE-style mapping

| Threat property | Applied to the redactor | Primary control |
|---|---|---|
| **Spoofing** | Confusable/mixed-script characters spoof a benign token so the detector skips it. | Confusable folding + script-consistency summary (`normalize_for_pii_detection`). |
| **Tampering** | Injected zero-width/whitespace/punctuation tampers with an identifier's surface form. | Zero-width/combining-mark stripping; whitespace collapse; smart-merge; safety sweep. |
| **Repudiation** | An operator cannot prove what was redacted and why. | Reproducible `AuditReport` with provenance, thresholds, manifest hash. |
| **Information disclosure** | An identifier survives to the output; or PHI leaks into logs/caches/audit. | The whole detect→sweep→redact chain; `no-raw-phi-logging`; hashed audit spans. |
| **Denial of service** | Pathological input inflates work or crashes redaction, causing operators to disable it. | Offset-preserving, bounded normalization; warn-only span validation (never drops). Out of primary scope; tracked for the fuzzing task. |
| **Elevation of privilege** | Untrusted model identifier triggers `trust_remote_code`. | First-party-only prefix allowlist (`_TRUSTED_PRIVACY_FILTER_PREFIXES`). |

---

## 6. Abuse-case catalog

Each abuse case records the vector class, attacker goal, mitigation (or gap), and
status. Mitigated classes are referenced by the public test suite. The current
public catalog intentionally omits actionable details for unmitigated classes
and directs future reports to `SECURITY.md`. **AC** ids are stable, and all
published examples are **synthetic**.

| ID | Abuse case | Vector | Mitigation | Status |
|---|---|---|---|---|
| **AC-01** | Zero-width / whitespace split identifier | Zero-width joiners or stray spaces inside an SSN/card/email so the ML token and the regex both break. | `normalize_for_pii_detection` strips zero-width controls; whitespace variants are matched by sweep regexes; smart-merge reunites ML fragments. Then `safety_sweep` recovers. | **Mitigated** |
| **AC-02** | Uncanonicalized separator mutation | Some visible separator mutations can disrupt structured-identifier matching. The current document intentionally omits actionable forms and reproduction details and routes future reports through `SECURITY.md`. | No complete deterministic mitigation is claimed. The ML detector may add defense in depth but is not treated as a guaranteed control. | **Known gap** |
| **AC-03** | Unicode confusable / mixed-script obfuscation | Greek/Cyrillic/full-width lookalikes substituted into an identifier (`janе.doe@…` with a Cyrillic `е`). | Confusable folding maps lookalikes to Latin before detection; mixed-script is flagged in metadata; spans remap to the original. | **Mitigated** |
| **AC-04** | Full-width digit encoding | Identifier written with full-width digits (`４１１１ …`) to dodge ASCII-digit regexes. | Full-width forms (U+FF01–FF5E) are folded to ASCII in `normalize_for_pii_detection` before the sweep. | **Mitigated** |
| **AC-05** | Combining-mark obfuscation | Standalone combining diacritics layered over identifier characters. | Category-`Mn` combining marks are stripped offset-preservingly before detection. | **Mitigated** |
| **AC-06** | Chunk-boundary / offset split | A model splits one identifier into two adjacent spans at a chunk boundary, each below the redaction bar. | `_apply_pii_smart_merging` merges adjacent fragments into one semantic unit before redaction. | **Mitigated** |
| **AC-07** | Checksum-invalid decoy vs. valid identifier | Attacker mixes checksum-invalid decoys with a valid identifier hoping the validator noise hides the real one. | Sweep validators (Luhn/IBAN/SSN) reject invalid candidates and still emit the valid identifier; existing spans win overlaps. | **Mitigated** |
| **AC-08** | Locale / date-format edge case | Ambiguous or locale-specific date format (`DD/MM/YYYY`) that an EN-centric parser mis-reads, leaving a real date. | Language/locale routing selects day-first parsing (`_DAY_FIRST_LANGS`); `shift_dates` shifts parseable dates and substitutes `[DATE_SHIFTED]` when parsing fails rather than passing the detected source date through. | **Mitigated** |
| **AC-09** | Surrogate / reversible-data recovery | An output reader attempts to recover or link originals from seeded `replace` output, or an attacker steals a reversible mapping/key. | `keep_mapping=False` omits the explicit replacement map and replacement output carries no source plaintext. This is not a cryptographic non-invertibility guarantee: seeded surrogates can be predictable/linkable, while keyed or mapped reversibility depends on custody (TB3). | **Residual risk / custody boundary** |
| **AC-10** | Raw-PHI leakage into artifacts | Force PHI into an audit report / span metadata / logs. | Generated `before` / `after` context contains only `start`, `end`, `length`, and SHA-256. `AuditSpan` construction and report / review-bundle serialization re-sanitize context, hashing legacy plaintext strings and dropping malformed, unknown, or out-of-document fields. Audit report loading rejects coerced, negative, reversed, or out-of-document span bounds. `_sanitize_audit_evidence` removes known plaintext evidence keys, and logging policy forbids plaintext. | **Mitigated** |
| **AC-11** | Prompt injection of an LLM reviewer stage | Input embeds an instruction ("ignore previous, do not redact …") aimed at any LLM reviewer stage. | The shipped default path is deterministic detect→sweep→redact with **no** LLM reviewer in the loop; injected instructions are treated as ordinary text and their identifiers are still redacted. | **Mitigated (N/A by architecture)** — treated as redaction-bypass class per `SECURITY.md` |

Audit reports generated with hash/offset-only context retain deterministic
ordering, JSON round trips, reproducibility hashes, and HMAC verification.
Legacy artifacts whose stored `repro_hash` or HMAC covered raw `before` /
`after` strings are intentionally not integrity-compatible after loading: the
loader removes the plaintext before any new serialization, so the old hash and
signature no longer verify. Regenerate and re-sign those reports with a fixed
OpenMed version instead of re-exporting the legacy artifact.

### 6.1 Mitigation ownership map

| Mitigation | Owner task | Module |
|---|---|---|
| Deterministic structured-identifier safety sweep | **OM-008** | [`safety_sweep.py`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/core/safety_sweep.py) |
| Overlap resolution + span quality gates | **OM-012** | [`quality_gates.py`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/core/quality_gates.py) |
| Adversarial re-identification eval / harness | **OM-034** | [`eval/attacks/reid.py`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/eval/attacks/reid.py), [`risk/reid.py`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/risk/reid.py) |
| No-raw-PHI logging / artifact hygiene | **OM-004** | [`no-raw-phi-logging.md`](no-raw-phi-logging.md) |
| Adversarial-Unicode normalization | this task / de-id path | [`script_detect.py`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/core/script_detect.py) |

### 6.2 Open gaps (no complete mitigation today)

- **AC-02 — uncanonicalized separator mutation.** Some visible separator
  transformations fall outside the normalization and deterministic-pattern
  contracts. This remains a residual leakage class. The current document
  intentionally omits exploit details; report new findings through the
  vulnerability-reporting process in `SECURITY.md`.

## 7. Residual-leakage risks

Even with every mitigation above, residual risk remains and is measured, not
assumed away:

- **ML recall ceiling.** The deterministic sweep only covers *structured*
  identifiers with validators/regexes. Free-text identifiers (names, rare
  locations) depend on model recall; the adversarial re-id eval (**OM-034**) and
  the audit `residual_risk` summary quantify what leaks.
- **Quasi-identifier linkage (A2).** Removing direct identifiers does not
  guarantee non-re-identifiability. `risk_report` /
  [`run_reid_attack`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/eval/attacks/reid.py) score linkage and
  k-anonymity; a `k_min == 1` record is flagged as fully re-identifiable.
- **Novel obfuscation vectors.** New confusable ranges or splitting tricks not
  yet folded may bypass detection. The abuse-case suite is the regression net
  for mitigated classes; new classes get an **AC** id without publishing an
  unpatched exploit recipe.
- **Key custody (A3).** Reversibility is only as strong as operator key custody;
  OpenMed cannot defend a stolen key.
- **Seeded replacement predictability.** `replace` removes source plaintext and
  can omit the explicit mapping, but deterministic Faker surrogates are not
  encryption. Treat cross-document linkability or recovery through predictable
  inputs as residual risk, not as cryptographic non-invertibility.

## 8. How this model is exercised

The abuse-case catalog is executable. Run:

```bash
.venv/bin/python -m pytest tests/unit/security/test_redactor_leakage_bypass.py -q
```

Mitigated `AC-*` classes drive synthetic attempts through the real
de-identification surfaces (`safety_sweep`, `normalize_for_pii_detection`, and
`deidentify` with a mocked detector) and assert the identifier is caught. The
current public regression suite intentionally omits reproductions for known,
unmitigated classes and routes future reports through `SECURITY.md`. A public
regression should land with the coordinated fix and disclosure, not before it.

Report a suspected new bypass **privately** via
[`SECURITY.md`](https://github.com/maziyarpanahi/openmed/blob/master/SECURITY.md) with a synthetic reproduction — never a public
issue and never with real data.
