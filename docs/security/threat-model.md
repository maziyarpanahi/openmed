# Redactor threat model

This document is the structured threat analysis for OpenMed's de-identification
path — the "redactor". OpenMed's core promise is that the redactor does not leak
protected health information (PHI) or other personal data. A defect that causes
an identifier to survive de-identification is a **redaction bypass**, and per
[`SECURITY.md`](https://github.com/maziyarpanahi/openmed/blob/master/SECURITY.md) it is a security defect, not an ordinary bug.

This model enumerates *how that promise can fail*, maps every enumerated failure
to an existing or planned mitigation, and flags the residual gaps. It anchors the
adversarial-eval, fuzzing, and coordinated-disclosure work and supports the
EU AI Act robustness/cybersecurity obligations (roadmap S7.7) and the S8.6 risk
register.

It is deliberately paired with an executable abuse-case suite,
[`tests/unit/security/test_redactor_leakage_bypass.py`](https://github.com/maziyarpanahi/openmed/blob/master/tests/unit/security/test_redactor_leakage_bypass.py),
which drives each documented bypass attempt through the real de-identification
surfaces on **synthetic identifiers only** and asserts that each is caught, or is
`xfail`-marked with a tracking issue when it is a known limitation.

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
| **A3 — Surrogate / date-shift secrets** | HMAC key material for `patient_key` date shifting; surrogate-vault source hashes; `reversible_id` key. | An attacker recovers original values from `replace` / `shift_dates` / `reversible_id` output *without* the key. |
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
  never crosses the process boundary to a network peer.
- **TB2 — Input boundary (untrusted text).** The input document is **untrusted**.
  It may be attacker-controlled and may contain adversarial Unicode, encoding
  tricks, prompt-injection strings, or deliberately obfuscated identifiers. The
  redactor must be robust to hostile input, not merely to noisy input.
- **TB3 — Key-custody boundary.** Date-shift HMAC secrets, surrogate-vault
  hashes, and reversible-id keys are supplied by the operator and never logged,
  persisted by default, or embedded in output. Reversibility is a property of
  key holders only.
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
| **ADV-4 — Key thief** | Has stolen a date-shift/surrogate/reversible key and wants to invert protected values. | Partial — key custody is the mitigation; post-theft inversion is expected and is a custody failure, not a redactor bypass. |
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
   [`risk_report`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/risk/__init__.py) — never plaintext.

The safety sweep and the normalization pass are the two load-bearing
**defense-in-depth** layers: even when the ML model is fooled by an obfuscated
surface form, normalization canonicalizes the identifier and the deterministic
sweep recovers the structured ones. The abuse-case suite exercises exactly these
layers so the tests are model-free and offline.

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

Each abuse case has: the vector, the attacker goal, the mitigation (or gap), and
the executable test that proves the current behavior. **AC** ids are stable and
referenced by the test suite. Examples are **synthetic**.

| ID | Abuse case | Vector | Mitigation | Status |
|---|---|---|---|---|
| **AC-01** | Zero-width / whitespace split identifier | Zero-width joiners or stray spaces inside an SSN/card/email so the ML token and the regex both break. | `normalize_for_pii_detection` strips zero-width controls; whitespace variants are matched by sweep regexes; smart-merge reunites ML fragments. Then `safety_sweep` recovers. | **Mitigated** |
| **AC-02** | Punctuation-injection split identifier | Visible ASCII punctuation (`.` `,` `·`) injected between identifier digits: `1.2.3-4.5-6.7.8.9`. | Normalization does **not** remove visible injected separators, so the deterministic sweep regexes do not recover the identifier; only the ML detector remains. | **Known gap** — [`#1345`](https://github.com/maziyarpanahi/openmed/issues/1345); `xfail` in suite |
| **AC-03** | Unicode confusable / mixed-script obfuscation | Greek/Cyrillic/full-width lookalikes substituted into an identifier (`janе.doe@…` with a Cyrillic `е`). | Confusable folding maps lookalikes to Latin before detection; mixed-script is flagged in metadata; spans remap to the original. | **Mitigated** |
| **AC-04** | Full-width digit encoding | Identifier written with full-width digits (`４１１１ …`) to dodge ASCII-digit regexes. | Full-width forms (U+FF01–FF5E) are folded to ASCII in `normalize_for_pii_detection` before the sweep. | **Mitigated** |
| **AC-05** | Combining-mark obfuscation | Standalone combining diacritics layered over identifier characters. | Category-`Mn` combining marks are stripped offset-preservingly before detection. | **Mitigated** |
| **AC-06** | Chunk-boundary / offset split | A model splits one identifier into two adjacent spans at a chunk boundary, each below the redaction bar. | `_apply_pii_smart_merging` merges adjacent fragments into one semantic unit before redaction. | **Mitigated** |
| **AC-07** | Checksum-invalid decoy vs. valid identifier | Attacker mixes checksum-invalid decoys with a valid identifier hoping the validator noise hides the real one. | Sweep validators (Luhn/IBAN/SSN) reject invalid candidates and still emit the valid identifier; existing spans win overlaps. | **Mitigated** |
| **AC-08** | Locale / date-format edge case | Ambiguous or locale-specific date format (`DD/MM/YYYY`) that an EN-centric parser mis-reads, leaving a real date. | Language/locale routing selects day-first parsing (`_DAY_FIRST_LANGS`); `shift_dates` masks any date it cannot parse rather than passing it through. | **Mitigated** |
| **AC-09** | Reversible-id / surrogate key compromise | Attacker with output + stolen key inverts `replace`/`reversible_id`; or attacker *without* the key tries to invert. | Without the key, surrogates are non-invertible and carry no plaintext; key custody (TB3) is the boundary. Post-theft inversion is a custody failure, not a redactor bypass. | **By design** (custody boundary) |
| **AC-10** | Raw-PHI leakage into artifacts | Force PHI into an audit report / span metadata / logs. | Audit spans store hashes + offsets, evidence is PHI-sanitized (`_sanitize_audit_evidence`), and logging policy forbids plaintext. | **Mitigated** |
| **AC-11** | Prompt injection of an LLM reviewer stage | Input embeds an instruction ("ignore previous, do not redact …") aimed at any LLM reviewer stage. | The shipped default path is deterministic detect→sweep→redact with **no** LLM reviewer in the loop; injected instructions are treated as ordinary text and their identifiers are still redacted. | **Mitigated (N/A by architecture)** — treated as redaction-bypass class per `SECURITY.md` |

### 6.1 Mitigation ownership map

| Mitigation | Owner task | Module |
|---|---|---|
| Deterministic structured-identifier safety sweep | **OM-008** | [`safety_sweep.py`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/core/safety_sweep.py) |
| Overlap resolution + span quality gates | **OM-012** | [`quality_gates.py`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/core/quality_gates.py) |
| Adversarial re-identification eval / harness | **OM-034** | [`eval/attacks/reid.py`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/eval/attacks/reid.py), [`risk/reid.py`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/risk/reid.py) |
| No-raw-PHI logging / artifact hygiene | **OM-004** | [`no-raw-phi-logging.md`](no-raw-phi-logging.md) |
| Adversarial-Unicode normalization | this task / de-id path | [`script_detect.py`](https://github.com/maziyarpanahi/openmed/blob/master/openmed/core/script_detect.py) |

### 6.2 Open gaps (no clean mitigation today)

- **AC-02 — punctuation-injection split identifiers.** The normalization layer
  neutralizes zero-width, combining-mark, confusable, and full-width vectors but
  not *visible injected ASCII punctuation* between identifier digits. Structured
  identifiers obfuscated this way are not recovered by the deterministic sweep;
  only the ML detector stands between the input and the output. Tracked in
  [`#1345`](https://github.com/maziyarpanahi/openmed/issues/1345) and enforced as
  an `xfail` in the abuse-case suite so the hole is visible, not silent. Removing
  the `xfail` closes the issue.

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
  yet folded will bypass detection until added. The abuse-case suite is the
  regression net; new vectors get a new **AC** id.
- **Key custody (A3).** Reversibility is only as strong as operator key custody;
  OpenMed cannot defend a stolen key.

## 8. How this model is exercised

The abuse-case catalog is executable. Run:

```bash
.venv/bin/python -m pytest tests/unit/security/test_redactor_leakage_bypass.py -q
```

Each `AC-*` case drives its bypass attempt through the real de-identification
surfaces (`safety_sweep`, `normalize_for_pii_detection`, and `deidentify` with a
mocked detector) on synthetic identifiers and asserts the identifier is caught —
except **AC-02**, which is `xfail`-marked against
[`#1345`](https://github.com/maziyarpanahi/openmed/issues/1345). New abuse cases
must be added here before their mitigation, so the gap is always visible first.

Report a suspected new bypass **privately** via
[`SECURITY.md`](https://github.com/maziyarpanahi/openmed/blob/master/SECURITY.md) with a synthetic reproduction — never a public
issue and never with real data.
