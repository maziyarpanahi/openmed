---
name: auditing-deid-leakage
description: "Adversarially scan already-de-identified clinical text for residual identifiers and emit a leakage report that blocks release on any hit. Use after OpenMed de-identification when the user asks to verify a redaction, prove no PHI/PII leaked, gate a dataset before sharing, or run a second-pass detector. Covers format and checksum detectors (SSN, Luhn for card numbers, MRN/account patterns, emails, phones, dates), entropy heuristics for high-randomness tokens, severity scoring, and a hard block-on-leak rule. This is the verification half of OpenMed's leakage-first ethos. Hand-off: re-run openmed.extract_pii on the de-id output and diff against expectations. License-free, local-first. Pairs after deidentifying-clinical-text."
license: Apache-2.0
metadata:
  project: OpenMed
  category: de-identification
  pairs: after
  version: "1.0"
---

# Auditing de-id leakage

De-identification is **verified, not assumed**. A model-driven redaction can
miss a structured identifier (an SSN typo'd with spaces, an account number in a
footer, a date in an odd format) — and a single residual identifier defeats the
whole release. This skill is the adversarial second pass: scan the *output* of
de-identification for anything that still looks like an identifier, score it, and
**block release on any leak**. It is the verification half of OpenMed's
leakage-first ethos — gate on leakage, not on F1.

## When to use

- Right after `deidentifying-clinical-text`, before the de-identified text leaves
  a trust boundary (export, share, train, publish).
- When the user wants proof that "no PHI leaked," a release gate, or a CI check
  that fails the build if any identifier survives.
- As a belt-and-suspenders detector independent of the model that produced the
  redaction — a deterministic checker catches different failures than the NER.

Run this on the **de-identified** text, not the original. The original is
expected to be full of identifiers.

## Quick start

Two complementary passes — a deterministic structural scan plus a model
second-pass diff:

```python
import re
import openmed

# Synthetic — the de-identified OUTPUT we are auditing for residual leaks.
deid_text = "Patient [NAME] seen on [DATE]. Backup contact 415-555-0184; acct 4111111111111111."

def luhn_ok(digits: str) -> bool:
    nums = [int(d) for d in digits]
    nums[-2::-2] = [(2 * d - 9 if 2 * d > 9 else 2 * d) for d in nums[-2::-2]]
    return sum(nums) % 10 == 0

DETECTORS = {
    "SSN":   (r"\b\d{3}-\d{2}-\d{4}\b", "critical", None),
    "EMAIL": (r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", "high", None),
    "PHONE": (r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "high", None),
    "DATE":  (r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "medium", None),
    "MRN":   (r"\bMRN[:#\s]*\d{5,}\b", "high", None),
    "CARD":  (r"\b(?:\d[ -]?){13,19}\b", "critical", luhn_ok),  # checksum-gated
}

findings = []
for label, (pattern, severity, checksum) in DETECTORS.items():
    for m in re.finditer(pattern, deid_text, flags=re.IGNORECASE):
        token = m.group()
        if checksum and not checksum(re.sub(r"\D", "", token)):
            continue  # fails Luhn -> not a real card number, skip
        findings.append({"label": label, "severity": severity,
                         "start": m.start(), "end": m.end()})  # offsets, not text

# Second-pass model detector: re-run PII extraction on the de-id output.
residual = openmed.extract_pii(deid_text)          # PredictionResult
for ent in residual.entities:
    findings.append({"label": ent.label, "severity": "high",
                     "start": ent.start, "end": ent.end})

leaked = bool(findings)
print({"leak": leaked, "count": len(findings)})    # report carries NO plaintext
assert not leaked, "Release BLOCKED: residual identifiers detected."
```

Note what the report records: **labels, severities, and offsets — never the
leaked plaintext**. Echoing the leaked identifier into a report or log re-creates
the exact PHI exposure you are auditing for.

## Workflow

1. **Run deterministic format + checksum detectors** on the de-identified text:
   SSN, email, phone, dates, MRN/account/ID patterns, and card numbers gated by
   the **Luhn checksum** so random 16-digit strings don't false-positive. These
   catch structured identifiers a model may skip.
2. **Add an entropy heuristic** for high-randomness tokens (long base36/base64
   strings, hex blobs) that match no known format but look like keys, tokens, or
   record locators. Flag for review rather than auto-block; entropy is noisy.
3. **Run a model second-pass:** re-run `openmed.extract_pii` on the *output* and
   treat any returned entity as a residual leak. Because it's a different
   detector than the one that did the redaction, it catches different misses.
4. **Score severity.** critical (SSN, card, full DOB+name co-occurrence) > high
   (email, phone, MRN, names) > medium (partial dates) > low (entropy-only).
5. **Block on any leak.** The gate is binary for release: if `findings` is
   non-empty at high/critical, fail the export. Surface a no-PHI report
   (counts + offsets + severities) so a reviewer can locate and re-redact.

## Hand-off to / from OpenMed

- **From** `deidentifying-clinical-text`: this skill consumes
  `result.deidentified_text`. Never audit `result.original_text`.
- **OpenMed second-pass detector:** `from openmed import extract_pii` — re-run it
  on the de-id output and diff. Equivalent MCP/REST surfaces detect PII spans for
  the same purpose. Any span returned on already-de-identified text is a leak.
- **To** `reviewing-reidentification-risk`: zero direct-identifier leaks is
  necessary but not sufficient — quasi-identifiers (age + ZIP + date) can still
  re-identify. Hand a clean-on-leakage dataset to QI risk scoring next.
- **To** `evaluating-with-leakage-gates`: wire this scan into the eval harness so
  a leakage regression fails CI, not just an F1 drop.

## Edge cases & gotchas

- **Never log the leaked value.** Report offsets, labels, hashes — not the text.
  A leakage report full of plaintext SSNs is itself a breach.
- **Checksum-gate card numbers.** Apply Luhn before flagging 13–19 digit runs, or
  every order number and account id becomes a false "card leak."
- **Surrogates are not leaks.** If de-id used `method="replace"`, the output
  contains *fake* names/emails by design. The model second-pass may flag them —
  diff against the known mapping/surrogate set so you don't block on synthetic
  data. True leaks are values present in the **original** text.
- **Locale-aware dates and IDs.** `dd/mm/yyyy`, `yyyy.mm.dd`, NHS/SIN/fiscal-code
  formats vary; tune detectors to the data's locale or you under-detect.
- **Entropy is advisory.** High-entropy ≠ identifier (could be a hash already).
  Route to human review, don't hard-block on entropy alone.
- **Local-first.** Run the whole scan on-device; do not ship the text to a cloud
  scanner to check whether it leaked.

## Standards & references

- HIPAA Safe Harbor — the 18 identifier categories that must be absent:
  https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/
- Luhn algorithm (ISO/IEC 7812-1) for payment-card checksum validation:
  https://www.iso.org/standard/70484.html
- NIST SP 800-188, *De-Identification of Personal Information*:
  https://csrc.nist.gov/pubs/sp/800/188/final
